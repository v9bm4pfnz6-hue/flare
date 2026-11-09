# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU equivalence checks for the verification distribution score preparation.

The candidate histogram and the score chain are part of the sampling numerics
contract: every accepted token depends on them bitwise. These tests compare the
production implementation in PreparedSpeculativeDistributions against a literal
reference of the previous full-vocabulary comparison loop, for CPU tensors,
across widths, prefix offsets, duplicate and invalid candidate ids, an inactive
row and both FP32 and BF16 logits.
"""

from __future__ import annotations

import pytest
import torch

from ignition.components.speculation import PreparedSpeculativeDistributions, _live, _sliced
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission


def example(*, batch=1, width=4, vocab=16, consumed=2, slot=2, live_count=None):
    state = DeviceRequestState.allocate("cpu", vocab_size=vocab)
    state.admit(slot, RequestAdmission(generation=7, seed=2**64 - 1, max_new_tokens=32,
        temperature=1, top_k=0, top_p=1, eos_token_ids=()), [1, 2], consumed_tokens=2,
        ple_history=(1, 2))
    state.consumed_tokens[slot] = consumed
    state.generated_tokens[slot] = 1
    state.pending_token[slot] = 3
    state.histogram[slot, 3] = 1
    slots = torch.full((batch,), -1, dtype=torch.int32)
    slots[0] = slot
    epochs = torch.full((batch,), 7, dtype=torch.int64)
    live = torch.tensor([batch if live_count is None else live_count], dtype=torch.int32)
    return state, slots, epochs, live


def reference_prepare_scores(op, logits):
    """Literal full-vocabulary reference of the previous histogram and score chain."""
    state, columns = op.state, op.columns
    safe, live = _live(state, op.slots, op.generations, op.live_rows, op.rows)
    counts = state.histogram[safe, None, :]
    prefix = columns + op.prefix_offset
    prefix_valid = torch.ones_like(op.valid)
    for index in range(columns.numel() - 1 + op.prefix_offset):
        ids = op.candidates[:, index]
        used = index < prefix
        valid = (ids >= 0) & (ids < state.vocab_size)
        prefix_valid &= valid[:, None] | ~used[None, :]
        matches = op.vocab_ids[None, None, :] == ids[:, None, None]
        additions = matches & valid[:, None, None] & used[None, :, None]
        counts = counts + additions.to(torch.int32)
    params = state.parameters[safe]
    scores = logits.float()
    penalty = params[:, None, 3:4]
    adjusted = torch.where(scores > 0, scores / penalty, scores * penalty)
    scores = torch.where(counts > 0, adjusted, scores)
    scores = scores - (counts > 0) * params[:, None, 4:5] - counts * params[:, None, 5:6]
    temperature = params[:, None, :1]
    scores = scores / torch.where(temperature > 0, temperature, 1)
    if op.grammar_masks is None:
        words = state.grammar_mask[safe][:, None, op.vocab_ids // 32]
    else:
        words = op.grammar_masks[safe][:, prefix][:, :, op.vocab_ids // 32]
    allowed = ((words >> (op.vocab_ids % 32)) & 1).bool() | ~state.grammar_enabled[safe, None, None].bool()
    scores = torch.where(allowed, scores, -torch.inf)
    maximum = _sliced(scores).amax(-1).amax(-1)
    valid = live[:, None] & prefix_valid & torch.isfinite(maximum)
    top_k = torch.where(state.top_k[safe] > 0, state.top_k[safe], state.vocab_size)[:, None].expand_as(valid).reshape(-1)
    top_p = params[:, 1:2].expand_as(valid).reshape(-1)
    return scores, maximum, valid, top_k, top_p, safe, params


def check_equivalence(width, prefix_offset, candidates, *, logits_dtype=torch.float32, vocab=16,
                      batch=1, penalty=(1.0, 1.0, 0.0, 2.0, 3.0, 0.5)):
    state, slots, epochs, live = example(batch=batch, width=width, vocab=vocab)
    state.parameters[2] = torch.tensor(penalty, dtype=torch.float32)
    torch.manual_seed(11 + width * 7 + prefix_offset)
    logits = torch.randn(batch, width, vocab).to(logits_dtype) * 3
    op = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, candidates,
        prefix_offset=prefix_offset)
    op.prepare_scores()
    scores, maximum, valid, top_k, top_p, safe, params = reference_prepare_scores(op, logits)
    assert torch.equal(op.scores, scores)
    assert torch.equal(op.maximum, maximum)
    assert torch.equal(op.valid, valid)
    assert torch.equal(op.packed_top_k, top_k)
    assert torch.equal(op.packed_top_p, top_p)


@pytest.mark.parametrize("width", [1, 2, 4, 6])
@pytest.mark.parametrize("prefix_offset", [0, 1, 2, 3])
def test_histogram_and_scores_match_full_vocabulary_reference(width, prefix_offset):
    if prefix_offset > width:
        return
    candidates = torch.tensor([[3, 4, 4, 9, 15, -1, 16, 5]])
    check_equivalence(width, prefix_offset, candidates)


@pytest.mark.parametrize("batch", [2, 4])
def test_histogram_matches_reference_per_row(batch):
    candidates = torch.tensor([[3, 3, 4, -1, 7, 31]] * batch)
    state, slots, epochs, live = example(batch=batch, width=6, vocab=32, live_count=batch - 1)
    slots[-1] = -1
    state.parameters[2] = torch.tensor([1.0, 1.0, 0.0, 2.0, 3.0, 0.5])
    torch.manual_seed(19)
    logits = torch.randn(batch, 6, 32)
    op = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, candidates)
    op.prepare_scores()
    scores, maximum, valid, top_k, top_p, safe, params = reference_prepare_scores(op, logits)
    assert torch.equal(op.scores, scores)
    assert torch.equal(op.maximum, maximum)
    assert torch.equal(op.valid, valid)
    assert bool(valid[-1].logical_not().all())


def test_histogram_matches_reference_with_bf16_logits():
    check_equivalence(6, 0, torch.tensor([[3, 4, 4, 9, 15, -1]]), logits_dtype=torch.bfloat16)


def test_histogram_matches_reference_at_serving_vocabulary_size():
    vocab = 163840
    batch, width = 2, 6
    state, slots, epochs, live = example(batch=batch, width=width, vocab=vocab)
    state.parameters[2] = torch.tensor([1.0, 1.0, 0.0, 2.0, 3.0, 0.5])
    torch.manual_seed(5)
    logits = (torch.randn(batch, width, vocab) * 2).to(torch.bfloat16)
    candidates = torch.tensor([[5, 5, 70000, 163839, -1, 12], [1, 2, 3, 4, 5, 6]])
    op = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, candidates)
    op.prepare_scores()
    scores, maximum, valid, top_k, top_p, safe, params = reference_prepare_scores(op, logits)
    assert torch.equal(op.scores, scores)
    assert torch.equal(op.maximum, maximum)
    assert torch.equal(op.valid, valid)
    assert torch.equal(op.packed_top_k, top_k)
    assert torch.equal(op.packed_top_p, top_p)