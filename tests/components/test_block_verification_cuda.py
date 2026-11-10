# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Token-wise block verification samples the target law and offers rows their real prefix."""

import itertools
import json
import math
import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import (PreparedStochasticBlockVerification, PreparedStochasticVerification,
                                             SpeculativeRngState)

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")

BATCH, VOCAB = 64, 3
TARGET = [[.70, .20, .10], [.10, .25, .65], [.35, .60, .05]]
DRAFT = [[.05, .85, .10], [.75, .15, .10], [.10, .15, .75]]


def harness(kind, width, limits=True):
    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=BATCH)
    for slot in range(BATCH):
        state.admit(slot, RequestAdmission(generation=7, seed=2**40 + 977 * slot, max_new_tokens=1000, top_k=0, top_p=1,
                                           eos_token_ids=()), [1, 2], consumed_tokens=2)
    slots = torch.arange(BATCH, device="cuda", dtype=torch.int32)
    epochs = torch.full((BATCH,), 7, device="cuda", dtype=torch.int64)
    live = torch.tensor([BATCH], device="cuda", dtype=torch.int32)
    q = torch.zeros((BATCH, width - 1, VOCAB), device="cuda")
    p = torch.zeros((BATCH, width, VOCAB), device="cuda")
    ids = torch.zeros((BATCH, width - 1), device="cuda", dtype=torch.int64)
    token_limits = torch.full((BATCH,), width, device="cuda", dtype=torch.int32) if limits else None
    rng = SpeculativeRngState.allocate("cuda", capacity=BATCH)
    rng.generation.fill_(7)
    verify = kind(q, ids, p, state, slots, epochs, live, rng=rng, token_limits=token_limits)
    verify.prepare()
    return verify, q, ids, p, rng, token_limits


def markov_rounds(kind, width, length, trials):
    """Decode `length` tokens per row over as many conditional rounds as verification needs."""
    verify, q, ids, p, rng, limits = harness(kind, width)
    target = torch.tensor(TARGET, device="cuda")
    draft = torch.tensor(DRAFT, device="cuda")
    rows = torch.arange(BATCH, device="cuda")
    observed = torch.zeros(VOCAB**length, dtype=torch.int64, device="cuda")
    emitted = rounds = rejected_then_accepted = 0
    powers = VOCAB ** torch.arange(length - 1, -1, -1, device="cuda")
    for _ in range(trials):
        pending = torch.zeros(BATCH, device="cuda", dtype=torch.int64)
        counts = torch.zeros(BATCH, device="cuda", dtype=torch.int64)
        sequence = torch.zeros((BATCH, length), device="cuda", dtype=torch.int64)
        while (counts < length).any():
            limits.copy_((length - counts).clamp(0, width))
            previous = pending
            for column in range(width):
                p[:, column] = target[previous]
                if column < width - 1:
                    q[:, column] = draft[previous]
                    ids[:, column] = torch.multinomial(draft[previous], 1)[:, 0]
                    previous = ids[:, column]
            out = verify()
            taken = out.proposed_count.long()
            live = limits > 0
            assert (out.valid == live).all()
            assert (taken[live] <= limits[live]).all()
            for row in live.nonzero()[:, 0].tolist():
                count, take = int(counts[row]), int(taken[row])
                sequence[row, count:count + take] = out.token_ids[row, :take]
                pending[row] = out.token_ids[row, take - 1]
                # The fixed-width library leaves dummy columns past a short tail.
                if kind is PreparedStochasticBlockVerification:
                    assert (out.token_ids[row, take:] == -1).all() and (out.token_ids[row, :take] >= 0).all()
                if take > 1:
                    assert (out.token_ids[row, :take - 1] == ids[row, :take - 1]).all()
            counts += taken
            emitted += int(taken.sum())
            rounds += int(live.sum())
            rng.target_offset.add_(out.rng_increment)
        observed += ((sequence * powers).sum(-1)).bincount(minlength=VOCAB**length)
    frequencies = observed.cpu().double() / observed.sum().item()
    return frequencies, emitted / rounds


def markov_law(length):
    law = []
    for tokens in itertools.product(range(VOCAB), repeat=length):
        probability, previous = 1., 0
        for token in tokens:
            probability *= TARGET[previous][token]
            previous = token
        law.append(probability)
    return torch.tensor(law, dtype=torch.float64)


@pytest.mark.parametrize("width", [2, 3, 5])
def test_block_joint_law_equals_target_and_outruns_chain(width):
    length, trials = 3, int(os.environ.get("BLOCK_LAW_TRIALS", 1500))
    frequencies, block_rate = markov_rounds(PreparedStochasticBlockVerification, width, length, trials)
    _, chain_rate = markov_rounds(PreparedStochasticVerification, width, length, trials // 3)
    samples = trials * BATCH
    error = float((frequencies - markov_law(length)).abs().max())
    bound = math.sqrt(math.log(2 * VOCAB**length / 1e-7) / (2 * samples))
    print(json.dumps(dict(width=width, samples=samples, max_joint_error=error, simultaneous_hoeffding_bound=bound,
                          tokens_per_round_block=block_rate, tokens_per_round_chain=chain_rate)))
    assert error < bound


def test_identical_distributions_accept_every_offered_draft():
    verify, q, ids, p, rng, limits = harness(PreparedStochasticBlockVerification, 5)
    law = torch.tensor([.2, .3, .5], device="cuda")
    q[:], p[:] = law, law
    ids[:] = torch.multinomial(law, BATCH * 4, replacement=True).view(BATCH, 4)
    limits[:] = torch.arange(BATCH, device="cuda", dtype=torch.int32) % 5 + 1
    out = verify()
    assert (out.accepted_drafts == limits - 1).all()
    offered = torch.arange(4, device="cuda")[None] < out.offered_drafts[:, None]
    assert (out.survival[:, :4][offered] == 1).all()


def test_zero_prefix_rejects_and_keeps_finite_residual():
    verify, q, ids, p, rng, limits = harness(PreparedStochasticBlockVerification, 5)
    q[:, :, 0] = 1.
    p[:, :, 1] = 1.
    # A later huge ratio cannot revive a zero prefix.
    q[:, 2] = torch.tensor([1e-6, 1. - 1e-6, 0.], device="cuda")
    p[:, 2] = torch.tensor([1., 0., 0.], device="cuda")
    out = verify()
    assert out.valid.all()
    assert (out.accepted_drafts == 0).all()
    assert (out.token_ids[:, 0] == 1).all()
    assert (out.survival[:, :4] == 0).all()


def test_tokens_do_not_depend_on_batch_slot_or_prefix_decisions_on_replay():
    verify, q, ids, p, rng, limits = harness(PreparedStochasticBlockVerification, 5)
    target = torch.tensor(TARGET, device="cuda")
    draft = torch.tensor(DRAFT, device="cuda")
    q[:] = draft[0]
    p[:] = target[0]
    ids[:] = 1
    rng.target_offset[:] = 12
    verify.state.seed[:] = 12345
    tokens = verify().token_ids.clone()
    assert (tokens == tokens[0]).all()
    again = verify().token_ids.clone()
    assert torch.equal(tokens, again)


def test_survival_is_non_increasing_and_bounds_chain():
    verify, q, ids, p, rng, limits = harness(PreparedStochasticBlockVerification, 5)
    draft = torch.tensor(DRAFT, device="cuda")
    target = torch.tensor(TARGET, device="cuda")
    q[:] = draft[0]
    p[:] = target[0]
    ids[:] = torch.multinomial(draft[0], BATCH * 4, replacement=True).view(BATCH, 4)
    out = verify()
    survival = out.survival[:, :4]
    assert ((survival >= 0) & (survival <= 1)).all()
    assert (survival[:, 1:] <= survival[:, :-1]).all()
    ratio = (p[:, :-1].gather(-1, ids[:, :, None]) / q.gather(-1, ids[:, :, None])).squeeze(-1).clamp(max=1.)
    assert (survival + 1e-6 >= ratio.cumprod(-1)).all()


def captured(verify, rng, *mutable):
    from ignition.preparation.compilation import compile_forward

    def run():
        verify()
        rng.target_offset.add_(verify.output.rng_increment)
    compiled = compile_forward(run)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    return graph


def test_compiled_graph_replay_samples_target_first_token_law():
    verify, q, ids, p, rng, limits = harness(PreparedStochasticBlockVerification, 5)
    draft, target = torch.tensor(DRAFT[0], device="cuda"), torch.tensor(TARGET[0], device="cuda")
    q[:], p[:] = draft, target
    graph = captured(verify, rng)
    tokens, accepted = [], []
    for _ in range(int(os.environ.get("BLOCK_REPLAYS", 2048))):
        # First-token marginal holds for any proposal history, so fresh drafts per replay.
        ids.copy_(torch.multinomial(draft, BATCH * 4, replacement=True).view(BATCH, 4))
        graph.replay()
        tokens.append(verify.output.token_ids[:, 0].clone())
        accepted.append(verify.output.accepted_drafts.clone())
    torch.cuda.synchronize()
    tokens = torch.cat(tokens)
    observed = tokens.bincount(minlength=VOCAB).cpu().double() / tokens.numel()
    error = float((observed - torch.tensor(TARGET[0], dtype=torch.float64)).abs().max())
    bound = math.sqrt(math.log(2 * VOCAB / 1e-7) / (2 * tokens.numel()))
    print(json.dumps(dict(compiled_first_token_error=error, bound=bound, samples=tokens.numel(),
                          mean_accepted=float(torch.cat(accepted).float().mean()))))
    assert error < bound
    graph.reset()


@pytest.mark.parametrize("kind", [PreparedStochasticVerification, PreparedStochasticBlockVerification])
def test_real_vocabulary_round_cost(kind):
    vocab = int(os.environ.get("BLOCK_VOCAB", 163840))
    state = DeviceRequestState.allocate("cuda", vocab_size=vocab)
    state.admit(0, RequestAdmission(generation=7, seed=99, max_new_tokens=1000, top_k=0, top_p=1, eos_token_ids=()),
                [1, 2], consumed_tokens=2)
    slots = torch.zeros(1, device="cuda", dtype=torch.int32)
    epochs = torch.full((1,), 7, device="cuda", dtype=torch.int64)
    live = torch.tensor([1], device="cuda", dtype=torch.int32)
    q = torch.rand((1, 4, vocab), device="cuda").pow(8)
    q /= q.sum(-1, keepdim=True)
    p = torch.rand((1, 5, vocab), device="cuda").pow(8)
    p /= p.sum(-1, keepdim=True)
    ids = torch.multinomial(q[0], 1).T.contiguous()
    limits = torch.full((1,), 5, device="cuda", dtype=torch.int32)
    rng = SpeculativeRngState.allocate("cuda")
    rng.generation.fill_(7)
    verify = kind(q, ids, p, state, slots, epochs, live, rng=rng, token_limits=limits)
    verify.prepare()
    graph = captured(verify, rng)
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 1000
    print(json.dumps(dict(verifier=kind.__name__, vocab=vocab, microseconds_per_round=elapsed * 1e6)))
    graph.reset()
