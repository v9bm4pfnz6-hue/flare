# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for draft vocabulary logit finalization."""

import pytest
import torch

from ignition.components.dflash2 import finalize_vocabulary_logits

# rows, vocabulary, multiplier, softcap. The first case is the serving draft
# block geometry; the others cover softcapped and odd vocabulary shapes.
CASES = (
    (6, 163840, 1.0, None),
    (6, 1000, 2.5, None),
    (5, 4096, 1.0, 30.0),
    (3, 777, 0.7, 15.0),
)


def _reference(gathered, logits, vocab, multiplier, softcap):
    unary = gathered[:, :vocab].float() * multiplier
    logits.copy_(unary if softcap is None else (unary / softcap).tanh() * softcap)


@pytest.mark.parametrize(("rows", "vocab", "multiplier", "softcap"), CASES)
def test_finalize_writes_the_reference_bits(rows, vocab, multiplier, softcap):
    generator = torch.Generator().manual_seed(rows * 7919 + vocab)
    gathered = torch.randn(rows, vocab, generator=generator).to(torch.bfloat16)
    gathered[0, 0] = float("nan")
    gathered[0, 1] = float("inf")
    gathered[0, 2] = float("-inf")
    gathered[-1, -1] = torch.tensor(-0.0, dtype=torch.bfloat16)
    gathered_before = gathered.clone()

    logits = torch.full((rows, vocab), float("nan"))
    reference = logits.clone()
    finalize_vocabulary_logits(gathered, logits, vocab, multiplier=multiplier, softcap=softcap)
    _reference(gathered, reference, vocab, multiplier, softcap)

    assert torch.equal(logits.view(torch.int32), reference.view(torch.int32))
    assert torch.equal(gathered.view(torch.int16), gathered_before.view(torch.int16))


def test_finalize_requires_one_fp32_row_per_gathered_row():
    gathered = torch.zeros(6, 128, dtype=torch.bfloat16)
    logits = torch.zeros(6, 128, dtype=torch.float32)
    with pytest.raises(ValueError):
        finalize_vocabulary_logits(gathered, logits, 129, multiplier=1.0, softcap=None)
