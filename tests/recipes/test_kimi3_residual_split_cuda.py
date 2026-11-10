# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The opt-in decode geometry of the residual mixture matches the one-program-per-row kernel."""

import os

import pytest
import torch

from ignition.recipes.kimi3 import residual
from ignition.recipes.kimi3.residual import ADD, KEEP, REPLACE, ResidualStream, score_weight

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA device")


def _ordered(x):
    i = x.view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def _stream(monkeypatch, prefix, blocks, split):
    stream = ResidualStream(prefix.clone(), blocks.clone(), torch.full_like(prefix, float("nan")),
                            split_rows=residual.SPLIT_ROWS if split else 0)
    assert (stream.split is not None) == split
    return stream


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 8, 11, 16])
@pytest.mark.parametrize("sources,mode,write", [(0, REPLACE, 0), (1, ADD, -1), (1, REPLACE, -1), (3, KEEP, 5),
                                                (4, ADD, 4), (7, ADD, -1), (8, ADD, -1), (8, REPLACE, -1)])
@torch.no_grad()
def test_split_mixture_matches_row_programs(monkeypatch, rows, sources, mode, write):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(rows * 131 + sources)
    blocks = torch.randn((rows, 8, 7168), dtype=torch.bfloat16, device=device, generator=generator)
    prefix, delta = torch.randn((2, rows, 7168), dtype=torch.bfloat16, device=device, generator=generator)
    score = score_weight(torch.randn(7168, device=device, generator=generator).bfloat16(),
                         (torch.randn(7168, device=device, generator=generator) * 0.02).bfloat16())
    norm = torch.randn(7168, dtype=torch.bfloat16, device=device, generator=generator)
    results = []
    for split in (False, True):
        stream = _stream(monkeypatch, prefix, blocks, split)
        mixture = torch.full_like(prefix, float("nan"))
        output = stream.mix(delta, mode, sources=sources, write=write, score=score, norm=norm, eps=1e-5,
                            output_eps=1e-5, mixture=mixture)
        results.append((stream.prefix, stream.blocks, output.clone(), mixture))
    (prefix0, blocks0, output0, mixture0), (prefix1, blocks1, output1, mixture1) = results
    assert torch.equal(prefix0, prefix1) and torch.equal(blocks0, blocks1)
    # Full-row FP32 sums are added per chunk, so a mixture may round to the neighbouring BF16 value; its RMSNorm
    # and weight multiply each round once more. Near-zero cancellations are judged like the checkpoint test does.
    for expected, actual, ulps in ((mixture0, mixture1, 1), (output0, output1, 3)):
        distance = (_ordered(expected) - _ordered(actual)).abs()
        large = expected.float().abs() >= expected.float().abs().amax(-1, keepdim=True) / 64
        assert int(distance[large].max()) <= ulps
        difference = (actual.float() - expected.float()).abs()
        ulp = expected.float().abs().clamp(min=2 ** -14) * torch.finfo(torch.bfloat16).eps
        assert bool((difference <= 4 * ulp).all()), float((difference / ulp).max())
        assert (actual != expected).float().mean().item() < 2e-3

    # CUDA graph replays advance the never-reset row epochs and reproduce the eager result bit for bit.
    stream = _stream(monkeypatch, prefix, blocks, True)
    eager = stream.mix(delta, mode, sources=sources, write=write, score=score, norm=norm, eps=1e-5,
                       output_eps=1e-5).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream.prefix.copy_(prefix)
        stream.mix(delta, mode, sources=sources, write=write, score=score, norm=norm, eps=1e-5, output_eps=1e-5)
    reference = stream.prefix.clone() if mode == KEEP else None
    for _ in range(64):
        stream.output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(stream.output, eager)
    if reference is not None:
        assert torch.equal(stream.prefix, reference)
