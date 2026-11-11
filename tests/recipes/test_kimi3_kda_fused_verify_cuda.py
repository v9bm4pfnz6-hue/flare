# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLARE_KIMI3_KDA_FUSED_VERIFY: one-launch KDA verification equals the three-launch path bitwise.

Synthetic BF16 weights at the Kimi K3.1 TP8 rank geometry (12 KDA heads, 8 verification columns).
Both layers go through compile_forward and CUDA-graph replay; the shared scratch arena is
poisoned before every run.
"""

import os

import pytest
import torch

cuda = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")

HEADS, WIDTH, HIDDEN, WINDOW = 12, 12 * 128, 1024, 8


def _weights(device, generator):
    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator) * scale).to(dtype).to(device)

    columns = (4 * WIDTH + 128 + HEADS + 7) // 8 * 8
    return {
        "input": randn(columns, HIDDEN, scale=HIDDEN ** -0.5),
        "forget": randn(WIDTH, 128, scale=128 ** -0.5),
        "output": randn(HIDDEN, WIDTH, scale=WIDTH ** -0.5),
        "convolution": randn(3 * WIDTH, 4, scale=0.3),
        "A_log": randn(HEADS, scale=0.5, dtype=torch.float32),
        "dt_bias": randn(WIDTH, scale=0.5, dtype=torch.float32),
        "norm": 1.0 + randn(128, scale=0.1, dtype=torch.float32),
    }


@cuda
@torch.no_grad()
@pytest.mark.parametrize("tokens", [1, 4, 8])
def test_fused_verify_matches_three_launches(tokens):
    from ignition.components.kda_speculation import KDARecords
    from ignition.components.recurrent import DeltaMetadata, DeltaState
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.kda import PreparedKDA

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(1110 + tokens)
    w = _weights(device, generator)
    source = (torch.randn((tokens, HIDDEN), generator=generator) * 0.5).to(torch.bfloat16).to(device)
    meta = DeltaMetadata.allocate(1, device, verification_columns=WINDOW)
    layers = {}
    with RecordedPreparationSession(device=device, checkpoint="synthetic-kda-fused-verify",
            workload_schema="kimi-kda-fused-verify", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        for name, flag in (("off", "0"), ("on", "1")):
            state = DeltaState.allocate(4, device, heads=HEADS, channels=3 * WIDTH, speculative_tokens=WINDOW - 1)
            records = KDARecords.allocate({0: w}, 4, HEADS, device, window=WINDOW)
            op = PreparedKDA(w, source, state, meta, mode="verify", eps=1e-5, session=session, name="verify_" + name,
                             workspace=arena, buffers={}, records=records.layer(0), fused_verify=flag == "1")
            layers[name] = (op, state, records)
        session.tune(show_progress=False)
        assert layers["off"][0].fused is None
        assert layers["on"][0].fused is not None
        meta.query_start_loc[1] = tokens
        meta.convolution_slots.fill_(1)
        meta.state_indices[:, 0].fill_(2)
        meta.accepted_tokens.fill_(3)
        meta.fresh.fill_(False)
        meta.num_seqs.fill_(1)
        meta.num_tokens.fill_(tokens)
        conv0 = (torch.randn(layers["on"][1].convolution.shape, generator=generator)).to(torch.bfloat16).to(device)
        rec0 = (torch.randn(layers["on"][1].recurrent.shape, generator=generator) * 0.1).to(device)

        results = {}
        for name, (op, state, records) in layers.items():
            compiled = compile_forward(lambda op=op: op(source))

            def reset():
                state.convolution.copy_(conv0)
                state.recurrent.copy_(rec0)
                records.corrections.fill_(float("nan"))
                records.keys_and_gates.fill_(float("nan"))
                arena.tensor.fill_(0xFF)

            reset()
            compiled()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = compiled()
            outs = []
            for _ in range(3):
                reset()
                graph.replay()
                torch.cuda.synchronize()
                outs.append((out.clone(), op.gated.clone(), state.convolution.clone(), state.recurrent.clone(),
                             records.corrections.clone(), records.keys_and_gates.clone()))
            for later in outs[1:]:
                for a, b in zip(outs[0], later, strict=True):
                    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            results[name] = outs[0]
        assert torch.isfinite(results["off"][0]).all()
        for a, b in zip(results["off"], results["on"], strict=True):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
