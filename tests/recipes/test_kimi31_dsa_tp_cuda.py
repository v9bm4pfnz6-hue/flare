# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Global DSA selection and LSE combination across context shards."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1", reason="requires four, eight, or sixteen assigned GPUs")


def _rank(rank, rendezvous, world, records):
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime, create_head_exchange
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dcp import ContextParallel, ContextShards
    from ignition.recipes.kimi3.dsa import PreparedDSA

    rows, heads = 2, 96
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=10))
    torch.manual_seed(3104)
    def random(*shape):
        return (torch.randn(shape, device=device) * .2).bfloat16()
    weights = dict(index_query=random(4096, 64), index_norm=torch.ones(128, device=device, dtype=torch.bfloat16),
                   index_bias=torch.zeros(128, device=device, dtype=torch.bfloat16))
    residual, key_input, head_weights = random(rows, 64), random(rows, 128), random(rows, 32)
    all_query, all_latent, all_keys = random(rows, heads, 576), random(records, 576), random(records, 128)
    local_heads = heads // world
    query = all_query[:, rank * local_heads:(rank + 1) * local_heads].contiguous()
    latent = all_latent[rank::world].contiguous().to(torch.float8_e4m3fn).view(-1, 128, 576)
    index_cache = torch.zeros((records // world // 64, 64 * 132), device=device, dtype=torch.uint8)
    output = torch.empty((rows, local_heads, 512), device=device, dtype=torch.bfloat16)
    slots = torch.full((rows,), -1, device=device, dtype=torch.int64)
    blocks = records // world // 128
    metadata = Metadata.allocate(rows, rows, blocks, device)
    metadata.page_table[0].copy_(torch.arange(blocks, device=device, dtype=torch.int32))
    metadata.page_table[1].copy_(torch.arange(blocks - 1, -1, -1, device=device, dtype=torch.int32))
    metadata.token_page_table.copy_(metadata.page_table)
    metadata.requests.copy_(torch.arange(rows, device=device))
    metadata.positions.fill_(records - 1)
    logical = torch.arange(records, device=device)
    # The second request uses a different physical page order, including for short prefixes.
    mapping = metadata.page_table.long()[:, logical // (128 * world)] * (128 * world) + logical % (128 * world)
    with PackageStorage(rank=rank, world=world, device=rank) as storage:
        coordinator = CollectivePreparation(dist.group.WORLD)
        # Decode rows exchange one FP32 score per candidate slot: 2 * 2048 BF16 columns per rank.
        selections = create_all_gather_runtime(dist.group.WORLD, device, max_rows=rows,
            columns=2 * 2048 * world, ipc=storage.peer_storage)
        exchange = create_head_exchange(dist.group.WORLD, device, max_rows=rows, heads=heads,
            head_dim=512, query_head_dim=576, ipc=storage.peer_storage)
        context = ContextParallel(ContextShards.on(rank, world, device), exchange, coordinator, {}, None,
                                  selections=selections)
        with RecordedPreparationSession(device=device, checkpoint=f"kimi-dsa-tp{world}", workload_schema="dsa-global-top2048",
                numerical_policy="fp8", coordinator=coordinator, compile_workers=4 * world) as session:
            arena, pool = ScratchArena(device), GraphBufferPool(device)
            session.register_scratch(arena)
            op = PreparedDSA(weights, residual, key_input, head_weights, query, output, slots, metadata,
                latent, index_cache, context=context, session=session, coordinator=coordinator,
                name="dsa", workspace=arena, buffers={}, allocator=pool.cursor())
            session.tune(show_progress=False)
            latent.copy_(all_latent[rank::world].to(torch.float8_e4m3fn).view_as(latent))
            op.stages[0][2].write_keys(all_keys[rank::world].contiguous(),
                torch.arange(records // world, device=device, dtype=torch.int64))
            compiled = compile_forward(op)
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            session.freeze()
            def quantized(x, floor):
                value = x.float()
                scale = torch.exp2(torch.ceil(torch.log2(value.abs().amax(-1).clamp_min(floor) / 448)))
                return (value / scale[..., None]).to(torch.float8_e4m3fn).float(), scale
            q, qs = quantized(torch.nn.functional.linear(residual, weights["index_query"]).view(rows, 32, 128), 1e-10)
            k, ks = quantized(all_keys, 1e-4)
            scores = torch.einsum("rhd,rkd->rhk", q, (k * ks[:, None])[mapping]).relu()
            scores = (scores * (head_weights.float() * qs * (128 * 32)**-.5)[:, :, None]).sum(1)
            for length in (records, 129, 0):
                metadata.positions.fill_(length - 1)
                dist.barrier()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                if not length:
                    assert not output.count_nonzero()
                    continue
                ids = scores[:, :length].topk(min(length, 2048), dim=1).indices
                physical = mapping.gather(1, ids)
                kv = all_latent.to(torch.float8_e4m3fn).float()[physical]
                logits = torch.einsum("rhd,rkd->rhk", query.to(torch.float8_e4m3fn).float(), kv) * 192**-.5
                expected = torch.einsum("rhk,rkd->rhd", logits.softmax(-1), kv[:, :, :512])
                torch.testing.assert_close(output.float(), expected, rtol=.03, atol=.002,
                    msg=lambda message: f"rank={rank} context={length}: {message}")
                selected = op.stages[0][3].output
                assert (selected[0] >= 0).sum() == min(length, 2048)
                assert selected[0][selected[0] >= 0].unique().numel() == min(length, 2048)
                if length >= 2048:
                    actual_scores = scores.gather(1, selected.long())
                    cutoff = scores[:, :length].topk(2048, dim=1).values[:, -1]
                    assert (actual_scores.amin(1) >= cutoff - 1e-3).all()
            graph.reset()
        selections.close()
        exchange.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("records", [16384, 1048576])
def test_global_selection_and_empty_context_shards(tmp_path, records):
    world = torch.cuda.device_count()
    assert world in (4, 8, 16)
    mp.spawn(_rank, args=((tmp_path / "rendezvous").as_uri(), world, records), nprocs=world, join=True)
