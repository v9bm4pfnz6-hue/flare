# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 checkpoint identity, attention partition and complete history storage."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.config import ModelConfig, DSparkConfig, LatentDSparkConfig
from ignition.recipes.kimi3.pool import continuation_layout, page_specs, page_bytes
from ignition.recipes.kimi31.recipe import RECIPE

FIXTURES = Path(__file__).parents[1] / "fixtures" / "kimi31"


@pytest.fixture
def raw():
    # Published configuration at moonshotai/Kimi-K3.1-Pro revision 6d0a0202956796543364d666ab19c6ad1a494ee1.
    return json.loads((FIXTURES / "config.json").read_text())


def test_layer_mapping_and_cache_accounting(raw):
    config = ModelConfig.from_dict(raw)
    config.validate_shard(TensorParallelShard(0, 16))
    assert config.layers.count("kda") == 69
    assert config.layers.count("dsa") == 24
    assert {i + 1 for i, kind in enumerate(config.layers) if kind == "dsa"} == set(raw["text_config"]["dsa_config"]["dsa_layers"])
    plain = RECIPE.plain_ranks[16]
    layout = continuation_layout(config, 16)
    assert page_bytes(config) == plain.page_bytes
    assert layout.row_bytes == plain.state_row_bytes
    for name, shape in page_specs(config):
        assert layout.region(name).shape == shape
    draft = DSparkConfig.from_dict(json.loads((FIXTURES / "draft-config.json").read_text()), config.text)
    assert isinstance(draft, LatentDSparkConfig)
    config = replace(config, draft=draft)
    draft.validate_shard(TensorParallelShard(0, 16))
    assert draft.auxiliary_layers == (75, 79, 83, 87, 91)
    assert draft.auxiliary_source == "residual_output"
    layout = continuation_layout(config, 16, speculative_tokens=7)
    assert layout.row_bytes == RECIPE.ranks[16].state_row_bytes
    assert page_bytes(config) == RECIPE.ranks[16].page_bytes
    assert len([name for name, _ in page_specs(config) if name.startswith("draft")]) == 5
    for name, shape in page_specs(config):
        assert layout.region(name).shape == shape


@pytest.mark.parametrize("field,value", [("index_topk", 512), ("index_rope_dim", 64),
    ("index_k_norm_type", "rmsnorm"), ("index_topk_freq", 2), ("dsa_layers", [4])])
def test_rejects_incompatible_sparse_attention(raw, field, value):
    raw["text_config"]["dsa_config"][field] = value
    with pytest.raises(ValueError, match="DSA|K3.1"):
        ModelConfig.from_dict(raw)


def test_dense_k3_cannot_silently_consume_sparse_checkpoint(raw):
    raw.update(model_type="kimi_k3", architectures=["KimiK3ForConditionalGeneration"])
    with pytest.raises(ValueError, match="DSA configuration"):
        ModelConfig.from_dict(raw)
    del raw["text_config"]["dsa_config"]
    config = ModelConfig.from_dict(raw)
    assert config.layers.count("mla") == 24
    assert not any(name.endswith(".index") for name, _ in page_specs(config))


@pytest.mark.parametrize("field,value", [("full_attention_causal", False), ("target_layer_ids", [93]),
    ("layer_types", ["sliding_attention"] * 5), ("vocab_size", 1)])
def test_draft_geometry_and_target_identity(raw, field, value):
    draft = json.loads((FIXTURES / "draft-config.json").read_text())
    draft[field] = value
    with pytest.raises(ValueError, match="DSpark"):
        DSparkConfig.from_dict(draft, raw["text_config"])


def test_recipe_preserves_image_capacity_and_sampling():
    assert RECIPE.max_images == 4
    assert RECIPE.sampling_defaults.top_p == .95
    assert RECIPE.sampling_defaults.temperature == 1.
    assert RECIPE.sampling_defaults.top_k == -1
    assert RECIPE.max_draft_tokens == 7


def test_gather_thread_tuning_keeps_other_kimi_collective_settings():
    from ignition.recipes.kimi3.recipe import RECIPE as KIMI3

    assert KIMI3.b12x_environment["B12X_PCIE_DCP_THREADS"] == "512"
    assert RECIPE.b12x_environment == {**KIMI3.b12x_environment, "B12X_PCIE_DCP_THREADS": "0",
                                       "B12X_PCIE_DCP_BLOCK_LIMIT": "64", "B12X_W4A16_PREFILL_SHAPES": "1",
                                       "B12X_W4A16_SMALL_M_SPLITK": "decode",
                                       "B12X_PCIE_DCP_PAIR_PUSH": "1", "B12X_PCIE_DCP_GATHER_ACQUIRE": "1",
                                       "B12X_PCIE_DCP_GATHER_ISLAND": "2", "B12X_PCIE_DCP_LSE_PUSH_TP9": "2",
                                       "B12X_TRELLIS_SWITCH_SCOPE": "prefill_safe", "B12X_TRELLIS_LUT_CP_ASYNC": "2",
                                       "B12X_TRELLIS_PREFETCH_SIDES": "1", "B12X_TRELLIS_INKERNEL_ROUTE_PACK": "2",
                                       "B12X_TRELLIS_FLOW_FLAGS": "1", "B12X_TRELLIS_TILE_CONFIG": "128,128,64,256",
                                       "B12X_W4A16_PREFILL_TABLE": "2", "B12X_W4A16_NONCOOP_PREFILL_SMS": "148"}
    assert not any(name.startswith(("B12X_TRELLIS", "B12X_PCIE_DCP_PAIR", "B12X_PCIE_DCP_GATHER", "B12X_PCIE_DCP_LSE"))
                   for name in KIMI3.b12x_environment)
    assert "B12X_W4A16_PREFILL_SHAPES" not in KIMI3.b12x_environment
    assert "B12X_W4A16_SMALL_M_SPLITK" not in KIMI3.b12x_environment
    assert RECIPE.b12x_environment["B12X_PCIE_DCP_BLOCK_LIMIT"] == "64"


def test_exl3_keeps_sparse_attention_and_bf16_projections(raw):
    raw["quantization_config"] = raw["text_config"]["quantization_config"] = {
        "quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json"},
    }
    config = ModelConfig.from_dict(raw)
    config.validate_shard(TensorParallelShard(0, 8))
    assert config.layers.count("dsa") == 24
    prefix = "language_model.model.layers.1.block_sparse_moe"
    for projection in ("w1", "w3", "w2"):
        assert config.projection_format(f"{prefix}.experts.0.{projection}") == "exl3"
    assert config.projection_format(f"{prefix}.shared_experts.up_proj") == "bf16"
    assert config.projection_format("language_model.model.layers.3.self_attn.indexer.wq_b") == "bf16"


@pytest.mark.parametrize("fields", [
    {"dense_format": "mxfp8"}, {"exl3": None}, {"exl3": {"manifest": "other.json"}},
])
def test_exl3_rejects_undeclared_retained_storage(raw, fields):
    raw["text_config"]["quantization_config"] = {
        "quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json"}, **fields,
    }
    with pytest.raises(ValueError, match="EXL3"):
        ModelConfig.from_dict(raw)


def test_tp8_costs_match_complete_history_layout(raw):
    config = ModelConfig.from_dict(raw)
    config.validate_shard(TensorParallelShard(0, 8))
    assert continuation_layout(config, 8).row_bytes == RECIPE.plain_ranks[8].state_row_bytes
    assert page_bytes(config) == RECIPE.plain_ranks[8].page_bytes
    draft = DSparkConfig.from_dict(json.loads((FIXTURES / "draft-config.json").read_text()), config.text)
    config = replace(config, draft=draft)
    assert continuation_layout(config, 8, speculative_tokens=7).row_bytes == RECIPE.ranks[8].state_row_bytes
    assert page_bytes(config) == RECIPE.ranks[8].page_bytes


def test_tp9_costs_match_complete_history_layout(raw):
    config = ModelConfig.from_dict(raw)
    assert continuation_layout(config, 9).row_bytes == RECIPE.plain_ranks[9].state_row_bytes
    assert page_bytes(config) == RECIPE.plain_ranks[9].page_bytes
    draft = DSparkConfig.from_dict(json.loads((FIXTURES / "draft-config.json").read_text()), config.text)
    config = replace(config, draft=draft)
    assert continuation_layout(config, 9, speculative_tokens=7).row_bytes == RECIPE.ranks[9].state_row_bytes
    assert page_bytes(config) == RECIPE.ranks[9].page_bytes


def test_tp9_expert_rotation_covers_every_hadamard_extent_without_empty_ranks(raw):
    from ignition.recipes.kimi3.config import expert_intermediate_range

    raw["text_config"]["quantization_config"] = {"quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json"}}
    config = ModelConfig.from_dict(raw)
    resident = [0] * 9
    for index in range(config.text["first_k_dense_replace"], config.text["num_hidden_layers"]):
        ranges = [expert_intermediate_range(TensorParallelShard(rank, 9), index, 3072) for rank in range(9)]
        for rank, (first, last) in enumerate(ranges):
            assert first % 128 == 0 and last > first
            assert (first // 32) % 4 == 0
            assert first // 1536 == (last - 1) // 1536
            resident[rank] += (last - first) // 32
        covered = [channel for first, last in ranges for channel in range(first, last, 128)]
        assert sorted(covered) == list(range(0, 3072, 128))
        config.validate_shard(TensorParallelShard(0, 9))
    assert max(resident) == 984 and min(resident) > 0



def test_tp8_preparation_keeps_context_parallel_page_geometry():
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(8)), prefill_chunk_size=256,
        max_active=1, vision=False, speculative_widths=())
    assert config.page_tokens == 1024
    assert config.preparation_pages == 3
    with pytest.raises(ValueError, match="distinct GPUs"):
        replace(config, devices=(0,) * 8)


@pytest.mark.parametrize("concurrency,expected", [(1, 3), (2, 3), (3, 5), (4, 5), (8, 9)])
def test_tuning_rows_cover_recurrence_without_snapshot_destinations(concurrency, expected):
    from ignition.components.capacity import request_capacity
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(8)), max_active=concurrency, vision=False)
    assert config.tuning_state_rows == expected
    assert config.tuning_state_rows >= 1 + request_capacity(concurrency)
    assert config.tuning_state_rows >= 3  # Null, one prefill state and its checkpoint.


def test_tp9_preparation_covers_the_partial_final_context_page():
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(9)), prefill_chunk_size=128,
        max_active=1, vision=False, speculative_widths=())
    assert config.context == 1048576
    assert config.page_tokens == 1152
    assert -(-config.context // config.page_tokens) == 911
    with pytest.raises(ValueError, match="128-aligned"):
        replace(config, context=config.context - 1)


def test_tp9_preparation_preserves_neutral_draft_temperature_by_default():
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(9)), max_active=1, vision=False)
    assert config.draft_temperature == 1.
    assert replace(config, draft_temperature=.7).draft_temperature == .7
    for temperature in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            replace(config, draft_temperature=temperature)


def test_exl3_rejects_shards_that_split_intermediate_hadamard_blocks(raw):
    raw["text_config"]["quantization_config"] = {"quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json"}}
    config = ModelConfig.from_dict(raw)
    with pytest.raises(ValueError, match="aligned rank partitions"):
        config.validate_shard(TensorParallelShard(0, 16))


@pytest.mark.parametrize("rows", (1, 8, 16, 32))
def test_tp9_route_gather_declares_all_speculative_rows(monkeypatch, rows):
    import torch
    from types import SimpleNamespace
    from b12x.comm import pcie
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.recipes.kimi3 import routing
    from ignition.recipes.kimi3.config import padded_width

    down_width = padded_width(3584, 9, 32)
    router_width = padded_width(896, 9, 8)
    runtime = SimpleNamespace(world_size=9, query_head_dim=2 * down_width + 4 * router_width,
                              exchange_group=object())
    plan = SimpleNamespace(request=lambda **kwargs: object())
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: tuple(range(9)))
    monkeypatch.setattr(pcie, "query_from_runtime", lambda runtime, **kwargs: kwargs)
    monkeypatch.setattr(pcie, "plan", lambda query, *, runtime: plan)
    monkeypatch.setattr(routing, "declare", lambda *args, **kwargs: None)

    def make(count, buffers, allocator):
        return routing.PreparedRouteGather(
            torch.zeros((count, down_width), dtype=torch.bfloat16),
            torch.zeros((count, router_width), dtype=torch.float32), torch.zeros(896, dtype=torch.float32),
            torch.zeros((count, 3584), dtype=torch.bfloat16), torch.zeros((count, 16), dtype=torch.float32),
            torch.zeros((count, 16), dtype=torch.int32), runtime=runtime, session=None, coordinator=None,
            name=f"route-{count}", buffers=buffers, allocator=allocator)

    pool = GraphBufferPool("cpu", chunk_bytes=256)
    buffers = {}
    largest = make(32, buffers, pool.cursor())
    nbytes = pool.nbytes
    gather = make(rows, buffers, pool.cursor())

    assert gather.rows == rows and gather.fused is None
    assert gather.logits.shape == (rows, 896)
    assert gather.gathered_latent.shape == (rows, down_width * 9)
    assert gather.gathered_logits.shape == (rows, router_width * 9)
    assert pool.nbytes == nbytes
    for field in ("logits", "gathered_latent", "gathered_logits"):
        assert getattr(gather, field).untyped_storage().data_ptr() == getattr(largest, field).untyped_storage().data_ptr()


@pytest.mark.parametrize("rows", (1, 4, 32))
def test_tp9_route_gather_pushes_packed_shards_through_a_column_channel(monkeypatch, rows):
    import torch
    from types import SimpleNamespace
    from b12x.comm import pcie
    from ignition.components import collectives
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.recipes.kimi3 import routing
    from ignition.recipes.kimi3.config import padded_width

    down_width = padded_width(3584, 9, 32)
    router_width = padded_width(896, 9, 8)
    columns = down_width + 2 * router_width
    runtime = SimpleNamespace(world_size=9, query_head_dim=2 * down_width + 4 * router_width,
                              exchange_group=object())
    channel = SimpleNamespace(world_size=9, heads_per_rank=1, query_head_dim=columns, max_batch_size=1024)
    declared, gathers = [], []
    plan = SimpleNamespace(request=lambda **kwargs: kwargs["name"])
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: tuple(range(9)))
    monkeypatch.setattr(pcie, "query_from_runtime", lambda runtime, **kwargs: kwargs)
    monkeypatch.setattr(pcie, "plan", lambda query, *, runtime: plan)
    monkeypatch.setattr(routing, "declare", lambda session, requests, **kwargs: declared.extend(requests))

    class Gather:
        def __init__(self, source, output, *, runtime, session, coordinator, name):
            gathers.append((tuple(source.shape), tuple(output.shape), source.dtype, runtime, name))

    monkeypatch.setattr(collectives, "PreparedAllGather", Gather)

    def make(count, buffers, allocator, **kwargs):
        return routing.PreparedRouteGather(
            torch.zeros((count, down_width), dtype=torch.bfloat16),
            torch.zeros((count, router_width), dtype=torch.float32), torch.zeros(896, dtype=torch.float32),
            torch.zeros((count, 3584), dtype=torch.bfloat16), torch.zeros((count, 16), dtype=torch.float32),
            torch.zeros((count, 16), dtype=torch.int32), runtime=runtime, session=None, coordinator=None,
            name=f"route-{count}", buffers=buffers, allocator=allocator, **kwargs)

    pool = GraphBufferPool("cpu", chunk_bytes=256)
    buffers = {}
    largest = make(32, buffers, pool.cursor(), gather_runtime=channel)
    nbytes = pool.nbytes
    del declared[:], gathers[:]
    gather = make(rows, buffers, pool.cursor(), gather_runtime=channel)

    # The channel carries the exchange; only the local selection is declared with the route runtime.
    assert gather.fused is None and declared == [f"route-{rows}/select"]
    assert gathers == [((rows, columns), (rows, 9 * columns), torch.bfloat16, channel, f"route-{rows}/gather")]
    assert gather.send.shape == (rows, columns) and gather.received.shape == (rows, 9 * columns)
    assert gather.logits.shape == (rows, 896) and not hasattr(gather, "gathered_latent")
    assert pool.nbytes == nbytes
    for field in ("logits", "send", "received"):
        assert getattr(gather, field).untyped_storage().data_ptr() == getattr(largest, field).untyped_storage().data_ptr()
    for wrong in (dict(heads_per_rank=2), dict(query_head_dim=columns + 8), dict(world_size=8),
                  dict(max_batch_size=rows - 1)):
        with pytest.raises(ValueError, match="column channel"):
            make(rows, {}, None, gather_runtime=SimpleNamespace(**{**vars(channel), **wrong}))


@pytest.mark.parametrize(("rows", "fused"), ((1, True), (4, True), (8, True), (9, False), (32, False)))
def test_tp9_route_gather_selects_inside_a_pushed_paired_exchange(monkeypatch, rows, fused):
    import sys
    import torch
    from types import SimpleNamespace
    from b12x.comm import pcie
    from ignition.components import collectives
    from ignition.recipes.kimi3 import routing
    from ignition.recipes.kimi3.config import padded_width

    down_width = padded_width(3584, 9, 32)
    router_width = padded_width(896, 9, 8)
    runtime = SimpleNamespace(world_size=9, query_head_dim=2 * down_width + 4 * router_width,
                              exchange_group=object())
    channel = SimpleNamespace(world_size=9, heads_per_rank=1, query_head_dim=down_width + 2 * router_width,
                              max_batch_size=1024)
    declared, planned, gathers = [], [], []
    plan = SimpleNamespace(request=lambda **kwargs: kwargs["name"])

    def query(runtime, *, surface, call):
        # b12x marks the paired surfaces of a push plan; other surfaces carry no transport.
        pair = surface.endswith(("all_gather_pair", "all_gather_pair_kimi_topk"))
        return SimpleNamespace(surface=surface, call={"pair_transport": "push"} if pair else {})

    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: tuple(range(9)))
    monkeypatch.setattr(pcie, "query_from_runtime", query)
    monkeypatch.setattr(pcie, "plan", lambda query, *, runtime: planned.append(query.surface) or plan)
    monkeypatch.setattr(routing, "declare", lambda session, requests, **kwargs: declared.extend(requests))
    monkeypatch.setitem(sys.modules, "b12x.comm.pcie.pcie_dcp_a2a", SimpleNamespace(PAIR_PUSH_MAX_ROWS=8))
    monkeypatch.setattr(collectives, "PreparedAllGather",
                        lambda source, output, **kwargs: gathers.append(kwargs["name"]) or object())

    gather = routing.PreparedRouteGather(
        torch.zeros((rows, down_width), dtype=torch.bfloat16),
        torch.zeros((rows, router_width), dtype=torch.float32), torch.zeros(896, dtype=torch.float32),
        torch.zeros((rows, 3584), dtype=torch.bfloat16), torch.zeros((rows, 16), dtype=torch.float32),
        torch.zeros((rows, 16), dtype=torch.int32), runtime=runtime, session=None, coordinator=None,
        name="route", buffers={}, gather_runtime=channel)

    if fused:
        # One exchange gathers the padded shards and selects: no separate selection, no column channel.
        assert gather.fused is not None and gather.gather is None
        assert planned == ["DcpAllToAll.all_gather_pair_kimi_topk"] and declared == ["route"] and not gathers
        assert gather.fused["out_down"].shape == (rows, 3584)
    else:
        assert gather.fused is None and gather.gather is not None
        assert planned == ["DcpAllToAll.kimi_topk16"] and declared == ["route/select"] and gathers == ["route/gather"]


def test_route_unpack_geometry_matches_rank_major_compaction():
    """The index arithmetic of the unpack kernel, replayed on the CPU against the paired path's compaction."""
    import torch
    from ignition.recipes.kimi3.config import padded_width

    for world, rows in ((9, 1), (9, 6), (9, 32), (12, 4)):
        down, router = padded_width(3584, world, 32), 2 * padded_width(896, world, 8)
        generator = torch.Generator().manual_seed(world * 131 + rows)
        local_down = torch.randint(-32768, 32768, (world, rows, down), generator=generator, dtype=torch.int16)
        local_router = torch.randint(-32768, 32768, (world, rows, router), generator=generator, dtype=torch.int16)
        # What the column gather delivers: every rank's packed row, rank-major.
        received = torch.cat([torch.cat((local_down[rank], local_router[rank]), dim=1) for rank in range(world)], dim=1)
        latent = torch.full((rows, 3584), -1, dtype=torch.int16)
        logits = torch.full((rows, 2 * 896), -1, dtype=torch.int16)
        column = torch.arange(512)
        for row in range(rows):
            for rank in range(world):
                source = received[row, rank * (down + router):]
                live = (column < down) & (rank * down + column < latent.shape[1])
                latent[row, rank * down + column[live]] = source[column[live]]
                live = (column < router) & (rank * router + column < logits.shape[1])
                logits[row, rank * router + column[live]] = source[down + column[live]]
        # The paired exchange gathers each field rank-major and keeps its logical prefix.
        assert torch.equal(latent, torch.cat(tuple(local_down), dim=1)[:, :3584])
        assert torch.equal(logits, torch.cat(tuple(local_router), dim=1)[:, :2 * 896])


def test_tp9_route_gather_rejects_rows_above_capacity():
    import torch
    from types import SimpleNamespace
    from ignition.recipes.kimi3.config import padded_width
    from ignition.recipes.kimi3.routing import PreparedRouteGather

    down_width, router_width = padded_width(3584, 9, 32), padded_width(896, 9, 8)
    runtime = SimpleNamespace(world_size=9, query_head_dim=2 * down_width + 4 * router_width)
    with pytest.raises(ValueError, match="1-32"):
        PreparedRouteGather(torch.zeros((33, down_width), dtype=torch.bfloat16),
            torch.zeros((33, router_width), dtype=torch.float32), torch.zeros(896),
            torch.zeros((33, 3584), dtype=torch.bfloat16), torch.zeros((33, 16)), torch.zeros((33, 16), dtype=torch.int32),
            runtime=runtime, session=None, coordinator=None, name="too-wide")


def test_tp9_router_fallback_reassembles_padded_rank_partitions(monkeypatch):
    import torch
    from ignition.recipes.kimi3 import decoder

    mlp = decoder.PreparedLatentMoE.__new__(decoder.PreparedLatentMoE)
    mlp.route_gather = None
    mlp.prefill_route_gather = None
    source = torch.zeros((2, 8), dtype=torch.bfloat16)
    mlp.router_projection = lambda value: torch.ones((2, 100), dtype=torch.float32)
    mlp.router_input = torch.zeros((2, 128), dtype=torch.float32)
    mlp.router_gather = lambda value: None
    mlp.gathered_logits = torch.zeros((2, 9 * 128), dtype=torch.float32)
    for rank in range(9):
        begin = rank * 128
        mlp.gathered_logits[:, begin:begin + 100] = torch.arange(rank * 100, rank * 100 + 100)
    mlp.router_columns = mlp.gathered_logits.view(2, 9, 128)[:, :, :100]
    mlp.router_output = torch.empty((2, 9, 100), dtype=torch.float32)
    mlp.router_logits = torch.empty((2, 896), dtype=torch.float32)
    expected = torch.arange(896, dtype=torch.float32).expand(2, 896)
    mlp.router = lambda logits: torch.testing.assert_close(logits, expected)
    mlp.source = torch.empty_like(source)
    mlp.hidden_columns = torch.arange(8)
    latent = torch.ones((2, 32), dtype=torch.bfloat16)
    mlp.down = lambda value: latent
    mlp.down_reduce = lambda value: None
    mlp.latent = latent
    mlp.latent_sum = latent
    mlp.experts = lambda: latent
    mlp.expert_reduce = lambda value: None
    mlp.latent_norm = torch.ones(32, dtype=torch.bfloat16)
    mlp.up_input = torch.empty_like(latent)
    mlp.latent_begin = torch.tensor([0], dtype=torch.int32)
    mlp.eps = 1e-6
    mlp.up = lambda value: None
    mlp.output = torch.zeros_like(source)
    mlp.shared = lambda value: torch.zeros_like(source)
    mlp.shared.concurrent = False
    monkeypatch.setattr(decoder, "rms_norm_columns", lambda source, norm, output, eps, begin: output.copy_(source))

    result = mlp(source)

    assert result is mlp.output
    torch.testing.assert_close(mlp.router_logits, expected)



def test_tp9_prefill_graph_selection_preserves_default_and_explicit_subset():
    from ignition.components.capacity import prefill_capacities
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(9)), max_active=4, vision=False)
    assert config.prefill_graph_capacities == prefill_capacities(4096)
    selected = replace(config, prefill_graph_capacities=(128, 4096))
    assert selected.prefill_graph_capacities == (128, 4096)
    for capacities in ((), (128,), (4096, 128), (128, 512), (128, 384, 4096)):
        with pytest.raises(ValueError, match="prefill graph selection"):
            replace(config, prefill_graph_capacities=capacities)


def test_tp9_prefill_gather_uses_bounded_decode_channel(monkeypatch):
    import torch
    from b12x.comm.pcie import DcpAllToAll
    from ignition.preparation.collectives import create_all_gather_runtime, create_head_exchange
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(9)), max_active=4, vision=False)
    assert config.decode_channel_rows == 1024
    assert config.prefill_gather_rows == 1024
    assert replace(config, devices=tuple(range(8))).prefill_gather_rows == 1024
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: len(config.devices))
    monkeypatch.setattr(DcpAllToAll, "from_exchange_group", classmethod(lambda cls, **kwargs: kwargs))
    group, device = object(), torch.device("cpu")
    embedding = create_all_gather_runtime(group, device, max_rows=config.prefill_gather_rows, columns=9 * 64)
    latent = create_head_exchange(group, device, max_rows=config.prefill_gather_rows, heads=9, head_dim=64)
    decode = create_head_exchange(group, device, max_rows=config.decode_channel_rows, heads=9, head_dim=64)
    assert embedding["max_batch_size"] == 1024
    assert latent["max_batch_size"] == 1024
    assert decode["max_batch_size"] == 1024


def test_tp9_dma_prefill_requires_aligned_ring_shards():
    from ignition.recipes.kimi3.bootstrap import PreparationConfig, _prefill_uses_dma

    for rows, expected in ((128, False), (4096, False), (4608, True), (9216, True)):
        assert _prefill_uses_dma(9, rows, 7168) is expected
        assert not _prefill_uses_dma(8, rows, 7168)
    config = PreparationConfig(devices=tuple(range(9)), max_active=4, vision=False,
        prefill_chunk_size=4608, prefill_graph_capacities=(128, 4608))
    assert config.prefill_graph_capacities == (128, 4608)


def test_tp9_small_prefill_graphs_reduce_on_the_direct_channel():
    from ignition.recipes.kimi3.bootstrap import DIRECT_PREFILL_MAX_ROWS, _prefill_reduces_on_ring

    assert DIRECT_PREFILL_MAX_ROWS == 128
    assert [capacity for capacity in (128, 256, 512, 1024, 2048, 4608) if not _prefill_reduces_on_ring(capacity)] == [128]


def test_tp9_lse_geometry_uses_measured_rows():
    from ignition.recipes.kimi3.dcp import lse_geometry

    assert lse_geometry(1, 9) == (512, 16)
    assert lse_geometry(8, 9) == (512, 8)
    assert lse_geometry(32, 9) == (384, 32)
    assert lse_geometry(32, 8) == (512, 8)
