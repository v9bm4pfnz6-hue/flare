# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi prefill gather transport: the setting, and the shard gather's path over the copy-engine ring."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3 import routing
from ignition.recipes.kimi3.bootstrap import PreparationConfig

WORLD = 9


def test_prefill_gather_defaults_to_kernel_pushes_and_accepts_the_ring():
    assert PreparationConfig().prefill_gather == "kernel"
    assert PreparationConfig(prefill_gather="ring").prefill_gather == "ring"
    for value in ("", "copy-engine", "ring "):
        with pytest.raises(ValueError, match="prefill gather must be"):
            PreparationConfig(prefill_gather=value)


def test_plan_carries_the_prefill_gather():
    import inspect

    from ignition import planning

    assert planning.Plan.__dataclass_fields__["prefill_gather"].default == "kernel"
    assert planning.validate_prefill_gather("ring") == "ring"
    with pytest.raises(planning.UsageError):
        planning.validate_prefill_gather("dma")
    assert inspect.signature(planning.fit).parameters["prefill_gather"].default == "kernel"
    assert inspect.signature(planning.plan).parameters["prefill_gather"].default == "kernel"


class _Configuration:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, flag=None):
        return SimpleNamespace(value=self.values.get(key, flag))


def test_preparation_setting_reads_the_prefill_gather():
    from ignition.cli.preparation import prefill_gather_mode

    assert prefill_gather_mode(_Configuration()) == "kernel"
    assert prefill_gather_mode(_Configuration({"preparation.prefill_gather": "ring"})) == "ring"
    assert prefill_gather_mode(_Configuration(), "ring") == "ring"
    with pytest.raises(Exception):
        prefill_gather_mode(_Configuration({"preparation.prefill_gather": "dma"}))


@pytest.mark.parametrize(("block", "width"), ((416, 3584), (208, 1792), (64, 576), (64, 520), (64, 64)))
def test_slot_blocks_land_in_rank_order_for_every_ring_position(block, width):
    rows, columns = 5, block + 24
    generator = torch.Generator().manual_seed(block + width)
    by_rank = torch.randn((WORLD, rows, columns), generator=generator).to(torch.bfloat16)
    expected = torch.cat([by_rank[rank, :, :block] for rank in range(WORLD)], dim=1)[:, :width]
    for first in range(WORLD):
        slots = torch.stack([by_rank[(first + slot) % WORLD] for slot in range(WORLD)])
        # A destination wider than its payload shows that nothing is written past the payload.
        backing = torch.full((rows, width + 40), 7.0, dtype=torch.bfloat16)
        routing._copy_slot_blocks(backing[:, :width], slots, block, first)
        assert torch.equal(backing[:, :width].view(torch.int16), expected.view(torch.int16)), first
        assert (backing[:, width:] == 7.0).all()


class _FakeRing:
    """Stands in for the prepared ring gather of one rank among nine."""

    created = []

    def __init__(self, shape, dtype, *, runtime, session, coordinator, name):
        self.rank, self.packed = runtime.rank, runtime.packed
        self.first_rank = (self.rank + 1) % WORLD
        self.slots = torch.zeros((WORLD, *shape), dtype=dtype)
        self.state, self.name = "idle", name
        _FakeRing.created.append(self)

    def __call__(self):
        assert self.state == "idle"
        # The caller packed this rank's shards into the last slot.
        assert torch.equal(self.slots[WORLD - 1].view(torch.int16), self.packed[self.rank].view(torch.int16))
        for slot in range(WORLD - 1):
            self.slots[slot].copy_(self.packed[(self.first_rank + slot) % WORLD])
        self.state = "gathered"
        return self.slots

    def release(self):
        assert self.state == "gathered"
        # The neighbours may overwrite the scratch from here on.
        self.slots.fill_(float("nan"))
        self.state = "idle"


@pytest.mark.parametrize("rank", range(WORLD))
def test_shard_gather_packs_into_and_reads_from_the_ring_before_releasing_it(monkeypatch, rank):
    from ignition.components import collectives

    rows, down, router, latent_width, experts = 512, 416, 104, 3584, 896
    generator = torch.Generator().manual_seed(11)
    downs = torch.randn((WORLD, rows, down), generator=generator).to(torch.bfloat16)
    routers = torch.randn((WORLD, rows, router), generator=generator)
    packed = torch.cat([downs, routers.view(WORLD, rows, 2 * router).view(torch.bfloat16)
                        if False else routers.contiguous().view(torch.bfloat16).view(WORLD, rows, 2 * router)], dim=2)
    monkeypatch.setattr(collectives, "PreparedRingGather", _FakeRing)
    _FakeRing.created.clear()
    ring = SimpleNamespace(rank=rank, packed=packed, gather_capacity=rows * (down + 2 * router) * 2)
    latent = torch.zeros((rows, latent_width), dtype=torch.bfloat16)
    logits = torch.zeros((rows, experts))
    gather = routing.PreparedLatentShardGather(downs[rank].contiguous(), routers[rank].contiguous(), latent, logits,
        runtime=SimpleNamespace(world_size=WORLD, max_batch_size=1152), session=None, coordinator=None,
        name="layer1.moe.latent_router_gather", buffers={}, ring=ring)
    assert gather.ring is _FakeRing.created[0] and gather.ring.name == "layer1.moe.latent_router_gather.ring"
    for _ in range(2):
        latent.zero_()
        logits.zero_()
        assert gather() is latent
        assert torch.equal(latent.view(torch.int16),
                           torch.cat(list(downs), dim=1)[:, :latent_width].contiguous().view(torch.int16))
        assert torch.equal(logits, torch.cat(list(routers), dim=1)[:, :experts])
        assert gather.ring.state == "idle"


def test_shard_gather_keeps_kernel_pushes_for_small_segments_and_short_channels(monkeypatch):
    from ignition.components import collectives

    built = []

    class FakeGather:
        def __init__(self, source, output, *, runtime, session, coordinator, name):
            built.append(name)

    monkeypatch.setattr(collectives, "PreparedAllGather", FakeGather)
    monkeypatch.setattr(collectives, "PreparedRingGather", _FakeRing)
    _FakeRing.created.clear()
    down, router = 416, 104
    runtime = SimpleNamespace(world_size=WORLD, max_batch_size=1152)

    def build(rows, capacity):
        return routing.PreparedLatentShardGather(
            torch.zeros((rows, down), dtype=torch.bfloat16), torch.zeros((rows, router)),
            torch.zeros((rows, 3584), dtype=torch.bfloat16), torch.zeros((rows, 896)),
            runtime=runtime, session=None, coordinator=None, name="g", buffers={},
            ring=SimpleNamespace(rank=0, packed=None, gather_capacity=capacity))

    # 256 rows pack 319,488 bytes per rank: below the ring's minimum.
    assert build(256, 1 << 30).ring is None
    # A channel whose shard capacity is smaller than the packed rows keeps the pushes.
    assert build(2048, 2048 * 1248 - 16).ring is None
    # A tree channel reports no gather capacity.
    assert build(2048, 0).ring is None
    assert not _FakeRing.created and built == ["g.0", "g.0", "g.1", "g.0", "g.1"]
    assert build(2048, 2048 * 1248).ring is not None


def test_configuration_built_in_default_resolves_the_prefill_gather():
    from pathlib import Path

    from ignition.cli.config import Configuration

    setting = Configuration(Path("/nonexistent/config.toml"), {}, False).get("preparation.prefill_gather")
    assert setting.value == "kernel"
    assert setting.source == "default"


def test_command_line_takes_the_prefill_gather_and_both_plan_builders_receive_it():
    """export derives its plan and debug-serve builds one directly; each must pass the setting on."""
    import inspect

    from ignition.cli import preparation, serve

    assert inspect.signature(preparation.derive).parameters["prefill_gather"].default is None
    source = inspect.getsource(serve.debug_serve)
    assert "prefill_gather=prefill_gather_mode(" in source and "args.prefill_gather" in source


def test_ring_gather_declares_a_plan_on_the_reduction_channel(monkeypatch):
    """The gather's plan is built from the channel a reduction uses, with the operation and segment in its query."""
    import torch.distributed as dist
    from b12x.comm.pcie import DmaAllReduce
    from b12x.comm.pcie._dma_preparation import query_from_metadata

    from ignition.components import collectives

    rows, columns = 512, 624
    runtime = DmaAllReduce.__new__(DmaAllReduce)
    runtime.world_size, runtime.rank, runtime.device = WORLD, 3, torch.device("cuda", 3)
    runtime.max_bytes = 4608 * 7168 * 2
    runtime.shard_capacity = runtime.max_bytes // WORLD
    runtime._fp8, runtime._bf16_gather, runtime._pieces_override, runtime._a2a_chunks_override = "", True, 0, 0
    runtime._fp8_stage, runtime._fp8_stage_stride = None, 0
    runtime._bf16_stage = torch.zeros(64, dtype=torch.uint8)
    runtime._tree, runtime._closed, runtime.group = None, False, object()
    backing = torch.zeros((WORLD, rows, columns), dtype=torch.bfloat16)
    monkeypatch.setattr(DmaAllReduce, "gather_slots", lambda self, shape, dtype: backing)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: list(range(WORLD)))
    declared = []
    monkeypatch.setattr(collectives, "declare",
                        lambda session, requests, coordinator=None: declared.append((requests, coordinator)))
    gather = collectives.PreparedRingGather((rows, columns), torch.bfloat16, runtime=runtime, session=None,
                                            coordinator="coordinator", name="layer1.moe.ring")
    assert gather.slots is backing and gather.first_rank == 4
    (requests, coordinator), = declared
    assert len(requests) == 1 and coordinator == "coordinator"
    query = gather.plan.query
    assert (query.call["operation"], query.call["shape"], query.call["dtype"]) == ("all_gather", (rows, columns), "bfloat16")
    assert (query.rank, query.world_size) == (3, WORLD)
    # The same channel declares a reduction with the same channel facts and storage.
    reduction = query_from_metadata(runtime, shape=(WORLD * 8, 64), dtype=torch.bfloat16)
    for key in ("max_bytes", "wire_mode", "bf16_gather", "pieces_override", "a2a_chunks_override"):
        assert query.call[key] == reduction.call[key]
    assert dict(query.setup) == dict(reduction.setup)
    # A segment larger than a shard of the channel is not a gather of this channel.
    with pytest.raises(ValueError, match="shard capacity"):
        collectives.PreparedRingGather((4608, 7168), torch.bfloat16, runtime=runtime, session=None,
                                       coordinator=None, name="too-large")
    tree = DmaAllReduce.__new__(DmaAllReduce)
    tree._tree, tree.shard_capacity = object(), runtime.shard_capacity
    with pytest.raises(ValueError, match="ring channel"):
        collectives.PreparedRingGather((rows, columns), torch.bfloat16, runtime=tree, session=None,
                                       coordinator=None, name="tree")
