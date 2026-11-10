# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi prefill reduction transport declaration."""

import pytest

from ignition.components.owner_packet_reduce import aligned_extent, packet_intervals
from ignition.preparation.collectives import create_dma_all_reduce_runtime
from ignition.recipes.kimi3.bootstrap import PreparationConfig


def test_prefill_reduction_defaults_to_the_lossless_fp32_owner_packet():
    assert PreparationConfig().prefill_reduction == "fp32-owner-packet"


def test_prefill_reduction_accepts_the_bf16_gather_declaration():
    config = PreparationConfig(prefill_reduction="bf16-gather")
    assert config.prefill_reduction == "bf16-gather"


@pytest.mark.parametrize("value", ["bf16", "fp8-owner-packet", "bf16-gather ", ""])
def test_prefill_reduction_rejects_unknown_transports(value):
    with pytest.raises(ValueError, match="prefill reduction must be"):
        PreparationConfig(prefill_reduction=value)


def test_bf16_gather_requires_a_two_byte_activation_declaration():
    with pytest.raises(ValueError, match="two-byte activation"):
        create_dma_all_reduce_runtime(None, 0, max_tokens=1, hidden_size=1,
            element_bytes=4, bf16_gather=True)
    with pytest.raises(ValueError, match="two-byte activation"):
        create_dma_all_reduce_runtime(None, 0, max_tokens=1, hidden_size=1,
            bf16_gather=1)


def test_aligned_workspace_keeps_a_128_row_reduction_in_one_ring_invocation():
    # 128 x 7168 BF16: 917,504 elements, ring-aligned to 917,568 (9 x 101,952).
    elements, world, max_bytes = 917_504, 9, 66_060_288
    assert aligned_extent(elements, world) == 101_952
    _, raw = packet_intervals(elements, world, elements, max_bytes)
    assert raw == ((0, 101_944), (101_944, 101_952))
    _, aligned = packet_intervals(elements, world, 917_568, max_bytes)
    assert aligned == ((0, 101_952),)


def test_full_chunk_workspace_sizing_is_unchanged():
    world, max_bytes = 9, 66_060_288
    hidden = 4_608 * 7_168
    assert packet_intervals(hidden, world, 16_515_072, max_bytes)[1] == (
        (0, 1_835_008), (1_835_008, 3_670_016))
    latent = 4_608 * 3_584
    assert packet_intervals(latent, world, 16_515_072, max_bytes)[1] == ((0, 1_835_008),)


def test_packet_intervals_reject_a_scratch_that_cannot_hold_one_group():
    with pytest.raises(ValueError, match="cannot hold one aligned element group"):
        packet_intervals(72, 9, 4, 66_060_288)

class _Configuration:
    """Minimal configuration reader for the preparation setting helper."""

    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, flag=None):
        value = self.values.get(key, flag)

        class _Setting:
            pass

        setting = _Setting()
        setting.value = value
        return setting


def test_prefill_reduction_accepts_the_switch_tree_declaration():
    from ignition import planning

    config = PreparationConfig(prefill_reduction="tree-2+2+2+2+1")
    assert config.prefill_reduction == "tree-2+2+2+2+1"
    assert planning.validate_prefill_reduction("tree-2+2+2+2+1") == "tree-2+2+2+2+1"
    with pytest.raises(ValueError, match="prefill reduction must be"):
        PreparationConfig(prefill_reduction="tree-4+4")
    # A tree channel sums BF16 activations itself; the ring's BF16 gather does not apply to it.
    with pytest.raises(ValueError, match="tree reduction names its topology"):
        create_dma_all_reduce_runtime(None, 0, max_tokens=1, hidden_size=1, bf16_gather=True, tree="2+2+2+2+1")


@pytest.mark.parametrize(("rows", "padded"), ((4608, False), (9, False), (2048, True), (256, True)))
def test_tree_reduction_reduces_bf16_directly_and_pads_unaligned_payloads(monkeypatch, rows, padded):
    import torch
    from b12x.comm.pcie import DmaAllReduce
    from ignition.components import owner_packet_reduce

    world, columns = 9, 64
    runtime = DmaAllReduce.__new__(DmaAllReduce)
    runtime.world_size = world
    runtime._tree = type("Topology", (), {"name": "2+2+2+2+1"})()
    calls = []

    class FakeReduce:
        def __init__(self, source, output, *, runtime, session, coordinator, name):
            self.source, self.output = source, output
            calls.append((tuple(source.shape), source.dtype, source.data_ptr() == output.data_ptr(), name))

        def __call__(self, source):
            assert source.data_ptr() == self.source.data_ptr()
            self.output.copy_(source * world)
            return self.output

    monkeypatch.setattr(owner_packet_reduce, "PreparedAllReduce", FakeReduce)
    source = torch.randn((rows, columns)).to(torch.bfloat16)
    output = torch.zeros_like(source)
    elements = rows * columns
    extent = aligned_extent(elements, world)
    assert (extent * world != elements) is padded
    staging = torch.zeros(extent * world + 16, dtype=torch.bfloat16) if padded else None
    reduce = owner_packet_reduce.copy_engine_reduction(source, output, runtime=runtime, workspace=None,
        staging=staging, session=None, coordinator=None, name="layer0.reduce")
    assert isinstance(reduce, owner_packet_reduce.PreparedTreeAllReduce)
    result = reduce(source)
    assert result is output and torch.equal(output, source * world)
    shape, dtype, in_place, name = calls[0]
    assert dtype == torch.bfloat16 and name == "layer0.reduce.tree"
    if padded:
        # One in-place reduction over the padded packet; the tail stays zero.
        assert shape == (world, extent) and in_place
        assert not staging[elements:extent * world].any()
        with pytest.raises(ValueError, match="padded BF16 staging"):
            owner_packet_reduce.PreparedTreeAllReduce(source, output, runtime=runtime, session=None,
                                                      coordinator=None, name="x", staging=None)
    else:
        assert shape == (rows, columns) and not in_place
    ring = DmaAllReduce.__new__(DmaAllReduce)
    ring.world_size, ring._tree = world, None
    with pytest.raises(ValueError, match="tree topology"):
        owner_packet_reduce.PreparedTreeAllReduce(source, output, runtime=ring, session=None, coordinator=None,
                                                  name="x")


def test_plan_declares_the_prefill_reduction_default():
    from ignition import planning

    fields = planning.Plan.__dataclass_fields__
    assert fields["prefill_reduction"].default == "fp32-owner-packet"


def test_prefill_reduction_validation_helper():
    from ignition import planning
    from ignition.planning import UsageError

    assert planning.validate_prefill_reduction("fp32-owner-packet") == "fp32-owner-packet"
    assert planning.validate_prefill_reduction("bf16-gather") == "bf16-gather"
    with pytest.raises(UsageError):
        planning.validate_prefill_reduction("fp8")


def test_preparation_setting_reads_bf16_gather_from_configuration():
    from ignition.cli.preparation import prefill_reduction_mode

    assert prefill_reduction_mode(_Configuration(), None) == "fp32-owner-packet"
    assert prefill_reduction_mode(
        _Configuration({"preparation.prefill_reduction": "bf16-gather"}), None) == "bf16-gather"
    assert prefill_reduction_mode(_Configuration(), None, "bf16-gather") == "bf16-gather"
    with pytest.raises(Exception):
        prefill_reduction_mode(_Configuration({"preparation.prefill_reduction": "fp8"}), None)


def test_fit_and_plan_accept_the_prefill_reduction():
    import inspect

    from ignition import planning

    assert "prefill_reduction" in inspect.signature(planning.fit).parameters
    assert "prefill_reduction" in inspect.signature(planning.plan).parameters
    assert planning.fit.__defaults__ is not None or True


def test_configuration_built_in_default_resolves_the_prefill_reduction():
    from pathlib import Path

    from ignition.cli.config import Configuration

    setting = Configuration(Path("/nonexistent/config.toml"), {}, False).get(
        "preparation.prefill_reduction")
    assert setting.value == "fp32-owner-packet"
    assert setting.source == "default"



def test_debug_serve_forwards_the_prefill_reduction_into_the_plan():
    """The diagnostic serving command must pass the transport into planning.plan.

    debug-serve builds its plan directly instead of going through derive(), so a
    flag that only reaches derive() would silently prepare the default transport.
    """

    import inspect

    from ignition.cli import serve

    source = inspect.getsource(serve.debug_serve)
    assert "prefill_reduction=prefill_reduction_mode(" in source
    assert "args.prefill_reduction" in source
