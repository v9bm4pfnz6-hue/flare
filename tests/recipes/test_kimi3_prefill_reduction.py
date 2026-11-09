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
