# SPDX-License-Identifier: Apache-2.0

"""Ring alignment for owner-packet reductions pads only unaligned payloads."""

import pytest

from ignition.components.owner_packet_reduce import aligned_extent


def test_world_multiple_payloads_keep_their_extent():
    assert aligned_extent(4608 * 7168, 9) * 9 == 4608 * 7168
    assert aligned_extent(144 * 7168, 9) * 9 == 144 * 7168
    assert aligned_extent(72, 9) * 9 == 72


def test_unaligned_payload_pads_to_whole_fp32_groups_per_rank():
    elements = 128 * 7168
    extent = aligned_extent(elements, 9)
    assert extent * 9 > elements
    assert extent * 9 - elements < 9 * 8
    assert extent % 8 == 0
    assert (extent * 9) % (9 * 8) == 0


def test_owner_packet_grouping_survives_every_kimi_prefill_capacity():
    for rows in (128, 256, 1024, 4096, 4608):
        elements = rows * 7168
        extent = aligned_extent(elements, 9)
        assert extent % 8 == 0
        assert extent * 9 >= elements


@pytest.mark.parametrize("elements,world", [(0, 9), (-8, 9), (72, 0), (72, -1)])
def test_aligned_extent_rejects_nonpositive_inputs(elements, world):
    with pytest.raises(ValueError):
        aligned_extent(elements, world)
