# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the fused DSA selection prologue and candidate packing.

The Triton kernels are GPU-only. These tests run the recipe launch wiring through a
torch emulation of each kernel and compare the written buffers bitwise against the
elementwise torch sequence the fused launches replace. The kernels themselves are
validated on GPUs by the ticket under optimization-team/tickets/dsa/.
"""

import pytest
import torch

from ignition.components import distributed_topk
from ignition.recipes.kimi3 import dsa


def _selection_metadata_emulation(arguments, metadata):
    positions, table, rank, pages, lengths, table_stride, pages_stride = arguments
    world, width = metadata["WORLD"], metadata["WIDTH"]
    owner = int(rank.reshape(-1)[0])
    for row in range(positions.shape[0]):
        length = max(int(positions[row]) + 1, 0)
        lengths[row] = length // world + int(length % world > owner)
        flat, output = table.reshape(-1), pages.reshape(-1)
        for column in range(width):
            page = int(flat[row * table_stride + column])
            target = row * pages_stride + 2 * column
            output[target] = 2 * page if page >= 0 else -1
            output[target + 1] = 2 * page + 1 if page >= 0 else -1


def _selection_maximum_emulation(arguments, metadata):
    lengths, active, count = arguments
    active[0] = lengths[:count].max()


def _pack_candidates_emulation(arguments, metadata):
    indices, scores, local, rank, row_stride = arguments
    world, topk = metadata["WORLD"], metadata["TOPK"]
    owner = int(rank.reshape(-1)[0])
    bits = scores.view(torch.int32)
    for row in range(indices.shape[0]):
        live = indices[row] >= 0
        local[row, :topk, 0] = torch.where(live, bits[row], torch.full_like(bits[row], -8388608))
        local[row, :topk, 1] = torch.where(
            live, indices[row] * world + owner, torch.full_like(indices[row], -1))


def _emulate(kernel, grid, arguments, **metadata):
    if kernel is dsa._selection_metadata:
        _selection_metadata_emulation(arguments, metadata)
    elif kernel is dsa._selection_maximum:
        _selection_maximum_emulation(arguments, metadata)
    elif kernel is distributed_topk._pack_candidates:
        _pack_candidates_emulation(arguments, metadata)
    else:
        raise AssertionError(f"unexpected kernel {kernel!r}")


@pytest.mark.parametrize("count,width,world,rank", [
    (1, 7, 9, 0), (2, 7, 9, 8), (6, 7, 9, 4), (8, 7, 9, 3),
    (24, 7, 9, 7), (32, 7, 9, 0), (24, 1, 1, 0), (3, 911, 9, 5),
])
def test_selection_metadata_matches_torch_sequence(monkeypatch, count, width, world, rank):
    monkeypatch.setattr(dsa, "_launch", _emulate)
    generator = torch.Generator().manual_seed(count * 131 + width * 17 + world * 7 + rank)
    positions = torch.randint(-1, 1 << 20, (count,), generator=generator, dtype=torch.int64)
    table = torch.randint(-1, 512, (count, width), generator=generator, dtype=torch.int32)
    capacity = dsa.SELECTION_ROWS
    pages = torch.full((capacity, 2 * width), 1234, dtype=torch.int32)
    lengths = torch.full((capacity,), 4321, dtype=torch.int32)
    active = torch.full((1,), 99, dtype=torch.int32)
    dsa._write_selection_metadata(positions, table, pages[:count], lengths[:count], active, world,
                                  torch.tensor([rank], dtype=torch.int64))
    expanded = table[:, :, None] * 2 + torch.arange(2, dtype=torch.int32)
    clamped = (positions + 1).clamp_min(0)
    expected_pages = torch.where(table[:, :, None] >= 0, expanded, -1).flatten(1)
    expected_lengths = (clamped // world + (clamped % world > rank).to(torch.int64)).to(torch.int32)
    assert torch.equal(pages[:count], expected_pages)
    assert torch.equal(lengths[:count], expected_lengths)
    assert active.item() == int(expected_lengths.max())
    # The buffers keep their declared capacity; rows beyond the slice are untouched.
    assert torch.all(pages[count:] == 1234)
    assert torch.all(lengths[count:] == 4321)


@pytest.mark.parametrize("rows,topk,world,rank", [
    (1, 512, 9, 0), (2, 2048, 9, 8), (7, 2048, 9, 4), (32, 2048, 9, 3), (1, 2048, 2, 1),
])
def test_candidate_packing_matches_torch_sequence(monkeypatch, rows, topk, world, rank):
    monkeypatch.setattr(distributed_topk, "_launch", _emulate)
    generator = torch.Generator().manual_seed(rows * 13 + topk + world * 7 + rank)
    indices = torch.randint(-1, 4608, (rows, topk), generator=generator, dtype=torch.int32)
    scores = torch.randn((rows, topk), generator=generator, dtype=torch.float32)
    local = torch.zeros((rows, topk, 2), dtype=torch.float32)
    distributed_topk._pack_local_candidates(indices, scores, local,
                                            torch.tensor([rank], dtype=torch.int64), world)
    live = indices >= 0
    bits = scores.view(torch.int32)
    expected = torch.zeros_like(local.view(torch.int32))
    expected[:, :, 0] = torch.where(live, bits, torch.full_like(bits, -8388608))
    expected[:, :, 1] = torch.where(live, indices.long() * world + rank,
                                    torch.full_like(indices, -1)).to(torch.int32)
    assert torch.equal(local.view(torch.int32), expected)


def test_record_mapper_writes_caller_buffers(monkeypatch):
    seen = {}

    def fake_launch(kernel, grid, arguments, **metadata):
        seen["kernel"], seen["grid"] = kernel, grid
        output, count = arguments[4], arguments[5]
        output.copy_(arguments[0])
        count.fill_(arguments[0].shape[1])

    monkeypatch.setattr(distributed_topk, "_launch", fake_launch)
    selected = torch.arange(12, dtype=torch.int32).reshape(2, 6)
    output = torch.full((2, 6), -7, dtype=torch.int32)
    counts = torch.full((2,), -7, dtype=torch.int32)
    result = distributed_topk.selected_local_records(
        selected, torch.zeros(2, dtype=torch.int64), torch.zeros((2, 1), dtype=torch.int32),
        world=9, rank=torch.tensor([0], dtype=torch.int64), page_size=128,
        output=output, counts=counts)
    assert result[0] is output and result[1] is counts
    assert torch.equal(output, selected) and torch.equal(counts, torch.full((2,), 6, dtype=torch.int32))
    assert seen["kernel"] is distributed_topk._records and seen["grid"] == (2,)
    with pytest.raises(ValueError):
        distributed_topk.selected_local_records(
            selected, torch.zeros(2, dtype=torch.int64), torch.zeros((2, 1), dtype=torch.int32),
            world=9, rank=torch.tensor([0], dtype=torch.int64), page_size=128,
            output=torch.zeros((2, 3), dtype=torch.int32), counts=counts)
