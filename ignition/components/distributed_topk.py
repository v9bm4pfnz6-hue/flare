# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact selection from context shards, with bounded candidate communication."""

import torch
import triton
import triton.language as tl
from .candidate_merge import PreparedCandidateMerge

from .collectives import PreparedAllGather


def _launch(kernel, grid, arguments, **metadata):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*arguments, **metadata)
    else:
        kernel[grid](*arguments, **metadata)


@triton.jit
def _pack_candidates(Indices, Scores, Packed, Rank, row_stride,
                     TOPK: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    # One packed slot pair per candidate: the score bits, and the global position bits of a
    # live candidate. Unavailable candidates carry -inf and -1, as the merge expects.
    row = tl.program_id(0)
    block = tl.program_id(1)
    rank = tl.load(Rank).to(tl.int32)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    mask = columns < TOPK
    index = tl.load(Indices + row * row_stride + columns, mask=mask, other=-1)
    score = tl.load(Scores + row * row_stride + columns, mask=mask, other=0.0)
    live = index >= 0
    slot = Packed + row * (2 * row_stride) + 2 * columns
    tl.store(slot, tl.where(live, score.to(tl.int32, bitcast=True), -8388608), mask=mask)
    tl.store(slot + 1, tl.where(live, index * WORLD + rank, -1), mask=mask)


def _pack_local_candidates(indices, scores, local, rank, world):
    """Write this rank's candidate score/global-position pairs for the all-gather.

    The kernel addresses candidates and packed slots with row strides only and assumes unit
    inner strides, matching contiguous caller-owned slices; validate that instead of letting a
    mismatched layout write out of place.
    """
    rows, topk = indices.shape
    if (indices.dtype != torch.int32 or scores.dtype != torch.float32
            or scores.shape != indices.shape
            or indices.stride(0) != scores.stride(0)
            or indices.stride(1) != 1 or scores.stride(1) != 1
            or local.dtype != torch.float32 or local.shape != (rows, topk, 2)
            or local.stride() != (2 * topk, 2, 1)):
        raise ValueError("candidate packing requires matching contiguous int32/FP32 candidate rows")
    _launch(_pack_candidates, (rows, triton.cdiv(topk, 1024)),
            (indices, scores, local.view(torch.int32), rank, indices.stride(0)),
            TOPK=topk, WORLD=world, BLOCK=1024, num_warps=4)


class PreparedDistributedTopK:
    """Merge each rank's local top-k into global logical positions.

    Context position ``p`` belongs to rank ``p % world`` at local offset ``p // world``.
    Every rank contributes k candidates: keeping only k/world could discard global winners.
    Candidate IDs travel as integer bits alongside FP32 scores; the collective transports bytes.
    Missing local positions have score -inf and ID -1. All ranks receive identical candidates.
    """

    def __init__(self, indices, scores, output, *, world, rank, runtime, session, coordinator,
                 name, buffers, workspace):
        if (indices.ndim != 2 or indices.dtype != torch.int32 or scores.shape != indices.shape
                or scores.dtype != torch.float32 or output.shape != indices.shape or output.dtype != indices.dtype
                or world < 2 or rank.shape != (1,) or rank.dtype != torch.int64):
            raise ValueError("distributed top-k requires matching int32 positions, FP32 scores and a rank tensor")
        rows, self.topk = indices.shape
        self.indices, self.scores, self.output = indices, scores, output
        self.world, self.rank = world, rank
        self.local = buffers.zeros((rows, self.topk, 2), torch.float32)
        self.gathered = buffers.zeros((rows, world * self.topk, 2), torch.float32)
        self.merge = PreparedCandidateMerge(self.gathered, output, session=session,
            workspace=workspace, name=name + ".merge")
        self.collective = PreparedAllGather(self.local.view(torch.bfloat16).view(rows, -1),
            self.gathered.view(torch.bfloat16).view(rows, -1), runtime=runtime, session=session,
            coordinator=coordinator, name=name)

    def __call__(self):
        _pack_local_candidates(self.indices, self.scores, self.local, self.rank, self.world)
        self.collective(self.local.view(torch.bfloat16).view(self.indices.shape[0], -1))
        self.merge()
        return self.output



@triton.jit
def _records(Selected, Positions, Pages, Rank, Output, Counts,
             selected_row_stride, selected_col_stride, position_stride,
             page_row_stride, page_col_stride,
             WIDTH: tl.constexpr, COLUMNS: tl.constexpr, WORLD: tl.constexpr,
             PAGE: tl.constexpr, BLOCK: tl.constexpr, TENSOR_RANK: tl.constexpr,
             STATIC_RANK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    selected = tl.load(Selected + row*selected_row_stride + col*selected_col_stride,
                       mask=col < WIDTH, other=-1)
    position = tl.load(Positions + row*position_stride)
    rank = STATIC_RANK
    if TENSOR_RANK:
        rank = tl.load(Rank)
    live = (col < WIDTH) & (selected >= 0) & (selected <= position) & (selected % WORLD == rank)
    count = tl.sum(live.to(tl.int32), 0)
    logical = tl.sort(tl.where(live, selected, 2147483647), descending=False)
    local = logical.to(tl.int64)//WORLD
    page_col = tl.minimum(local//PAGE, COLUMNS-1)
    page = tl.load(Pages + row*page_row_stride + page_col*page_col_stride,
                   mask=(col < WIDTH) & (col < count), other=-1).to(tl.int64)
    physical = tl.where((col < count) & (page >= 0), page*PAGE + local % PAGE, -1)
    tl.store(Output + row*WIDTH + col, physical.to(tl.int32), mask=col < WIDTH)
    tl.store(Counts + row, count)


def selected_local_records(selected, positions, pages, *, world, rank, page_size, output=None, counts=None):
    """Compact selected global positions into this rank's physical records.

    ``pages`` is a model-page table per query row. The selected positions are independently
    bounded by that row's causal position; negative positions denote inactive rows. Callers may
    pass contiguous int32 ``output`` and ``counts`` buffers to skip a staging copy.
    """
    if output is None:
        output = torch.empty_like(selected, memory_format=torch.contiguous_format)
    elif (output.shape != selected.shape or output.dtype != selected.dtype
          or not output.is_contiguous() or output.device != selected.device):
        raise ValueError("record mapper output must be a contiguous selection-shaped int32 buffer")
    if counts is None:
        count = torch.empty((selected.shape[0],), dtype=torch.int32, device=selected.device)
    elif (counts.shape != (selected.shape[0],) or counts.dtype != torch.int32
          or not counts.is_contiguous() or counts.device != selected.device):
        raise ValueError("record mapper counts must be a contiguous int32 row buffer")
    else:
        count = counts
    tensor_rank = isinstance(rank, torch.Tensor)
    _launch(_records, (selected.shape[0],), (
        selected, positions, pages, rank if tensor_rank else positions, output, count,
        *selected.stride(), positions.stride(0), *pages.stride(),
        selected.shape[1], pages.shape[1], world, page_size,
        triton.next_power_of_2(selected.shape[1]), tensor_rank,
        0 if tensor_rank else rank), num_warps=8)
    return output, count
