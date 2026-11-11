# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared per-replay DSA selection metadata for decode and speculative verification."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .config import PAGE_TOKENS
from .dsa import SELECTION_ROWS


@dataclass(frozen=True)
class SelectionMetadata:
    """Expanded 64-record index pages, local cache lengths, per-slice widths, and physical write slots."""

    pages: torch.Tensor
    lengths: torch.Tensor
    active: torch.Tensor
    slots: torch.Tensor | None = None


@triton.jit
def _prepare_selection_rows(Table, Positions, OwnerPtr, PagesOut, LengthsOut, SlotsOut,
                            table_row_stride, table_col_stride, pages_out_stride,
                            WIDTH: tl.constexpr, WORLD: tl.constexpr, PAGE: tl.constexpr,
                            TENSOR_OWNER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    owner = tl.load(OwnerPtr).to(tl.int64) if TENSOR_OWNER else tl.zeros((), dtype=tl.int64)
    pos = tl.load(Positions + row).to(tl.int64)

    # 1) Expand model page table [rows, WIDTH] -> 64-record index pages [rows, 2 * WIDTH]
    c_base = 0
    while c_base < WIDTH:
        cols = c_base + tl.arange(0, BLOCK)
        mask = cols < WIDTH
        p = tl.load(Table + row * table_row_stride + cols * table_col_stride, mask=mask, other=-1).to(tl.int32)
        even = tl.where(p >= 0, p * 2, -1)
        odd = tl.where(p >= 0, p * 2 + 1, -1)
        tl.store(PagesOut + row * pages_out_stride + cols * 2, even, mask=mask)
        tl.store(PagesOut + row * pages_out_stride + cols * 2 + 1, odd, mask=mask)
        c_base += BLOCK

    # 2) Local visible token count: shards.local_tokens(pos + 1).clamp(min=0)
    total_len = pos + 1
    if total_len <= 0:
        local_len = tl.zeros((), dtype=tl.int32)
    else:
        q = total_len // WORLD
        rem = total_len - q * WORLD
        local_len = (q + tl.where(rem > owner, 1, 0)).to(tl.int32)
    tl.store(LengthsOut + row, local_len)

    # 3) Local physical slot: shards.token_slots(positions, token_page_table)
    clamped_pos = tl.maximum(pos, 0)
    local_idx = clamped_pos // WORLD
    page_col = tl.minimum(local_idx // PAGE, WIDTH - 1)
    tok_page = tl.load(Table + row * table_row_stride + page_col * table_col_stride).to(tl.int64)
    rem_pos = pos - (pos // WORLD) * WORLD
    live_slot = (pos >= 0) & (rem_pos == owner) & (tok_page >= 0)
    slot = tl.where(live_slot, tok_page * PAGE + (local_idx % PAGE), -1)
    tl.store(SlotsOut + row, slot)


@triton.jit
def _compute_stage_active(Lengths, ActiveOut, rows, STAGE_CAP: tl.constexpr, BLOCK: tl.constexpr):
    stage = tl.program_id(0)
    offs = stage * STAGE_CAP + tl.arange(0, BLOCK)
    vals = tl.load(Lengths + offs, mask=(tl.arange(0, BLOCK) < STAGE_CAP) & (offs < rows), other=0)
    tl.store(ActiveOut + stage, tl.max(vals, axis=0))


class PreparedSelectionMetadata:
    """Compute the layer-independent DSA selection inputs and write slots once per replay."""

    def __init__(self, metadata, shards, *, rows: int, device, allocator=None, slice_rows: int = SELECTION_ROWS):
        table = metadata.token_page_table
        if (rows <= 0 or table.shape[0] < rows or metadata.positions.shape[0] < rows
                or slice_rows <= 0):
            raise ValueError("selection metadata requires positive row/slice counts covered by the batch inputs")
        width = table.shape[1]
        capacity = min(rows, slice_rows)
        stages = -(-rows // capacity)

        def allocate(shape, dtype):
            # Every graph variant allocates from its own cursor rather than sharing a cross-variant dict.
            return (torch.zeros(shape, dtype=dtype, device=device) if allocator is None
                    else allocator.zeros(shape, dtype))

        self.metadata = metadata
        self.shards = shards
        self.rows = rows
        self.capacity = capacity
        self.stages = stages
        self.width = width
        self.output = SelectionMetadata(
            pages=allocate((rows, 2 * width), torch.int32),
            lengths=allocate((rows,), torch.int32),
            active=allocate((stages,), torch.int32),
            slots=allocate((rows,), torch.int64),
        )
        self.pages = self.output.pages
        self.lengths = self.output.lengths
        self.active = self.output.active
        self.slots = self.output.slots

    def __call__(self) -> SelectionMetadata:
        m, out = self.metadata, self.output
        table = m.token_page_table[:self.rows]
        positions = m.positions[:self.rows]
        owner = self.shards.owner
        tensor_owner = isinstance(owner, torch.Tensor)
        col_block = min(1024, max(32, triton.next_power_of_2(min(self.width, 512))))
        torch.library.wrap_triton(_prepare_selection_rows)[(self.rows,)](
            table, positions, owner if tensor_owner else positions,
            out.pages, out.lengths, out.slots,
            table.stride(0), table.stride(1), out.pages.stride(0),
            self.width, self.shards.world, PAGE_TOKENS, tensor_owner, col_block,
            num_warps=4,
        )
        stage_block = max(16, triton.next_power_of_2(self.capacity))
        torch.library.wrap_triton(_compute_stage_active)[(self.stages,)](
            out.lengths, out.active, self.rows, self.capacity, stage_block, num_warps=1,
        )
        return out
