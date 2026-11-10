# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Layer-independent DSA selection metadata for one decode or verification step."""

from dataclasses import dataclass

import torch

from .dsa import SELECTION_ROWS


@dataclass(frozen=True)
class SelectionMetadata:
    """Expanded index page table, per-row shard lengths, and per-slice active widths."""

    pages: torch.Tensor
    lengths: torch.Tensor
    active: torch.Tensor


class PreparedSelectionMetadata:
    """Derive the DSA selection metadata that every DSA layer of one step shares.

    The page-table expansion, the per-rank visible length and the active width depend only
    on the request metadata, not on the layer, so the decoder writes them once per replay
    into graph-owned buffers and the layers bind slices of them. Values are identical to
    the per-layer sequence in dsa.PreparedDSA: two planar index pages per model page, the
    shard's visible token count, and the maximum count within each 32-row slice.
    """

    def __init__(self, metadata, shards, *, rows, device, buffers, allocator=None):
        if rows <= 0:
            raise ValueError("shared DSA selection metadata requires query rows")
        self.metadata, self.shards, self.rows = metadata, shards, rows
        self.capacity = min(rows, SELECTION_ROWS)
        self.stages = -(-rows // self.capacity)
        self.halves = torch.arange(2, dtype=torch.int32, device=device)
        width = metadata.token_page_table.shape[1]

        def allocate(key, shape, dtype):
            existing = buffers.get(key)
            if existing is None or existing.shape[0] < shape[0]:
                buffers[key] = (torch.zeros(shape, dtype=dtype, device=device) if allocator is None
                                else allocator.zeros(shape, dtype))
            return buffers[key][:shape[0]]

        self.pages = allocate("selection_pages", (rows, 2 * width), torch.int32)
        self.lengths = allocate("selection_lengths", (rows,), torch.int32)
        self.active = allocate("selection_active", (self.stages,), torch.int32)
        self.metadata_view = SelectionMetadata(self.pages, self.lengths, self.active)

    def __call__(self):
        m = self.metadata
        positions = m.positions[:self.rows]
        table = m.token_page_table[:self.rows]
        expanded = table[:, :, None] * 2 + self.halves
        self.pages[:self.rows].copy_(torch.where(table[:, :, None] >= 0, expanded, -1).flatten(1))
        lengths = self.shards.local_tokens((positions + 1).clamp_min(0)).to(torch.int32)
        self.lengths[:self.rows].copy_(lengths)
        for stage in range(self.stages):
            values = lengths[stage * self.capacity:(stage + 1) * self.capacity]
            self.active[stage:stage + 1].copy_(values.max().reshape(1))
        return self.metadata_view
