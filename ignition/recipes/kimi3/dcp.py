# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Decode context parallel placement of Kimi K3 latent records.

As in vLLM's Kimi K3 serving (decode context parallel size 16, KV interleave 1), ranks own single
interleaved tokens: global position t belongs to rank t % W at local index t // W. Every rank addresses
its local records through the shared page tables: a page holds PAGE_TOKENS local records, which cover
PAGE_TOKENS * W global positions. Decode and verification gather every rank's query heads, attend over
local records and combine the partial outputs by their log-sum-exp; prefill copies the request's earlier
context from every rank's cache into a slab one window at a time and attends with this rank's heads only.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .config import PAGE_TOKENS

# One E4M3 latent record: 512 normalized latent values and 64 unrotated key values.
RECORD_BYTES = 576
# Earlier context positions a prefill layer gathers from every rank at once.
CONTEXT_WINDOW = 131072


@dataclass(frozen=True)
class ContextShards:
    """One rank's share of the context. Compiled forwards are shared by every rank, so the rank they use
    is a device tensor bound per rank rather than a traced constant."""

    rank: int = 0
    world: int = 1
    index: torch.Tensor | None = None

    def __post_init__(self):
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError("Kimi K3 context shards require a rank within a positive world")
        if self.world > 1 and (self.index is None or self.index.shape != (1,) or self.index.dtype != torch.int64):
            raise ValueError("Kimi K3 context shards require the rank as a one-element int64 device tensor")

    @classmethod
    def on(cls, rank, world, device):
        return cls(rank, world, torch.tensor([rank], dtype=torch.int64, device=device))

    @property
    def owner(self):
        return 0 if self.index is None else self.index

    @property
    def page_tokens(self):
        return PAGE_TOKENS * self.world

    def token_slots(self, positions, tables):
        """Local physical record of each global position on this rank, -1 where another rank owns it.

        `tables` holds the page-table row of every position; negative positions are inactive.
        """
        local = positions.clamp(min=0) // self.world
        column = (local // PAGE_TOKENS).clamp(0, tables.shape[-1] - 1)
        page = tables.gather(-1, column[..., None].long()).squeeze(-1).long()
        live = (positions >= 0) & (positions % self.world == self.owner) & (page >= 0)
        return torch.where(live, page * PAGE_TOKENS + local % PAGE_TOKENS, -1)

    def local_tokens(self, lengths):
        """This rank's share of the first `lengths` global positions."""
        return lengths // self.world + (lengths % self.world > self.owner).to(lengths.dtype)


@triton.jit
def _pull_context(Addresses, Table, Window, Slab, columns, WORLD: tl.constexpr, PAGE: tl.constexpr,
                  TOKENS: tl.constexpr, WORDS: tl.constexpr, WORD_BLOCK: tl.constexpr):
    # Copy the records of global positions in [Window[0], Window[1]) into the slab from its first row, reading
    # each from its owner's cache: this rank's own memory or a peer's through CUDA IPC.
    first = tl.program_id(0) * TOKENS
    begin = tl.load(Window).to(tl.int64)
    end = tl.load(Window + 1).to(tl.int64)
    if begin + first < end:
        word = tl.arange(0, WORD_BLOCK)
        for offset in tl.range(0, TOKENS):
            token = begin + first + offset
            if token < end:
                owner = token % WORLD
                local = token // WORLD
                page = tl.load(Table + tl.minimum(local // PAGE, columns - 1)).to(tl.int64)
                source = tl.load(Addresses + owner).to(tl.pointer_type(tl.int32))
                values = tl.load(source + (page * PAGE + local % PAGE) * WORDS + word, mask=word < WORDS)
                tl.store(Slab + (first + offset) * WORDS + word, values, mask=word < WORDS)


@triton.jit
def _write_records(Latent, Indices, Table, latent_stride, page_stride, WIDTH: tl.constexpr, RECORDS: tl.constexpr,
                   BLOCK: tl.constexpr):
    # Write one token's record, saturated to E4M3 with a unit scale, at its slot: one record of a page of RECORDS.
    row = tl.program_id(0)
    index = tl.load(Indices + row).to(tl.int64)
    if index >= 0:
        column = tl.arange(0, BLOCK)
        values = tl.load(Latent + row * latent_stride + column, mask=column < WIDTH, other=0.).to(tl.float32)
        values = tl.minimum(tl.maximum(values, -448.), 448.)
        record = Table + index // RECORDS * page_stride + index % RECORDS * WIDTH
        tl.store(record + column, values.to(Table.dtype.element_ty), mask=column < WIDTH)


def _launch(kernel, grid, args, **meta):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **meta)
    else:
        kernel[grid](*args, **meta)


_PULL_TOKENS = 16


def pull_context(addresses, table, window, slab, shards):
    """Copy every rank's records of the positions in `window`, a [begin, end) int64 pair no wider than the slab,
    into the slab [rows, 144] int32 words from its first row."""
    rows = slab.shape[0]
    _launch(_pull_context, (triton.cdiv(rows, _PULL_TOKENS),), (addresses, table, window, slab, table.shape[0]),
            WORLD=shards.world, PAGE=PAGE_TOKENS, TOKENS=_PULL_TOKENS, WORDS=RECORD_BYTES // 4, WORD_BLOCK=256,
            num_warps=4)


@triton.jit
def _pull_index_context(Addresses, Table, End, Destination, columns,
                        WORLD: tl.constexpr, PAGE: tl.constexpr, ROWS: tl.constexpr):
    # Preserve the original FP8 bytes and FP32 scales in planar 64-record pages.
    tokens = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    words = tl.arange(0, 32)
    end = tl.load(End).to(tl.int64)
    local = tokens // WORLD
    live = tokens < end
    block = local // PAGE
    page = tl.load(Table + tl.minimum(block, columns - 1), mask=live, other=-1).to(tl.int64)
    live &= (block < columns) & (page >= 0)
    source = tl.load(Addresses + tokens % WORLD).to(tl.pointer_type(tl.int32))
    index_page = page * 2 + local % PAGE // 64
    src = index_page * (64 * 33) + local % 64 * 32
    dst = tokens // 64 * (64 * 33) + tokens % 64 * 32
    keys = tl.load(source[:, None] + src[:, None] + words[None, :], mask=live[:, None], other=0)
    tl.store(Destination + dst[:, None] + words[None, :], keys, mask=live[:, None])
    scale = tl.load(source + index_page * (64 * 33) + 64 * 32 + local % 64, mask=live, other=0)
    tl.store(Destination + tokens // 64 * (64 * 33) + 64 * 32 + tokens % 64, scale, mask=live)


def pull_index_context(addresses, table, end, destination, shards):
    """Copy committed index history into a contiguous, byte-identical planar index cache."""
    records = destination.shape[0] * 64
    _launch(_pull_index_context, (triton.cdiv(records, 8),),
            (addresses, table, end, destination.view(torch.int32), table.numel()),
            WORLD=shards.world, PAGE=PAGE_TOKENS, ROWS=8, num_warps=4)


def write_records(latent, indices, table):
    """Write each token's BF16 record [tokens, 576], saturated to E4M3 as vLLM's FP8 cache stores it, at its
    slot of `table`: a row of [rows, 576], or page * records + record of [pages, records, 576] with any page
    stride, such as rings in continuation rows; negative indices skip their tokens."""
    pages = table if table.ndim == 3 else table[:, None]
    if pages.shape[2] != RECORD_BYTES or pages.stride(2) != 1 or pages.shape[1] > 1 and pages.stride(1) != RECORD_BYTES:
        raise ValueError("record tables hold contiguous 576-byte records")
    _launch(_write_records, (latent.shape[0],), (latent, indices, pages, latent.stride(0), pages.stride(0)),
            WIDTH=RECORD_BYTES, RECORDS=pages.shape[1], BLOCK=triton.next_power_of_2(RECORD_BYTES), num_warps=4)


@dataclass(frozen=True)
class ContextParallel:
    """Shared context placement and head/LSE exchanges for latent attention.

    Peer page addresses are {layer: [world] int64}. Dense prefill uses a bounded
    E4M3 context slab [window + chunk, 576]. Gathered sparse prefill instead uses
    full-context latent/index slabs and exchanges final selected positions;
    decode and verification retain context-sharded candidates and head/LSE exchange.
    """

    shards: ContextShards
    heads: object
    coordinator: object
    peers: dict
    slab: torch.Tensor
    window: int = CONTEXT_WINDOW
    candidates: object = None
    prefill_layout: str = "dcp"
    index_peers: dict | None = None
    index_slab: torch.Tensor | None = None
    selections: object = None
    paired_queries: object = None
    index_exchange: object = None


# Measured 2026-11-09 on the nine-GPU RTX PRO 6000 TP9 topology. Each geometry
# matched 512x8 bitwise and did not regress in two interleaved 50-sample runs;
# evidence is validation/lse-geometry-summary-20261109.json in the candidate tree.
_LSE_GEOMETRY_BY_ROWS = {
    1: (512, 16),
    2: (384, 32),
    3: (512, 16),
    4: (512, 16),
    6: (128, 32),
    7: (384, 16),
    9: (384, 16),
    10: (128, 32),
    11: (384, 16),
    12: (256, 32),
    13: (384, 32),
    14: (384, 32),
    15: (384, 16),
    16: (512, 16),
    17: (384, 16),
    18: (512, 16),
    19: (512, 16),
    20: (512, 16),
}


def lse_geometry(rows, world):
    """Return the measured TP9 DCP LSE reduce-scatter launcher geometry."""
    if world != 9:
        return 512, 8
    if 21 <= rows <= 32:
        return 384, 32
    return _LSE_GEOMETRY_BY_ROWS.get(rows, (512, 8))
