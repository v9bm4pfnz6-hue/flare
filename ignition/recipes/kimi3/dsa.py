# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated Kimi DSA: learned selection over context-sharded FP8 keys and latent records."""

import torch
import triton
import triton.language as tl
from b12x.attention import dense_mla

from ignition.components.collectives import PreparedAllGather, PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.distributed_topk import PreparedDistributedTopK, selected_local_records
from ignition.components.fp8_indexer import PreparedFP8Indexer

from .config import PAGE_TOKENS
from .dcp import ContextShards, RECORD_BYTES, lse_geometry, pull_context
from .projections import prepare_projection

# Slice selection and its cross-rank candidate exchange so prefill scratch is context-independent.
# Slice selection and its cross-rank candidate exchange so prefill scratch is context-independent.
SELECTION_ROWS = 32
INDEX_HEADS, INDEX_DIM, TOPK = 32, 128, 2048


def _launch(kernel, grid, arguments, **metadata):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*arguments, **metadata)
    else:
        kernel[grid](*arguments, **metadata)


@triton.jit
def _selection_metadata(Positions, Table, Rank, Pages, Lengths, table_stride, pages_stride,
                        WORLD: tl.constexpr, STEPS: tl.constexpr, WIDTH: tl.constexpr,
                        BLOCK: tl.constexpr):
    # One program per query row: expand the model page table into the two planar FP8 index
    # pages a model page owns, and derive this shard's visible length of the row's prefix.
    row = tl.program_id(0)
    rank = tl.load(Rank)
    length = tl.maximum(tl.load(Positions + row).to(tl.int64) + 1, 0)
    local = length // WORLD + (length % WORLD > rank).to(tl.int64)
    tl.store(Lengths + row, local.to(tl.int32))
    source = Table + row * table_stride
    target = Pages + row * pages_stride
    for step in tl.static_range(STEPS):
        columns = step * BLOCK + tl.arange(0, BLOCK)
        mask = columns < WIDTH
        page = tl.load(source + columns, mask=mask, other=-1)
        live = page >= 0
        doubled = 2 * page
        # An inactive page table entry maps both halves of the row to -1, as the torch form does.
        tl.store(target + 2 * columns, tl.where(live, doubled, -1), mask=mask)
        tl.store(target + 2 * columns + 1, tl.where(live, doubled + 1, -1), mask=mask)


@triton.jit
def _selection_maximum(Lengths, Active, count, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(Lengths + offsets, mask=offsets < count, other=0)
    tl.store(Active, tl.max(values, 0))


def _write_selection_metadata(positions, table, pages, lengths, active, world, rank):
    """Write the index page table, shard lengths and active width of one selection slice.

    The integer values are identical to the elementwise torch sequence this replaces
    (where(table >= 0, table * 2 + halves, -1), local_tokens(positions + 1) and
    lengths.max()); the buffers keep their declared capacities and rows beyond
    ``count`` are untouched.
    """
    count = positions.shape[0]
    width = pages.shape[1] // 2
    if count == 0 or width == 0:
        return
    if count > SELECTION_ROWS:
        raise ValueError("selection metadata rows exceed the prepared selection slice")
    _launch(_selection_metadata, (count,),
            (positions, table, rank, pages, lengths, table.stride(0), pages.stride(0)),
            WORLD=world, STEPS=(width + 255) // 256, WIDTH=width, BLOCK=256, num_warps=4)
    _launch(_selection_maximum, (1,), (lengths, active, count),
            BLOCK=SELECTION_ROWS, num_warps=1)


class PreparedDSA:
    def __init__(self, weights, query_residual, key_input, weight_input, query, output, slots,
                 metadata, cache, index_cache, *, context, session, coordinator, name, workspace,
                 buffers, allocator=None, eps=1e-5, mode="decode", latent=None, peers=None, index_peers=None):
        self.prefill = None
        if mode == "prefill" and context is not None and context.prefill_layout == "query_split":
            from .dsa_prefill import PreparedGatheredDSA

            self.prefill = PreparedGatheredDSA(weights, query_residual, key_input, weight_input,
                query, output, slots, metadata, cache, index_cache, context=context, session=session,
                coordinator=coordinator, name=name, workspace=workspace, buffers=buffers,
                latent=latent, peers=peers, index_peers=index_peers, eps=eps)
            return
        self.gathered_context = mode == "prefill" and context is not None and context.prefill_layout == "ckv"
        self.context, self.peers = context, peers
        if self.gathered_context:
            if metadata.starts.numel() != 1 or peers is None or context.slab is None:
                raise ValueError("gathered DSA prefill requires one request and a shared context slab")
            self.window = torch.zeros(2, dtype=torch.int64, device=query.device)
        self.metadata, self.query, self.output, self.slots = metadata, query, output, slots
        self.query_residual = query_residual
        self.eps = eps
        self.key_input, self.weight_input = key_input, weight_input
        self.norm_weight, self.norm_bias = weights["index_norm"], weights["index_bias"]
        self.shards = ContextShards() if context is None else context.shards
        rows, heads, _ = query.shape
        self.rows = rows
        self.step = min(rows, SELECTION_ROWS)
        device = query.device
        # Compiled forwards are shared by every rank, so the shard index stays a device tensor.
        index = self.shards.owner
        self.rank = index if isinstance(index, torch.Tensor) else torch.zeros((1,), dtype=torch.int64, device=device)

        def allocate(key, shape, dtype):
            existing = buffers.get(key)
            if existing is None or existing.shape[0] < shape[0]:
                buffers[key] = (torch.zeros(shape, dtype=dtype, device=device) if allocator is None
                                else allocator.zeros(shape, dtype))
            return buffers[key][:shape[0]]

        self.index_query = allocate("index_query", (rows, INDEX_HEADS, INDEX_DIM), torch.bfloat16)
        from ignition.components.weights import OutputPartitionedBF16Weight
        self.partitioned = isinstance(weights["index_query"], OutputPartitionedBF16Weight)
        self.local_index_query = self.index_query
        if self.partitioned:
            if context is None or context.paired_queries is None or self.gathered_context:
                raise ValueError("partitioned index decode requires paired context-parallel query exchange")
            self.local_index_query = allocate("index_query_local",
                (rows, INDEX_HEADS//self.shards.world, INDEX_DIM), torch.bfloat16)
        self.key = allocate("index_key", (rows, INDEX_DIM), torch.bfloat16)
        self.project = prepare_projection(query_residual, weights["index_query"], self.local_index_query.view(rows, -1),
            session=session, name=name + ".index_query", workspace=workspace)
        capacity = self.step
        self.pages = allocate("index_pages", (capacity, metadata.token_page_table.shape[1] * 2), torch.int32)
        self.lengths = allocate("index_lengths", (capacity,), torch.int32)
        self.active = allocate("index_active", (1,), torch.int32)
        self.local = allocate("index_local", (capacity, TOPK), torch.int32)
        self.scores = allocate("index_scores", (capacity, TOPK), torch.float32)
        self.selected = allocate("index_selected", (capacity, TOPK), torch.int32)
        self.physical = allocate("index_physical", (capacity, TOPK), torch.int32)
        self.counts = allocate("index_counts", (capacity,), torch.int32)
        self.offsets = torch.arange(capacity + 1, device=device, dtype=torch.int32)
        self.stages = []
        # Sequential slices and layers reuse the same selection buffers; each operator owns its binding.
        from ignition.preparation.scratch import GraphBufferPool
        if "index_workspace" not in buffers:
            # Each slice has at most 32 rows. Large per-dtype chunks would reserve
            # hundreds of MiB for query scales and other small intermediates.
            buffers["index_workspace"] = GraphBufferPool(device, chunk_bytes=2 << 20)
        pool = buffers["index_workspace"]
        cache_records = (context.slab if self.gathered_context else cache).view(-1, 1, RECORD_BYTES)
        selection_views = {}
        for begin in range(0, rows, capacity):
            end = min(begin + capacity, rows)
            count = end - begin
            stage_name = name + f".rows{begin}"
            cursor = pool.cursor()
            # Identical stage regions share tensor identities as well as their backing storage.
            if count not in selection_views:
                selection_views[count] = (self.pages[:count], self.lengths[:count], self.local[:count],
                    self.scores[:count], self.selected[:count], self.physical[:count], self.counts[:count],
                    self.offsets[:count + 1])
            pages, lengths, local, scores, selected, physical, counts, offsets = selection_views[count]
            selector = PreparedFP8Indexer(self.index_query[begin:end], weight_input[begin:end], index_cache,
                pages, lengths, self.active, local, mode="decode",
                session=session, name=stage_name + ".index", workspace=workspace, buffers=cursor,
                scores=scores, rotation="none")
            merge = None
            if self.shards.world > 1:
                merge = PreparedDistributedTopK(local, scores, selected,
                    world=self.shards.world, rank=self.shards.index, runtime=context.candidates,
                    session=session, coordinator=coordinator, name=stage_name + ".candidates", buffers=cursor,
                    workspace=workspace)
            gather = combine = paired = None
            q, attended = query[begin:end], output[begin:end]
            if self.gathered_context:
                quantized = cursor.zeros(q.shape, torch.float8_e4m3fn)
                q, lse = quantized, None
            elif context is not None:
                quantized = cursor.zeros(q.shape, torch.float8_e4m3fn)
                q = cursor.zeros((count, heads * self.shards.world, RECORD_BYTES), torch.float8_e4m3fn)
                attended = cursor.zeros((count, heads * self.shards.world, 512), torch.bfloat16)
                lse = cursor.zeros((count, heads * self.shards.world), torch.float32)
                if self.partitioned:
                    from ignition.components.peer_rows import PreparedPairedGather

                    paired = PreparedPairedGather(quantized.view(count, -1),
                        self.local_index_query[begin:end].view(count, -1), q.view(count, -1),
                        self.index_query[begin:end].view(count, -1), runtime=context.paired_queries,
                        session=session, coordinator=coordinator, name=stage_name + ".queries", buffers=cursor)
                else:
                    gather = PreparedAllGather(quantized.view(count, -1), q.view(count, -1), runtime=context.heads,
                        coordinator=coordinator, session=session, name=stage_name + ".gather_heads")
                threads, block_limit = lse_geometry(count, self.shards.world)
                combine = PreparedLseReduceScatter(attended, lse, output[begin:end], runtime=context.heads,
                    session=session, coordinator=coordinator, name=stage_name + ".combine",
                    threads=threads, block_limit=block_limit)
            else:
                quantized = lse = None
            caps = dense_mla.Caps(device=device, mode="decode", q_dtype=q.dtype, kv_dtype=cache.dtype,
                num_q_heads=q.shape[1], max_total_q=count, max_batch=count, page_size=1,
                max_cache_tokens=TOPK, max_page_table_width=TOPK, num_cache_pages=cache_records.shape[0],
                use_cuda_graph=True, partial_dtype=torch.float32)
            attention = PreparedDenseMLA(q, cache_records, attended,
                physical, counts, offsets, caps=caps,
                scale=192**-.5, session=session, name=stage_name + ".sparse_attention", workspace=workspace, lse=lse)
            self.stages.append((begin, end, selector, merge, quantized, gather, attention, combine, lse, paired))

    def __call__(self):
        if self.prefill is not None:
            return self.prefill()
        m, shards = self.metadata, self.shards
        self.project(self.query_residual)
        self.key.copy_(torch.nn.functional.layer_norm(self.key_input.float(), (INDEX_DIM,),
            self.norm_weight.float(), self.norm_bias.float(), self.eps).to(self.key.dtype))
        self.stages[0][2].write_keys(self.key, self.slots)
        for begin, end, selector, merge, quantized, gather, attention, combine, lse, paired in self.stages:
            count = end - begin
            positions, pages = m.positions[begin:end], m.token_page_table[begin:end]
            _write_selection_metadata(positions, pages, self.pages[:count], self.lengths[:count],
                self.active, shards.world, self.rank)
            if paired is not None:
                quantized.copy_(self.query[begin:end].float().clamp(-448., 448.).to(torch.float8_e4m3fn))
                paired(quantized.view(count, -1), self.local_index_query[begin:end].view(count, -1))
            selector()
            selected = self.local[:count] if merge is None else merge()
            if self.gathered_context:
                # The candidate exchange orders all ranks' cache writes before peer reads.
                if begin == 0:
                    self.window[0:1].zero_()
                    self.window[1:].copy_((m.positions.max() + 1).clamp_min(0).reshape(1))
                    pull_context(self.peers, m.page_table[0], self.window,
                                 self.context.slab.view(torch.int32), shards)
                live = (selected >= 0) & (selected <= positions[:, None])
                physical = torch.sort(torch.where(live, selected, torch.iinfo(torch.int32).max), dim=1).values
                lengths = live.sum(1).to(torch.int32)
                physical = torch.where(physical != torch.iinfo(torch.int32).max, physical, -1)
                self.physical[:count].copy_(physical)
                self.counts[:count].copy_(lengths)
            else:
                # The record mapper writes the attention operands directly, without a staging copy.
                selected_local_records(selected, positions, pages, world=shards.world,
                    rank=shards.owner, page_size=PAGE_TOKENS,
                    output=self.physical[:count], counts=self.counts[:count])
            if quantized is not None and paired is None:
                quantized.copy_(self.query[begin:end].float().clamp(-448., 448.).to(torch.float8_e4m3fn))
            if gather is not None:
                gather(quantized.view(count, -1))
            attention()
            if combine is not None:
                combine(attention.output, lse)
        return self.output
