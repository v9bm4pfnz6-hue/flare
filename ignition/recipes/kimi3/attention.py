# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated NoPE MLA over paged E4M3 latent records, sharded over decode context ranks.

As vLLM's Kimi K3 serving stores its latent cache (--kv-cache-dtype fp8), every record is E4M3 with a
unit scale. Without context ranks a rank attends with its heads over the whole cache. With them
(decode context parallel, vLLM's --decode-context-parallel-size 16), each rank stores the positions it
owns; decode and verification quantize their absorbed queries, gather every rank's heads, attend over
this rank's records with b12x dense MLA and keep this rank's heads by combining the partial outputs by
their LSE (b12x DcpAllToAll). Prefill copies the request's earlier context from every rank's cache into
a slab beside the chunk's own records and attends with this rank's heads.
"""

import torch
from b12x.attention import dense_mla

from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllGather, PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.linear import BatchedUnquantizedProjection

from .config import PAGE_TOKENS
from .dcp import RECORD_BYTES, ContextShards, lse_geometry, pull_context, write_records
from .pointwise import clamp_cast_fp8_e4m3, rms_norm_columns, sigmoid_gate, unpack_and_rms_norm_qkv
from .projections import prepare_projection

FP8 = torch.float8_e4m3fn


class PreparedMLA:
    def __init__(self, weights, source, metadata, cache, *, mode, session, name, workspace, buffers, latents=None,
                 coordinator=None, context=None, peers=None, allocator=None, index_cache=None, index_peers=None,
                 eps=1e-6, rotary=None, gated=True, partial_budget_bytes=None, selection=None, gather_ring=None,
                 dual_wave=False, out_pad_rows=0):
        tokens, hidden = source.shape
        heads = weights["key"].shape[0]
        device = source.device

        def buffer(key, *shape, dtype=None, rows=None):
            # Graphs share each buffer at its declared capacity, else their own rows; a larger graph replaces
            # it and earlier graphs keep their bindings.
            if key not in buffers or buffers[key].shape[0] < tokens:
                shape = (max(rows or tokens, tokens), *shape)
                dtype = source.dtype if dtype is None else dtype
                buffers[key] = (torch.zeros(shape, dtype=dtype, device=device) if allocator is None
                                else allocator.zeros(shape, dtype))
            return buffers[key][:tokens]

        latent_width = weights["query_norm"].shape[0] + RECORD_BYTES
        self.query_norm = buffer("query_norm", latent_width - RECORD_BYTES)
        self.query = buffer("query", heads * 192).view(tokens, heads, 192)
        self.latent = buffer("latent", RECORD_BYTES)
        self.absorbed = buffer("absorbed", heads, RECORD_BYTES)
        self.attended = buffer("attended", heads, 512)
        self.expanded = buffer("expanded", heads * 128)
        self.weights, self.metadata, self.mode, self.cache = weights, metadata, mode, cache
        self.eps, self.rotary, self.gated = eps, rotary, gated
        # Each rank projects only its query and KV latent rows beside its output gate, and the ranks gather the
        # latents, as vLLM's VLLM_KIMI_SHARD_QKV_A does: every rank keeps and reads one latent-projection shard.
        # A single rank (``latents`` None) projects every latent row itself.
        self.latent_gather = None
        local_width = (weights["latent_gate"].shape[0] if isinstance(weights["latent_gate"], torch.Tensor)
                       else weights["latent_gate"].out_features)
        query_width = latent_width - RECORD_BYTES
        if latents is None:
            self.local_q, self.local_kv = query_width, RECORD_BYTES
            self.local = buffer("local", local_width)
            self.input = self.local[:, :latent_width]
            self.query_input = self.input[:, :query_width]
            self.kv_input = self.input[:, query_width:]
            self.gate = self.local[:, latent_width:latent_width + heads * 128]
        else:
            rows, width = latents.max_batch_size, latents.query_head_dim
            world = latents.world_size
            self.local_q = -(-query_width // world)
            self.local_kv = -(-RECORD_BYTES // world)
            self.input = buffer("input", world * (self.local_q + self.local_kv))
            self.query_input = self.input[:, :world * self.local_q]
            self.kv_input = self.input[:, world * self.local_q:]
            self.local = buffer("local", local_width, rows=rows)
            self.gate = self.local[:, width:width + heads * 128]
            # The exchange reads contiguous rows, which a single row's latent columns already are.
            from .routing import RING_COLUMN_GATHER_MIN_BYTES
            self.latent_ring = None
            nbytes = tokens * width * source.element_size()
            if (gather_ring is not None and world > 1 and nbytes % 16 == 0
                    and RING_COLUMN_GATHER_MIN_BYTES <= nbytes <= gather_ring.gather_capacity):
                from ignition.components.collectives import PreparedRingGather
                self.latent_ring = PreparedRingGather((tokens, width), source.dtype, runtime=gather_ring,
                    session=session, coordinator=coordinator, name=name + ".gather_latents.ring")
                self.staged = False
            else:
                self.staged = tokens > 1
                self.send = buffer("send", width, rows=rows) if self.staged else self.local[:, :width]
                self.gathered_latents = buffer("gathered_latents", world * width, rows=rows)
                self.latent_gather = PreparedChunkedAllGather(self.send, self.gathered_latents, runtime=latents,
                    session=session, coordinator=coordinator, name=name + ".gather_latents")
        self.dual_wave = bool(dual_wave and mode == 'prefill' and tokens % 2 == 0)
        self.projections = {key: prepare_projection(x, weights[key], y, session=session,
            name=name + "." + key, workspace=workspace)
            for key, x, y in (("latent_gate", source, self.local), ("q_b", self.query_norm, self.query.view(tokens, -1)))}
        if self.dual_wave:
            self.half = tokens // 2
            pad_r = max(self.half, int(out_pad_rows))
            self.out_pads = (
                buffer('output_w0', hidden, rows=pad_r)[:pad_r],
                buffer('output_w1', hidden, rows=pad_r)[:pad_r],
            )
            self.out_halves = (
                prepare_projection(self.expanded[:self.half], weights['output'], self.out_pads[0][:self.half],
                                   session=session, name=name + '.output.w0', workspace=workspace),
                prepare_projection(self.expanded[self.half:], weights['output'], self.out_pads[1][:self.half],
                                   session=session, name=name + '.output.w1', workspace=workspace),
            )
        else:
            out_buf = buffer("output", hidden)
            self.output_buffer = out_buf
            self.projections["output"] = prepare_projection(self.expanded, weights["output"], out_buf,
                                                            session=session, name=name + ".output", workspace=workspace)
        self.shards = ContextShards() if context is None else context.shards
        self.peers = peers
        self.shared_slots = selection is not None and getattr(selection, 'slots', None) is not None
        self.slots = (selection.slots[:tokens] if self.shared_slots
                      else torch.full((tokens,), -1, dtype=torch.int64, device=device))
        self.absorb = BatchedUnquantizedProjection(
            self.query[:, :, :128].transpose(0, 1), weights['key'],
            self.absorbed[:, :, :512].transpose(0, 1), major='n', session=session, name=name + '.absorb')
        self.expand = BatchedUnquantizedProjection(
            self.attended.transpose(0, 1), weights['value'],
            self.expanded.view(tokens, heads, 128).transpose(0, 1), major='n', session=session, name=name + '.expand')
        world, width = self.shards.world, metadata.page_table.shape[1]
        common = dict(device=device, kv_dtype=FP8, page_size=PAGE_TOKENS, use_cuda_graph=True,
                      partial_dtype=torch.float32)
        if partial_budget_bytes is not None:
            if type(partial_budget_bytes) is not int or partial_budget_bytes < 0:
                raise ValueError("MLA partial-output budget must be a nonnegative byte count")
            # One partial row stores every participating head's FP32 latent output and LSE.
            # The tuner still chooses the split count; every choice reads the full context.
            common["budget"] = dense_mla.Budget(
                max_partial_rows=partial_budget_bytes // (heads * world * (512 + 1) * 4))
        scale = 192 ** -.5
        self.sparse = None
        if index_cache is not None:
            from .dsa import PreparedDSA

            index_begin = (latent_width if latents is None else latents.query_head_dim) + heads * 128
            self.sparse = PreparedDSA(weights, self.query_norm, self.local[:, index_begin:index_begin + 128],
                self.local[:, index_begin + 128:index_begin + 160], self.absorbed, self.attended, self.slots,
                metadata, cache, index_cache, context=context, session=session, coordinator=coordinator,
                name=name, workspace=workspace, buffers=buffers.setdefault("selection", {}), allocator=allocator, eps=eps,
                mode=mode, latent=self.latent, peers=peers, index_peers=index_peers, selection=selection,
                gather_ring=gather_ring)
        elif world == 1:
            # Every rank holds every record: attend with this rank's heads; b12x quantizes the BF16 query.
            self.lengths = torch.zeros_like(metadata.lengths)
            caps = dense_mla.Caps(mode=mode if mode in ("decode", "verify") else "extend", q_dtype=torch.bfloat16,
                num_q_heads=heads, max_total_q=tokens, max_batch=metadata.lengths.numel(),
                max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width, num_cache_pages=cache.shape[0],
                **common)
            self.attention = PreparedDenseMLA(self.absorbed, cache, self.attended, metadata.page_table, self.lengths,
                metadata.query_start_loc, caps=caps, scale=scale, session=session, name=name + ".attention",
                workspace=workspace)
        elif mode == "prefill":
            self.prefill = PreparedContextPrefill(self.absorbed, self.attended, metadata, context, peers,
                session=session, name=name, workspace=workspace, buffer=buffer, scale=scale, common=common)
        else:
            # Each query row attends to the records this rank owns at or before its position, as its own sequence.
            total, rows = heads * world, context.heads.max_batch_size
            self.quantized = buffer("quantized", heads, RECORD_BYTES, dtype=FP8, rows=rows)
            self.gathered = buffer("gathered", total, RECORD_BYTES, dtype=FP8, rows=rows)
            self.partial = buffer("partial", total, 512, rows=rows)
            self.lse = buffer("lse", total, dtype=torch.float32, rows=rows)
            self.token_pages = buffer("token_pages", width, dtype=torch.int32, rows=rows)
            self.token_lengths = buffer("token_lengths", dtype=torch.int32, rows=rows)
            self.token_starts = torch.arange(tokens + 1, dtype=torch.int32, device=device)
            self.gather = PreparedAllGather(self.quantized.view(tokens, -1), self.gathered.view(tokens, -1),
                runtime=context.heads, session=session, coordinator=context.coordinator, name=name + ".gather_heads")
            caps = dense_mla.Caps(mode="decode", q_dtype=FP8, num_q_heads=total, max_total_q=tokens,
                max_batch=tokens, max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width,
                num_cache_pages=cache.shape[0], **common)
            self.attention = PreparedDenseMLA(self.gathered, cache, self.partial, self.token_pages,
                self.token_lengths, self.token_starts, caps=caps, scale=scale, session=session,
                name=name + ".attention", workspace=workspace, lse=self.lse)
            threads, block_limit = lse_geometry(tokens, self.shards.world)
            self.combine = PreparedLseReduceScatter(self.partial, self.lse, self.attended, runtime=context.heads,
                session=session, coordinator=context.coordinator, name=name + ".combine",
                threads=threads, block_limit=block_limit)

    def pre_output(self, source):
        p, m, shards = self.projections, self.metadata, self.shards
        p["latent_gate"](source)
        if getattr(self, 'latent_ring', None) is not None:
            from .routing import _copy_slot_blocks
            slots = self.latent_ring.slots
            world, tokens, width = slots.shape
            slots[world - 1].copy_(self.local[:, :width])
            self.latent_ring()
            first = self.latent_ring.first_rank
            q, kv = self.local_q, self.local_kv
            _copy_slot_blocks(self.query_input, slots[:, :, :q], q, first)
            _copy_slot_blocks(self.kv_input, slots[:, :, q:q + kv], kv, first)
            self.latent_ring.release()
            rms_norm_columns(self.query_input[:, :self.query_norm.shape[1]], self.weights['query_norm'], self.query_norm, self.eps)
            rms_norm_columns(self.kv_input[:, :512], self.weights['kv_norm'], self.latent[:, :512], self.eps)
            self.latent[:, 512:].copy_(self.kv_input[:, 512:])
        elif self.latent_gather is not None:
            tokens, width = self.send.shape
            if self.staged:
                self.send.copy_(self.local[:, :width])
            self.latent_gather(self.send)
            world = self.gathered_latents.shape[1] // width
            gathered = self.gathered_latents.view(tokens, world, width)
            unpack_and_rms_norm_qkv(
                gathered, self.local_q, self.local_kv,
                self.weights['query_norm'], self.weights['kv_norm'],
                self.query_norm, self.latent, self.eps)
        else:
            rms_norm_columns(self.query_input[:, :self.query_norm.shape[1]], self.weights['query_norm'], self.query_norm, self.eps)
            rms_norm_columns(self.kv_input[:, :512], self.weights['kv_norm'], self.latent[:, :512], self.eps)
            self.latent[:, 512:].copy_(self.kv_input[:, 512:])
        p["q_b"](self.query_norm)
        if self.rotary is not None:
            self.rotary.apply(self.query, m.positions)
            self.rotary.apply(self.latent, m.positions)
        self.absorb(self.query[:, :, :128].transpose(0, 1))
        self.absorbed[:, :, 512:].copy_(self.query[:, :, 128:])
        if not self.shared_slots:
            self.slots.copy_(shards.token_slots(m.positions, m.token_page_table))
        write_records(self.latent, self.slots, self.cache.view(-1, RECORD_BYTES))
        if self.sparse is not None:
            self.sparse()
        elif shards.world == 1:
            self.lengths.copy_(torch.where(m.lengths > 0, m.starts + m.lengths, 0))
            self.attention()
        elif self.mode == "prefill":
            self.prefill(self.latent)
        else:
            tokens = self.absorbed.shape[0]
            clamp_cast_fp8_e4m3(self.absorbed, self.quantized)
            self.gather(self.quantized.view(tokens, -1))
            self.token_pages.copy_(m.token_page_table)
            self.token_lengths.copy_(shards.local_tokens(m.positions + 1).to(torch.int32))
            self.attention()
            self.combine(self.partial, self.lse)
        self.expand(self.attended.transpose(0, 1))
        if self.gated:
            sigmoid_gate(self.expanded, self.gate)
        return self.expanded

    @property
    def can_split_sparse(self) -> bool:
        return bool(
            self.dual_wave and self.sparse is not None
            and getattr(self.sparse, 'prefill', None) is not None
            and self.sparse.prefill.ring_gather is not None
            and self.half % self.sparse.prefill.step == 0
        )

    def prepare_sparse_dual(self, source):
        p, m, shards = self.projections, self.metadata, self.shards
        p['latent_gate'](source)
        if getattr(self, 'latent_ring', None) is not None:
            from .routing import _copy_slot_blocks
            slots = self.latent_ring.slots
            world, tokens, width = slots.shape
            slots[world - 1].copy_(self.local[:, :width])
            self.latent_ring()
            q, kv = self.local_q, self.local_kv
            _copy_slot_blocks(self.query_input, slots[:, :, :q], q, self.latent_ring.first_rank)
            _copy_slot_blocks(self.kv_input, slots[:, :, q:q + kv], kv, self.latent_ring.first_rank)
            self.latent_ring.release()
        rms_norm_columns(self.query_input[:, :self.query_norm.shape[1]], self.weights['query_norm'], self.query_norm, self.eps)
        rms_norm_columns(self.kv_input[:, :512], self.weights['kv_norm'], self.latent[:, :512], self.eps)
        self.latent[:, 512:].copy_(self.kv_input[:, 512:])
        p['q_b'](self.query_norm)
        if self.rotary is not None:
            self.rotary.apply(self.query, m.positions)
            self.rotary.apply(self.latent, m.positions)
        self.absorbed[:, :, :512].copy_(torch.bmm(self.query[:, :, :128].transpose(0, 1), self.weights['key']).transpose(0, 1))
        self.absorbed[:, :, 512:].copy_(self.query[:, :, 128:])
        self.slots.copy_(shards.token_slots(m.positions, m.token_page_table))
        write_records(self.latent, self.slots, self.cache.view(-1, RECORD_BYTES))
        self.sparse.prefill.select_and_gather()

    def run_sparse_out_half(self, wave: int):
        sl = slice(0, self.half) if wave == 0 else slice(self.half, 2 * self.half)
        self.sparse.prefill.attend_range(sl.start, sl.stop)
        att_sub = self.attended[sl]
        att_rows, att_heads = att_sub.shape[:2]
        self.expanded[sl].view(att_rows, att_heads, 128).copy_(
            torch.bmm(att_sub.transpose(0, 1), self.weights['value']).transpose(0, 1))
        if self.gated:
            sigmoid_gate(self.expanded[sl], self.gate[sl])
        return self.out_halves[wave](self.expanded[sl])

    def run_out_half(self, wave: int):
        sl = slice(0, self.half) if wave == 0 else slice(self.half, 2 * self.half)
        return self.out_halves[wave](self.expanded[sl])

    def __call__(self, source):
        self.pre_output(source)
        return self.projections["output"](self.expanded)


class PreparedContextPrefill:
    """One packed prefill request's attention with this rank's heads over its whole context, whose positions
    every rank of the decode context group shares.

    Earlier positions arrive from every rank's cache one window at a time, back from the chunk: the latest
    window and the chunk's own records attend causally, each earlier window without a mask, and the partial
    outputs combine by their log-sum-exp, as vLLM's chunked MLA prefill does. Earlier chunks' records were
    written before this forward's collectives ordered the ranks.
    """

    def __init__(self, absorbed, attended, metadata, context, peers, *, session, name, workspace, buffer, scale,
                 common):
        tokens, heads = absorbed.shape[:2]
        device, world, width = absorbed.device, context.shards.world, metadata.page_table.shape[1]
        slab, window = context.slab, context.window
        pages = slab.shape[0] // PAGE_TOKENS
        if (peers is None or peers.shape != (world,) or slab.shape[0] % PAGE_TOKENS or window % PAGE_TOKENS
                or slab.shape[0] < window + tokens):
            raise ValueError("context-parallel prefill requires every rank's cache address and a context slab")
        self.metadata, self.peers, self.shards, self.attended = metadata, peers, context.shards, attended
        self.slab, self.window = slab, window
        self.words = slab.view(torch.uint8).view(torch.int32).view(slab.shape[0], RECORD_BYTES // 4)[:window]
        self.windows = -(-width * PAGE_TOKENS * world // window)
        self.offsets = torch.arange(self.windows, dtype=torch.int64, device=device) * window
        self.bounds = torch.zeros((self.windows, 2), dtype=torch.int64, device=device)
        self.records = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.lengths = torch.zeros_like(metadata.lengths)
        self.lse = buffer("prefill_lse", heads, dtype=torch.float32)
        cache = slab.view(pages, PAGE_TOKENS, RECORD_BYTES)
        caps = dense_mla.Caps(mode="extend", q_dtype=torch.bfloat16, num_q_heads=heads, max_total_q=tokens,
            max_batch=1, max_cache_tokens=pages * PAGE_TOKENS, max_page_table_width=pages, num_cache_pages=pages,
            **common)
        self.table = torch.arange(pages, dtype=torch.int32, device=device)[None]
        self.attention = PreparedDenseMLA(absorbed, cache, attended, self.table, self.lengths,
            metadata.query_start_loc, caps=caps, scale=scale, session=session, name=name + ".attention",
            workspace=workspace, lse=self.lse)
        self.context_attention = None
        if self.windows > 1:
            columns = window // PAGE_TOKENS
            self.merged = buffer("merged", heads, 512, dtype=torch.float32)
            self.window_lengths = torch.zeros_like(metadata.lengths)
            self.window_lse = buffer("window_lse", heads, dtype=torch.float32)
            self.window_attended = buffer("window_attended", heads, 512)
            self.window_table = torch.arange(columns, dtype=torch.int32, device=device)[None]
            caps = dense_mla.Caps(mode="extend", causal=False, q_dtype=torch.bfloat16, num_q_heads=heads,
                max_total_q=tokens, max_batch=1, max_cache_tokens=window, max_page_table_width=columns,
                num_cache_pages=pages, **common)
            self.context_attention = PreparedDenseMLA(absorbed, cache, self.window_attended, self.window_table,
                self.window_lengths, metadata.query_start_loc, caps=caps, scale=scale, session=session,
                name=name + ".context_attention", workspace=workspace, lse=self.window_lse)

    def __call__(self, latent):
        m, table = self.metadata, self.metadata.page_table[0]
        ends = m.starts[:1].to(torch.int64) - self.offsets
        self.bounds.copy_(torch.stack(((ends - self.window).clamp(min=0), ends.clamp(min=0)), -1))
        pull_context(self.peers, table, self.bounds[0], self.words, self.shards)
        # The chunk's records follow the latest window in the slab.
        latest = self.bounds[0, 0]
        self.records.copy_(torch.where(m.positions >= 0, m.positions - latest, -1))
        write_records(latent, self.records, self.slab)
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts - latest + m.lengths, 0))
        self.attention()
        if self.context_attention is None:
            return self.attended
        self.merged.copy_(self.attended.float())
        for index in range(1, self.windows):
            bounds = self.bounds[index]
            pull_context(self.peers, table, bounds, self.words, self.shards)
            present = bounds[1] > bounds[0]
            self.window_lengths.copy_((bounds[1] - bounds[0]).to(self.window_lengths.dtype).expand_as(
                self.window_lengths))
            self.context_attention()
            # An empty window contributes nothing; a query row without positions keeps a zero output.
            peak = torch.maximum(self.lse, torch.where(present, self.window_lse, -torch.inf)).clamp(min=-1e30)
            kept = torch.exp(self.lse - peak)
            added = torch.where(present, torch.exp(self.window_lse - peak), 0.)
            total = (kept + added).clamp(min=1e-30)
            self.merged.copy_((self.merged * kept[..., None] + torch.where(present, self.window_attended.float(),
                                                                           0.) * added[..., None]) / total[..., None])
            self.lse.copy_(peak + torch.log(total))
        return self.attended.copy_(self.merged)
