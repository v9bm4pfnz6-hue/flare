# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 decode routing: latent-down and router shards gathered with expert selection.

Each rank projects its rows of the routed-expert down projection and its experts' router logits.
One b12x exchange gathers both shards; a single decode row also selects Kimi's 16 experts inside
that exchange, and 2-32 rows select them with b12x's prepared top-16 afterwards. This replaces
the router all-gather, the separate top-k, and the latent all-reduce of the general path.

Given a column-gather channel of the packed shard width, the exchange instead pushes each rank's
packed row into its peers and every rank unpacks what it received: the paired exchange reads its
peers' rows through their mappings, one 16-byte transaction at a time from a single block.

When b12x plans the paired exchange itself as a push (one block per row writes the row into every
peer), that exchange also selects the experts for up to eight rows and clips the padded last shard
of a world that does not divide the widths, and it is used in place of both forms above.
"""

import itertools
import weakref

import torch
import triton
import triton.language as tl
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare

MAX_ROWS = 32
# Smallest per-rank segment a prefill gather moves on the copy-engine ring. Eight ring hops have
# a fixed cost that kernel pushes do not; below this size the pushes are expected to finish first.
RING_GATHER_MIN_BYTES = 512 << 10
RING_COLUMN_GATHER_MIN_BYTES = 240 << 10
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def _launch(kernel, grid, arguments, **metadata):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*arguments, **metadata)
    else:
        kernel[grid](*arguments, **metadata)


@triton.jit
def _pack_route(Down, Router, Send, down_stride, router_stride, send_stride,
                DOWN: tl.constexpr, ROUTER: tl.constexpr, BLOCK: tl.constexpr):
    # One program per row: the BF16 latent shard, then the FP32 router shard, as 16-bit words.
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    latent = tl.load(Down + row * down_stride + column, mask=column < DOWN, other=0)
    tl.store(Send + row * send_stride + column, latent, mask=column < DOWN)
    router = tl.load(Router + row * router_stride + column, mask=column < ROUTER, other=0)
    tl.store(Send + row * send_stride + DOWN + column, router, mask=column < ROUTER)


@triton.jit
def _unpack_route(Received, Latent, Logits, received_stride, latent_stride, logits_stride,
                  DOWN: tl.constexpr, ROUTER: tl.constexpr, LATENT: tl.constexpr, LOGITS: tl.constexpr,
                  BLOCK: tl.constexpr):
    # One program per row and source rank. Rank r's packed row holds latent columns
    # [r * DOWN, (r + 1) * DOWN) and router words [r * ROUTER, (r + 1) * ROUTER); the last rank's
    # padded shard is clipped at the logical widths.
    row = tl.program_id(0)
    rank = tl.program_id(1)
    column = tl.arange(0, BLOCK)
    source = Received + row * received_stride + rank * (DOWN + ROUTER)
    live = (column < DOWN) & (rank * DOWN + column < LATENT)
    latent = tl.load(source + column, mask=live, other=0)
    tl.store(Latent + row * latent_stride + rank * DOWN + column, latent, mask=live)
    live = (column < ROUTER) & (rank * ROUTER + column < LOGITS)
    router = tl.load(source + DOWN + column, mask=live, other=0)
    tl.store(Logits + row * logits_stride + rank * ROUTER + column, router, mask=live)


def _copy_rank_major_blocks(destination, received, block):
    """Copy rank-major blocks of a [rows, world, columns] gather into rows.

    Source rank r owns received[:, r, :block]; destination columns
    [r * block, (r + 1) * block) receive it. Every full block is copied by
    one strided copy, and a clipped final rank is copied separately, so the
    total copy count per chunk is two instead of one per rank.
    """
    rows = destination.shape[0]
    world = received.shape[1]
    full = min(world, destination.shape[1] // block)
    if full:
        view = torch.as_strided(destination, (rows, full, block),
                                (destination.stride(0), block, 1))
        view.copy_(received[:, :full, :block])
    if full < world:
        begin = full * block
        count = destination.shape[1] - begin
        if count > 0:
            destination[:, begin:begin + count].copy_(received[:, full, :count])


@triton.jit
def _pack_shard_gather_slot(DownI16, RouterI16, OwnSlotI16, down_stride, router_stride, slot_stride,
                            DOWN_COLS: tl.constexpr, TOTAL_COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    in_down = cols < DOWN_COLS
    in_router = (cols >= DOWN_COLS) & (cols < TOTAL_COLS)
    d_val = tl.load(DownI16 + row * down_stride + cols, mask=in_down, other=0)
    r_val = tl.load(RouterI16 + row * router_stride + (cols - DOWN_COLS), mask=in_router, other=0)
    val = tl.where(in_down, d_val, r_val)
    tl.store(OwnSlotI16 + row * slot_stride + cols, val, mask=cols < TOTAL_COLS)


@triton.jit
def _unpack_shard_gather_slots(SlotsI16, LatentI16, LogitsI16, slot_rank_stride, slot_row_stride, lat_stride, log_stride,
                               FIRST_RANK: tl.constexpr, WORLD: tl.constexpr, DOWN_COLS: tl.constexpr,
                               LATENT_WIDTH: tl.constexpr, ROUTER_I16_COLS: tl.constexpr,
                               LOGITS_I16_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.program_id(1).to(tl.int64)
    rank = (FIRST_RANK + slot) % WORLD
    cols = tl.arange(0, BLOCK)
    base = SlotsI16 + slot * slot_rank_stride + row * slot_row_stride
    d_dst = rank * DOWN_COLS + cols
    d_valid = (cols < DOWN_COLS) & (d_dst < LATENT_WIDTH)
    d_val = tl.load(base + cols, mask=d_valid, other=0)
    tl.store(LatentI16 + row * lat_stride + d_dst, d_val, mask=d_valid)
    r_dst = rank * ROUTER_I16_COLS + cols
    r_valid = (cols < ROUTER_I16_COLS) & (r_dst < LOGITS_I16_WIDTH)
    r_val = tl.load(base + DOWN_COLS + cols, mask=r_valid, other=0)
    tl.store(LogitsI16 + row * log_stride + r_dst, r_val, mask=r_valid)


def _copy_slot_blocks(destination, slots, block, first_rank):
    """Copy rank blocks of a ring gather into rows.

    `slots` is [world, rows, columns]; slot s holds the columns of rank (first_rank + s) % world,
    whose first `block` columns belong in destination columns [rank * block, (rank + 1) * block),
    clipped at the destination's width. Slots of consecutive ranks are copied by one strided copy:
    at most two runs of whole blocks, and one clipped block.
    """
    world, rows, width = slots.shape[0], destination.shape[0], destination.shape[1]
    full = min(world, width // block)
    for slot, rank, count in ((0, first_rank, world - first_rank), (world - first_rank, 0, first_rank)):
        count = min(rank + count, full) - rank
        if count > 0:
            view = torch.as_strided(destination[:, rank * block:], (rows, count, block),
                                    (destination.stride(0), block, 1))
            view.copy_(slots[slot:slot + count, :, :block].permute(1, 0, 2))
    if full < world and width > full * block:
        destination[:, full * block:].copy_(slots[(full - first_rank) % world, :, :width - full * block])


class PreparedLatentShardGather:
    """Gather latent-output column partitions and FP32 router logits.

    Each rank owns output columns of both projections. Router bytes travel in
    BF16 views without conversion. Kernel pushes carry the packed shards in chunks: serial
    layers reuse the supplied send and receive buffers, and each chunk is consumed before the
    next overwrites them. Given a copy-engine ring channel whose shard capacity holds the packed
    shards of all rows, and at least RING_GATHER_MIN_BYTES of them per rank, the shards instead
    travel that ring whole, packed into and read from the channel's own scratch.
    """

    def __init__(self, local_down, local_router, latent, logits, *, runtime,
                 session, coordinator, name, buffers, allocator=None, ring=None):
        from ignition.components.collectives import PreparedAllGather

        rows, self.down_columns = local_down.shape
        self.world = runtime.world_size
        self.router_columns = local_router.shape[1]
        self.columns = self.down_columns + 2 * self.router_columns
        if (local_down.dtype != torch.bfloat16 or local_router.dtype != torch.float32
                or latent.shape[0] != rows or not 0 < latent.shape[1] <= self.world * self.down_columns
                or logits.shape[0] != rows or not 0 < logits.shape[1] <= self.world * self.router_columns
                or latent.dtype != torch.bfloat16 or logits.dtype != torch.float32
                or local_router.shape[0] != rows
                or not all(value.is_contiguous() and value.device == local_down.device
                           for value in (local_down, local_router, latent, logits))):
            raise ValueError("latent routing requires matching contiguous BF16 latent and FP32 router partitions")
        self.local_down, self.local_router = local_down, local_router
        self.latent, self.logits = latent, logits
        self.ring = None
        if ring is not None and RING_GATHER_MIN_BYTES <= rows * self.columns * 2 <= ring.gather_capacity:
            from ignition.components.collectives import PreparedRingGather

            self.ring = PreparedRingGather((rows, self.columns), torch.bfloat16, runtime=ring,
                session=session, coordinator=coordinator, name=name + ".ring")
            return
        chunk_rows = min(rows, runtime.max_batch_size)

        def allocate(field, columns):
            if field not in buffers or buffers[field].shape[0] < chunk_rows:
                shape = (chunk_rows, columns)
                buffers[field] = (torch.zeros(shape, dtype=torch.bfloat16, device=local_down.device)
                                  if allocator is None else allocator.zeros(shape, torch.bfloat16))
            return buffers[field][:chunk_rows]

        self.send = allocate("send", self.columns)
        self.received = allocate("received", self.columns * self.world)
        self.bounds = [(begin, min(rows, begin + chunk_rows)) for begin in range(0, rows, chunk_rows)]
        self.parts = [PreparedAllGather(self.send[:end-begin], self.received[:end-begin],
            runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.{index}")
            for index, (begin, end) in enumerate(self.bounds)]

    def pack_ring(self):
        own_i16 = self.ring.slots[self.world - 1].view(torch.int16)
        down_i16 = self.local_down.view(torch.int16)
        router_i16 = self.local_router.view(torch.int16)
        rows = down_i16.shape[0]
        r_cols = 2 * self.router_columns
        total = self.down_columns + r_cols
        block = triton.next_power_of_2(total)
        torch.library.wrap_triton(_pack_shard_gather_slot)[(rows,)](
            down_i16, router_i16, own_i16,
            down_i16.stride(0), router_i16.stride(0), own_i16.stride(0),
            self.down_columns, total, block, num_warps=4)

    def ring_and_unpack(self):
        slots_i16 = self.ring.slots.view(torch.int16)
        self.ring()
        lat_i16 = self.latent.view(torch.int16)
        log_i16 = self.logits.view(torch.int16)
        rows = lat_i16.shape[0]
        r_cols = 2 * self.router_columns
        block = triton.next_power_of_2(max(self.down_columns, r_cols))
        torch.library.wrap_triton(_unpack_shard_gather_slots)[(rows, self.world)](
            slots_i16, lat_i16, log_i16,
            slots_i16.stride(0), slots_i16.stride(1), lat_i16.stride(0), log_i16.stride(0),
            self.ring.first_rank, self.world, self.down_columns, lat_i16.shape[1],
            r_cols, log_i16.shape[1], block, num_warps=4)
        self.ring.release()
        return self.latent

    def __call__(self):
        if self.ring is not None:
            self.pack_ring()
            self.ring_and_unpack()
            return self.latent
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            count = end - begin
            self.send[:count, :self.down_columns].copy_(self.local_down[begin:end])
            self.send[:count, self.down_columns:].copy_(self.local_router[begin:end].view(torch.bfloat16))
            part(self.send[:count])
            received = self.received[:count].view(count, self.world, self.columns)
            _copy_rank_major_blocks(self.latent[begin:end], received, self.down_columns)
            _copy_rank_major_blocks(self.logits[begin:end].view(torch.bfloat16),
                                    received[:, :, self.down_columns:], 2 * self.router_columns)
        return self.latent


class PreparedRingColumnGather:
    """All-gather column shards [rows, local_cols] -> [rows, out_cols] on the copy-engine ring when >= 240 KiB."""

    def __init__(self, source, output, *, runtime, session, coordinator, name, ring=None):
        from ignition.components.collectives import PreparedChunkedAllGather, PreparedRingGather

        rows, self.local_cols = source.shape
        self.output = output
        self.world = 1 if runtime is None else runtime.world_size
        self.ring = None
        nbytes = rows * self.local_cols * source.element_size()
        if (ring is not None and self.world > 1 and source.is_contiguous() and output.is_contiguous()
                and nbytes % 16 == 0 and RING_COLUMN_GATHER_MIN_BYTES <= nbytes <= ring.gather_capacity):
            self.ring = PreparedRingGather((rows, self.local_cols), source.dtype, runtime=ring,
                session=session, coordinator=coordinator, name=name + '.ring')
            return
        self.fallback = PreparedChunkedAllGather(source, output, runtime=runtime,
            session=session, coordinator=coordinator, name=name)

    def __call__(self, source):
        if self.ring is None:
            return self.fallback(source)
        slots = self.ring.slots
        slots[self.world - 1].copy_(source)
        self.ring()
        _copy_slot_blocks(self.output, slots, self.local_cols, self.ring.first_rank)
        self.ring.release()
        return self.output


class PreparedExpertReduceNorm:
    """8-hop column-sharded expert reduce-scatter + lossless FP32 scalar norm exchange (skips 8 AG hops)."""

    GROUP_ROWS = 32

    def __init__(self, expert_output, weight, up_input, *, eps, dma_runtime, stats_runtime,
                 session, coordinator, name, buffers, allocator=None):
        from ignition.components.collectives import PreparedAllGather, PreparedReduceScatter

        rows, self.full_width = expert_output.shape
        _, self.local_width = up_input.shape
        self.world = dma_runtime.world_size
        self.eps, self.weight, self.up_input = float(eps), weight, up_input
        if rows % self.GROUP_ROWS != 0 or (rows * self.local_width) % 128 != 0:
            raise ValueError("PreparedExpertReduceNorm requires 32-row groups and 128-element aligned shards")

        def allocate(key, shape, dtype):
            cur = buffers.get(key)
            if cur is None or cur.numel() < torch.Size(shape).numel():
                buffers[key] = (torch.zeros(shape, dtype=dtype, device=expert_output.device)
                                if allocator is None else allocator.zeros(shape, dtype))
            return buffers[key].view(-1)[:torch.Size(shape).numel()].view(shape)

        self.packed = allocate('expert_rs_packed', (self.world, rows, self.local_width), torch.bfloat16)
        self.reduced_shard = allocate('expert_rs_shard', (rows, self.local_width), torch.bfloat16)
        batches = rows // self.GROUP_ROWS
        self.local_stats = allocate('expert_rs_local_stats', (batches, self.GROUP_ROWS), torch.float32)
        self.gathered_stats_bf16 = allocate('expert_rs_gathered_stats',
                                            (batches, self.world * self.GROUP_ROWS * 2), torch.bfloat16)
        self.reduce_scatter = PreparedReduceScatter(self.packed, self.reduced_shard, runtime=dma_runtime,
            session=session, coordinator=coordinator, name=name + '.expert_rs')
        self.stats_gather = PreparedAllGather(
            self.local_stats.view(torch.bfloat16).view(batches, self.GROUP_ROWS * 2),
            self.gathered_stats_bf16, runtime=stats_runtime, session=session,
            coordinator=coordinator, name=name + '.expert_norm_stats')

    def __call__(self, expert_output):
        from .pointwise import distributed_rms_norm, pack_expert_columns, partial_sum_sq

        pack_expert_columns(expert_output, self.packed)
        self.reduce_scatter(self.packed)
        partial_sum_sq(self.reduced_shard, self.local_stats.view(-1))
        batches = self.local_stats.shape[0]
        self.stats_gather(self.local_stats.view(torch.bfloat16).view(batches, self.GROUP_ROWS * 2))
        return distributed_rms_norm(self.reduced_shard, self.gathered_stats_bf16.view(torch.float32),
            self.weight, self.up_input, full_width=self.full_width, world=self.world,
            group_rows=self.GROUP_ROWS, eps=self.eps)


def create_route_runtime(group, device, *, ipc=None):
    """Allocate the b12x exchange that carries 1-32 rows of K3 latent and router shards."""
    import torch.distributed as dist
    from b12x.comm.pcie import DcpAllToAll
    from .config import padded_width

    world_size = dist.get_world_size(group)
    if world_size < 2:
        raise ValueError("K3 route exchange requires multiple ranks")
    latent_width = padded_width(3584, world_size, 32)
    router_width = padded_width(896, world_size, 8)
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device, max_batch_size=MAX_ROWS,
        total_heads=world_size, head_dim=8, query_head_dim=2 * latent_width + 4 * router_width,
        stream_affine=False, ipc=ipc)


def _selects_in_pushed_exchange(pcie, runtime, local_down, local_router, rows):
    """Whether b12x plans this channel's paired exchange as a push that also selects `rows` rows' experts.

    The transport is a b12x preparation setting; its plan query reports it. Only that transport runs one
    block per row, so only it selects experts for more than one row and for padded shards.
    """
    query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair",
        call=dict(local_first=local_down, local_second=local_router, threads=512))
    if getattr(query, "call", {}).get("pair_transport") != "push":
        return False
    from b12x.comm.pcie.pcie_dcp_a2a import PAIR_PUSH_MAX_ROWS

    return rows <= PAIR_PUSH_MAX_ROWS


class PreparedRouteGather:
    """Gather latent and FP32 router shards into full rows, then select 16 experts per row.

    The latent, expert weights and expert IDs are caller-owned outputs read by the routed experts.
    Every rank must declare the same operation names and replay them serially. Transport buffers
    may share graph-owned storage across sequential layers and capacities because compaction
    finishes before another operation consumes that storage.
    """

    def __init__(self, local_down, local_router, correction_bias, latent, weights, ids, *, runtime, session,
                 coordinator, name, buffers=None, allocator=None, gather_runtime=None):
        import torch.distributed as dist
        from b12x.comm import pcie

        rows = local_down.shape[0]
        world = runtime.world_size
        self.down_width, self.router_width = local_down.shape[1], local_router.shape[1]
        expected = ((local_down, (rows, self.down_width), torch.bfloat16),
                    (local_router, (rows, self.router_width), torch.float32),
                    (correction_bias, (896,), torch.float32), (latent, (rows, 3584), torch.bfloat16),
                    (weights, (rows, 16), torch.float32), (ids, (rows, 16), torch.int32))
        wrong = [(tuple(t.shape), t.dtype, t.is_contiguous(), shape, dtype) for t, shape, dtype in expected
                 if t.shape != shape or t.dtype != dtype or not t.is_contiguous()]
        if (not 1 <= rows <= MAX_ROWS or wrong
                or 2 * self.down_width + 4 * self.router_width != runtime.query_head_dim):
            raise ValueError(f"K3 route gather requires contiguous 1-32 row latent, router and route tensors: {wrong}")
        self.runtime, self.rows, self.world = runtime, rows, world
        self.local_down, self.local_router, self.correction_bias = local_down, local_router, correction_bias
        self.latent, self.weights, self.ids = latent, weights, ids
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        requests = []
        self.fused = self.gather = None
        exact = self.down_width * world == 3584 and self.router_width * world == 896
        if (rows == 1 and exact) or _selects_in_pushed_exchange(pcie, runtime, local_down, local_router, rows):
            self.fused = dict(local_down=local_down, local_router=local_router, correction_bias=correction_bias,
                              out_down=latent, topk_weights=weights, topk_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair_kimi_topk",
                                                          call=self.fused), runtime=runtime)
            requests.append(self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.fused), output=latent, owners=(runtime, latent, weights, ids)),
                collective=CollectiveRequirement(key=name, ranks=ranks)))
        else:
            buffers = {} if buffers is None else buffers
            def allocate(field, columns, dtype):
                if field not in buffers or buffers[field].shape[0] < rows:
                    shape = (rows, columns)
                    buffers[field] = (torch.zeros(shape, dtype=dtype, device=latent.device)
                                      if allocator is None else allocator.zeros(shape, dtype))
                return buffers[field][:rows]
            self.logits = allocate("logits", 896, torch.float32)
            self.select = dict(router_logits=self.logits, correction_bias=correction_bias,
                               output_weights=weights, output_ids=ids)
            self.select_plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.kimi_topk16",
                                                                 call=self.select), runtime=runtime)
            if gather_runtime is not None:
                from ignition.components.collectives import PreparedAllGather

                # A packed row holds the BF16 latent shard, then the FP32 router shard as two BF16 lanes.
                columns = self.down_width + 2 * self.router_width
                if (gather_runtime.world_size != world or gather_runtime.heads_per_rank != 1
                        or gather_runtime.query_head_dim != columns or rows > gather_runtime.max_batch_size):
                    raise ValueError("K3 route gather requires a column channel of one packed latent and router "
                                     "shard per rank")
                self.send = allocate("send", columns, torch.bfloat16)
                self.received = allocate("received", columns * world, torch.bfloat16)
                self.gather = PreparedAllGather(self.send, self.received, runtime=gather_runtime, session=session,
                    coordinator=coordinator, name=name + "/gather")
            else:
                self.gathered_latent = allocate("latent", self.down_width * world, torch.bfloat16)
                self.gathered_logits = allocate("router", self.router_width * world, torch.float32)
                self.pair = dict(local_first=local_down, local_second=local_router, out_first=self.gathered_latent,
                                 out_second=self.gathered_logits, threads=512)
                self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair",
                                                              call=self.pair), runtime=runtime)
                requests.append(self.plan.request(name=name + "/gather", prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run(**self.pair), output=self.gathered_latent,
                    owners=(runtime, self.gathered_latent, self.gathered_logits)),
                    collective=CollectiveRequirement(key=name + "/gather", ranks=ranks)))
            requests.append(self.select_plan.request(name=name + "/select", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.select), output=weights, owners=(runtime, self.logits, weights, ids))))
        declare(session, tuple(requests), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def compact(self):
        """Copy each rank's gathered shard into its logical column range.

        The exchange writes rank-major padded rows: source rank r occupies
        padded columns [r * down_width, (r + 1) * down_width), and the last
        rank's block is clipped at the logical width. The caller-owned rows
        use the same rank-major geometry up to their logical width, so
        copying the logical prefix of each gathered row reproduces every
        per-rank copy with one strided copy per tensor. No value, value
        order, destination or dtype changes.
        """
        self.latent.copy_(self.gathered_latent[:, :self.latent.shape[1]])
        self.logits.copy_(self.gathered_logits[:, :self.logits.shape[1]])

    def exchange(self):
        """Push this rank's packed shards into its peers and unpack every rank's into full rows.

        The packed row and the received rows are rewritten by every call before they are read. The
        unpacked latent and logits hold the bits `compact` copies out of the paired exchange.
        """
        rows, down, router = self.rows, self.down_width, 2 * self.router_width
        block = triton.next_power_of_2(max(down, router))
        send = self.send.view(torch.int16)
        _launch(_pack_route, (rows,), (self.local_down.view(torch.int16), self.local_router.view(torch.int16), send,
            self.local_down.stride(0), 2 * self.local_router.stride(0), send.stride(0)),
            DOWN=down, ROUTER=router, BLOCK=block, num_warps=4)
        received = self.gather(self.send).view(torch.int16)
        _launch(_unpack_route, (rows, self.world), (received, self.latent.view(torch.int16),
            self.logits.view(torch.int16), received.stride(0), self.latent.stride(0), 2 * self.logits.stride(0)),
            DOWN=down, ROUTER=router, LATENT=self.latent.shape[1], LOGITS=2 * self.logits.shape[1],
            BLOCK=block, num_warps=4)

    def __call__(self):
        if self.gather is None:
            _route_gather(self.local_down, self.local_router, self.latent, self.weights, self.ids, self.handle)
        else:
            self.exchange()
            _route_select(self.logits, self.weights, self.ids, self.handle)
        return self.latent


@torch.library.custom_op("flare::kimi3_route_gather", mutates_args=("latent", "weights", "ids"))
def _route_gather(local_down: torch.Tensor, local_router: torch.Tensor, latent: torch.Tensor,
                  weights: torch.Tensor, ids: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    runtime = owner.runtime
    if owner.fused is not None:
        runtime.all_gather_pair_kimi_topk(local_down, local_router, owner.correction_bias, latent, weights, ids,
                                          plan=owner.plan)
        return
    runtime.all_gather_pair(local_down, local_router, owner.gathered_latent, owner.gathered_logits,
                            plan=owner.plan, threads=512)
    owner.compact()
    runtime.kimi_topk16(owner.logits, owner.correction_bias, weights, ids, plan=owner.select_plan)


@_route_gather.register_fake
def _fake_route_gather(local_down, local_router, latent, weights, ids, handle):
    pass


@torch.library.custom_op("flare::kimi3_route_select", mutates_args=("weights", "ids"))
def _route_select(logits: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.kimi_topk16(logits, owner.correction_bias, weights, ids, plan=owner.select_plan)


@_route_select.register_fake
def _fake_route_select(logits, weights, ids, handle):
    pass
