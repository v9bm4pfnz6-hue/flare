# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 decode routing: latent-down and router shards gathered with expert selection.

Each rank projects its rows of the routed-expert down projection and its experts' router logits.
One b12x exchange gathers both shards; a single decode row also selects Kimi's 16 experts inside
that exchange, and 2-32 rows select them with b12x's prepared top-16 afterwards. This replaces
the router all-gather, the separate top-k, and the latent all-reduce of the general path.
"""

import itertools
import weakref

import torch
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare

MAX_ROWS = 32
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


class PreparedLatentShardGather:
    """Gather latent-output column partitions and FP32 router logits in chunks.

    Each rank owns output columns of both projections. Router bytes travel in
    BF16 views without conversion. Serial layers reuse the supplied send and
    receive buffers; each chunk is consumed before the next overwrites them.
    """

    def __init__(self, local_down, local_router, latent, logits, *, runtime,
                 session, coordinator, name, buffers, allocator=None):
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

    def __call__(self):
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            count = end - begin
            self.send[:count, :self.down_columns].copy_(self.local_down[begin:end])
            self.send[:count, self.down_columns:].copy_(self.local_router[begin:end].view(torch.bfloat16))
            part(self.send[:count])
            received = self.received[:count].view(count, self.world, self.columns)
            for rank in range(self.world):
                down_begin = rank * self.down_columns
                down_count = min(self.down_columns, self.latent.shape[1] - down_begin)
                if down_count > 0:
                    self.latent[begin:end, down_begin:down_begin + down_count].copy_(
                        received[:, rank, :down_count])
                router_begin = rank * self.router_columns
                router_count = min(self.router_columns, self.logits.shape[1] - router_begin)
                if router_count > 0:
                    self.logits[begin:end, router_begin:router_begin + router_count].view(torch.bfloat16).copy_(
                        received[:, rank, self.down_columns:self.down_columns + 2 * router_count])
        return self.latent


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


class PreparedRouteGather:
    """Gather latent and FP32 router shards into full rows, then select 16 experts per row.

    The latent, expert weights and expert IDs are caller-owned outputs read by the routed experts.
    Every rank must declare the same operation names and replay them serially. Transport buffers
    may share graph-owned storage across sequential layers and capacities because compaction
    finishes before another operation consumes that storage.
    """

    def __init__(self, local_down, local_router, correction_bias, latent, weights, ids, *, runtime, session,
                 coordinator, name, buffers=None, allocator=None):
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
        self.fused = None
        if rows == 1 and self.down_width * world == 3584 and self.router_width * world == 896:
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
            self.gathered_latent = allocate("latent", self.down_width * world, torch.bfloat16)
            self.gathered_logits = allocate("router", self.router_width * world, torch.float32)
            self.pair = dict(local_first=local_down, local_second=local_router, out_first=self.gathered_latent,
                             out_second=self.gathered_logits, threads=512)
            self.select = dict(router_logits=self.logits, correction_bias=correction_bias,
                               output_weights=weights, output_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair",
                                                          call=self.pair), runtime=runtime)
            self.select_plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.kimi_topk16",
                                                                 call=self.select), runtime=runtime)
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

    def __call__(self):
        _route_gather(self.local_down, self.local_router, self.latent, self.weights, self.ids, self.handle)
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
