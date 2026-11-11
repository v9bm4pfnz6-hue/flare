# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 attention residuals and latent MoE composed into complete forwards."""

from dataclasses import dataclass, field

import torch
import triton

from ignition.components.collectives import PreparedChunkedAllGather, PreparedChunkedAllReduce
from ignition.components.owner_packet_reduce import aligned_extent, copy_engine_reduction, direct_aligned_extent
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.kda_speculation import PreparedKDAReplay
from ignition.preparation.tuning import register_finalizer

from .attention import PreparedMLA
from .kda import PreparedKDA
from .pointwise import rms_norm_columns, situ
from .projections import prepare_projection
from .residual import ADD, REPLACE, ResidualStream
from .routing import (
    MAX_ROWS as ROUTE_ROWS,
    PreparedExpertReduceNorm,
    PreparedLatentShardGather,
    PreparedRingColumnGather,
    PreparedRouteGather,
)


# A concurrent shared-expert MLP forks onto the auxiliary stream after its input is written and joins before its
# output is added; the explicit mutated outputs order both against the routed branch through compilation.
@torch.library.custom_op("flare::kimi3_start_shared_expert", mutates_args=("outputs", "workspace"))
def _start_shared_expert(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor | None],
                         outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], auxiliary_stream: int) -> None:
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    def project(value, index, output):
        if scales[index] is None:
            bf16_out(value, weights[index], output, plans[index])
        else:
            packed_out_workspace(value, weights[index], scales[index], None, None, output,
                                 workspace, "none", None, plans[index], "any")

    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)
    gate_up, intermediate, projected = outputs
    with torch.cuda.stream(auxiliary):
        project(source, 0, gate_up)
        situ(gate_up, intermediate)
        project(intermediate, 1, projected)


@_start_shared_expert.register_fake
def _fake_start_shared_expert(source, weights, scales, outputs, workspace, plans, auxiliary_stream) -> None:
    pass


@torch.library.custom_op("flare::kimi3_join_shared_expert", mutates_args=("outputs", "workspace"))
def _join_shared_expert(source: torch.Tensor, routed: torch.Tensor, outputs: list[torch.Tensor],
                        workspace: torch.Tensor, auxiliary_stream: int) -> None:
    main = torch.cuda.current_stream(source.device)
    main.wait_stream(torch.cuda.ExternalStream(auxiliary_stream, device=source.device))


@_join_shared_expert.register_fake
def _fake_join_shared_expert(source, routed, outputs, workspace, auxiliary_stream) -> None:
    pass


@torch.library.custom_op("flare::kimi3_start_shared_expert_stage1", mutates_args=("outputs", "workspace"))
def _start_shared_expert_stage1(source: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None,
                                outputs: list[torch.Tensor], workspace: torch.Tensor, plan: int,
                                auxiliary_stream: int) -> None:
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)
    gate_up, intermediate, _ = outputs
    with torch.cuda.stream(auxiliary):
        if scale is None:
            bf16_out(source, weight, gate_up, plan)
        else:
            packed_out_workspace(source, weight, scale, None, None, gate_up, workspace, "none", None, plan, "any")
        situ(gate_up, intermediate)


@_start_shared_expert_stage1.register_fake
def _fake_start_shared_expert_stage1(source, weight, scale, outputs, workspace, plan, auxiliary_stream) -> None:
    pass


@torch.library.custom_op("flare::kimi3_start_shared_expert_stage2", mutates_args=("outputs", "workspace"))
def _start_shared_expert_stage2(dependency: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None,
                                outputs: list[torch.Tensor], workspace: torch.Tensor, plan: int,
                                auxiliary_stream: int) -> None:
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    main = torch.cuda.current_stream(dependency.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=dependency.device)
    auxiliary.wait_stream(main)
    _, intermediate, projected = outputs
    with torch.cuda.stream(auxiliary):
        if scale is None:
            bf16_out(intermediate, weight, projected, plan)
        else:
            packed_out_workspace(intermediate, weight, scale, None, None, projected, workspace, "none", None, plan, "any")


@_start_shared_expert_stage2.register_fake
def _fake_start_shared_expert_stage2(dependency, weight, scale, outputs, workspace, plan, auxiliary_stream) -> None:
    pass


@dataclass(frozen=True)
class DecoderStorage:
    scratch: dict
    hidden_columns: torch.Tensor
    latent_begin: torch.Tensor
    allocator: object = None
    stage_allocators: dict = field(default_factory=dict)

    @classmethod
    def allocate(cls, config, shard, device, *, scratch=None, allocator=None):
        # Rank-specific offsets stay in tensors so every rank replays rank zero's compiled forward.
        return cls({} if scratch is None else scratch,
            torch.arange(*shard.padded_partition(config.hidden, alignment=64), device=device).clamp(
                max=config.hidden - 1),
            torch.tensor(shard.padded_partition(config.text["routed_expert_hidden_size"], alignment=32)[:1],
                         dtype=torch.int32, device=device), allocator)

    def scratch_tensors(self):
        return tuple(self.scratch.values())

    def stage_allocator(self, name):
        """Serial attention and MLP stages share storage beyond persistent decoder values."""
        if self.allocator is None:
            return None
        if name not in self.stage_allocators:
            self.stage_allocators[name] = self.allocator.fork()
        return self.stage_allocators[name]


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, world_size, device, *, activations=None, buffers=None):
        def zeros(shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        vocabulary = -(-config.vocabulary // (world_size * 64)) * 64
        return cls(zeros((tokens, config.hidden)) if activations is None else activations, zeros((rows, config.hidden)),
                   zeros((rows, vocabulary)), zeros((rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


def retained_auxiliary_rows(weights, tokens, *, mode):
    """Bound accepted-prefill features where the projection preserves enclosing-row arithmetic."""
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from ignition.components.sm103_projection import is_sm103
    from ignition.recipes.kimi31.dflash_config import DFlash2Config

    if (mode == "prefill" and isinstance(weights.config.draft, DFlash2Config)
            and weights.draft is not None and isinstance(weights.draft.projection, MXFP8LinearWeight)
            and not is_sm103(weights.embedding.device)):
        from ignition.recipes.kimi31.dflash_config import CONTEXT_ROW_ALIGNMENT
        return min(tokens, weights.config.draft.window + CONTEXT_ROW_ALIGNMENT)
    return tokens


class PreparedMLP:
    """A SiTU MLP; with an overlapping workspace, ``start`` and ``finish`` run it on the auxiliary stream."""

    def __init__(self, weights, source, *, session, name, buffers, workspace=None, allocator=None,
                 parallel_plans=()):
        rows, device = source.shape[0], source.device
        def buffer(field, width):
            if field not in buffers or buffers[field].shape[0] < rows:
                buffers[field] = (torch.zeros((rows, width), dtype=source.dtype, device=device) if allocator is None
                                  else allocator.zeros((rows, width), source.dtype))
            return buffers[field][:rows]
        width = (weights["gate_up"].shape[0] if isinstance(weights["gate_up"], torch.Tensor)
                 else weights["gate_up"].out_features)
        self.input = buffer("input", width // 2)
        self.gate_up = prepare_projection(source, weights["gate_up"], buffer("gate_up", width),
                                          session=session, name=name + ".gate_up", workspace=workspace)
        self.down = prepare_projection(self.input, weights["down"], buffer("output", source.shape[1]),
                                       session=session, name=name + ".down", workspace=workspace)
        self.concurrent = bool(parallel_plans) and workspace is not None and workspace.overlap
        if self.concurrent:
            self.arena, self.parallel_plans = workspace, tuple(parallel_plans)
            self.stream_handle = workspace.auxiliary_stream.cuda_stream
            self.weights, self.scales = [], []
            for projection in (self.gate_up, self.down):
                if isinstance(projection, UnquantizedProjection):
                    self.weights.append(projection.weight)
                    self.scales.append(None)
                else:
                    values, scales, _ = projection.parts
                    self.weights.append(values)
                    self.scales.append(scales.view(torch.uint8))
                for main in self.parallel_plans:
                    workspace.register_parallel(main, projection.plan)
            self.outputs = [self.gate_up.output, self.input, self.down.output]
            register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        side_bytes = max(p.plan.memory_requirements().scratch_nbytes for p in (self.gate_up, self.down))
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def __call__(self, source):
        situ(self.gate_up(source), self.input)
        return self.down(self.input)

    def start(self, source):
        _start_shared_expert(source, self.weights, self.scales, self.outputs, self.workspace,
                             [self.gate_up.plan.handle, self.down.plan.handle], self.stream_handle)

    def start_stage1(self, source):
        _start_shared_expert_stage1(source, self.weights[0], self.scales[0], self.outputs, self.workspace,
                                    self.gate_up.plan.handle, self.stream_handle)

    def join_stage1(self, source, routed):
        _join_shared_expert(source, routed, self.outputs, self.workspace, self.stream_handle)

    def start_stage2(self, dependency):
        _start_shared_expert_stage2(dependency, self.weights[1], self.scales[1], self.outputs, self.workspace,
                                    self.down.plan.handle, self.stream_handle)

    def finish(self, source, routed):
        _join_shared_expert(source, routed, self.outputs, self.workspace, self.stream_handle)
        return self.down.output


class PreparedLatentMoE:
    def __init__(self, weights, source, config, shard, *, runtime, router_runtime, coordinator, session, name, workspace, buffers, storage,
                 route_runtime=None, prefill_route_runtime=None, allocator=None, accumulation_buffers=None,
                 reduction_staging=None, route_gather_runtime=None, copy_engine=False, gather_ring=None,
                 stats_runtime=None, concurrent_shared=True):
        tokens, hidden = source.shape
        c, device = config.text, source.device
        latent = c["routed_expert_hidden_size"]
        local_latent = weights["latent_norm"].shape[0]

        def buffer(field, width, dtype=torch.bfloat16):
            # Decode-only fields first appear in a decoder of any row count, so each holds every route row.
            if field not in buffers or buffers[field].shape[0] < tokens:
                shape = (max(tokens, ROUTE_ROWS), width)
                buffers[field] = (torch.zeros(shape, dtype=dtype, device=device) if allocator is None
                                  else allocator.zeros(shape, dtype))
            return buffers[field][:tokens]

        self.source = buffer("source", storage.hidden_columns.numel())
        self.hidden_columns, self.latent_begin = storage.hidden_columns, storage.latent_begin
        self.latent = buffer("latent", latent)
        # The routed experts write where prefill reduces the latent down-projection from, and their sum
        # replaces the latent rows they consumed.
        self.expert_output = buffer("down", latent)
        self.latent_sum = self.latent
        self.up_input = buffer("up_input", local_latent)
        self.output = buffer("output", hidden)
        self.up = prepare_projection(self.up_input, weights["latent_up"], self.output,
                                      session=session, name=name + ".latent_up", workspace=workspace)
        # Each rank normalizes the full latent row and keeps the columns its up-projection shard reads.
        self.latent_norm, self.eps = weights["latent_norm"], c["rms_norm_eps"]
        reduction = copy_engine_reduction if copy_engine else PreparedChunkedAllReduce
        reduction_args = (dict(workspace=accumulation_buffers, staging=reduction_staging) if copy_engine else {})
        self.expert_reduce_norm = None
        if (copy_engine and getattr(runtime, 'direct_mode', '') and stats_runtime is not None
                and tokens >= 512 and tokens % 32 == 0 and (tokens * local_latent) % 128 == 0):
            self.expert_reduce_norm = PreparedExpertReduceNorm(self.expert_output, self.latent_norm, self.up_input,
                eps=self.eps, dma_runtime=runtime, stats_runtime=stats_runtime, session=session,
                coordinator=coordinator, name=name, buffers=buffers, allocator=allocator)
            self.expert_reduce = None
        else:
            self.expert_reduce = reduction(self.expert_output, self.latent_sum, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".expert_reduce", **reduction_args)
        local_logits = buffer("local_router", weights["router"].shape[0], torch.float32)
        self.router_projection = UnquantizedProjection(source, weights["router"], local_logits, session=session, name=name + ".router")
        if "routes" not in buffers:
            buffers["routes"] = RouterScratch.allocate(tokens, device, experts=c["num_experts"], top_k=c["num_experts_per_token"])
        routes = buffers["routes"].rows(tokens)
        self.route_gather = None
        self.prefill_route_gather = None
        if route_runtime is not None and tokens <= ROUTE_ROWS:
            # Decode rows gather row-sharded latent and router shards and select experts in one exchange.
            self.down = prepare_projection(source, weights["latent_down_rows"],
                buffer("local_down", local_latent), session=session, name=name + ".latent_down_rows",
                workspace=workspace)
            self.route_gather = PreparedRouteGather(self.down.output, local_logits, weights["correction"],
                self.latent, routes.weights, routes.ids, runtime=route_runtime, session=session,
                coordinator=coordinator, name=name + ".route_gather",
                buffers=buffers.setdefault("route_gather_transport", {}), allocator=allocator,
                gather_runtime=route_gather_runtime)
        else:
            logits = buffer("router", c["num_experts"], torch.float32)
            self.router_logits = logits
            if prefill_route_runtime is not None:
                # Each output column uses the full input width. Only finished
                # BF16 values cross ranks, without rounded partial dot products.
                self.down = prepare_projection(source, weights["latent_down_rows"],
                    buffer("local_down", local_latent), session=session,
                    name=name + ".latent_down_rows", workspace=workspace)
                self.prefill_route_gather = PreparedLatentShardGather(self.down.output, local_logits,
                    self.latent, logits, runtime=prefill_route_runtime, session=session,
                    coordinator=coordinator, name=name + ".latent_router_gather",
                    buffers=buffers.setdefault("latent_router_transport", {}), allocator=allocator,
                    ring=gather_ring)
            else:
                self.down = prepare_projection(self.source, weights["latent_down"], buffer("down", latent),
                    session=session, name=name + ".latent_down", workspace=workspace)
                self.down_reduce = reduction(self.down.output, self.latent, runtime=runtime,
                    coordinator=coordinator, session=session, name=name + ".down_reduce", **reduction_args)
                local_experts = local_logits.shape[1]
                router_width = (local_experts + 31) // 32 * 32
                self.router_input = buffer("router_input", router_width, torch.float32)
                self.gathered_logits = buffer("gathered_router", router_width * shard.world_size, torch.float32)
                self.router_columns = self.gathered_logits.view(tokens, shard.world_size, router_width)[:, :, :local_experts]
                self.router_output = buffer("router_padded", shard.world_size * local_experts,
                    torch.float32).view(tokens, shard.world_size, local_experts)
                # Gathering BF16 views transports the FP32 logits without conversion or arithmetic.
                self.router_gather = PreparedChunkedAllGather(self.router_input.view(torch.bfloat16), self.gathered_logits.view(torch.bfloat16),
                    runtime=router_runtime, coordinator=coordinator, session=session, name=name + ".router_gather")
            self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
                renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], session=session, name=name + ".route")
        self.experts = PreparedRoutedExperts(self.latent, weights["experts"], routes.ids, routes.weights,
            self.expert_output, workspace=workspace, session=session, name=name + ".experts")
        # Gathered routing and routed experts have independent operands and scratch from the shared expert branch.
        self.shared = PreparedMLP(weights["shared"], source, session=session, name=name + ".shared",
                                  buffers=buffers.setdefault("shared", {}), allocator=allocator,
                                  workspace=workspace, parallel_plans=(self.router_projection.plan, self.down.plan,
                                      self.experts.plan, self.up.plan)
                                  if concurrent_shared and (self.route_gather is not None or self.prefill_route_gather is not None) else ())

    def phase_down_pack(self, source):
        self.router_projection(source)
        self.down(source)
        self.prefill_route_gather.pack_ring()

    def phase_route_ring(self):
        self.prefill_route_gather.ring_and_unpack()

    def phase_router(self):
        self.router(self.router_logits)

    def phase_experts(self):
        return self.experts()

    def phase_expert_reduce(self):
        if getattr(self, 'expert_reduce_norm', None) is not None:
            self.expert_reduce_norm(self.expert_output)
        else:
            self.expert_reduce(self.expert_output)
            rms_norm_columns(self.latent_sum, self.latent_norm, self.up_input, self.eps, begin=self.latent_begin)

    def phase_up_combine(self, shared_out):
        self.up(self.up_input)
        self.output.add_(shared_out)
        return self.output

    def __call__(self, source):
        two_stage = self.shared.concurrent and self.prefill_route_gather is not None
        if self.shared.concurrent and not two_stage:
            self.shared.start(source)
        logits = self.router_projection(source)
        if self.route_gather is not None:
            self.down(source)
            self.route_gather()
        elif self.prefill_route_gather is not None:
            self.down(source)
            if two_stage:
                # Stage 1 (gate_up + SiTU) runs on the low-priority overlap stream concurrently with
                # the copy-engine route gather and joins before the cooperative routed-expert kernel.
                self.shared.start_stage1(source)
            self.prefill_route_gather()
            self.router(self.router_logits)
            if two_stage:
                self.shared.join_stage1(source, self.latent)
        else:
            self.router_input[:, :logits.shape[1]].copy_(logits)
            self.router_gather(self.router_input.view(torch.bfloat16))
            self.router_output.copy_(self.router_columns)
            local_experts = self.router_output.shape[-1]
            for rank in range(self.router_output.shape[1]):
                begin = rank * local_experts
                count = min(local_experts, self.router_logits.shape[1] - begin)
                if count > 0:
                    self.router_logits[:, begin:begin + count].copy_(self.router_output[:, rank, :count])
            self.router(self.router_logits)
            self.source.copy_(source.index_select(1, self.hidden_columns))
            self.down_reduce(self.down(self.source))
        routed = self.experts()
        if two_stage:
            # Stage 2 (shared down projection) runs concurrently with the copy-engine expert reduction.
            self.shared.start_stage2(routed)
        if getattr(self, 'expert_reduce_norm', None) is not None:
            self.expert_reduce_norm(routed)
        else:
            self.expert_reduce(routed)
            rms_norm_columns(self.latent_sum, self.latent_norm, self.up_input, self.eps, begin=self.latent_begin)
        self.up(self.up_input)
        shared = self.shared.finish(source, self.output) if self.shared.concurrent else self.shared(source)
        self.output.add_(shared)
        return self.output


class PreparedDecoder:
    """Kimi K3's decoder over packed rows.

    ``verify`` rows are speculative candidates: KDA layers record them without advancing committed states
    until ``commit`` replays each request's accepted prefix. With a draft model, every forward also keeps
    DSpark's AttnRes mixtures entering its auxiliary layers in ``auxiliary``.
    """

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime, embedding_runtime, router_runtime, vision_state,
                 coordinator, session, name, workspace, buffers, kda_records=None, route_runtime=None, context=None,
                 latent_runtimes=(), scoring=None, prefill_route_runtime=None, ring_gather=False, stats_runtime=None):
        # Verification records its candidate rows; other modes ignore records.
        if mode not in ("prefill", "decode", "verify") or mode == "verify" and kda_records is None:
            raise ValueError("Kimi K3 verification decodes recorded candidate rows")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.activations
        allocator = buffers.allocator
        accumulation_buffers = None
        from b12x.comm.pcie import DmaAllReduce

        reduction_staging = None
        copy_engine = isinstance(runtime, DmaAllReduce)
        if copy_engine:
            if mode != "prefill":
                raise ValueError("Kimi copy-engine reductions are prepared only for prefill")
            # Size the FP32 packet workspace to the ring-aligned extent, not the raw payload:
            # an unaligned prefill shape otherwise reduces its 64-element alignment tail as a
            # second, fully synchronized ring invocation.
            direct_ring = bool(getattr(runtime, 'direct_mode', ''))
            extent_fn = direct_aligned_extent if direct_ring else aligned_extent
            ring_elements = extent_fn(source.numel(), runtime.world_size) * runtime.world_size
            if runtime.tree is None and not direct_ring:
                # Tree and direct BF16/FP24 channels reduce BF16 activations without FP32 packet copies.
                count = min(ring_elements, runtime.max_bytes // 4)
                key = "prefill_reduction_workspace"
                if key not in buffers.scratch or buffers.scratch[key].numel() < count:
                    buffers.scratch[key] = (torch.zeros(count, dtype=torch.float32, device=source.device)
                        if allocator is None else allocator.zeros((count,), torch.float32))
                accumulation_buffers = buffers.scratch[key]
            if ring_elements != source.numel():
                key = "prefill_reduction_staging"
                if key not in buffers.scratch or buffers.scratch[key].numel() < ring_elements:
                    buffers.scratch[key] = (torch.zeros(ring_elements, dtype=source.dtype, device=source.device)
                        if allocator is None else allocator.zeros((ring_elements,), source.dtype))
                reduction_staging = buffers.scratch[key]
        active_gather_ring = runtime if copy_engine and ring_gather else None
        if ("local_embedding" not in buffers.scratch
                or buffers.scratch["local_embedding"].shape[0] < source.shape[0]):
            shape = (source.shape[0], weights.embedding.shape[1])
            buffers.scratch["local_embedding"] = (torch.zeros(shape, dtype=source.dtype, device=source.device)
                if allocator is None else allocator.zeros(shape, source.dtype))
        self.local_embedding = buffers.scratch["local_embedding"][:source.shape[0]]
        emb_gather_cls = None if active_gather_ring is None else (
            lambda src, out, **kw: PreparedRingColumnGather(src, out, ring=active_gather_ring, **kw))
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, self.local_embedding, source,
            runtime=embedding_runtime, coordinator=coordinator,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding", gather_cls=emb_gather_cls)
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), source.device,
            verification_columns=kda_records.corrections.shape[3] if mode == "verify" else 1)
        self.buffers, self.layers = buffers.scratch, []
        def zeros(shape):
            return (torch.zeros(shape, dtype=source.dtype, device=source.device) if allocator is None
                    else allocator.zeros(shape, source.dtype))

        # Residual blocks stay live across layers; reserve them before smaller layer intermediates.
        tokens, hidden = source.shape
        if "residual" not in self.buffers:
            committed = -(-len(weights.layers) // config.text["attn_res_block_size"])
            self.buffers["residual"] = dict(blocks=zeros((tokens, committed, hidden)), prefix=zeros(source.shape))
        self.auxiliary = None
        if "total" not in self.buffers:
            self.buffers["total"] = zeros(source.shape)
        self.auxiliary_layers = config.draft.auxiliary_layers if weights.draft is not None else ()
        self.auxiliary_count = None
        self.auxiliary_window, self.auxiliary_alignment = None, 1
        if self.auxiliary_layers:
            auxiliary_rows = retained_auxiliary_rows(weights, tokens, mode=mode)
            if auxiliary_rows < tokens:
                if metadata.lengths.numel() != 1 or config.draft.auxiliary_source != "attention_input":
                    raise ValueError("bounded Kimi draft features require a single accepted prefill and attention-input taps")
                self.auxiliary_count = metadata.lengths
                from ignition.recipes.kimi31.dflash_config import CONTEXT_ROW_ALIGNMENT
                self.auxiliary_window = config.draft.window
                self.auxiliary_alignment = CONTEXT_ROW_ALIGNMENT
            if "auxiliary" not in self.buffers:
                self.buffers["auxiliary"] = zeros((auxiliary_rows, len(self.auxiliary_layers) * hidden))
            self.auxiliary = self.buffers["auxiliary"][:auxiliary_rows]
        # One request-metadata derivation per replay, shared by every DSA layer.
        self.selection = None
        if mode != "prefill" and "dsa" in config.layers:
            from .dcp import ContextShards
            from .selection import PreparedSelectionMetadata

            shards = context.shards if context is not None else ContextShards()
            self.selection = PreparedSelectionMetadata(metadata, shards, rows=source.shape[0],
                device=source.device, allocator=allocator)
        self.dual_wave = bool(
            mode == 'prefill' and copy_engine and getattr(runtime, 'direct_mode', '')
            and active_gather_ring is not None and stats_runtime is not None
            and workspace is not None and workspace.overlap
            and tokens >= 2048 and tokens % 512 == 0
            and metadata.lengths.numel() == 1
            and (not self.auxiliary_layers or config.draft.auxiliary_source == 'attention_input')
        )
        if self.dual_wave:
            self.half_tokens = tokens // 2
            pad_cand = -(-self.half_tokens // runtime.world_size) * runtime.world_size
            self.half_pad_rows = pad_cand if pad_cand * hidden * 2 <= runtime.max_bytes else self.half_tokens
            for key_dw, r_cap in (('dual_mlp_in', tokens), ('dual_total_w0', self.half_pad_rows), ('dual_total_w1', self.half_pad_rows)):
                if key_dw not in self.buffers or self.buffers[key_dw].shape[0] < r_cap:
                    self.buffers[key_dw] = zeros((r_cap, hidden))
            self.dual_mlp_in = self.buffers['dual_mlp_in'][:tokens]
            self.dual_total_w0 = self.buffers['dual_total_w0'][:self.half_pad_rows]
            self.dual_total_w1 = self.buffers['dual_total_w1'][:self.half_pad_rows]
            self.delta_a = DeltaMetadata.allocate(1, source.device)
            self.delta_b = DeltaMetadata.allocate(1, source.device)
            self.comm_priority_stream = torch.cuda.Stream(device=source.device, priority=-1)
            self.comm_stream_handle = self.comm_priority_stream.cuda_stream
            self.comp_stream_handle = workspace.auxiliary_stream.cuda_stream
            self.dual_layers = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            kind = config.layers[index]
            kwargs = dict(mode=mode, session=session, name=prefix + ".attention", workspace=workspace,
                          buffers=self.buffers.setdefault(kind, {}), allocator=buffers.stage_allocator(kind))
            if self.dual_wave:
                half = self.half_tokens
                pad_r = self.half_pad_rows
                attn_in = self.buffers['moe']['output'][:tokens]
                mlp_in = self.dual_mlp_in
                tot_w0 = self.dual_total_w0
                tot_w1 = self.dual_total_w1
                reduction_args = dict(workspace=accumulation_buffers, staging=reduction_staging)
                if kind == 'kda':
                    kda_dw = self.buffers.setdefault('kda_dw', {})
                    for wk in ('w0', 'w1'):
                        kb = kda_dw.setdefault(wk, {})
                        if 'output' not in kb or kb['output'].shape[0] < pad_r:
                            kb['output'] = buffers.stage_allocator('kda').zeros((pad_r, hidden), torch.bfloat16)
                    kda_a = PreparedKDA(w['attention'], attn_in[:half], pool.recurrent(index), self.delta_a,
                        eps=config.text['rms_norm_eps'], mode=mode, session=session, name=prefix + '.attention.w0',
                        workspace=workspace, buffers=kda_dw['w0'], allocator=buffers.stage_allocator('kda'))
                    kda_b = PreparedKDA(w['attention'], attn_in[half:], pool.recurrent(index), self.delta_b,
                        eps=config.text['rms_norm_eps'], mode=mode, session=session, name=prefix + '.attention.w1',
                        workspace=workspace, buffers=kda_dw['w1'], allocator=buffers.stage_allocator('kda'))
                    out_p0 = kda_dw['w0']['output'][:pad_r]
                    out_p1 = kda_dw['w1']['output'][:pad_r]
                    attention = (kda_a, kda_b, out_p0, out_p1)
                else:
                    attention = PreparedMLA(w['attention'], attn_in, metadata, pool.attention(index), context=context,
                                index_cache=pool.index(index) if kind == 'dsa' else None,
                                eps=config.text['rms_norm_eps'],
                                peers=context.peers.get(index) if context is not None else None,
                                index_peers=(context.index_peers.get(index)
                                             if context is not None and context.index_peers is not None else None),
                                selection=self.selection, coordinator=coordinator,
                                latents=next((r for r in latent_runtimes if source.shape[0] <= r.max_batch_size),
                                             latent_runtimes[-1] if latent_runtimes else None),
                                gather_ring=active_gather_ring, dual_wave=True, out_pad_rows=pad_r, **kwargs)
                    out_p0, out_p1 = attention.out_pads
                attn_red_a = copy_engine_reduction(out_p0, tot_w0, runtime=runtime,
                    coordinator=coordinator, session=session, name=prefix + '.attn_reduce.w0', **reduction_args)
                attn_red_b = copy_engine_reduction(out_p1, tot_w1, runtime=runtime,
                    coordinator=coordinator, session=session, name=prefix + '.attn_reduce.w1', **reduction_args)
                if 'mlp' in w:
                    dense_buf = self.buffers.setdefault('dense_dw', {})
                    for wk in ('w0', 'w1'):
                        wb = dense_buf.setdefault(wk, {})
                        if 'output' not in wb or wb['output'].shape[0] < pad_r:
                            wb['output'] = buffers.stage_allocator('dense').zeros((pad_r, hidden), torch.bfloat16)
                    mlp_a = PreparedMLP(w['mlp'], mlp_in[:half], session=session, name=prefix + '.mlp.w0',
                                        buffers=dense_buf['w0'], allocator=buffers.stage_allocator('dense'),
                                        workspace=workspace)
                    mlp_b = PreparedMLP(w['mlp'], mlp_in[half:], session=session, name=prefix + '.mlp.w1',
                                        buffers=dense_buf['w1'], allocator=buffers.stage_allocator('dense'),
                                        workspace=workspace)
                    mlp_out_p0 = dense_buf['w0']['output'][:pad_r]
                    mlp_out_p1 = dense_buf['w1']['output'][:pad_r]
                    mlp_red_a = copy_engine_reduction(mlp_out_p0, tot_w0, runtime=runtime,
                        coordinator=coordinator, session=session, name=prefix + '.mlp_reduce.w0', **reduction_args)
                    mlp_red_b = copy_engine_reduction(mlp_out_p1, tot_w1, runtime=runtime,
                        coordinator=coordinator, session=session, name=prefix + '.mlp_reduce.w1', **reduction_args)
                    self.dual_layers.append((w, kind, attention, ('dense', mlp_a, mlp_b, mlp_out_p0, mlp_out_p1), attn_red_a, attn_red_b, mlp_red_a, mlp_red_b))
                else:
                    moe_dw = self.buffers.setdefault('moe_dw', {})
                    for wk in ('w0', 'w1'):
                        wb = moe_dw.setdefault(wk, {})
                        if 'output' not in wb or wb['output'].shape[0] < pad_r:
                            wb['output'] = buffers.stage_allocator('moe').zeros((pad_r, hidden), torch.bfloat16)
                    moe_a = PreparedLatentMoE(w, mlp_in[:half], config, weights.shard, runtime=runtime,
                        router_runtime=router_runtime, prefill_route_runtime=prefill_route_runtime,
                        coordinator=coordinator, session=session, name=prefix + '.moe.w0', workspace=workspace,
                        buffers=moe_dw['w0'], storage=buffers, allocator=buffers.stage_allocator('moe'),
                        accumulation_buffers=accumulation_buffers, reduction_staging=reduction_staging,
                        copy_engine=True, gather_ring=active_gather_ring, stats_runtime=stats_runtime,
                        concurrent_shared=False)
                    moe_b = PreparedLatentMoE(w, mlp_in[half:], config, weights.shard, runtime=runtime,
                        router_runtime=router_runtime, prefill_route_runtime=prefill_route_runtime,
                        coordinator=coordinator, session=session, name=prefix + '.moe.w1', workspace=workspace,
                        buffers=moe_dw['w1'], storage=buffers, allocator=buffers.stage_allocator('moe'),
                        accumulation_buffers=accumulation_buffers, reduction_staging=reduction_staging,
                        copy_engine=True, gather_ring=active_gather_ring, stats_runtime=stats_runtime,
                        concurrent_shared=False)
                    moe_out_p0 = moe_dw['w0']['output'][:pad_r]
                    moe_out_p1 = moe_dw['w1']['output'][:pad_r]
                    mlp_red_a = copy_engine_reduction(moe_out_p0, tot_w0, runtime=runtime,
                        coordinator=coordinator, session=session, name=prefix + '.mlp_reduce.w0', **reduction_args)
                    mlp_red_b = copy_engine_reduction(moe_out_p1, tot_w1, runtime=runtime,
                        coordinator=coordinator, session=session, name=prefix + '.mlp_reduce.w1', **reduction_args)
                    self.dual_layers.append((w, kind, attention, ('moe', moe_a, moe_b, moe_out_p0, moe_out_p1), attn_red_a, attn_red_b, mlp_red_a, mlp_red_b))
                continue
            attention = (PreparedKDA(w["attention"], source, pool.recurrent(index), self.delta,
                eps=config.text["rms_norm_eps"], records=kda_records.layer(index) if mode == "verify" else None,
                fused_verify=True, **kwargs) if kind == "kda" else
                PreparedMLA(w["attention"], source, metadata, pool.attention(index), context=context,
                            index_cache=pool.index(index) if kind == "dsa" else None,
                            eps=config.text["rms_norm_eps"],
                            peers=context.peers.get(index) if context is not None else None,
                            index_peers=(context.index_peers.get(index)
                                         if context is not None and context.index_peers is not None else None),
                            selection=self.selection,
                            coordinator=coordinator,
                            # The smallest latent exchange that carries the step, else the largest in chunks.
                            latents=next((r for r in latent_runtimes if source.shape[0] <= r.max_batch_size),
                                         latent_runtimes[-1] if latent_runtimes else None),
                            gather_ring=active_gather_ring,
                            **kwargs))
            if "mlp" in w:
                # The dense MLP writes the MoE layers' output, which its reduction consumes before the next mixture.
                dense = self.buffers.setdefault("dense", {})
                if "output" in self.buffers.get("moe", {}):
                    dense.setdefault("output", self.buffers["moe"]["output"])
                mlp = PreparedMLP(w["mlp"], source, session=session, name=prefix + ".mlp", buffers=dense,
                                  allocator=buffers.stage_allocator("dense"))
            else:
                # Shared experts write the sum buffer, which the preceding mixture consumed and the MoE
                # reduction overwrites only after their output is added.
                if "total" in self.buffers:
                    self.buffers.setdefault("moe", {}).setdefault("shared", {}).setdefault("output", self.buffers["total"])
                mlp = PreparedLatentMoE(w, source, config, weights.shard, runtime=runtime,
                    router_runtime=router_runtime, route_runtime=route_runtime if mode != "prefill" else None,
                    prefill_route_runtime=prefill_route_runtime if mode == "prefill" else None,
                    # Decode rows push their packed shards through the channel prefill chunks gather on.
                    route_gather_runtime=prefill_route_runtime if mode != "prefill" else None,
                    coordinator=coordinator, session=session, name=prefix + ".moe", workspace=workspace,
                    buffers=self.buffers.setdefault("moe", {}), storage=buffers, allocator=buffers.stage_allocator("moe"),
                    accumulation_buffers=accumulation_buffers, reduction_staging=reduction_staging,
                    copy_engine=copy_engine,
                    # Large prefill gathers may ride the ring that reduces this graph.
                    gather_ring=active_gather_ring,
                    stats_runtime=stats_runtime if mode == 'prefill' else None)
            # The sum buffer is consumed before the next attention or MLP reduction.
            total = self.buffers["total"][:source.shape[0]]
            reduction = copy_engine_reduction if copy_engine else PreparedChunkedAllReduce
            reduction_args = (dict(workspace=accumulation_buffers, staging=reduction_staging) if copy_engine else {})
            reduce = reduction(source, total, runtime=runtime, coordinator=coordinator,
                               session=session, name=prefix + ".reduce", **reduction_args)
            self.layers.append((w, attention, mlp, reduce))
        if allocator is not None:
            allocator.join(*buffers.stage_allocators.values())
        # The running block sum and committed blocks persist through one forward; every forward
        # rewrites them from its embeddings before reading them.
        r = self.buffers["residual"]
        # The normalized mixture entering attention takes the MoE output, which the previous reduction
        # consumed; the one entering the MLP takes the attention output, which its reduction consumed.
        self.mlp_input = self.buffers["kda"]["output"][:tokens]
        self.residual = ResidualStream(r["prefix"][:tokens], r["blocks"][:tokens], self.buffers["moe"]["output"][:tokens])
        self.scoring = None if scoring is None else scoring.prepare(self.residual.output, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator, name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[:kda_records.corrections.shape[1]] for i in indices]
            self.replay = PreparedKDAReplay(pool.backing, states, kda_records, self.layers[indices[0]][1].plan,
                self.delta.state_indices[:, 0], session=session, workspace=workspace)
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64, device=source.device)
            self.convolution_shape = regions[0].shape

    def commit(self, counts):
        self.replay(counts)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, self.metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self):
        m, d, c = self.metadata, self.delta, self.weights.config.text
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        torch.eq(m.starts, 0, out=d.fresh)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        local = self.embedding.lookup()
        if self.vision_state is not None:
            self.vision_state.inject(m, local)
        # Each layer's residual source is the running sum of its block plus every committed block;
        # a block commits its running sum when the next block starts.
        size, eps = c["attn_res_block_size"], c["rms_norm_eps"]
        delta, mode = self.embedding.gather(local), REPLACE
        if self.dual_wave:
            return self._call_dual_wave(delta, mode, size, eps)
        if self.selection is not None:
            self.selection()
        for index, (w, attention, mlp, reduce) in enumerate(self.layers):
            committed, boundary = -(-index // size), index % size == 0
            mixture = None
            if (index in self.auxiliary_layers
                    and self.weights.config.draft.auxiliary_source == "attention_input"):
                column = self.auxiliary_layers.index(index) * self.residual.prefix.shape[1]
                mixture = self.auxiliary[:, column:column + self.residual.prefix.shape[1]]
            hidden = self.residual.mix(delta, mode, sources=committed, write=index // size if boundary else -1,
                score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps, mixture=mixture,
                mixture_count=self.auxiliary_count if mixture is not None else None,
                mixture_window=self.auxiliary_window, mixture_alignment=self.auxiliary_alignment)
            attended = reduce(attention(hidden))
            hidden = self.residual.mix(attended, REPLACE if boundary else ADD, sources=committed + boundary, write=-1,
                score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps,
                output=self.mlp_input)
            delta, mode = reduce(mlp(hidden)), ADD
            if (index in self.auxiliary_layers
                    and self.weights.config.draft.auxiliary_source == "residual_output"):
                column = self.auxiliary_layers.index(index) * self.residual.prefix.shape[1]
                self.auxiliary[:, column:column + self.residual.prefix.shape[1]].copy_(self.residual.prefix + delta)
        hidden = self.residual.mix(delta, mode, sources=-(-len(self.layers) // size), write=-1,
            score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps)
        if self.scoring is not None:
            self.scoring()
        torch.index_select(hidden, 0, m.sample_indices, out=self.scratch.hidden)
        self.head(self.scratch.hidden)

    def _call_dual_wave(self, delta, mode, size, eps):
        """Interleave half-chunk (Wave A [0:H], Wave B [H:2H]) SM compute on comp_stream with PCIe DMA rings on main."""
        half = self.half_tokens
        m = self.metadata
        la = m.lengths.clamp(min=0, max=half).to(torch.int32)
        lb = (m.lengths - half).clamp(min=0, max=half).to(torch.int32)
        act_a = la > 0
        act_b = lb > 0
        da, db = self.delta_a, self.delta_b
        da.query_start_loc[0].zero_(); da.query_start_loc[1:].copy_(la)
        da.convolution_slots.copy_(torch.where(act_a, m.state_rows, -1))
        torch.eq(m.starts, 0, out=da.fresh)
        da.state_indices[:, 0].copy_(m.state_rows)
        da.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        da.final_state_indices.copy_(torch.where(act_a, m.state_rows, 0))
        da.num_tokens.copy_(la[:1]); da.num_seqs.copy_(act_a.to(torch.int32)[:1])

        db.query_start_loc[0].zero_(); db.query_start_loc[1:].copy_(lb)
        db.convolution_slots.copy_(torch.where(act_b, m.state_rows, -1))
        db.fresh.zero_()
        db.state_indices[:, 0].copy_(m.state_rows)
        db.initial_state_indices.copy_(torch.where(act_b, m.state_rows, 0))
        db.final_state_indices.copy_(torch.where(act_b, m.state_rows, 0))
        db.num_tokens.copy_(lb[:1]); db.num_seqs.copy_(act_b.to(torch.int32)[:1])

        main = torch.cuda.current_stream(delta.device)
        comm = torch.cuda.ExternalStream(self.comm_stream_handle, device=delta.device)
        comp = torch.cuda.ExternalStream(self.comp_stream_handle, device=delta.device)
        res_a = self.residual.substream(0, half)
        res_b = self.residual.substream(half, 2 * half)
        attn_in = self.residual.output
        mlp_in = self.dual_mlp_in
        tot_a = self.dual_total_w0[:half]
        tot_b = self.dual_total_w1[:half]
        n_layers = len(self.dual_layers)

        def on_comp(fn):
            comp.wait_stream(main)
            with torch.cuda.stream(comp):
                fn()
            main.wait_stream(comp)

        def overlap(comm_fn, comp_fn):
            comm.wait_stream(main)
            comp.wait_stream(main)
            with torch.cuda.stream(comm):
                comm_fn()
            with torch.cuda.stream(comp):
                comp_fn()
            main.wait_stream(comm)
            main.wait_stream(comp)

        def attn_mix_kwargs(idx, w, wave):
            committed, boundary = -(-idx // size), idx % size == 0
            mixture = None
            if idx in self.auxiliary_layers and self.weights.config.draft.auxiliary_source == 'attention_input':
                col = self.auxiliary_layers.index(idx) * self.residual.prefix.shape[1]
                mix_full = self.auxiliary[:, col : col + self.residual.prefix.shape[1]]
                mixture = mix_full if self.auxiliary_count is not None else (
                    mix_full[:half] if wave == 0 else mix_full[half:])
            return dict(
                sources=committed,
                write=idx // size if boundary else -1,
                score=w['attention_res_score'],
                norm=w['input_layernorm'],
                eps=eps,
                output_eps=eps,
                mixture=mixture,
                mixture_count=self.auxiliary_count if mixture is not None else None,
                mixture_window=self.auxiliary_window,
                mixture_alignment=self.auxiliary_alignment,
            )

        delta_a, delta_b = delta[:half], delta[half:]
        mix_a_done = False
        kda_a_done = False
        mix_b_done = False
        for index, (w, kind, attention, mlp_spec, attn_red_a, attn_red_b, mlp_red_a, mlp_red_b) in enumerate(self.dual_layers):
            committed, boundary = -(-index // size), index % size == 0
            out_p0, out_p1 = attention[2:] if kind == 'kda' else attention.out_pads
            if not mix_a_done:
                on_comp(lambda w=w, index=index, da=delta_a, md=mode: res_a.mix(da, md, **attn_mix_kwargs(index, w, 0)))
            if kind == 'kda':
                kda_a, kda_b = attention[:2]
                if not kda_a_done:
                    on_comp(lambda kda_a=kda_a: kda_a(attn_in[:half]))
                def run_kda_wave_b(w=w, index=index, kda_b=kda_b, db=delta_b, md=mode, mix_b_done=mix_b_done):
                    if not mix_b_done:
                        res_b.mix(db, md, **attn_mix_kwargs(index, w, 1))
                    kda_b(attn_in[half:])
                overlap(
                    lambda attn_red_a=attn_red_a, out_p0=out_p0: attn_red_a(out_p0),
                    run_kda_wave_b,
                )
            elif attention.can_split_sparse:
                if not mix_b_done:
                    on_comp(lambda w=w, index=index, db=delta_b, md=mode: res_b.mix(db, md, **attn_mix_kwargs(index, w, 1)))
                attention.prepare_sparse_dual(attn_in)
                on_comp(lambda attention=attention: attention.run_sparse_out_half(0))
                overlap(
                    lambda attn_red_a=attn_red_a, out_p0=out_p0: attn_red_a(out_p0),
                    lambda attention=attention: attention.run_sparse_out_half(1),
                )
            else:
                if not mix_b_done:
                    on_comp(lambda w=w, index=index, db=delta_b, md=mode: res_b.mix(db, md, **attn_mix_kwargs(index, w, 1)))
                attention.pre_output(attn_in)
                on_comp(lambda attention=attention: attention.run_out_half(0))
                overlap(
                    lambda attn_red_a=attn_red_a, out_p0=out_p0: attn_red_a(out_p0),
                    lambda attention=attention: attention.run_out_half(1),
                )

            mlp_mode = REPLACE if boundary else ADD
            mlp_sources = committed + boundary
            mlp_kind, m_a, m_b, mlp_p0, mlp_p1 = mlp_spec
            if mlp_kind == 'dense':
                def run_dense_a(w=w, m_a=m_a):
                    res_a.mix(tot_a, mlp_mode, sources=mlp_sources, write=-1,
                              score=w['mlp_res_score'], norm=w['post_attention_layernorm'],
                              eps=eps, output_eps=eps, output=mlp_in[:half])
                    m_a(mlp_in[:half])
                overlap(
                    lambda attn_red_b=attn_red_b, out_p1=out_p1: attn_red_b(out_p1),
                    run_dense_a,
                )
                def run_dense_b(w=w, m_b=m_b):
                    res_b.mix(tot_b, mlp_mode, sources=mlp_sources, write=-1,
                              score=w['mlp_res_score'], norm=w['post_attention_layernorm'],
                              eps=eps, output_eps=eps, output=mlp_in[half:])
                    m_b(mlp_in[half:])
                overlap(
                    lambda mlp_red_a=mlp_red_a, mlp_p0=mlp_p0: mlp_red_a(mlp_p0),
                    run_dense_b,
                )
                next_idx = index + 1
                nw, nkind, nattn = self.dual_layers[next_idx][:3]
                def pre_next_a(nw=nw, nkind=nkind, nattn=nattn, next_idx=next_idx):
                    res_a.mix(tot_a, ADD, **attn_mix_kwargs(next_idx, nw, 0))
                    if nkind == 'kda':
                        nattn[0](attn_in[:half])
                overlap(
                    lambda mlp_red_b=mlp_red_b, mlp_p1=mlp_p1: mlp_red_b(mlp_p1),
                    pre_next_a,
                )
                delta_a, delta_b, mode = tot_a, tot_b, ADD
                mix_a_done = True
                kda_a_done = nkind == 'kda'
                mix_b_done = False
                continue

            # MoE Layer: fully overlapped dual-wave pipeline
            def prep_moe_a(w=w, m_a=m_a):
                res_a.mix(tot_a, mlp_mode, sources=mlp_sources, write=-1,
                          score=w['mlp_res_score'], norm=w['post_attention_layernorm'],
                          eps=eps, output_eps=eps, output=mlp_in[:half])
                m_a.phase_down_pack(mlp_in[:half])
            on_comp(prep_moe_a)

            overlap(m_a.phase_route_ring, lambda m_a=m_a: m_a.shared(mlp_in[:half]))

            def run_exp_a(m_a=m_a):
                m_a.phase_router()
                m_a.phase_experts()
            overlap(
                lambda attn_red_b=attn_red_b, out_p1=out_p1: attn_red_b(out_p1),
                run_exp_a,
            )

            # Overlap m_a.phase_expert_reduce (PCIe) || prep_moe_b (SM)!
            def prep_moe_b(w=w, m_b=m_b):
                res_b.mix(tot_b, mlp_mode, sources=mlp_sources, write=-1,
                          score=w['mlp_res_score'], norm=w['post_attention_layernorm'],
                          eps=eps, output_eps=eps, output=mlp_in[half:])
                m_b.phase_down_pack(mlp_in[half:])
            overlap(m_a.phase_expert_reduce, prep_moe_b)

            # Overlap m_b.phase_route_ring (PCIe) || (m_a.up_combine + m_b.shared) (SM)!
            def run_up_a_sh_b(m_a=m_a, m_b=m_b):
                m_a.phase_up_combine(m_a.shared.down.output)
                m_b.shared(mlp_in[half:])
            overlap(m_b.phase_route_ring, run_up_a_sh_b)

            # Now m_b.phase_route_ring is complete -> safely run m_b.phase_router() + m_b.phase_experts() || mlp_red_a (PCIe)!
            def run_exp_b(m_b=m_b):
                m_b.phase_router()
                m_b.phase_experts()
            overlap(
                lambda mlp_red_a=mlp_red_a, mlp_p0=mlp_p0: mlp_red_a(mlp_p0),
                run_exp_b,
            )

            has_next = index + 1 < n_layers
            if has_next:
                nw, nkind, nattn = self.dual_layers[index + 1][:3]
                overlap(
                    m_b.phase_expert_reduce,
                    lambda nw=nw, next_idx=index + 1: res_a.mix(tot_a, ADD, **attn_mix_kwargs(next_idx, nw, 0)),
                )
            else:
                overlap(m_b.phase_expert_reduce, lambda: None)

            on_comp(lambda m_b=m_b: m_b.phase_up_combine(m_b.shared.down.output))
            if has_next and nkind == 'kda':
                overlap(
                    lambda mlp_red_b=mlp_red_b, mlp_p1=mlp_p1: mlp_red_b(mlp_p1),
                    lambda nkda_a=nattn[0]: nkda_a(attn_in[:half]),
                )
                kda_a_done = True
            else:
                overlap(lambda mlp_red_b=mlp_red_b, mlp_p1=mlp_p1: mlp_red_b(mlp_p1), lambda: None)
                kda_a_done = False
            delta_a, delta_b, mode = tot_a, tot_b, ADD
            mix_a_done = has_next
            mix_b_done = False

        final_sources = -(-n_layers // size)
        res_a.mix(delta_a, mode, sources=final_sources, write=-1,
                  score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps)
        res_b.mix(delta_b, mode, sources=final_sources, write=-1,
                  score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps)
        if self.scoring is not None:
            self.scoring()
        torch.index_select(attn_in, 0, self.metadata.sample_indices, out=self.scratch.hidden)
        self.head(self.scratch.hidden)
