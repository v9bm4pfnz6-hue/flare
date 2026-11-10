# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Kimi K3 text and image graphs with tensor-parallel decoder storage."""

import contextlib
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity, speculative_token_capacity
from ignition.components.parallel import TensorParallelShard
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
                                              create_all_reduce_runtime, create_single_round_all_reduce_runtime,
                                              create_dma_all_reduce_runtime, create_head_exchange)
from .routing import create_route_runtime
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import GraphBufferPool, ScratchArena, scratch_views
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS, LatentDSparkConfig, padded_width
from ignition.recipes.kimi31.dflash_config import DFlash2Config
from .contract import prepare as prepare_contract
from .decoder import DecoderStorage, PreparedDecoder, Scratch, retained_auxiliary_rows
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights
from .vision import (PATCH_CAPACITIES, FEATURES, HIDDEN, PreparedVision, VisionInputs, VisionScratch,
                     resolve_reference_attention)
from .vision_state import PreparedVisionGeneration
from ignition.components.vision_state import PrefillVisionInputs, VisionState


# Most rows of one all-reduce; larger prefill reductions run in chunks of this many rows.
REDUCE_ROWS = 1024


# World 9 reduces a prefill graph of up to this many rows through the direct channel and a
# larger one through the copy-engine ring. The ring's sixteen sequential hops have a fixed cost
# that dominates small payloads; a direct reduction of 7168 columns takes 0.38 ms at 128 rows
# and 0.77 ms at 256. Measured on complete launches: 128 rows 0.232 s on the ring and 0.222 s
# direct; 256 rows 0.351 s on the ring and 0.376 s direct.
DIRECT_PREFILL_MAX_ROWS = 128


def _prefill_uses_dma(world_size, rows, hidden):
    return world_size == 9 and (rows * hidden) % (world_size * 8) == 0


def _prefill_reduces_on_ring(capacity):
    """Whether a world-9 prefill graph of this capacity reduces on the copy-engine ring."""
    return capacity > DIRECT_PREFILL_MAX_ROWS


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    image_concurrency: int | None = None
    compile_workers: int | None = None
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = tuple(range(16))
    context: int = 1048576
    prefill_chunk_size: int = 4096
    prefill_graph_capacities: tuple[int, ...] | None = None
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    # Red Hat DSpark checkpoint, either external or in the target's dspark directory.
    draft_checkpoint: Path | None = None
    draft_head_quantization: str | None = None
    draft_temperature: float = 1.
    # Verification widths count the pending input and up to seven drafts.
    speculative_widths: tuple[int, ...] = tuple(range(1, 9))
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    speculative_graph_budget: int = 64
    block_verification: bool = False
    # Store eligible attention, shared-expert, latent MoE, and DSpark projections as MXFP8.
    online_mxfp8: bool = False
    # Shared experts overlap routing and routed experts on an auxiliary stream.
    stream_overlap: bool = True
    # Prefill reduction transport. "fp32-owner-packet" is the lossless ordered FP32 copy-engine
    # ring. "bf16-gather" runs the same ordered FP32 reduce-scatter and forwards a single BF16
    # rounding of each fully reduced owner chunk in the all-gather phase: the serving BF16
    # activations are bit-identical and the all-gather wire carries half the bytes.
    # "tree-2+2+2+2+1" routes the reduction over the PCIe switch tree of nine ranks on four leaf
    # pairs and one single rank: BF16 values to the rank that sums them, FP32 partial sums to the
    # owner, one BF16 rounding per owner. Same arithmetic precision; the FP32 additions group
    # by leaf instead of by ring order, so results are not bit-identical to the ring's.
    prefill_reduction: str = "fp32-owner-packet"
    # Ranks share each page's context (decode context parallel over every rank, KV interleave 1).
    context_parallel: bool = True
    # The MoonViT encoder and its graphs prepare beside the text model unless the plan serves text only.
    vision: bool = True

    def __post_init__(self):
        if self.prefill_reduction not in ("fp32-owner-packet", "bf16-gather", "tree-2+2+2+2+1"):
            raise ValueError(
                "prefill reduction must be fp32-owner-packet, bf16-gather or tree-2+2+2+2+1"
            )
        if self.draft_head_quantization not in (None, "mxfp8", "nvfp4"):
            raise ValueError("Kimi draft head conversion must be mxfp8 or nvfp4")
        if self.draft_head_quantization is not None and not self.drafting:
            raise ValueError("Kimi draft head conversion requires draft weights")
        if not 0 < self.draft_temperature < float("inf"):
            raise ValueError("Kimi draft temperature must be finite and positive")
        if self.speculative_widths and max(self.speculative_widths) > 8:
            raise ValueError("Kimi K3 draft model supports at most seven draft tokens")
        from ignition.components.capacity import image_capacity

        if self.vision:
            images = image_capacity(self.max_active, self.image_concurrency)
        else:
            if self.image_concurrency is not None and (type(self.image_concurrency) is not int
                                                       or self.image_concurrency != 0):
                raise ValueError("image concurrency must be zero or omitted when vision is disabled")
            images = 0
        object.__setattr__(self, "image_concurrency", images)
        if (len(self.devices) not in (8, 9, 16) or len(set(self.devices)) != len(self.devices)
                or any(d < 0 for d in self.devices)):
            raise ValueError("Kimi preparation requires eight, nine or sixteen distinct GPUs")
        request_capacity(self.max_active)
        if self.drafting:
            if not self.speculative_widths or not set(self.speculative_widths) <= set(range(1, 9)):
                raise ValueError("Kimi K3 drafts verify one through eight rows per request")
            if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
                raise ValueError("fixed verification width must belong to the captured widths")
            _ = self.speculative_round_shapes
        if not self.page_tokens <= self.context <= 1048576 or self.context % 128:
            raise ValueError(f"Kimi K3 context must be 128-aligned in [{self.page_tokens},1048576]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Kimi K3 prefill chunk must be 128-aligned and fit the context")
        selected = prefill_capacities(self.prefill_chunk_size) if self.prefill_graph_capacities is None else self.prefill_graph_capacities
        if (not isinstance(selected, tuple) or not selected or tuple(sorted(set(selected))) != selected
                or any(type(capacity) is not int for capacity in selected)
                or not set(selected) <= set(prefill_capacities(self.prefill_chunk_size))
                or selected[-1] != self.prefill_chunk_size):
            raise ValueError("Kimi prefill graph selection must be distinct increasing standard capacities, including the chunk")
        object.__setattr__(self, "prefill_graph_capacities", selected)
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def decode_channel_rows(self):
        return max(min(max(self.prefill_chunk_size, self.decode_tokens), REDUCE_ROWS), self.decode_tokens)

    @property
    def prefill_gather_rows(self):
        return self.decode_channel_rows

    @property
    def drafting(self):
        return self.draft_checkpoint is not None

    @property
    def page_tokens(self):
        """Global positions per KV page: PAGE_TOKENS local records on each rank that shares the context."""
        return PAGE_TOKENS * (len(self.devices) if self.context_parallel else 1)

    @property
    def preparation_pages(self):
        """Back one prefill chunk and two boundary pages per request row."""
        return -(-self.prefill_chunk_size // self.page_tokens) + 2 * request_capacity(self.max_active)

    @property
    def tuning_state_rows(self):
        """Null plus live decode rows, or one prefill row and its checkpoint."""
        return max(3, 1 + request_capacity(self.max_active))

    @property
    def speculative_round_shapes(self):
        from ignition.components.capacity import parallel_speculative_shapes

        return parallel_speculative_shapes(self.max_active, self.speculative_widths,
            self.speculative_token_capacity, self.speculative_graph_budget,
            fixed_width=self.speculative_fixed_width) if self.drafting else ()

    @property
    def decode_tokens(self):
        return max((packed for _, _, packed in self.speculative_round_shapes), default=request_capacity(self.max_active))

    @property
    def decode_samples(self):
        return max((batch * width for batch, width, _ in self.speculative_round_shapes), default=request_capacity(self.max_active))

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.drafting else 0

    @property
    def completion_token_capacity(self):
        return max((width for _, width, _ in self.speculative_round_shapes), default=1)


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [shard.header_sha256 for shard in manifest.shards]
    with package_storage.transient():
        peers = None if group is None or package_storage.peer_storage is None else (group, package_storage.peer_storage)
        weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, online_mxfp8=config.online_mxfp8,
                                    draft_checkpoint=config.draft_checkpoint, peers=peers,
                                    vision=config.vision, draft_head_quantization=config.draft_head_quantization,
                                    index_partitioning=config.context_parallel,
                                    weight_storage=package_storage.packed_weights())
        if weights.config.model_type == "kimi_k31":
            from ignition.recipes.kimi31.recipe import RECIPE
        del manifest
        release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    layout = continuation_layout(weights.config, shard.world_size, speculative_tokens=config.draft_tokens)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Kimi K3 continuation")
    slots, blocks = request_capacity(config.max_active), -(-config.context // config.page_tokens)
    drafting = weights.draft is not None
    max_tokens = max(config.prefill_chunk_size, config.decode_tokens)
    state_rows = slots + snapshots
    capture_state_rows = min(state_rows + 1, 1 + 2 * slots)
    pages = config.preparation_pages
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, shard.world_size, state_rows + 1, pages, device,
            speculative_tokens=config.draft_tokens, storage=package_storage,
            maximum_pages=total_memory // page_bytes(weights.config),
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(capture_state_rows, config.tuning_state_rows))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max_tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity), device=device)
        coordinator = CollectivePreparation(group)
        # Serial embedding and context gathers share bounded row staging.
        channel_rows = config.decode_channel_rows
        prefill_gather_rows = config.prefill_gather_rows
        from b12x.comm.pcie.pcie_twoshot_bf16 import SUPPORTED_WORLD_SIZES

        # Aligned prefill chunks can use the copy-engine ring with FP32 owner packets instead of
        # the main channel. Only a world without that ring has to size its channel for a complete
        # prefill chunk, because TP9's direct channel then also serves prefill reductions.
        dma_prefill = _prefill_uses_dma(shard.world_size, config.prefill_chunk_size, weights.config.hidden)
        reduction_rows = config.decode_tokens if shard.world_size in SUPPORTED_WORLD_SIZES else channel_rows
        if shard.world_size == 9 and not dma_prefill:
            reduction_rows = config.prefill_chunk_size
        elif dma_prefill:
            # The direct channel also carries the prefill graphs that stay off the ring.
            reduction_rows = max([reduction_rows, *(capacity for capacity in config.prefill_graph_capacities
                                                    if not _prefill_reduces_on_ring(capacity))])
        collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning,
            max_tokens=reduction_rows, hidden_size=weights.config.hidden, ipc=package_storage.peer_storage)
        prefill_collective = (create_dma_all_reduce_runtime(group, device,
            max_tokens=config.prefill_chunk_size, hidden_size=weights.config.hidden,
            ipc=package_storage.peer_storage, wire_mode="",
            bf16_gather=config.prefill_reduction == "bf16-gather",
            tree=(config.prefill_reduction.removeprefix("tree-")
                  if config.prefill_reduction.startswith("tree-") else None)) if dma_prefill else
            create_single_round_all_reduce_runtime(group, device,
                max_tokens=config.prefill_chunk_size, hidden_size=weights.config.hidden,
                ipc=package_storage.peer_storage)
            if shard.world_size in SUPPORTED_WORLD_SIZES else None)
        text = weights.config.text
        local_vocabulary = padded_width(weights.config.vocabulary, shard.world_size, 64)
        local_hidden = padded_width(weights.config.hidden, shard.world_size, 64)
        local_latent = padded_width(text["routed_expert_hidden_size"], shard.world_size, 32)
        local_router = padded_width(text["num_experts"], shard.world_size, 8)
        vocabulary_collective = create_all_gather_runtime(group, device,
            max_rows=max(config.decode_samples, slots * (config.draft_tokens or 1)),
            columns=local_vocabulary * shard.world_size, ipc=package_storage.peer_storage)
        embedding_collective = create_all_gather_runtime(group, device, max_rows=prefill_gather_rows,
            columns=local_hidden * shard.world_size, ipc=package_storage.peer_storage)
        context_collective = markov_collective = draft_context_collective = convolution_collective = None
        if drafting:
            # Embedding lookup and accepted-feature projection run serially and
            # gather identical hidden-column partitions on every rank.
            context_collective = embedding_collective
            if isinstance(weights.config.draft, DFlash2Config):
                draft_context_collective = create_head_exchange(group, device, max_rows=channel_rows,
                    heads=shard.world_size, head_dim=padded_width(weights.config.draft.layers * 576, shard.world_size, 8),
                    ipc=package_storage.peer_storage)
                convolution_collective = create_head_exchange(group, device,
                    max_rows=slots*(config.draft_tokens+1), heads=shard.world_size,
                    head_dim=padded_width(4 * weights.config.hidden // 16, shard.world_size, 8), ipc=package_storage.peer_storage)
            else:
                markov_collective = create_all_gather_runtime(group, device, max_rows=slots,
                    columns=2 * local_vocabulary * shard.world_size, ipc=package_storage.peer_storage)
        # Prefill rows beyond a channel's capacity gather in chunks, as they reduce.
        router_columns = 2 * shard.world_size * ((local_router + 31) // 32 * 32)
        router_collective = create_all_gather_runtime(group, device, max_rows=prefill_gather_rows,
            columns=router_columns, ipc=package_storage.peer_storage)
        vision_collectives = {}
        if config.vision and weights.vision.shard.world_size > 1:
            from .vision import HEAD_SIZE, INTERMEDIATE, VISION_HEADS

            # Vision and text execute serially. Reuse channels with identical column geometry.
            available = {weights.config.hidden: embedding_collective, router_columns: router_collective}
            padded_heads = -(-VISION_HEADS // shard.world_size) * shard.world_size * HEAD_SIZE
            for columns in sorted({1024, padded_heads, INTERMEDIATE, HIDDEN}):
                runtime = available.get(columns)
                if runtime is None:
                    runtime = create_all_gather_runtime(group, device, max_rows=min(1024, channel_rows),
                        columns=columns, ipc=package_storage.peer_storage)
                vision_collectives[columns] = runtime
        route_collective = (create_route_runtime(group, device, ipc=package_storage.peer_storage)
                            if shard.world_size > 1 else None)
        prefill_route_collective = (create_head_exchange(group, device, max_rows=prefill_gather_rows,
            heads=shard.world_size, head_dim=local_latent + 2 * local_router,
            ipc=package_storage.peer_storage) if shard.world_size == 9 else None)
        context = None
        if config.context_parallel and shard.world_size > 1:
            from .dcp import RECORD_BYTES, ContextParallel, ContextShards
            from .dsa import SELECTION_ROWS, TOPK

            # Decode and verification gather every rank's absorbed E4M3 query heads and combine latent outputs.
            heads = padded_width(text["num_attention_heads"], shard.world_size) * shard.world_size
            sparse = "dsa" in weights.config.layers
            selection_rows = min(max_tokens, SELECTION_ROWS)
            draft_rows = (request_capacity(config.max_active) * config.draft_tokens
                          if isinstance(weights.config.draft, LatentDSparkConfig) and drafting else 0)
            context = ContextParallel(ContextShards.on(rank, shard.world_size, device),
                heads=create_head_exchange(group, device, max_rows=max(config.decode_tokens,
                    selection_rows if sparse else 0, draft_rows), heads=heads, head_dim=512,
                                           query_head_dim=RECORD_BYTES, ipc=package_storage.peer_storage),
                coordinator=coordinator,
                peers=pool.share_pages(package_storage, group), slab=None,
                candidates=(create_all_gather_runtime(group, device, max_rows=selection_rows,
                    columns=4 * TOPK * shard.world_size, ipc=package_storage.peer_storage) if sparse else None),
                prefill_layout="query_split" if sparse else "dcp",
                index_peers=pool.share_pages(package_storage, group, suffix=".index") if sparse else None,
                selections=(create_all_gather_runtime(group, device,
                    max_rows=min(32, (config.prefill_chunk_size + shard.world_size - 1) // shard.world_size),
                    columns=2 * TOPK * shard.world_size, ipc=package_storage.peer_storage) if sparse else None))
            from ignition.components.weights import OutputPartitionedBF16Weight
            if any(isinstance(layer["attention"].get("index_query"), OutputPartitionedBF16Weight)
                   for layer in weights.layers):
                from ignition.components.peer_rows import SharedPeerRows

                context = replace(context,
                    paired_queries=create_all_gather_runtime(group, device, max_rows=selection_rows,
                        columns=(heads * RECORD_BYTES + 32 * 128 * 2)//2, ipc=package_storage.peer_storage),
                    index_exchange=SharedPeerRows(group, device, rows=config.prefill_chunk_size,
                        row_bytes=(32//shard.world_size)*132, ipc=package_storage.peer_storage))
        # Every MLA step gathers each rank's query and KV latent rows, padded to 16 bytes: one exchange for up to
        # eight rows, as vLLM's b12x latent transport carries, and one that prefill steps use in chunks.
        latent = padded_width(text["q_lora_rank"], shard.world_size) + padded_width(
            text["kv_lora_rank"] + text["qk_rope_head_dim"], shard.world_size)
        latent_runtimes = tuple(create_head_exchange(group, device, max_rows=rows, heads=shard.world_size,
                                                     head_dim=-(-latent // 8) * 8, ipc=package_storage.peer_storage)
                                for rows in sorted({min(8, max_tokens), prefill_gather_rows}))
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "headers": headers, "vision": weights.config.vision,
            "draft": None if not drafting else weights.config.draft.raw,
            "world_size": shard.world_size, "contract": "kimi3.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device, checkpoint=identity,
            workload_schema="kimi3", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        deferred_pool = package_storage.scratch_pool()
        with package_storage.scratch_pool() as scratch_pool:
            # AttnRes blocks live through a text forward, not across graphs.
            # Reserve their large slab first so serial vision graphs can reuse it.
            residual_blocks = graph_buffers.cursor().zeros((max_tokens,
                -(-len(weights.layers) // weights.config.text["attn_res_block_size"]),
                weights.config.hidden), torch.bfloat16)
            # Vision and text graphs execute serially and share disposable workspace.
            text_shape = (max(config.prefill_chunk_size, config.decode_tokens), weights.config.hidden)
            # KDA and MLA layers execute serially and share their attention output.
            text_shapes = (text_shape,) * 4
            if drafting:
                auxiliary_rows = max(config.decode_tokens,
                    retained_auxiliary_rows(weights, config.prefill_chunk_size, mode="prefill"))
                text_shapes += ((auxiliary_rows, len(weights.config.draft.auxiliary_layers) * text_shape[1]),)
            patches = PATCH_CAPACITIES if config.vision else ()
            cursor = graph_buffers.cursor()
            cursor.zeros(residual_blocks.shape, torch.bfloat16)
            shared_scratch = cursor.zeros((max(0 if not patches else
                VisionScratch.workspace_elements(max(patches), weights.vision.shard),
                sum(rows * columns for rows, columns in text_shapes)),), torch.bfloat16)
            if patches:
                # Reserve the largest vision slabs before smaller text allocations share their chunks.
                vision_cursor = graph_buffers.cursor()
                vision_work = vision_cursor.zeros(shared_scratch.shape, torch.bfloat16)
                VisionScratch.allocate(max(patches), device, shard=weights.vision.shard,
                                       work=vision_work, buffers=vision_cursor)
                cursor = graph_buffers.cursor()
                cursor.zeros(residual_blocks.shape, torch.bfloat16)
                shared_scratch = cursor.zeros(shared_scratch.shape, torch.bfloat16)
            activation, total, kda_output, moe_output, *auxiliary = scratch_views(shared_scratch, text_shapes)
            mla_output = kda_output
            prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, weights.config, shard.world_size, device,
                                               activations=activation[:config.prefill_chunk_size], buffers=cursor)
            decode_storage = Scratch.allocate(config.decode_tokens, config.decode_samples, weights.config, shard.world_size,
                                              device, buffers=cursor)
            vision_storage = {}
            if context is not None and "mla" in weights.config.layers:
                from .dcp import CONTEXT_WINDOW, RECORD_BYTES

                # Serial prefill layers gather their context one window at a time into one slab beside the chunk's
                # own records; it is released while forwards compile.
                window = min(CONTEXT_WINDOW, config.context)
                context = replace(context, window=window, slab=torch.zeros(
                    (window + config.prefill_chunk_size, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device))
            if context is not None and "dsa" in weights.config.layers:
                # Slabs are reused by serial sparse layers. Native KV remains context-sharded.
                context = replace(context,
                    slab=torch.zeros((config.context, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device),
                    index_slab=torch.zeros((config.context // 64, 64 * 132), dtype=torch.uint8, device=device))
        vision_inputs = vision_prefill = vision_state = None
        if patches:
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            # Frontend uploads and published features are not kernel-tuning
            # operands. Keep their addresses while deferring physical backing.
            with deferred_pool:
                vision_inputs = VisionInputs.allocate(device)
                vision_state = VisionState(inputs, vision_prefill, requests, image_slots=config.image_concurrency,
                    image_token=163605, features=FEATURES, hidden=padded_width(HIDDEN, shard.world_size, 64))
        vision_buffers = {} if not patches else {**vision_inputs.buffers(), **vision_prefill.buffers()}
        buffers = dict(inputs.buffers(), **vision_buffers, **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    forwards, signatures, metadata = {}, {}, {}
    vision_encoders = []
    # The first residual mixture replaces its prefix with the embedding. That
    # activation has no subsequent consumer, so it can hold the running sum.
    residual = dict(blocks=residual_blocks, prefix=activation)
    decoder_buffers = DecoderStorage.allocate(weights.config, shard, device, allocator=cursor, scratch={
        "residual": residual, "total": total, "kda": {"output": kda_output},
        "mla": {"output": mla_output}, "dsa": {"output": mla_output}, "moe": {"output": moe_output},
        "local_embedding": cursor.zeros((max_tokens, weights.embedding.shape[1]), torch.bfloat16),
        **({"auxiliary": auxiliary[0]} if drafting else {})})
    prefill_sizes, batch_sizes = config.prefill_graph_capacities, batch_capacities(config.max_active)
    collectives = dict(runtime=collective, embedding_runtime=embedding_collective, router_runtime=router_collective,
        route_runtime=route_collective, prefill_route_runtime=prefill_route_collective,
        vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
        vision_state=vision_state, context=context, latent_runtimes=latent_runtimes)
    speculation = None
    if drafting:
        from .dspark import SpeculativePreparation

        extra = dict(markov_runtime=markov_collective)
        if isinstance(weights.config.draft, DFlash2Config):
            from ignition.recipes.kimi31.dflash import SpeculativePreparation
            extra = dict(latent_runtime=latent_runtimes[0], context_latent_runtime=draft_context_collective,
                         convolution_runtime=convolution_collective,
                         context_buffers=cursor.fork())
        speculation = SpeculativePreparation(weights, pool, requests, inputs, decode_storage, completion=completion,
            frame=frame, sequence=sequence, session=session, workspace=arena, collectives=collectives,
            decoder_buffers=decoder_buffers, draft_tokens=config.draft_tokens, context_runtime=context_collective,
            page_tokens=config.page_tokens, draft_temperature=config.draft_temperature, **extra)
        speculation.block_verification = config.block_verification
    modes = (True, False) if drafting else (False,)
    count = (len(modes) * (prefill_graph_count(prefill_sizes, scoring) + 4 * len(batch_sizes)) + len(batch_sizes)
             + len(config.speculative_round_shapes) + len(patches))
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                packed_token_capacity=packed_tokens)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            # Speculative packages also hold plain graphs, selected when Flare loads without drafts.
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for kind, mode, sizes in prefill_modes(prefill_sizes, () if enabled else reversed(batch_sizes), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    prefill_runtime = (prefill_collective if mode == "prefill"
                                       and prefill_collective is not None
                                       and (capacity == config.prefill_chunk_size if not dma_prefill
                                            else _prefill_reduces_on_ring(capacity))
                                       else collective)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        session=session, name=name, workspace=arena, buffers=decoder_buffers,
                        **dict(collectives, runtime=prefill_runtime))
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" and patches else None,
                        page_tokens=config.page_tokens)
                    # Plain graphs do not update draft context.
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=(None if speculation is None or not enabled
                                         else speculation.forwards.context))
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1,
                        (*forward_inputs, *(vision_prefill.buffers() if mode == "prefill" and patches else ())),
                        enabled=enabled)
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound

            speculation.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
            for batch, width, packed in sorted(config.speculative_round_shapes, key=lambda shape: (shape[2], shape[0]), reverse=True):
                name = f"speculative{batch}x{width}-tokens{packed}"
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                    buffers=speculation.sampling_buffers.cursor(), target_token_capacity=packed)
                forward.prepare()
                add(name, forward, "speculative", batch, width, (*inputs.buffers(), "sampling_grammar_mask"), enabled=True, packed_tokens=packed)

        def vision_gather(source, output, *, name):
            from ignition.components.collectives import PreparedAllGather

            return PreparedAllGather(source, output, runtime=vision_collectives[output.shape[1]],
                session=session, coordinator=coordinator, name=name)

        for capacity in reversed(patches):
            name = f"vision{capacity}"
            with scratch_pool:
                cursor = graph_buffers.cursor()
                work = cursor.zeros(shared_scratch.shape, torch.bfloat16)
                vision_storage[capacity] = VisionScratch.allocate(capacity, device, shard=weights.vision.shard,
                                                                 work=work, buffers=cursor)
            encoder = PreparedVision(weights.vision, vision_storage[capacity], vision_inputs,
                session=session, workspace=arena, name=name, buffers=cursor,
                gather_factory=vision_gather if weights.vision.shard.world_size > 1 else None,
                chunk_rows=min(1024, channel_rows))
            vision_encoders.append(encoder)
            forward = PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence,
                                                feature_range=shard.padded_partition(HIDDEN, alignment=64), buffers=cursor)
            add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name, activation_mode="auto")
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled, page_tokens=config.page_tokens)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()),
                        enabled=enabled)
    from ignition.components.weights import OutputPartitionedBF16Weight
    for layer in weights.layers:
        index_weight = layer["attention"].get("index_query")
        if isinstance(index_weight, OutputPartitionedBF16Weight):
            index_weight.reference_cpu = None
    deferred_pool.release_backing()
    try:
        with phase(progress, "Prepare and tune kernels") as stage, transient:
            contextual = (partial(resolve_reference_attention, encoders=vision_encoders, coordinator=coordinator)
                          if any(encoder.reference_attention_name is not None for encoder in vision_encoders) else None)
            tuning = session.tune(show_progress=progress is not None, console=stage.console, contextual=contextual,
                                  finalist_policies={} if contextual is not None else None)
    finally:
        deferred_pool.materialize()
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()
        deferred_pool.release_backing()

    def materialize():
        scratch_pool.materialize()
        deferred_pool.materialize()
        # Snapshot destinations coexist with live rows during maintenance capture,
        # but no tuning fixture accesses them. Grow before any complete forward.
        package_storage.grow("STATE_ROWS", capture_state_rows)
        pool.memory_capacity[0] = capture_state_rows
        if vision_inputs is not None:
            vision_inputs.vision_slot.fill_(-1)
        pool.clear_for_preparation()

    # Large vision allocations must precede smaller text allocations in the shared graph pool.
    vision_names = tuple(f"vision{capacity}" for capacity in reversed(patches))
    capture_order = (*vision_names, *(name for name in forwards if name not in vision_names))
    capture_catalog = {name: forwards[name] for name in capture_order}
    try:
        communication = contextlib.nullcontext() if prefill_collective is None else prefill_collective.capture()
        with communication:
            graphs, compiled = capture_forwards(capture_catalog, signatures, session=session, device=device,
                stream=stream, pool=graph_pool, collective=collective, coordinator=coordinator,
                compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
                retain_graphs=config.retain_graphs, materialize=materialize,
                warmup_storage=package_storage,
                compile_kinds=("prefill", "scored_prefill", "decode", "resume", "vision", "speculative"))
    finally:
        scratch_pool.materialize()
        deferred_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    from .tokenizer import convert, runtime_assets
    assets = runtime_assets(convert(snapshot))
    if weights.config.model_type == "kimi_k31":
        from ignition.recipes.kimi31 import recipe as kimi31_recipe

        assets["chat_template"] = {"path": str(Path(kimi31_recipe.__file__).with_name("assets") / "chat_template.jinja")}
    eos = weights.config.text["eos_token_id"]
    eos_ids = list(dict.fromkeys([*(eos if isinstance(eos, list) else [eos]), 163586]))
    frontend = prepare_frontend(snapshot, eos_token_ids=eos_ids, image_profile="moonvit" if patches else None)
    frontend["template_encoding"] = 1
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=config.page_tokens, frontend=frontend,
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency,
                                        vocabulary=vocabulary, page_tokens=config.page_tokens,
                                        draft_lag=0 if drafting else None, vision=bool(patches)))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=config.devices[rank], weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config), page_tokens=config.page_tokens,
        kv_bytes=pages * page_bytes(weights.config), completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    if scoring is not None:
        memory.update(scoring.memory_report())
    if drafting:
        server_config.update(speculative_widths=tuple(sorted({width for _, width, _ in config.speculative_round_shapes})),
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths),
                                                                  config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_round_shapes=config.speculative_round_shapes,
                      speculative_graph_budget=config.speculative_graph_budget)
    draft_disposable = [] if speculation is None else [speculation.draft_scratch.scratch_tensors(), speculation.context_buffers,
        speculation.kda_records.corrections, speculation.kda_records.keys_and_gates]
    return PreparedModel(rebuild_weights=declare_weights(weights, components=('embedding', 'norm', 'head', 'output_res_score', 'vision', 'draft')), device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        close_resources=(*(() if scoring is None else (scoring,)),
                         *(() if prefill_collective is None else (prefill_collective,)),
                         *(() if context is None or context.index_exchange is None else
                           (context.index_exchange, context.paired_queries))),
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges([prefill_storage, decode_storage, graph_buffers.chunks,
            None if speculation is None else speculation.sampling_buffers.chunks, arena.tensor, metadata,
                                           *(() if context is None else (context.slab, context.index_slab)),
                                           *(() if context is None or context.index_exchange is None else
                                             (context.index_exchange.staging,)),
                                           *(() if not patches else (vision_storage, vision_state.arena)),
                                           decoder_buffers.scratch_tensors(), *draft_disposable]), *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage, context,
                   decode_storage, metadata, compiled, stream, graph_pool, session, coordinator, collective,
                   vocabulary_collective, embedding_collective, router_collective, route_collective, prefill_route_collective,
                   prefill_collective,
                   vision_inputs, vision_prefill, vision_state, vision_storage, deferred_pool, scratch_pool, shared_scratch, graph_buffers,
                   speculation, context_collective, markov_collective, latent_runtimes,
                   draft_context_collective, convolution_collective, vision_collectives))
