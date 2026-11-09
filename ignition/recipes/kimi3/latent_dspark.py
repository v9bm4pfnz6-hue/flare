# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Full-context MLA proposals and target-feedback cache writes for Kimi DSpark."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ignition.components.collectives import PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.mlp import MLPScratch
from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

from .attention import PreparedMLA
from .config import padded_width
from .dcp import ContextShards, RECORD_BYTES, write_records
from .dspark import projection
from .pointwise import rms_norm_columns


@triton.jit
def _rotate_yarn(Values, Positions, Frequencies, row_stride, head_stride,
                 OFFSET: tl.constexpr, SCALE: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    pair = tl.arange(0, 32)
    angle = tl.maximum(tl.load(Positions + row), 0).to(tl.float32) * tl.load(Frequencies + pair)
    # Full-range reduction matters at positions near one million; approximate sin.approx is not sufficient.
    cos, sin = libdevice.cos(angle) * SCALE, libdevice.sin(angle) * SCALE
    address = Values + row * row_stride + head * head_stride + OFFSET + 2 * pair
    first, second = tl.load(address).to(tl.float32), tl.load(address + 1).to(tl.float32)
    tl.store(address, first * cos - second * sin)
    tl.store(address + 1, second * cos + first * sin)


class YarnRotary:
    """DeepSeek YaRN frequencies with interleaved FP32 rotation and BF16 outputs."""

    def __init__(self, config, device):
        from transformers import PretrainedConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        c = config.draft.text
        rope = dict(c["rope_parameters"])
        # The serving reference selects DeepSeek YaRN, which derives magnitude from
        # mscale/mscale_all_dim and does not consume the legacy attn_factor field.
        rope.pop("attn_factor", None)
        canonical = PretrainedConfig(rope_parameters=rope, hidden_size=c["hidden_size"],
            num_attention_heads=c["num_attention_heads"], head_dim=c["qk_rope_head_dim"],
            max_position_embeddings=c["max_position_embeddings"])
        self.inverse, self.scale = ROPE_INIT_FUNCTIONS["yarn"](canonical, device=device)

    def apply(self, tensor, positions):
        if tensor.ndim not in (2, 3) or tensor.shape[-1] < 64 or tensor.stride(-1) != 1:
            raise ValueError("YaRN rotary inputs require contiguous 64-channel tails")
        torch.library.wrap_triton(_rotate_yarn)[(tensor.shape[0], tensor.shape[1] if tensor.ndim == 3 else 1)](
            tensor, positions, self.inverse, tensor.stride(0), tensor.stride(1) if tensor.ndim == 3 else 0,
            tensor.shape[-1] - 64, self.scale, num_warps=1, enable_fp_fusion=False)


@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class DraftWeights:
    projection: object
    hidden_norm: torch.Tensor
    context_kv: torch.Tensor
    norm: torch.Tensor
    embedding: torch.Tensor
    head: None
    markov_embedding: torch.Tensor
    markov_head: torch.Tensor
    layers: tuple[LayerWeights, ...]

    @classmethod
    def load(cls, snapshot, config, device, shard, *, target_embedding, target_head, progress=None,
             online_mxfp8=False):
        from ignition.components.weights import quantize_bf16_linear_as_mxfp8

        packed = quantize_bf16_linear_as_mxfp8 if online_mxfp8 else lambda x: x
        d, hidden = config.draft, config.hidden
        d.validate_shard(shard)
        h0, h1 = shard.padded_partition(d.heads)
        q_range, kv_range = shard.padded_partition(d.text["q_lora_rank"]), shard.padded_partition(RECORD_BYTES)
        intermediate = shard.padded_partition(d.text["intermediate_size"], alignment=32)
        with DirectWeightLoader(inspect_indexed_checkpoint(snapshot), device, progress=progress,
                                progress_label="Load draft weights") as loader:
            def matrix(name, rows=None, columns=None):
                value = loader.load_padded_tensor(name + ".weight", rows=rows, columns=columns)
                if value.ndim != 2 or value.dtype != torch.bfloat16:
                    raise ValueError(f"latent DSpark requires BF16 matrices: {name}")
                return value
            project = matrix("context_proj", rows=shard.padded_partition(hidden, alignment=64))
            if project.shape[1] != hidden * len(d.auxiliary_layers):
                raise ValueError("draft context projection does not match the selected target features")
            hidden_norm, norm = loader.load_tensors(("context_norm.weight", "final_norm.weight"))
            layers, context = [], []
            for i in range(d.layers):
                prefix = f"layers.{i}"
                a = prefix + ".self_attn"
                latent = (matrix(a + ".q_a_proj", rows=q_range), matrix(a + ".kv_a_proj_with_mqa", rows=kv_range))
                width = sum(x.shape[0] for x in latent)
                fused = torch.cat((*latent, latent[0].new_zeros((-width % 8, hidden))))
                kv = matrix(a + ".kv_b_proj", rows=(h0 * 256, h1 * 256)).view(h1 - h0, 256, 512)
                attention = dict(latent_gate=packed(fused),
                    q_b=packed(matrix(a + ".q_b_proj", rows=(h0 * 192, h1 * 192))),
                    key=kv[:, :128].contiguous(), value=kv[:, 128:].transpose(1, 2).contiguous(),
                    query_norm=loader.load_tensor(a + ".q_a_layernorm.weight"),
                    kv_norm=loader.load_tensor(a + ".kv_a_layernorm.weight"),
                    output=packed(matrix(a + ".o_proj", columns=(h0 * 128, h1 * 128))))
                context.append(matrix(a + ".kv_a_proj_with_mqa"))
                mlp = MLPWeights(packed(torch.cat([matrix(prefix + ".mlp." + n, rows=intermediate)
                    for n in ("gate_proj", "up_proj")])),
                    packed(matrix(prefix + ".mlp.down_proj", columns=intermediate)))
                layers.append(LayerWeights(attention, loader.load_tensor(prefix + ".input_layernorm.weight"),
                    loader.load_tensor(prefix + ".post_attention_layernorm.weight"), mlp))
            markov_embedding = matrix("markov_head.markov_w1")
            markov_head = matrix("markov_head.markov_w2", rows=shard.padded_partition(config.vocabulary, alignment=64))
            if markov_embedding.shape != (config.vocabulary, d.markov_rank) or markov_head.shape[1] != d.markov_rank:
                raise ValueError("draft Markov projections disagree with their declared rank")
        # The checkpoint declares a confidence head; shared proposal sampling computes its own probabilities.
        return cls(packed(project), hidden_norm, torch.cat(context), norm, target_embedding, None,
                   markov_embedding, markov_head, tuple(layers))


@dataclass(frozen=True)
class DraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    output: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    attention_buffers: dict

    @classmethod
    def allocate(cls, config, shard, tokens, block, device):
        def zeros(columns):
            return torch.zeros((tokens, columns), device=device, dtype=torch.bfloat16)
        return cls(zeros(padded_width(config.hidden, shard.world_size, 64)), zeros(config.hidden), zeros(config.hidden),
            MLPScratch.allocate(tokens, config.hidden, padded_width(config.draft.text["intermediate_size"], shard.world_size, 32), device),
            zeros(config.hidden), zeros(padded_width(config.vocabulary, shard.world_size, 64)), zeros(config.vocabulary), {})

    def rows(self, tokens, samples):
        if samples != tokens or not 0 < tokens <= self.embedding.shape[0]:
            raise ValueError("latent draft scratch requires one sample per physical token")
        return type(self)(self.local_embedding[:tokens], self.embedding[:tokens], self.output[:tokens],
            self.mlp.rows(tokens), self.hidden[:tokens], self.local_logits[:tokens], self.logits[:tokens], self.attention_buffers)

    def scratch_tensors(self):
        return (self,)


class DraftAttention(torch.nn.Module):
    def __init__(self, index, weights, config, scratch, metadata, pool, block, *, runtime, coordinator,
                 session, name, workspace, latent_runtimes, context):
        super().__init__()
        rows = scratch.embedding.shape[0]
        latents = next((r for r in latent_runtimes if rows <= r.max_batch_size),
                       latent_runtimes[-1] if latent_runtimes else None)
        self.attention = PreparedMLA(weights.attention, scratch.embedding, metadata, pool.draft_attention(index),
            mode="decode" if block == 1 else "verify", session=session, name=name, workspace=workspace,
            buffers=scratch.attention_buffers, latents=latents, coordinator=coordinator, context=context,
            eps=config.draft.text["rms_norm_eps"], rotary=YarnRotary(config, scratch.embedding.device), gated=False,
            # Full-context drafts share a bounded partial-output arena across their five layers.
            partial_budget_bytes=64 << 20)
        self.reduce = PreparedAllReduce(self.attention.projections["output"].output, scratch.output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce")

    def forward(self, hidden_states, **kwargs):
        return self.reduce(self.attention(hidden_states)), None


class PreparedContext:
    """Write accepted target feedback into every draft layer's full latent history."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime,
                 coordinator, buffers, context=None):
        self.metadata, self.pool, self.weights = metadata, pool, weights
        self.shards = ContextShards() if context is None else context.shards
        self.eps = weights.config.draft.text["rms_norm_eps"]
        rows, hidden, device = auxiliary.shape[0], weights.config.hidden, auxiliary.device
        def buffer(key, columns, dtype=torch.bfloat16):
            if key not in buffers or buffers[key].shape[0] < rows:
                buffers[key] = torch.zeros((rows, columns), dtype=dtype, device=device)
            return buffers[key][:rows]
        self.local = buffer("local", padded_width(hidden, weights.shard.world_size, 64))
        self.projected, self.normalized = buffer("projected", hidden), buffer("normalized", hidden)
        self.project = projection(auxiliary, weights.draft.projection, self.local,
            session=session, name=name + ".project", workspace=workspace)
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".gather")
        self.kv = buffer("kv", weights.config.draft.layers * RECORD_BYTES)
        self.project_kv = projection(self.normalized, weights.draft.context_kv, self.kv,
            session=session, name=name + ".kv", workspace=workspace)
        self.latent = buffer("latent", RECORD_BYTES)
        self.slots = buffer("slots", 1, torch.int64).view(-1)
        self.rotary = YarnRotary(weights.config, device)

    def __call__(self, auxiliary, counts):
        m, w = self.metadata, self.weights.draft
        self.gather(self.project(auxiliary))
        rms_norm_columns(self.projected, w.hidden_norm, self.normalized, self.eps)
        self.project_kv(self.normalized)
        live = (m.positions >= 0) & (m.positions - m.starts[m.requests] < counts[m.requests])
        self.slots.copy_(torch.where(live, self.shards.token_slots(m.positions, m.token_page_table), -1))
        for index, layer in enumerate(w.layers):
            value = self.kv[:, index * RECORD_BYTES:(index + 1) * RECORD_BYTES]
            rms_norm_columns(value[:, :512], layer.attention["kv_norm"], self.latent[:, :512], self.eps)
            self.latent[:, 512:].copy_(value[:, 512:])
            self.rotary.apply(self.latent, m.positions)
            write_records(self.latent, self.slots, self.pool.draft_attention(index))

    def scratch_tensors(self):
        return (self.local, self.projected, self.normalized, self.kv, self.latent, self.slots)
