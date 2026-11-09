# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 direct transfers with sharded attention, latent projections, and experts."""

from dataclasses import dataclass, replace
import sys

import torch

from ignition.components.weights import (DirectWeightLoader, SharedColumnReads, fp4_down_staging_bytes,
                                         quantize_bf16_linear_as_mxfp8)

from .config import DSparkConfig, LatentDSparkConfig, ModelConfig, expert_intermediate_range
from .residual import score_weight

PREFIX = "language_model.model"
# MXFP4 and Trellis routed experts retain BF16 activations.
EXPERT_ACTIVATION = "a16"


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    output_res_score: torch.Tensor
    layers: tuple[dict, ...]
    scale_scratch: tuple[torch.Tensor, ...] | None
    allocated_bytes: int
    vision: object = None
    draft: object = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, draft_checkpoint=None, online_mxfp8=False,
             peers=None, vision=True, draft_head_quantization=None, index_partitioning=False, weight_storage=None):
        """``peers``, a host's Gloo group and peer storage, reads column-sharded projections once per host.

        ``draft_checkpoint`` loads the declared sliding-window or full-context MLA DSpark model.
        Without ``vision``, the vision encoder stays on disk.
        """
        before = torch.cuda.memory_allocated(device)
        pack = (lambda value: value) if weight_storage is None else weight_storage.tree
        with KimiWeightLoader(manifest, device, shard=shard, progress=progress, online_mxfp8=online_mxfp8,
                              peers=peers, index_partitioning=index_partitioning) as loader:
            config = replace(loader.config, draft=None)
            embedding = pack(loader.linear(f"{PREFIX}.embed_tokens",
                                           columns=shard.padded_partition(config.hidden, alignment=64)))
            norm, res_norm, res_proj = loader.load_tensors(tuple(f"{PREFIX}.{n}.weight" for n in
                ("norm", "output_attn_res_norm", "output_attn_res_proj")))
            norm, output_res_score = pack(norm), pack(score_weight(res_norm, res_proj))
            del res_norm, res_proj
            head = pack(loader.linear("language_model.lm_head",
                                      rows=shard.padded_partition(config.vocabulary, alignment=64)))
            layers = []
            for index in range(len(config.layers)):
                layer = loader.load_layer(index)
                # Expert owners contain prepared layouts and handles. Only the
                # dense numeric weights are repacked, one layer at a time.
                layers.append({key: value if key in ("experts", "expert_trellis") else pack(value)
                               for key, value in layer.items()})
                del layer
        if vision:
            from .vision import VisionWeights
            # Preserve the replicated BF16 checkpoint path. Sharded vision uses
            # packed MXFP8 projections and an unpartitioned attention schedule.
            vision = pack(VisionWeights.load(manifest, device, progress=progress,
                                        shard=shard if online_mxfp8 else None, online_mxfp8=online_mxfp8))
        else:
            vision = None
        draft = None
        if draft_checkpoint is not None:
            from .dspark_weights import DraftWeights

            config = replace(config, draft=DSparkConfig.load(draft_checkpoint, config))
            from ignition.recipes.kimi31.dflash_config import DFlash2Config

            extra = {}
            if isinstance(config.draft, DFlash2Config):
                from ignition.recipes.kimi31.dflash_weights import DraftWeights
                extra["draft_head_quantization"] = draft_head_quantization
            elif draft_head_quantization is not None:
                raise ValueError("Kimi draft head conversion requires the MLA DFlash2 draft")
            elif isinstance(config.draft, LatentDSparkConfig):
                from .latent_dspark import DraftWeights

                extra["target_embedding"] = embedding
            draft = pack(DraftWeights.load(draft_checkpoint, config, device, shard, target_head=head, progress=progress,
                                      online_mxfp8=online_mxfp8, **extra))
        if weight_storage is not None:
            weight_storage.freeze()
        return cls(config, shard, embedding, norm, head, output_res_score, tuple(layers),
                   loader.scale_scratch, torch.cuda.memory_allocated(device) - before, vision=vision, draft=draft)


class KimiWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None, online_mxfp8=False, peers=None,
                 index_partitioning=False):
        super().__init__(manifest, device, prefixes=prefixes, progress=progress)
        self.peers, self.shared_columns = peers, None
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        # Online conversion stores attention, shared-expert, and latent projections as E4M3 values
        # with E8M0 block-32 scales; their activations stay BF16.
        self.online_mxfp8 = online_mxfp8
        index_heads = self.config.text.get("dsa_config", {}).get("index_num_heads", 32)
        # Output-partitioned DSA also partitions prefill rows into complete strided owner sets.
        # Non-divisible topologies keep the release's replicated BF16 index projection instead.
        self.index_partitioning = (index_partitioning and shard.world_size > 1 and index_heads % shard.world_size == 0
                                   and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1)))
        self.config.validate_shard(shard)
        self.scale_scratch = None
        if self.config.text["quantization_config"].get("quant_method") == "exl3":
            expected = tuple(range(self.config.text["first_k_dense_replace"],
                                   self.config.text["num_hidden_layers"]))
            if self.exl3_manifest.geometry.moe_layer_indices != expected:
                raise ValueError("EXL3 manifest must cover every routed-expert layer")

    def _selected_names(self):
        return {n for n in super()._selected_names() if n.startswith("language_model.")}

    def __exit__(self, *exc):
        shared, self.shared_columns = self.shared_columns, None
        if exc[0] is None and shared is not None:
            try:
                shared.close()
            except BaseException:
                super().__exit__(*sys.exc_info())
                raise
        return super().__exit__(*exc)

    def _shared_reads(self):
        """Read column-sharded projections once per host; staging holds this rank's whole expert down projections."""
        if self.peers is None:
            return None
        if self.shared_columns is None:
            group, peer_storage = self.peers
            c = self.config.text
            capacity = -(-c["num_experts"] // self.shard.world_size) * fp4_down_staging_bytes(
                c["routed_expert_hidden_size"], c["moe_intermediate_size"], 32)
            self.shared_columns = SharedColumnReads(group, peer_storage, self.device, capacity)
        return self.shared_columns

    def linear(self, prefix, *, rows=None, columns=None):
        source = self._sources[prefix + ".weight"]
        if source.ndim != 2 or source.dtype != torch.bfloat16:
            raise ValueError(f"{prefix}: expected unquantized BF16 projection")
        r0, r1 = rows or (0, source.shape[0])
        c0, c1 = columns or (0, source.shape[1])
        if min(r0, c0) < 0 or r1 < r0 or c1 < c0:
            raise ValueError(f"{prefix}: invalid projection partition")
        shared = self._shared_reads() if rows is None and columns is not None else None
        if shared is not None:
            out = torch.zeros((source.shape[0], c1 - c0), dtype=source.dtype, device=self.device)
            count = max(0, min(c1, source.shape[1]) - c0)
            if count:
                shared.read_columns(self, source, (c0, c0 + count), out[:, :count])
            return out
        if r1 > source.shape[0] or c1 > source.shape[1]:
            out = torch.zeros((r1 - r0, c1 - c0), dtype=source.dtype, device=self.device)
            rows_count = max(0, min(r1, source.shape[0]) - r0)
            columns_count = max(0, min(c1, source.shape[1]) - c0)
            if rows_count and columns_count:
                self._copy(out[:rows_count, :columns_count],
                           source[r0:r0 + rows_count, c0:c0 + columns_count])
                self._session.flush()
            return out
        return self.load_tensor(prefix + ".weight", extent=(slice(r0, r1), slice(c0, c1)))

    def tensor_rows(self, name, rows):
        source = self._sources.get(name)
        if source is None or rows[1] <= source.shape[0]:
            return self.load_tensor(name, extent=slice(*rows))
        return super().load_padded_tensor(name, rows=rows)

    def fused(self, prefixes, ranges):
        values = [self.linear(p, rows=r) for p, r in zip(prefixes, ranges, strict=True)]
        return torch.cat(values)

    def mlp(self, prefix, intermediate, *, online_mxfp8=False):
        interval = self.shard.padded_partition(intermediate, alignment=32)
        weights = dict(gate_up=self.fused([prefix + ".gate_proj", prefix + ".up_proj"], [interval] * 2),
                       down=self.linear(prefix + ".down_proj", columns=interval))
        return {key: quantize_bf16_linear_as_mxfp8(weight, retain_scale_rows=False)
                for key, weight in weights.items()} if online_mxfp8 else weights

    def load_attention(self, index):
        c, shard = self.config.text, self.shard
        prefix = f"{PREFIX}.layers.{index}"
        a = prefix + ".self_attn"
        heads = c["linear_attn_config"]["num_heads"] if self.config.layers[index] == "kda" else c["num_attention_heads"]
        h0, h1 = shard.padded_partition(heads)
        head_rows = h0 * 128, h1 * 128
        attention = {}
        if self.config.layers[index] == "kda":
            # Every projection of the layer input in one GEMM: query, key, value, forget-gate input, output gate and
            # beta. Each output segment starts on a 16-byte boundary and zero rows pad the width to eight.
            projected = self.fused([a + "." + n for n in ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_proj", "b_proj")],
                [head_rows] * 3 + [None, head_rows, (h0, h1)])
            padding = -projected.shape[0] % 8
            projected = torch.cat((projected, projected.new_zeros((padding, projected.shape[1]))))
            attention["input"] = projected
            attention["forget"] = self.linear(a + ".f_b_proj", rows=head_rows)
            conv = [self.tensor_rows(a + "." + n + ".weight", rows=head_rows)
                    for n in ("q_conv1d", "k_conv1d", "v_conv1d")]
            attention["convolution"] = torch.cat(conv).squeeze(1)
            log = self._sources[a + ".A_log"]
            if tuple(log.shape) not in ((heads,), ((heads + 127) // 128 * 128,)) or log.dtype != torch.float32:
                raise ValueError(f"{a}.A_log must cover the logical heads, optionally padded to 128")
            attention["A_log"] = self.tensor_rows(a + ".A_log", rows=(h0, h1))
            attention["dt_bias"] = self.tensor_rows(a + ".dt_bias", rows=head_rows)
            attention["norm"] = self.load_tensor(a + ".o_norm.weight")
        else:
            attention["q_b"] = self.linear(a + ".q_b_proj", rows=(h0 * 192, h1 * 192))
            kv = self.linear(a + ".kv_b_proj", rows=(h0 * 256, h1 * 256)).view(h1 - h0, 256, 512)
            attention["key"] = kv[:, :128].contiguous()
            attention["value"] = kv[:, 128:].transpose(1, 2).contiguous()
            attention["query_norm"] = self.load_tensor(a + ".q_a_layernorm.weight")
            attention["kv_norm"] = self.load_tensor(a + ".kv_a_layernorm.weight")
        if self.config.layers[index] != "kda":
            # This rank's query and KV latent rows beside its output gate; the ranks gather the latents, as
            # vLLM's VLLM_KIMI_SHARD_QKV_A does. Zero rows pad the latent shard to the exchange's 16-byte width.
            latent = (self.linear(a + ".q_a_proj", rows=shard.padded_partition(c["q_lora_rank"])),
                      self.linear(a + ".kv_a_proj_with_mqa",
                                  rows=shard.padded_partition(c["kv_lora_rank"] + c["qk_rope_head_dim"])))
            gate = self.linear(a + ".g_proj", rows=head_rows)
            width = sum(x.shape[0] for x in latent)
            suffix = ()
            if self.config.layers[index] == "dsa":
                # The key and head-weight projections share the layer input with the MLA projections.
                suffix = (self.linear(a + ".indexer.wk"), self.linear(a + ".indexer.weights_proj"))
                full_index = self.linear(a + ".indexer.wq_b")
                if self.index_partitioning:
                    from ignition.components.weights import OutputPartitionedBF16Weight

                    first, end = shard.partition(32)
                    attention["index_query"] = OutputPartitionedBF16Weight(
                        full_index[first*128:end*128].clone(), full_index.shape[0], full_index.cpu())
                else:
                    attention["index_query"] = full_index
                attention["index_norm"] = self.load_tensor(a + ".indexer.k_norm.weight")
                attention["index_bias"] = self.load_tensor(a + ".indexer.k_norm.bias")
            attention["latent_gate"] = torch.cat((*latent, gate.new_zeros((-width % 8, gate.shape[1])), gate, *suffix))
        attention["output"] = self.linear(a + ".o_proj", columns=head_rows)
        if self.online_mxfp8:
            for key in ("input", "forget", "latent_gate", "q_b", "output"):
                if key in attention:
                    attention[key] = quantize_bf16_linear_as_mxfp8(attention[key], retain_scale_rows=False)
        return attention

    def load_layer(self, index):
        c, shard = self.config.text, self.shard
        prefix = f"{PREFIX}.layers.{index}"
        result = {name: self.load_tensor(f"{prefix}.{name}.weight") for name in
            ("input_layernorm", "post_attention_layernorm")}
        for score, norm, projection in (("attention_res_score", "self_attention_res_norm", "self_attention_res_proj"),
                                        ("mlp_res_score", "mlp_res_norm", "mlp_res_proj")):
            result[score] = score_weight(*self.load_tensors((f"{prefix}.{norm}.weight", f"{prefix}.{projection}.weight")))
        result["attention"] = self.load_attention(index)
        if index < c["first_k_dense_replace"]:
            result["mlp"] = self.mlp(prefix + ".mlp", c["intermediate_size"])
            return result
        m = prefix + ".block_sparse_moe"
        result["shared"] = self.mlp(m + ".shared_experts", c["num_shared_experts"] * c["moe_intermediate_size"],
                                    online_mxfp8=self.online_mxfp8)
        result["router"] = self.linear(m + ".gate", rows=shard.padded_partition(c["num_experts"], alignment=8))
        result["correction"] = self.load_tensor(m + ".gate.e_score_correction_bias")
        result["latent_down"] = self.linear(m + ".routed_expert_down_proj",
                                            columns=shard.padded_partition(c["hidden_size"], alignment=64))
        # Decode rows project their own latent rows and gather them with the router shards.
        result["latent_down_rows"] = self.linear(m + ".routed_expert_down_proj",
                                                 rows=shard.padded_partition(c["routed_expert_hidden_size"], alignment=32))
        result["latent_up"] = self.linear(m + ".routed_expert_up_proj",
                                          columns=shard.padded_partition(c["routed_expert_hidden_size"], alignment=32))
        if self.online_mxfp8:
            for key in ("latent_down", "latent_down_rows", "latent_up"):
                result[key] = quantize_bf16_linear_as_mxfp8(result[key], retain_scale_rows=False)
        # Each rank applies the latent norm's weights to the columns its up-projection shard reads.
        result["latent_norm"] = self.tensor_rows(m + ".routed_expert_norm.weight",
            rows=shard.padded_partition(c["routed_expert_hidden_size"], alignment=32))
        prefixes = [f"{m}.experts.{i}" for i in range(c["num_experts"])]
        formats = {self.config.projection_format(f"{p}.{projection}") for p in prefixes for projection in ("w1", "w2", "w3")}
        if formats == {"exl3"}:
            result["experts"], result["expert_trellis"] = self.load_exl3_experts(index,
                num_experts=c["num_experts"], hidden_size=c["routed_expert_hidden_size"],
                intermediate_size=c["moe_intermediate_size"],
                intermediate_range=expert_intermediate_range(shard, index, c["moe_intermediate_size"]),
                activation="situ")
            return result
        if len(formats) != 1 or not formats <= {"mxfp4", "mxfp4_csf"}:
            raise ValueError(f"{m}: expert storage disagrees with quantization configuration")
        hidden, intermediate = c["routed_expert_hidden_size"], c["moe_intermediate_size"] // shard.world_size
        if self.scale_scratch is None:
            def buffer(rows, columns):
                return torch.empty((c["num_experts"], (rows + 255) // 256 * 256,
                                    (columns + 7) // 8 * 8), dtype=torch.uint8, device=self.device)
            self.scale_scratch = (buffer(2 * intermediate, hidden // 32), buffer(hidden, intermediate // 32))
        if formats == {"mxfp4"}:
            # Native E8M0 block scales are compressed while the experts are prepared, so both
            # checkpoint encodings keep the same resident form; gate/up follow b12x's w31 order.
            result["experts"] = self.load_fp4_experts(prefixes, hidden_size=hidden,
                intermediate_size=c["moe_intermediate_size"], format="mxfp4", activation_mode=EXPERT_ACTIVATION,
                intermediate_range=shard.padded_partition(c["moe_intermediate_size"], alignment=32),
                projection_names=("w1", "w3", "w2"), weight_name="weight_packed", activation="situ",
                w13_layout="w31", scale_compression="csf", scale_scratch=self.scale_scratch,
                shared_columns=self._shared_reads())
            return result
        result["experts"] = self.load_mxfp4_csf_experts(prefixes, hidden_size=hidden,
            intermediate_size=c["moe_intermediate_size"], intermediate_range=shard.padded_partition(c["moe_intermediate_size"], alignment=32),
            projection_names=("w1", "w3", "w2"), weight_name="weight_packed",
            activation="situ", activation_mode=EXPERT_ACTIVATION, scale_scratch=self.scale_scratch)
        return result
