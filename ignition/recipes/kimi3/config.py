# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi hybrid decoder geometry, sparse selection and checkpoint quantization contracts."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from ignition.components.parallel import TensorParallelShard


def padded_width(extent, world_size, alignment=1):
    """Equal aligned rank widths, including the zero-filled tail of a non-divisible extent."""
    return -(-extent // (world_size * alignment)) * alignment


def expert_intermediate_range(shard, layer_index, intermediate_size):
    """Rotate complete 128-channel expert blocks across nine ranks without splitting Hadamard extents."""
    alignment = 128
    if shard.world_size != 9:
        return shard.partition(intermediate_size, alignment=alignment)
    if intermediate_size % (2 * alignment):
        raise ValueError("TP9 expert rotation requires two aligned intermediate halves")
    owner = (shard.rank - layer_index) % shard.world_size
    if owner < 5:
        half, half_rank, ranks_in_half = 0, owner, 5
    else:
        half, half_rank, ranks_in_half = 1, owner - 5, 4
    blocks_per_half = intermediate_size // (2 * alignment)
    quotient, remainder = divmod(blocks_per_half, ranks_in_half)
    blocks = quotient + int(half_rank < remainder)
    begin = half_rank * quotient + min(half_rank, remainder)
    first = (half * blocks_per_half + begin) * alignment
    return first, first + blocks * alignment


PAGE_TOKENS = 128
# Optional speculators-format draft directory within a target checkpoint.
DRAFT_DIRECTORY = "dspark"
DSPARK_REPOSITORY = "RedHatAI/Kimi-K3-speculator.dspark"
DSPARK_REVISION = "38a88101e0d46bb22134b9da340f381b954d40d4"
LATENT_DSPARK_REPOSITORY = "Inferact/Kimi-K3.1-Pro-DSpark-preview"
LATENT_DSPARK_REVISION = "6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94"


@dataclass(frozen=True)
class DSparkConfig:
    """A speculators-format DSpark draft: a Qwen3 sliding-window stack over Kimi K3 auxiliary states.

    The block's anchor predicts the first draft (``sample_from_anchor``), every later block position holds
    the mask token, and a rank-``markov_rank`` transition bias from the previous token joins each
    position's logits while drafts are sampled left to right.
    """

    raw: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target.text)

    @classmethod
    def from_dict(cls, raw, target):
        if raw.get("architectures") == ["DFlash2DraftModel"]:
            from ignition.recipes.kimi31.dflash_config import DFlash2Config
            return DFlash2Config.from_dict(raw, target)
        if raw.get("model_type") == "k3_dspark":
            return LatentDSparkConfig.from_dict(raw, target)
        layer = raw.get("transformer_layer_config", {})
        aux = raw.get("aux_hidden_state_layer_ids", ())
        rope = layer.get("rope_parameters", {})
        if (raw.get("architectures") != ["DSparkDraftModel"] or raw.get("speculators_model_type") != "dspark"
                or layer.get("model_type") != "qwen3" or layer.get("hidden_act") != "silu"
                or layer.get("attention_bias") or layer.get("hidden_size") != target["hidden_size"]
                or layer.get("vocab_size") != target["vocab_size"]
                or raw.get("draft_vocab_size") != target["vocab_size"]
                or layer.get("layer_types") != ["sliding_attention"] * layer.get("num_hidden_layers", 0)
                or not 0 < layer.get("sliding_window", 0) or layer.get("sliding_window") % PAGE_TOKENS
                or raw.get("sliding_window_non_causal") is not False or raw.get("sample_from_anchor") is not True
                or rope.get("rope_type") != "default" or set(rope) != {"rope_type", "rope_theta"}
                or layer.get("num_attention_heads", 0) % max(1, layer.get("num_key_value_heads", 0))
                or raw.get("markov_head_type") != "vanilla" or not 0 < raw.get("markov_rank", 0)
                or not 0 <= raw.get("mask_token_id", -1) < target["vocab_size"]
                or not aux or list(aux) != sorted(set(aux))
                or any(not 0 < i < target["num_hidden_layers"] for i in aux)):
            raise ValueError("unsupported Kimi K3 DSpark draft geometry")
        return cls(raw)

    @property
    def text(self):
        return self.raw["transformer_layer_config"]

    @property
    def auxiliary_layers(self):
        """Decoder layers whose AttnRes input the draft reads; each follows the preceding layer's output."""
        return tuple(self.raw["aux_hidden_state_layer_ids"])

    @property
    def auxiliary_source(self):
        return "attention_input"

    @property
    def noise_token(self):
        return self.raw["mask_token_id"]

    @property
    def markov_rank(self):
        return self.raw["markov_rank"]

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def heads(self):
        return self.text["num_attention_heads"]

    @property
    def kv_heads(self):
        return self.text["num_key_value_heads"]

    @property
    def head_dim(self):
        return self.text["head_dim"]

    @property
    def window(self):
        return self.text["sliding_window"]

    @property
    def theta(self):
        return self.text["rope_parameters"]["rope_theta"]

    def validate_shard(self, shard: TensorParallelShard):
        shard.padded_partition(self.heads)
        shard.padded_partition(self.kv_heads)
        shard.padded_partition(self.text["intermediate_size"], alignment=32)
        shard.padded_partition(self.text["hidden_size"], alignment=8)


@dataclass(frozen=True)
class LatentDSparkConfig:
    """Causal MLA DSpark with full target-derived context and a shared target vocabulary head."""

    raw: dict

    @classmethod
    def from_dict(cls, raw, target):
        if raw.get("architectures") != ["K3DSparkModel"] or raw.get("model_type") != "k3_dspark":
            raise ValueError("latent DSpark requires K3DSparkModel")
        for field, expected in dict(hidden_act="silu", attention_bias=False, sample_from_anchor=True,
                markov_head_type="vanilla", full_attention_causal=True, rope_interleave=True,
                kv_lora_rank=512, q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64,
                v_head_dim=128).items():
            if raw.get(field) != expected:
                raise ValueError(f"unsupported latent DSpark {field}: {raw.get(field)!r}")
        layers = raw.get("target_layer_ids", ())
        if (raw.get("hidden_size") != target["hidden_size"] or raw.get("target_hidden_size") != target["hidden_size"]
                or raw.get("vocab_size") != target["vocab_size"] or raw.get("draft_vocab_size") != target["vocab_size"]
                or raw.get("target_num_hidden_layers") != target["num_hidden_layers"]
                or raw.get("num_attention_heads") != target["num_attention_heads"]
                or raw.get("max_position_embeddings") != target["max_position_embeddings"]
                or raw.get("num_target_layers") != len(layers) or not layers or list(layers) != sorted(set(layers))
                or any(type(i) is not int or not 0 <= i < target["num_hidden_layers"] for i in layers)
                or type(raw.get("num_hidden_layers")) is not int or raw["num_hidden_layers"] <= 0
                or raw.get("layer_types") != ["full_attention"] * raw["num_hidden_layers"]
                or raw.get("sliding_window") is not None or raw.get("use_sliding_window") is not False
                or raw.get("markov_rank", 0) <= 0 or not 0 <= raw.get("mask_token_id", -1) < target["vocab_size"]):
            raise ValueError("latent DSpark geometry or target feature mapping is incompatible")
        rope = raw.get("rope_parameters", {})
        if (rope.get("rope_type") != "yarn" or rope.get("factor", 0) <= 0
                or rope.get("rope_theta", 0) <= 0 or rope.get("original_max_position_embeddings", 0) <= 0
                or rope.get("mscale") != 1. or rope.get("mscale_all_dim") != 0.):
            raise ValueError("latent DSpark requires a complete YaRN rotary configuration")
        return cls(raw)

    @property
    def text(self):
        return self.raw

    @property
    def auxiliary_layers(self):
        return tuple(self.raw["target_layer_ids"])

    @property
    def auxiliary_source(self):
        return "residual_output"

    @property
    def noise_token(self):
        return self.raw["mask_token_id"]

    @property
    def markov_rank(self):
        return self.raw["markov_rank"]

    @property
    def layers(self):
        return self.raw["num_hidden_layers"]

    @property
    def heads(self):
        return self.raw["num_attention_heads"]

    def validate_shard(self, shard):
        for extent, alignment in ((self.heads, 1), (self.raw["hidden_size"], 64),
                (self.raw["intermediate_size"], 32), (self.raw["q_lora_rank"], 1),
                (self.raw["kv_lora_rank"] + self.raw["qk_rope_head_dim"], 1)):
            shard.padded_partition(extent, alignment=alignment)


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    vision: dict
    draft: DSparkConfig | LatentDSparkConfig | None = None
    model_type: str = "kimi_k3"

    @classmethod
    def load(cls, snapshot):
        snapshot = Path(snapshot)
        config = cls.from_dict(json.loads((snapshot / "config.json").read_text()))
        draft = snapshot / DRAFT_DIRECTORY / "config.json"
        if not draft.is_file():
            return config
        return cls(config.text, config.vision, DSparkConfig.from_dict(json.loads(draft.read_text()), config.text),
                   config.model_type)

    @classmethod
    def from_dict(cls, raw):
        architectures = {"kimi_k3": "KimiK3ForConditionalGeneration",
                         "kimi_k31": "KimiK31ForConditionalGeneration"}
        model_type = raw.get("model_type")
        if model_type not in architectures or raw.get("architectures") != [architectures[model_type]]:
            raise ValueError("Kimi requires matching K3 or K3.1 model_type and architecture")
        c = raw["text_config"]
        required = dict(model_type="kimi_linear", hidden_act="situ", activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0, mla_use_nope=True, mla_use_output_gate=True,
            latent_moe_use_norm=True, moe_renormalize=True, moe_router_activation_func="sigmoid",
            num_expert_group=1, topk_group=1, topk_method="noaux_tc", moe_layer_freq=1,
            num_nextn_predict_layers=0, tie_word_embeddings=False)
        for name, value in required.items():
            if c.get(name) != value:
                raise ValueError(f"unsupported Kimi K3 {name}: {c.get(name)!r}")
        for name in ("hidden_size", "num_hidden_layers", "intermediate_size", "moe_intermediate_size",
                     "num_attention_heads", "routed_expert_hidden_size", "num_experts", "num_experts_per_token",
                     "num_shared_experts", "attn_res_block_size", "vocab_size"):
            if type(c.get(name)) is not int or c[name] <= 0:
                raise ValueError(f"Kimi K3 requires positive {name}")
        if not 0 < c["num_experts_per_token"] <= c["num_experts"] <= 1024:
            raise ValueError("Kimi K3 routing requires top_k <= experts <= 1024")
        if not 0 <= c["first_k_dense_replace"] <= c["num_hidden_layers"]:
            raise ValueError("dense layer count exceeds the decoder")
        if (c["kv_lora_rank"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]) != (512, 128, 64, 128):
            raise ValueError("Kimi K3 requires MLA dimensions (512,128,64,128)")
        kda = c["linear_attn_config"]
        if any(kda.get(k) != v for k, v in dict(head_dim=128, short_conv_kernel_size=4,
                use_full_rank_gate=True, gate_lower_bound=-5.0).items()):
            raise ValueError("unsupported Kimi K3 KDA geometry")
        linear, full = kda["kda_layers"], kda["full_attn_layers"]
        if sorted(linear + full) != list(range(1, c["num_hidden_layers"] + 1)):
            raise ValueError("Kimi K3 attention layers must partition the one-based decoder indices")
        dsa = c.get("dsa_config")
        if model_type == "kimi_k31":
            if not isinstance(dsa, dict) or dsa.get("dsa_layers") != full:
                raise ValueError("Kimi K3.1 DSA layers must match the non-KDA layer indices")
            for name, value in dict(index_topk=2048, index_head_dim=128, index_num_heads=32,
                    index_rope_dim=0, index_k_norm_type="layernorm", index_topk_freq=1,
                    index_skip_topk_offset=1).items():
                if dsa.get(name) != value:
                    raise ValueError(f"unsupported Kimi K3.1 {name}: {dsa.get(name)!r}")
        elif dsa is not None:
            raise ValueError("DSA configuration requires the Kimi K3.1 architecture")
        q = c["quantization_config"]
        if q.get("quant_method") == "exl3":
            exl3 = q.get("exl3")
            if (q.get("dense_format") != "bf16" or not isinstance(exl3, dict)
                    or exl3.get("manifest") != "exl3-manifest.json"):
                raise ValueError("Kimi EXL3 requires exl3-manifest.json and BF16 retained projections")
        else:
            if (q.get("quant_method") != "compressed-tensors" or q.get("format") != "mxfp4-pack-quantized"
                    or q.get("quantization_status") != "compressed" or q.get("kv_cache_scheme") is not None
                    or q.get("weight_scale_encoding") not in (None, "csf_32x2_v1")):
                raise ValueError("Kimi K3 requires MXFP4 with native/CSF scales or EXL3 experts")
            groups = list(q.get("config_groups", {}).values())
            if len(groups) != 1:
                raise ValueError("Kimi K3 requires one MXFP4 quantization group")
            group, w = groups[0], groups[0].get("weights", {})
            if (group.get("targets") != ["Linear"] or group.get("format") != q["format"]
                    or group.get("input_activations") is not None or group.get("output_activations") is not None
                    or any(w.get(k) != v for k, v in dict(num_bits=4, group_size=32, strategy="group",
                        type="float", symmetric=True, dynamic=False, scale_dtype="torch.uint8").items())):
                raise ValueError("unsupported Kimi K3 MXFP4 group")
        vision = raw["vision_config"]
        expected = dict(vt_hidden_size=1024, vt_intermediate_size=4096, vt_num_attention_heads=12,
            vt_num_hidden_layers=27, qkv_hidden_size=1536, patch_size=14, norm_type="rmsnorm",
            mlp_type="mlp2", activation_func="gelu_pytorch_tanh", attn_bias=False, linear_bias=False,
            patch_embed_proj_bias=False, init_pos_emb_height=64, init_pos_emb_width=64, init_pos_emb_time=4,
            pos_emb_type="divided_fixed", pos_emb_interpolation_mode="bilinear", merge_kernel_size=[2, 2],
            merge_type="sd2_tpool", mm_projector_type="patchmergerv2", mm_hidden_size=1024,
            text_hidden_size=c["hidden_size"], projector_hidden_act="gelu", projector_ln_eps=1e-5)
        for name, value in expected.items():
            if vision.get(name) != value:
                raise ValueError(f"unsupported Kimi K3 vision {name}: {vision.get(name)!r}")
        if raw.get("media_placeholder_token_id") != 163605 or raw.get("image_placeholder") != "<|kimi_image_placeholder|>":
            raise ValueError("unsupported Kimi K3 image placeholder")
        return cls(c, vision, model_type=model_type)

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def layers(self):
        linear = set(self.text["linear_attn_config"]["kda_layers"])
        sparse = set((self.text.get("dsa_config") or {}).get("dsa_layers", ()))
        return tuple("kda" if i + 1 in linear else "dsa" if i + 1 in sparse else "mla"
                     for i in range(self.text["num_hidden_layers"]))

    def validate_shard(self, shard: TensorParallelShard):
        c = self.text
        if c["quantization_config"].get("quant_method") == "exl3":
            for index in range(c["first_k_dense_replace"], c["num_hidden_layers"]):
                expert_intermediate_range(shard, index, c["moe_intermediate_size"])
        else:
            shard.partition(c["moe_intermediate_size"], alignment=32)
        for extent, alignment in ((c["num_attention_heads"], 1),
                (c["linear_attn_config"]["num_heads"], 1), (c["hidden_size"], 64),
                (c["routed_expert_hidden_size"], 32), (c["moe_intermediate_size"], 32),
                (c["intermediate_size"], 32), (c["num_shared_experts"] * c["moe_intermediate_size"], 32),
                (c["num_experts"], 8), (c["q_lora_rank"], 1),
                (c["kv_lora_rank"] + c["qk_rope_head_dim"], 1), (c["vocab_size"], 64)):
            shard.padded_partition(extent, alignment=alignment)

    def projection_format(self, name):
        q = self.text["quantization_config"]
        if q.get("quant_method") == "exl3":
            return "exl3" if re.search(r"\.block_sparse_moe\.experts\.\d+\.w[123]$", name) else "bf16"
        if any(re.search(pattern[3:], name) if pattern.startswith("re:") else pattern == name for pattern in q["ignore"]):
            return "bf16"
        return "mxfp4" if q.get("weight_scale_encoding") is None else "mxfp4_csf"
