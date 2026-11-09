"""Conversion scopes preserve projection groups outside the requested policy."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.weights import KimiWeightLoader


@dataclass
class Converted:
    shape: tuple


class MetadataLoader(KimiWeightLoader):
    """Exercise loader composition without reading a checkpoint or using CUDA."""

    def __init__(self, scope, online=True):
        self.online_mxfp8 = online
        self.index_partitioning = False
        self.device, self.shard = torch.device("meta"), TensorParallelShard(0, 16)
        self.scale_scratch = None
        self.config = SimpleNamespace(layers=("kda", "dsa"),
            text=dict(linear_attn_config=dict(num_heads=16), num_attention_heads=16,
                      q_lora_rank=16, kv_lora_rank=512, qk_rope_head_dim=64,
                      hidden_size=64, routed_expert_hidden_size=128, num_experts=256,
                      moe_intermediate_size=512, num_shared_experts=2,
                      first_k_dense_replace=0), projection_format=lambda name: "mxfp4")
        self._sources = {"language_model.model.layers.0.self_attn.A_log":
                         SimpleNamespace(shape=(16,), dtype=torch.float32)}

    def linear(self, prefix, *, rows=None, columns=None):
        name = prefix.rsplit(".", 1)[-1]
        output, source = {
            "q_proj": (2048, 64), "k_proj": (2048, 64), "v_proj": (2048, 64),
            "g_proj": (2048, 64), "b_proj": (16, 64), "f_a_proj": (32, 64),
            "f_b_proj": (2048, 32), "o_proj": (64, 2048), "q_b_proj": (3072, 16),
            "kv_b_proj": (4096, 512), "q_a_proj": (16, 64),
            "kv_a_proj_with_mqa": (576, 64), "wq_b": (4096, 16), "wk": (128, 64),
            "weights_proj": (32, 64), "gate_proj": (1024, 64), "up_proj": (1024, 64),
            "down_proj": (64, 1024), "gate": (256, 64),
            "routed_expert_down_proj": (128, 64), "routed_expert_up_proj": (64, 128),
        }[name]
        if rows is not None:
            output = rows[1]-rows[0]
        if columns is not None:
            source = columns[1]-columns[0]
        return torch.empty((output, source), device="meta", dtype=torch.bfloat16)

    def load_tensor(self, name, *, extent=None):
        if "conv1d.weight" in name:
            value = torch.empty((2048, 1, 4), device="meta", dtype=torch.bfloat16)
        elif name.endswith("A_log"):
            value = torch.empty(16, device="meta", dtype=torch.float32)
        else:
            value = torch.empty(2048, device="meta", dtype=torch.bfloat16)
        return value if extent is None else value[extent]

    def load_tensors(self, names):
        return tuple(torch.empty(64, device="meta", dtype=torch.bfloat16) for _ in names)

    def load_fp4_experts(self, *args, **kwargs):
        return "experts"

    def _shared_reads(self):
        return None


@pytest.mark.parametrize("online", [True, False])
def test_conversion_selects_actual_loader_projection_groups(monkeypatch, online):
    monkeypatch.setattr("ignition.recipes.kimi3.weights.quantize_bf16_linear_as_mxfp8",
                        lambda tensor, *, retain_scale_rows: Converted(tuple(tensor.shape)))
    loader = MetadataLoader("all", online)
    for index, kind in enumerate(loader.config.layers):
        layer = loader.load_layer(index)
        converted = {name for name, value in layer["attention"].items() if isinstance(value, Converted)}
        expected = ({"input", "forget", "output"} if kind == "kda"
                    else {"latent_gate", "q_b", "output"}) if online else set()
        assert converted == expected
        assert all(isinstance(value, Converted) == online for value in layer["shared"].values())
        assert all(isinstance(layer[name], Converted) == online
                   for name in ("latent_down", "latent_down_rows", "latent_up"))
        assert isinstance(layer["router"], torch.Tensor)
        assert layer["experts"] == "experts"


def test_tp9_latent_projection_partitions_satisfy_mxfp8_and_collective_widths(monkeypatch):
    monkeypatch.setattr("ignition.recipes.kimi3.weights.quantize_bf16_linear_as_mxfp8",
                        lambda tensor, *, retain_scale_rows: Converted(tuple(tensor.shape)))
    loader = MetadataLoader("all")
    loader.shard = TensorParallelShard(0, 9)
    loader.config.text["hidden_size"] = 7168
    loader.config.text["routed_expert_hidden_size"] = 3584
    layer = loader.load_layer(0)
    assert layer["latent_down"].shape == (128, 832)
    assert layer["latent_down_rows"].shape == (416, 64)
    assert layer["latent_up"].shape == (64, 416)
    assert layer["latent_norm"].shape == (416,)
    assert layer["latent_down_rows"].shape[1] % 32 == 0
    assert layer["latent_up"].shape[1] % 32 == 0
