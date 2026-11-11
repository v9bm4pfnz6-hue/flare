# SPDX-License-Identifier: Apache-2.0

"""MLA draft geometry, TP storage and accepted-window semantics."""

from dataclasses import replace
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from tests.triton_interpreter import interpreted, run_interpreted
from ignition.recipes.kimi3.config import DSparkConfig, ModelConfig
from ignition.recipes.kimi3.pool import continuation_layout, page_bytes
from ignition.recipes.kimi31.dflash_config import DFlash2Config
from ignition.recipes.kimi31.recipe import RECIPE


def configuration():
    target = ModelConfig.from_dict(json.loads((Path(__file__).parents[1]/"fixtures/kimi31/config.json").read_text()))
    raw = dict(architectures=["DFlash2DraftModel"], model_type="qwen3", dtype="bfloat16", hidden_act="silu",
        attention_bias=False, attention_dropout=0., is_causal=True, use_sliding_window=True,
        tie_word_embeddings=False, mla_use_nope=False, rope_interleave=True, q_lora_rank=1536,
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, qk_head_dim=192, v_head_dim=128,
        hidden_size=7168, num_attention_heads=96, vocab_size=163840, draft_vocab_size=163840,
        max_position_embeddings=1048576, num_hidden_layers=5, intermediate_size=14336, num_target_layers=93,
        sliding_window=1024, layer_types=["sliding_attention"]*5, rms_norm_eps=1e-5,
        dflash_config=dict(attention_mode="mla", aux_hidden_stream="attn_res", block_size=5,
            conv_kernel_size=2, conv_group_size=16, selector_rank=256, selector_top_k=32,
            mask_token_id=163824, target_layer_ids=[7,23,51,67,83,91]),
        rope_parameters=dict(rope_type="yarn", factor=32., rope_theta=50000.,
            original_max_position_embeddings=32768, mscale=1., mscale_all_dim=1., beta_fast=32, beta_slow=1))
    return target, raw


@pytest.mark.parametrize("world", [8,9,16])
def test_sharded_draft_preserves_target_pages(world):
    target, raw = configuration()
    draft = DSparkConfig.from_dict(raw, target.text)
    assert isinstance(draft, DFlash2Config)
    assert draft.auxiliary_layers == (8,24,52,68,84,92)
    assert draft.auxiliary_source == "attention_input"
    draft.validate_shard(TensorParallelShard(0, world))
    model = replace(target, draft=draft)
    layout = continuation_layout(model, world, speculative_tokens=7)
    costs = RECIPE.speculative_methods["dflash2"].ranks[world]
    assert page_bytes(model) == page_bytes(target) == costs.page_bytes
    assert layout.row_bytes == costs.state_row_bytes
    for i in range(draft.layers):
        region = layout.region(f"draft{i}.latent")
        assert region.shape == (1024,576)
        assert region.dtype == torch.float8_e4m3fn
    assert draft.attention_scale == pytest.approx(192**-.5*(1+.1*__import__('math').log(32))**2)


@pytest.mark.parametrize("change", [dict(is_causal=False),dict(mla_use_nope=True),dict(sliding_window=1000),
    dict(num_target_layers=92),dict(hidden_size=8192),dict(rope_interleave=False)])
def test_rejects_incompatible_draft(change):
    target, raw = configuration()
    raw.update(change)
    with pytest.raises(ValueError):
        DFlash2Config.from_dict(raw, target.text)


def test_private_latent_window_cases_in_triton_interpreter():
    run_interpreted(__file__, 7, "test_private_latent_window_excludes_inactive_and_preserves_history")


@interpreted
@pytest.mark.parametrize("start,count", [(0,5),(17,5),(1023,5),(1024,5),(1025,3),(500001,5),(1025,0)])
def test_private_latent_window_excludes_inactive_and_preserves_history(start, count):
    from ignition.recipes.kimi31.dflash import stage_latent_window

    window, block, hidden = 1024,5,576
    history = torch.arange(window).remainder(61).bfloat16()[None,:,None].expand(2,window,hidden).clone()
    latent = torch.arange(block).add(100).bfloat16()[:,None].expand(block,hidden).clone()
    saved = history.clone()
    output = torch.full((9,128,576), float('nan'), dtype=torch.bfloat16)
    metadata = SimpleNamespace(starts=torch.tensor([start]), lengths=torch.tensor([count],dtype=torch.int32),
        state_rows=torch.tensor([1]), query_start_loc=torch.tensor([0,count],dtype=torch.int32))
    stage_latent_window(history,latent,output,metadata)
    actual = output.view(-1,576)
    prefix = min(start,window)
    expected = torch.zeros_like(actual)
    if count:
        expected[:prefix] = history[1,torch.arange(start-prefix,start).remainder(window)]
        expected[prefix:prefix+count] = latent[:count]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(history,saved,rtol=0,atol=0)
