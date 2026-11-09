# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi preparation backing follows logical page geometry, independently of serving context."""

import pytest
from types import SimpleNamespace

from ignition.recipes.kimi3.bootstrap import PreparationConfig


@pytest.mark.parametrize("context_parallel,page_tokens,pages", [(True, 2048, 10), (False, 128, 40)])
def test_preparation_pages_cover_prefill_and_request_boundaries(context_parallel, page_tokens, pages):
    short = PreparationConfig(context=4096, context_parallel=context_parallel)
    long = PreparationConfig(context=1048576, context_parallel=context_parallel)
    assert short.page_tokens == long.page_tokens == page_tokens
    assert short.preparation_pages == long.preparation_pages == pages
    assert (pages - 8) * page_tokens >= short.prefill_chunk_size
    assert (pages - 9) * page_tokens < short.prefill_chunk_size


@pytest.mark.parametrize("chunk,pages", [(128, 9), (2048, 9), (2176, 10), (4096, 10)])
def test_prefill_backing_rounds_up_to_context_parallel_pages(chunk, pages):
    assert PreparationConfig(prefill_chunk_size=chunk).preparation_pages == pages


@pytest.mark.parametrize("declared_context", [262144, 1048576])
def test_recipe_preserves_declared_addressability_beyond_estimated_kv(monkeypatch, declared_context):
    from ignition.preparation import rank_group
    from ignition.recipes.kimi3.recipe import prepare_variant

    def inspect_config(snapshot, config, prepare_rank, **kwargs):
        return config

    monkeypatch.setattr(rank_group, "prepare_group", inspect_config)
    plan = SimpleNamespace(
        recipe=SimpleNamespace(context=declared_context), context=47104,
        world_size=8, speculation=None, collective_tuning="static", collective_ladder=None,
        prompt_logprobs=False, compile_workers=1, autotune_effort="off", tuning_replica=None,
        native_dense_activations=False, lock_gpu_clocks=False, concurrency=1,
        image_concurrency=0, prefill_chunk=1024, prefill_graph_capacities=None, snapshots=1,
        costs=SimpleNamespace(state_row_bytes=58_361_088), headroom_bytes=0, reserve_bytes=0,
        online_mxfp8=True, stream_overlap=True, context_parallel=True,
        language_model_only=True, checkpoint=None,
    )
    config = prepare_variant(
        "checkpoint", plan,
        bootstrap=SimpleNamespace(PreparationConfig=PreparationConfig, prepare_rank=None),
        draft_repository=None, draft_revision=None, compile_forwards=True,
        kernel_cache=None, progress=None,
    )
    assert config.context == declared_context
    # Physical preparation backing remains one chunk plus boundary pages.
    assert config.preparation_pages == 3
