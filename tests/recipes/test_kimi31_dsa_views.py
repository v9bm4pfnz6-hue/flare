# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for tensor views retained by sparse-attention stage construction.

Native operator constructors record their operands here. These tests exercise
the recipe's real tensor construction without preparing or executing kernels.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3 import dsa
from ignition.recipes.kimi3.dcp import ContextShards, RECORD_BYTES


@pytest.fixture
def construct(monkeypatch):
    def selector(query, weights, cache, pages, lengths, active, output, **kwargs):
        return SimpleNamespace(query=query, weights=weights, cache=cache, pages=pages,
            lengths=lengths, active=active, output=output, scores=kwargs["scores"],
            name=kwargs["name"], buffers=kwargs["buffers"], plan=object())

    def merge(indices, scores, output, **kwargs):
        return SimpleNamespace(indices=indices, scores=scores, output=output,
            name=kwargs["name"], buffers=kwargs["buffers"], plan=object())

    def attention(query, cache, output, pages, lengths, starts, **kwargs):
        return SimpleNamespace(query=query, cache=cache, output=output, pages=pages,
            lengths=lengths, starts=starts, lse=kwargs["lse"], name=kwargs["name"], plan=object())

    def collective(source, output, **kwargs):
        return SimpleNamespace(source=source, output=output, name=kwargs["name"], plan=object())

    def combine(partial, lse, output, **kwargs):
        return SimpleNamespace(partial=partial, lse=lse, output=output, name=kwargs["name"], plan=object())

    monkeypatch.setattr(dsa, "prepare_projection", lambda *args, **kwargs: None)
    monkeypatch.setattr(dsa, "PreparedFP8Indexer", selector)
    monkeypatch.setattr(dsa, "PreparedScoreExchangeTopK", merge)
    monkeypatch.setattr(dsa, "PreparedDenseMLA", attention)
    monkeypatch.setattr(dsa, "PreparedAllGather", collective)
    monkeypatch.setattr(dsa, "PreparedLseReduceScatter", combine)
    monkeypatch.setattr(dsa.dense_mla, "Caps", lambda **kwargs: SimpleNamespace(**kwargs))

    def build(rows, world=2):
        context = None if world == 1 else SimpleNamespace(
            shards=ContextShards.on(0, world, "cpu"), selections=object(), heads=object())
        cache = torch.zeros((2, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn)
        op = dsa.PreparedDSA(
            {"index_query": None, "index_norm": torch.ones(128), "index_bias": torch.zeros(128)},
            torch.zeros((rows, 8)), torch.zeros((rows, 128)), torch.zeros((rows, 32)),
            torch.zeros((rows, 1, RECORD_BYTES), dtype=torch.bfloat16),
            torch.zeros((rows, 1, 512), dtype=torch.bfloat16), torch.zeros(rows, dtype=torch.int64),
            SimpleNamespace(token_page_table=torch.zeros((rows, 2), dtype=torch.int32)),
            cache, torch.zeros((4, 64 * 132), dtype=torch.uint8), context=context,
            session=None, coordinator=None, name="attention", workspace=None, buffers={})
        return op, cache

    return build


def _selection_operands(stage):
    selector, merge, attention = stage[2], stage[3], stage[6]
    return (selector.pages, selector.lengths, selector.output, selector.scores,
            merge.indices, merge.scores, merge.output, attention.pages,
            attention.lengths, attention.starts, attention.cache)


def _assert_view(view, expected):
    assert view.untyped_storage()._cdata == expected.untyped_storage()._cdata
    assert view.storage_offset() == expected.storage_offset()
    assert view.shape == expected.shape
    assert view.stride() == expected.stride()
    assert view.dtype == expected.dtype
    assert view._version == expected._version


@pytest.mark.parametrize("rows,world", [(1, 1), (33, 1), (65, 2), (4096, 2)])
def test_stages_reuse_identical_views_and_preserve_partial_geometry(construct, rows, world):
    op, cache = construct(rows, world)
    by_count = {}
    for begin, end, selector, merge, quantized, gather, attention, combine, lse, paired, _branch in op.stages:
        count = end - begin
        assert paired is None
        views = (selector.pages, selector.lengths, selector.output, selector.scores,
                 None if merge is None else merge.output, attention.pages, attention.lengths, attention.starts)
        expected = (op.pages[:count], op.lengths[:count], op.local[:count], op.scores[:count],
                    op.selected[:count], op.physical[:count], op.counts[:count], op.offsets[:count + 1])
        for view, reference in zip(views, expected, strict=True):
            if view is not None:
                _assert_view(view, reference)
                before = view._version
                reference.add_(1)
                assert view._version == before + 1
        if count in by_count:
            assert all(value is prior for value, prior in zip(views, by_count[count], strict=True))
        else:
            by_count[count] = views
        _assert_view(attention.cache, cache.view(-1, 1, RECORD_BYTES))
        assert attention.cache is op.stages[0][6].cache
        _assert_view(selector.query, op.index_query[begin:end])
        _assert_view(selector.weights, op.weight_input[begin:end])
        if merge is not None:
            assert selector.output is merge.indices and selector.scores is merge.scores
            assert selector.buffers is merge.buffers
            assert gather.output.untyped_storage()._cdata == attention.query.untyped_storage()._cdata
            assert combine.partial is attention.output and combine.lse is lse
            _assert_view(combine.output, op.output[begin:end])
    assert len({id(stage[2].plan) for stage in op.stages}) == len(op.stages)
    assert len({id(stage[6].plan) for stage in op.stages}) == len(op.stages)
    assert len({id(stage[2].buffers) for stage in op.stages}) == len(op.stages)
    if rows == 4096:
        assert len({id(value) for stage in op.stages for value in _selection_operands(stage)}) == 9
    if rows > 32 and rows % 32:
        assert op.stages[-1][2].pages is not op.stages[0][2].pages


def test_worker_serialization_preserves_shared_view_identities(construct):
    from ignition.preparation.forward_metadata import dumps, loads

    op, _ = construct(65)
    operands = tuple(_selection_operands(stage) for stage in op.stages)
    restored = loads(dumps(operands))
    for before, after in zip(operands, restored, strict=True):
        for reference, view in zip(before, after, strict=True):
            assert view.shape == reference.shape and view.stride() == reference.stride()
            assert view.dtype == reference.dtype and view.storage_offset() == reference.storage_offset()
    assert all(left is right for left, right in zip(restored[0], restored[1], strict=True))
    assert restored[0][0] is not restored[2][0]
    assert restored[0][0].untyped_storage()._cdata == restored[2][0].untyped_storage()._cdata
    assert restored[0][0].untyped_storage()._cdata != restored[0][1].untyped_storage()._cdata
    assert restored[0][-1] is restored[2][-1]
    assert restored[0][2] is restored[0][4]
    assert restored[0][3] is restored[0][5]
    # A view derived after deserialization retains the canonical tensor's version counter.
    canonical = restored[0][1]
    derived = canonical[:]
    before = canonical._version
    with canonical.fake_mode:
        derived.add_(1)
    assert canonical._version == before + 1 and restored[1][1]._version == canonical._version


class _StageComposition:
    def __init__(self, stages):
        self.stages = stages

    def __call__(self):
        result = torch.zeros(())
        for operands in self.stages:
            operands[1].add_(1)
            for value in operands:
                result = result + value.float().sum()
        return result


def test_dynamo_deduplicates_stage_view_sources(construct):
    op, _ = construct(512)
    shared = tuple(_selection_operands(stage) for stage in op.stages)
    distinct = tuple(tuple(value.view_as(value) for value in stage) for stage in shared)
    counts = []

    def backend(graph, inputs):
        counts.append(sum(node.op == "placeholder" for node in graph.graph.nodes))
        return graph.forward

    expected = _StageComposition(distinct)()
    op.lengths.zero_()
    actual = torch.compile(_StageComposition(shared), backend=backend, fullgraph=True, dynamic=False)()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    op.lengths.zero_()
    reference = torch.compile(_StageComposition(distinct), backend=backend, fullgraph=True, dynamic=False)()
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    assert counts == [9, 16 * 11]
