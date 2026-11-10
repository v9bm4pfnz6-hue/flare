# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the shared DSA selection metadata and candidate packing.

The Triton candidate-packing kernel is GPU-only, so its launch wiring is exercised
through a torch emulation and compared bitwise with the elementwise torch sequence it
replaces. The shared selection metadata is plain torch and is compared directly.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import distributed_topk
from ignition.recipes.kimi3 import dsa
from ignition.recipes.kimi3.dcp import ContextShards, RECORD_BYTES
from ignition.recipes.kimi3.selection import PreparedSelectionMetadata


def _per_layer_metadata(metadata, shards, rows):
    expanded = metadata.token_page_table[:rows, :, None] * 2 + torch.arange(2, dtype=torch.int32)
    pages = torch.where(metadata.token_page_table[:rows, :, None] >= 0, expanded, -1).flatten(1)
    lengths = shards.local_tokens((metadata.positions[:rows] + 1).clamp_min(0)).to(torch.int32)
    capacity = min(rows, dsa.SELECTION_ROWS)
    active = torch.stack([lengths[stage:stage + capacity].max().reshape(1)[0]
                          for stage in range(0, rows, capacity)])
    return pages, lengths, active


@pytest.mark.parametrize("rows,width,world,rank", [
    (1, 7, 9, 0), (7, 7, 9, 8), (24, 7, 9, 4), (32, 911, 9, 3),
    (33, 7, 9, 5), (40, 7, 2, 1), (96, 911, 9, 8), (32, 1, 1, 0),
])
def test_shared_selection_metadata_matches_per_layer_sequence(rows, width, world, rank):
    generator = torch.Generator().manual_seed(rows * 977 + width * 13 + world * 7 + rank)
    metadata = SimpleNamespace(
        positions=torch.randint(-1, 1 << 20, (rows,), generator=generator, dtype=torch.int64),
        token_page_table=torch.randint(-1, 512, (rows, width), generator=generator, dtype=torch.int32))
    shards = ContextShards.on(rank, world, "cpu")
    buffers = {}
    component = PreparedSelectionMetadata(metadata, shards, rows=rows, device="cpu")
    view = component()
    expected = _per_layer_metadata(metadata, shards, rows)
    assert torch.equal(view.pages, expected[0])
    assert torch.equal(view.lengths, expected[1])
    assert torch.equal(view.active, expected[2])
    # A second call rewrites the same buffers from new metadata values.
    metadata.positions.add_(3)
    view = component()
    expected = _per_layer_metadata(metadata, shards, rows)
    assert torch.equal(view.pages, expected[0])
    assert torch.equal(view.lengths, expected[1])
    assert torch.equal(view.active, expected[2])


def test_hoisted_buffers_are_allocated_from_the_variant_cursor():
    from ignition.preparation.scratch import GraphBufferPool

    pool = GraphBufferPool("cpu")
    def metadata(rows):
        return SimpleNamespace(positions=torch.arange(rows, dtype=torch.int64),
            token_page_table=torch.zeros((rows, 3), dtype=torch.int32))
    first = pool.cursor()
    PreparedSelectionMetadata(metadata(24), ContextShards.on(0, 9, "cpu"),
                              rows=24, device="cpu", allocator=first)
    second = pool.cursor()
    component = PreparedSelectionMetadata(metadata(6), ContextShards.on(0, 9, "cpu"),
                                          rows=6, device="cpu", allocator=second)
    later = second.zeros((1 << 16,), torch.int32)
    def span(tensor):
        return tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
    later_start, later_end = span(later)
    for tensor in (component.pages, component.lengths, component.active):
        start, end = span(tensor)
        assert end <= later_start or later_end <= start, "hoisted buffer overlaps this variant's later allocation"


def _construct_shared(monkeypatch, rows, world):
    def selector(query, weights, cache, pages, lengths, active, output, **kwargs):
        return SimpleNamespace(query=query, weights=weights, cache=cache, pages=pages,
            lengths=lengths, active=active, output=output, scores=kwargs["scores"])

    def merge(indices, scores, output, **kwargs):
        return SimpleNamespace(indices=indices, scores=scores, output=output)

    def attention(query, cache, output, pages, lengths, starts, **kwargs):
        return SimpleNamespace(query=query, cache=cache, output=output, pages=pages,
            lengths=lengths, starts=starts, lse=kwargs["lse"])

    def collective(source, output, **kwargs):
        return SimpleNamespace(source=source, output=output)

    def combine(partial, lse, output, **kwargs):
        return SimpleNamespace(partial=partial, lse=lse, output=output)

    monkeypatch.setattr(dsa, "prepare_projection", lambda *args, **kwargs: None)
    monkeypatch.setattr(dsa, "PreparedFP8Indexer", selector)
    monkeypatch.setattr(dsa, "PreparedDistributedTopK", merge)
    monkeypatch.setattr(dsa, "PreparedDenseMLA", attention)
    monkeypatch.setattr(dsa, "PreparedAllGather", collective)
    monkeypatch.setattr(dsa, "PreparedLseReduceScatter", combine)
    monkeypatch.setattr(dsa.dense_mla, "Caps", lambda **kwargs: SimpleNamespace(**kwargs))
    capacity = dsa.SELECTION_ROWS
    metadata = SimpleNamespace(positions=torch.arange(rows, dtype=torch.int64),
        token_page_table=torch.zeros((rows, 2), dtype=torch.int32))
    selection_rows = rows if rows <= capacity else capacity
    shared = SimpleNamespace(
        pages=torch.zeros((rows, 4), dtype=torch.int32),
        lengths=torch.zeros((rows,), dtype=torch.int32),
        active=torch.zeros((-(-rows // selection_rows),), dtype=torch.int32))
    context = None if world == 1 else SimpleNamespace(
        shards=ContextShards.on(0, world, "cpu"), candidates=object(), heads=object())
    op = dsa.PreparedDSA(
        {"index_query": None, "index_norm": torch.ones(128), "index_bias": torch.zeros(128)},
        torch.zeros((rows, 8)), torch.zeros((rows, 128)), torch.zeros((rows, 32)),
        torch.zeros((rows, 1, RECORD_BYTES), dtype=torch.bfloat16),
        torch.zeros((rows, 1, 512), dtype=torch.bfloat16), torch.zeros(rows, dtype=torch.int64), metadata,
        torch.zeros((2, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn),
        torch.zeros((4, 64 * 132), dtype=torch.uint8), context=context,
        session=None, coordinator=None, name="attention", workspace=None, buffers={}, selection=shared)
    return op, shared, capacity


@pytest.mark.parametrize("rows,world", [(8, 9), (33, 9), (96, 9), (8, 1)])
def test_layer_binds_shared_selection_slices(monkeypatch, rows, world):
    op, shared, capacity = _construct_shared(monkeypatch, rows, world)
    assert len(op.stages) == -(-rows // capacity)
    for begin, end, selector, merge, quantized, gather, attention, combine, lse, paired, branch in op.stages:
        stage = begin // capacity
        assert torch.equal(selector.pages, shared.pages[begin:end])
        assert torch.equal(selector.lengths, shared.lengths[begin:end])
        assert torch.equal(selector.active, shared.active[stage:stage + 1])


def _pack_candidates_emulation(arguments, metadata):
    indices, scores, local, rank, row_stride = arguments
    world, topk = metadata["WORLD"], metadata["TOPK"]
    owner = int(rank.reshape(-1)[0])
    bits = scores.view(torch.int32)
    for row in range(indices.shape[0]):
        live = indices[row] >= 0
        local[row, :topk, 0] = torch.where(live, bits[row], torch.full_like(bits[row], -8388608))
        local[row, :topk, 1] = torch.where(
            live, indices[row] * world + owner, torch.full_like(indices[row], -1))


def _emulate(kernel, grid, arguments, **metadata):
    if kernel is distributed_topk._pack_candidates:
        _pack_candidates_emulation(arguments, metadata)
    else:
        raise AssertionError(f"unexpected kernel {kernel!r}")


@pytest.mark.parametrize("rows,topk,world,rank", [
    (1, 512, 9, 0), (2, 2048, 9, 8), (7, 2048, 9, 4), (32, 2048, 9, 3), (1, 2048, 2, 1),
])
def test_candidate_packing_matches_torch_sequence(monkeypatch, rows, topk, world, rank):
    monkeypatch.setattr(distributed_topk, "_launch", _emulate)
    generator = torch.Generator().manual_seed(rows * 13 + topk + world * 7 + rank)
    indices = torch.randint(-1, 4608, (rows, topk), generator=generator, dtype=torch.int32)
    scores = torch.randn((rows, topk), generator=generator, dtype=torch.float32)
    local = torch.zeros((rows, topk, 2), dtype=torch.float32)
    distributed_topk._pack_local_candidates(indices, scores, local,
                                            torch.tensor([rank], dtype=torch.int64), world)
    live = indices >= 0
    bits = scores.view(torch.int32)
    expected = torch.zeros_like(local.view(torch.int32))
    expected[:, :, 0] = torch.where(live, bits, torch.full_like(bits, -8388608))
    expected[:, :, 1] = torch.where(live, indices.long() * world + rank,
                                    torch.full_like(indices, -1)).to(torch.int32)
    assert torch.equal(local.view(torch.int32), expected)


def test_record_mapper_writes_caller_buffers(monkeypatch):
    def fake_launch(kernel, grid, arguments, **metadata):
        output, count = arguments[4], arguments[5]
        output.copy_(arguments[0])
        count.fill_(arguments[0].shape[1])

    monkeypatch.setattr(distributed_topk, "_launch", fake_launch)
    selected = torch.arange(12, dtype=torch.int32).reshape(2, 6)
    output = torch.full((2, 6), -7, dtype=torch.int32)
    counts = torch.full((2,), -7, dtype=torch.int32)
    result = distributed_topk.selected_local_records(
        selected, torch.zeros(2, dtype=torch.int64), torch.zeros((2, 1), dtype=torch.int32),
        world=9, rank=torch.tensor([0], dtype=torch.int64), page_size=128,
        output=output, counts=counts)
    assert result[0] is output and result[1] is counts
    assert torch.equal(output, selected) and torch.equal(counts, torch.full((2,), 6, dtype=torch.int32))
    with pytest.raises(ValueError):
        distributed_topk.selected_local_records(
            selected, torch.zeros(2, dtype=torch.int64), torch.zeros((2, 1), dtype=torch.int32),
            world=9, rank=torch.tensor([0], dtype=torch.int64), page_size=128,
            output=torch.zeros((2, 3), dtype=torch.int32), counts=counts)


def test_query_gather_branch_orders_after_merge_and_joins_before_attention(monkeypatch):
    events = []

    class _Stream:
        cuda_stream = 7

    class FakeArena:
        overlap = True
        auxiliary_stream = _Stream()

    class _Selector:
        def __init__(self, *args, **kwargs):
            pass

        def write_keys(self, *args):
            pass

        def __call__(self):
            events.append("selector")

    class _Merge:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self):
            events.append("merge")
            return None

    class _Attention:
        def __init__(self, *args, **kwargs):
            self.output = None

        def __call__(self):
            events.append("attention")

    class _Collective:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args):
            events.append("gather")

    class _Combine:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args):
            events.append("combine")

    def records(selected, positions, pages, **kwargs):
        events.append("records")
        return kwargs["output"], kwargs["counts"]

    def start(handle, stream, query, quantized):
        events.append("start")

    def join(dependency, stream):
        events.append("join")

    def project(*args, **kwargs):
        class _Projection:
            def __call__(self, source):
                events.append("project")

        return _Projection()

    monkeypatch.setattr(dsa, "ScratchArena", FakeArena)
    monkeypatch.setattr(dsa, "prepare_projection", project)
    monkeypatch.setattr(dsa, "PreparedFP8Indexer", _Selector)
    monkeypatch.setattr(dsa, "PreparedDistributedTopK", _Merge)
    monkeypatch.setattr(dsa, "PreparedDenseMLA", _Attention)
    monkeypatch.setattr(dsa, "PreparedAllGather", _Collective)
    monkeypatch.setattr(dsa, "PreparedLseReduceScatter", _Combine)
    monkeypatch.setattr(dsa, "selected_local_records", records)
    monkeypatch.setattr(dsa, "_start_dsa_query_branch", start)
    monkeypatch.setattr(dsa, "_join_dsa_query_branch", join)
    monkeypatch.setattr(dsa.dense_mla, "Caps", lambda **kwargs: SimpleNamespace(**kwargs))
    rows, world = 8, 2
    context = SimpleNamespace(shards=ContextShards.on(0, world, "cpu"), candidates=object(),
                              heads=object())
    metadata = SimpleNamespace(positions=torch.arange(rows, dtype=torch.int64),
        token_page_table=torch.zeros((rows, 2), dtype=torch.int32))
    op = dsa.PreparedDSA(
        {"index_query": None, "index_norm": torch.ones(128), "index_bias": torch.zeros(128)},
        torch.zeros((rows, 8)), torch.zeros((rows, 128)), torch.zeros((rows, 32)),
        torch.zeros((rows, 1, RECORD_BYTES), dtype=torch.bfloat16),
        torch.zeros((rows, 1, 512), dtype=torch.bfloat16), torch.zeros(rows, dtype=torch.int64),
        metadata, torch.zeros((2, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn),
        torch.zeros((4, 64 * 132), dtype=torch.uint8), context=context, session=None,
        coordinator=None, name="attention", workspace=FakeArena(), buffers={})
    assert op.stream_handle == 7
    op()
    assert "start" in events and "join" in events
    assert events.index("merge") < events.index("start") < events.index("join") < events.index("attention")
