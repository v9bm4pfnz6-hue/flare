# SPDX-License-Identifier: Apache-2.0

"""The score-exchange top-k declares one FP32 score per candidate slot and runs pack, gather, share in order."""
from types import SimpleNamespace

import pytest
import torch

from ignition.components import candidate_share, distributed_topk


class _Buffers:
    def zeros(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype)


def test_score_exchange_gathers_scores_and_selects_this_rank_share(monkeypatch):
    rows, topk, world = 3, 2048, 9
    events = []

    class Gather:
        def __init__(self, source, output, *, runtime, session, coordinator, name):
            self.arguments = (source, output, runtime, name)
            assert source.dtype == output.dtype == torch.float32
            assert source.shape == (rows, topk) and output.shape == (rows, world * topk)

        def __call__(self, source):
            assert source is self.arguments[0]
            events.append("gather")

    class Share:
        def __init__(self, gathered, indices, scores, rank, output, *, world, session, workspace, name):
            self.arguments = (gathered, indices, scores, rank, output, world, name)

        def __call__(self):
            events.append("share")

    monkeypatch.setattr(distributed_topk, "PreparedAllGather", Gather)
    monkeypatch.setattr(candidate_share, "PreparedCandidateShare", Share)
    monkeypatch.setattr(distributed_topk, "_pack_local_keys",
                        lambda indices, scores, keys: events.append(("pack", indices, scores, keys)))
    indices = torch.zeros((rows, topk), dtype=torch.int32)
    scores = torch.zeros((rows, topk))
    output = torch.zeros((rows, topk), dtype=torch.int32)
    rank = torch.tensor([5])
    runtime = SimpleNamespace()
    exchange = distributed_topk.PreparedScoreExchangeTopK(indices, scores, output, world=world, rank=rank,
        runtime=runtime, session=None, coordinator=None, name="layer3.rows0.candidate_scores",
        buffers=_Buffers(), workspace=None)
    assert exchange.collective.arguments[2:] == (runtime, "layer3.rows0.candidate_scores")
    gathered = exchange.collective.arguments[1]
    assert exchange.share.arguments[0] is gathered
    assert exchange.share.arguments[1:] == (indices, scores, rank, output, world, "layer3.rows0.candidate_scores.share")
    assert exchange() is output
    assert [event if isinstance(event, str) else event[0] for event in events] == ["pack", "gather", "share"]
    assert events[0][1:] == (indices, scores, exchange.keys)
    with pytest.raises(ValueError):
        distributed_topk.PreparedScoreExchangeTopK(indices, scores, output, world=1, rank=rank, runtime=runtime,
            session=None, coordinator=None, name="n", buffers=_Buffers(), workspace=None)


def test_share_scratch_holds_the_order_and_the_values_of_the_k_greatest_scores():
    query = candidate_share.Query(rows=24, world=9, topk=2048)
    assert query.width == 18432 and query.scratch_bytes == 2 * 24 * 2048 * 4
    order, values = candidate_share._buffers(torch.zeros(query.scratch_bytes, dtype=torch.uint8), query)
    assert order.dtype == torch.int32 and values.dtype == torch.float32
    assert order.shape == values.shape == (24, 2048)
    assert values.data_ptr() - order.data_ptr() == 24 * 2048 * 4
