# SPDX-License-Identifier: Apache-2.0

"""A rank's share of a top-k over every rank's candidates, taken from the gathered scores alone."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer
from .native_operator import native_contract, native_plan


@triton.jit
def _select_share(TopValues, Keys, Indices, Scores, Rank, Output, row_stride,
                  K: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    # One program per row. Keys holds the score of every candidate slot of every rank, -inf for a
    # slot without a candidate; TopValues the K greatest keys of the row. With at most K
    # candidates in all, every candidate of this rank is selected and TopValues is not used.
    # Otherwise the cut-off is the least of the K greatest keys, and a candidate of this rank is
    # selected when its score beats the cut-off, or equals it while the K places are not yet
    # filled by greater scores, by equal scores of earlier ranks and by equal scores in earlier
    # slots of this rank.
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < K
    rank = tl.load(Rank).to(tl.int32)
    top = tl.load(TopValues + row*K + col, mask=mask, other=float('inf'))
    cutoff = tl.min(top, 0)
    total = 0
    greater = 0
    earlier = 0
    for peer in tl.static_range(WORLD):
        value = tl.load(Keys + (row*WORLD + peer)*K + col, mask=mask, other=-float('inf'))
        total += tl.sum((value > -float('inf')).to(tl.int32), 0)
        greater += tl.sum((value > cutoff).to(tl.int32), 0)
        earlier += tl.sum((value == cutoff).to(tl.int32), 0) * (rank > peer).to(tl.int32)
    index = tl.load(Indices + row*row_stride + col, mask=mask, other=-1)
    score = tl.load(Scores + row*row_stride + col, mask=mask, other=-float('inf'))
    valid = mask & (index >= 0) & (score > -float('inf'))
    equal = valid & (score == cutoff)
    places = K - greater - earlier
    chosen = valid & ((total <= K) | (score > cutoff) | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= places)))
    tl.store(Output + row*K + col, tl.where(chosen, index*WORLD + rank, -1), mask=mask)


@dataclass(frozen=True)
class Query:
    rows: int
    world: int
    topk: int

    @property
    def width(self):
        return self.world*self.topk

    @property
    def scratch_bytes(self):
        # The order and the values of the K greatest keys of every row.
        return 2*self.rows*self.topk*4


CONTRACT = native_contract(component_id="flare.candidate_share", query_type=Query,
                           tactics=lambda q: (0,), semantic_version=1)


def _buffers(scratch, query):
    count = query.rows*query.topk
    return (scratch.view(torch.int32)[:count].view(query.rows, query.topk),
            scratch.view(torch.float32)[count:2*count].view(query.rows, query.topk))


@program_cache(scope="preparation")
def compile_share(payload, ordinal):
    q = Query(**dict(payload))
    gathered = torch.empty((q.rows, q.width), dtype=torch.float32, device="meta")
    lengths = torch.empty((q.rows,), dtype=torch.int32, device="meta")
    indices = torch.empty((q.rows, q.topk), dtype=torch.int32, device="meta")
    scores = torch.empty((q.rows, q.topk), dtype=torch.float32, device="meta")
    rank = torch.empty((1,), dtype=torch.int64, device="meta")
    output = torch.empty((q.rows, q.topk), dtype=torch.int32, device="meta")
    order, top_values = _buffers(torch.empty(q.scratch_bytes, dtype=torch.uint8, device="meta"), q)
    with torch.cuda.device(ordinal), compile_only_launches():
        topk = {}
        run_row_topk(row_logits=gathered, lengths=lengths, topk=q.topk, output_values=top_values,
                     output_indices=order, write_values=True, launcher_sink=topk)
        select = launch_triton(_select_share, (q.rows, 1, 1), top_values, gathered, indices, scores, rank, output,
                               q.topk, q.topk, q.world, triton.next_power_of_2(q.topk), num_warps=4)
    return dict(topk=topk, select=select)


@dataclass(frozen=True)
class State:
    query: Query
    programs: dict

    def run(self, gathered, lengths, indices, scores, rank, output, scratch):
        q, p = self.query, self.programs
        order, top_values = _buffers(scratch, q)
        run_row_topk(row_logits=gathered, lengths=lengths, topk=q.topk, output_values=top_values,
                     output_indices=order, write_values=True, launcher=p["topk"])
        p["select"][(q.rows, 1, 1)](top_values, gathered, indices, scores, rank, output, indices.stride(0),
                                    q.topk, q.world, triton.next_power_of_2(q.topk))


@torch.library.custom_op("flare::select_candidate_share", mutates_args=("output", "scratch"))
def _execute(gathered: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor,
             rank: torch.Tensor, output: torch.Tensor, scratch: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, gathered.device).run(
        gathered, lengths, indices, scores, rank, output, scratch)


@_execute.register_fake
def _fake(gathered, lengths, indices, scores, rank, output, scratch, handle):
    pass


class PreparedCandidateShare:
    """This rank's candidates among the top-k of every rank's candidates.

    `gathered` is [rows, world * topk] FP32: the score of every candidate slot of every rank in
    rank order, -inf for a slot without a candidate. `indices` and `scores` are this rank's own
    slots and `rank` its one-element int64 rank. A slot holds a candidate when its index is not
    negative and its score is greater than -inf. Each output row holds `index * world + rank`
    for every candidate of this rank that belongs to the top-k of all candidates and -1 in every
    other slot.

    For finite scores the selected set is the one PreparedCandidateMerge selects from the same
    candidates gathered as score/ID pairs, restricted to this rank: scores above the cut-off, then
    equal scores in gathered order, which gives earlier ranks and earlier slots the remaining
    places. Every rank reads the same scores, so the shares of the ranks are disjoint and their
    union is that selection. Scores must not be NaN.
    """

    def __init__(self, gathered, indices, scores, rank, output, *, world, session, workspace, name):
        if (indices.ndim != 2 or indices.dtype != torch.int32 or scores.shape != indices.shape
                or scores.dtype != torch.float32 or indices.stride() != scores.stride() or indices.stride(1) != 1
                or indices.shape[1] not in (512, 1024, 2048) or world < 2
                or gathered.shape != (indices.shape[0], world*indices.shape[1]) or gathered.dtype != torch.float32
                or not gathered.is_contiguous() or output.shape != indices.shape or output.dtype != torch.int32
                or not output.is_contiguous() or rank.shape != (1,) or rank.dtype != torch.int64
                or any(value.device != gathered.device for value in (indices, scores, rank, output))):
            raise ValueError("a candidate share requires the FP32 scores of every rank's slots, this rank's int32 "
                             "positions and FP32 scores, its int64 rank and an int32 output of the same slots")
        self.gathered, self.indices, self.scores, self.rank, self.output = gathered, indices, scores, rank, output
        q = Query(indices.shape[0], world, indices.shape[1])
        # The row top-k reads every gathered slot. This constant belongs to the operator: storage
        # shared with other operators would not keep it.
        self.lengths = lengths = torch.full((q.rows,), q.width, dtype=torch.int32, device=gathered.device)
        payload = CONTRACT.encode_query(q)

        def materialize(tactic):
            programs = compile_share(payload, gathered.device.index)
            load_programs(programs)
            return attach_programs(State(q, programs), *programs["topk"].values(), programs["select"])

        self.plan = native_plan(contract=CONTRACT, query=q, device=gathered.device,
            scratch_bytes=lambda tactic: q.scratch_bytes, materialize=materialize,
            compile_jobs=lambda tactic, device: (CompileJob.create(
                "ignition.components.candidate_share:compile_share", payload, device.ordinal),))

        def call(state):
            def inputs():
                keys = torch.empty_like(gathered).normal_()
                own = torch.arange(q.topk, dtype=torch.int32, device=gathered.device).repeat(q.rows, 1)
                return keys, own, keys[:, :q.topk].clone(), torch.zeros_like(rank), torch.empty_like(output)
            keys, own, values, first, target = benchmark_resource(session, ("candidate_share", name), inputs)
            scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            return PreparedCall(run=lambda: state.run(keys, lengths, own, values, first, target, scratch),
                                output=target,
                                owners=prime_owners(workspace, scratch, keys, lengths, own, values, first, target))

        def prepare(state):
            scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            return PreparedCall(run=lambda: state.run(gathered, lengths, indices, scores, rank, output, scratch),
                                owners=prime_owners(workspace, scratch, gathered, lengths, indices, scores, rank,
                                                    output))

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=call),))
        register_finalizer(session, lambda: self._bind(workspace), workspace=workspace)

    def _bind(self, workspace):
        self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)

    def __call__(self):
        _execute(self.gathered, self.lengths, self.indices, self.scores, self.rank, self.output, self.scratch,
                 self.plan.handle)
        return self.output
