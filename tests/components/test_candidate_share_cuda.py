"""One GPU: every rank's candidate share equals the candidate merge's selection restricted to the rank."""
import os

import pytest
import torch

from ignition.components.candidate_merge import PreparedCandidateMerge
from ignition.components.candidate_share import PreparedCandidateShare
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(
    os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires one assigned CUDA GPU")

WORLD, TOPK = 9, 2048
KINDS = ("random", "ties", "short", "exact-k", "one-owner", "all-equal", "minus-inf", "mostly-minus-inf")


@pytest.mark.parametrize("rows", [1, 4, 24])
def test_share_matches_merge_for_every_rank(rows):
    device = torch.device("cuda:0")
    packed = torch.zeros((rows, WORLD * TOPK, 2), dtype=torch.float32, device=device)
    gathered = torch.zeros((rows, WORLD * TOPK), dtype=torch.float32, device=device)
    indices = torch.zeros((rows, TOPK), dtype=torch.int32, device=device)
    scores = torch.zeros((rows, TOPK), dtype=torch.float32, device=device)
    rank = torch.zeros((1,), dtype=torch.int64, device=device)
    merged = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
    shared = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="candidate-share-fixture",
            workload_schema="score-candidates", numerical_policy="fp32", compile_workers=2) as session:
        workspace = ScratchArena(device)
        session.register_scratch(workspace)
        merge = PreparedCandidateMerge(packed, merged, session=session, workspace=workspace, name="merge", fused=True)
        share = PreparedCandidateShare(gathered, indices, scores, rank, shared, world=WORLD, session=session,
                                       workspace=workspace, name="share")
        session.tune(show_progress=False)
        assert share.plan.component_id == "flare.candidate_share"
        graphs = {}
        for name, operation in (("merge", merge), ("share", share)):
            forward = compile_forward(operation)
            forward()
            graphs[name] = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graphs[name]):
                forward()
        session.freeze()
        for case, kind in enumerate(KINDS):
            generator = torch.Generator(device=device).manual_seed(4000 + 31 * case + rows)
            all_indices = torch.full((WORLD, rows, TOPK), -1, dtype=torch.int32, device=device)
            all_scores = torch.randn((WORLD, rows, TOPK), device=device, generator=generator) * 50
            for owner in range(WORLD):
                for row in range(rows):
                    if kind == "short":
                        count = int(torch.randint(0, TOPK // WORLD, (1,), device=device, generator=generator))
                    elif kind == "exact-k":
                        count = TOPK // WORLD + (TOPK % WORLD if owner == 0 else 0)
                    elif kind == "one-owner":
                        count = TOPK if owner == 4 else int(torch.randint(0, TOPK // 4, (1,), device=device, generator=generator))
                    else:
                        count = int(torch.randint(TOPK // 2, TOPK + 1, (1,), device=device, generator=generator))
                    slots = torch.randperm(TOPK, device=device, generator=generator)[:count]
                    all_indices[owner, row, slots] = torch.randperm(4 * TOPK, device=device, generator=generator)[:count].int()
                    if kind == "ties":
                        values = torch.randint(-2, 3, (count,), device=device, generator=generator).float()
                    elif kind == "all-equal":
                        values = torch.full((count,), 1.5, device=device)
                    else:
                        values = torch.randn(count, device=device, generator=generator) + (100. if kind == "one-owner" and owner == 4 else 0.)
                    if kind.endswith("minus-inf") and count:
                        lost = count // 2 if kind == "minus-inf" else count - count // 20
                        values[torch.randperm(count, device=device, generator=generator)[:lost]] = -torch.inf
                    all_scores[owner, row, slots] = values
            live = all_indices >= 0
            finite = live & (all_scores > -torch.inf)
            pairs = packed.view(rows, WORLD, TOPK, 2)
            pairs[..., 0].copy_(torch.where(live, all_scores, -torch.inf).permute(1, 0, 2))
            ranks = torch.arange(WORLD, device=device, dtype=torch.int32)[:, None, None]
            positions = all_indices * WORLD + ranks
            pairs.view(torch.int32)[..., 1].copy_(torch.where(live, positions, -1).permute(1, 0, 2))
            gathered.view(rows, WORLD, TOPK).copy_(torch.where(finite, all_scores, -torch.inf).permute(1, 0, 2))
            merge.scratch.fill_(255)
            merged.fill_(-777)
            graphs["merge"].replay()
            torch.cuda.synchronize()
            reference = merged.clone()
            for owner in range(WORLD):
                indices.copy_(all_indices[owner])
                scores.copy_(all_scores[owner])
                rank.fill_(owner)
                share.scratch.fill_(255)
                shared.fill_(-777)
                graphs["share"].replay()
                torch.cuda.synchronize()
                assert int((shared < -1).sum()) == 0
                for row in range(rows):
                    # A candidate with a score of -inf is no candidate of the score exchange.
                    kept = set(positions[owner, row][finite[owner, row]].tolist())
                    expected = sorted(int(value) for value in reference[row] if int(value) in kept)
                    assert sorted(int(value) for value in shared[row] if value >= 0) == expected, (kind, owner, row)
        for graph in graphs.values():
            graph.reset()
