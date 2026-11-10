"""FLARE_CANDIDATE_MERGE_FUSED folds the threshold launch into the cutoff counts with identical output."""
import os

import pytest
import torch

from ignition.components.candidate_merge import PreparedCandidateMerge
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(
    os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires one assigned CUDA GPU")


@pytest.mark.parametrize("rows,width", [(1, 16384), (5, 16384), (8, 8192)])
def test_fused_threshold_matches_three_launch_tail(rows, width):
    device, topk = torch.device("cuda:0"), 2048
    packed = torch.zeros((rows, width, 2), dtype=torch.float32, device=device)
    packed.view(torch.int32)[:, :, 1].fill_(-1)
    with RecordedPreparationSession(device=device, checkpoint="candidate-merge-fused-fixture",
            workload_schema="finite-score-candidates", numerical_policy="fp32", compile_workers=2) as session:
        workspace = ScratchArena(device)
        session.register_scratch(workspace)
        operations = {}
        for flag in ("0", "1"):
            output = torch.empty((rows, topk), dtype=torch.int32, device=device)
            operations[flag] = (PreparedCandidateMerge(packed, output, session=session, workspace=workspace,
                                                       name="merge" + flag, fused=flag == "1"), output)
        session.tune(show_progress=False)
        assert operations["0"][0].plan.component_id == "flare.candidate_merge"
        assert operations["1"][0].plan.component_id == "flare.candidate_merge_fused"
        graphs = {}
        for flag, (operation, _) in operations.items():
            forward = compile_forward(operation)
            forward()
            graphs[flag] = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graphs[flag]):
                forward()
        session.freeze()
        for valid in (0, 1, 2047, 2048, 2049, 5000, width):
            for ties in (False, True):
                generator = torch.Generator(device=device).manual_seed(97 * valid + ties)
                packed[:, :, 0].fill_(-torch.inf)
                packed.view(torch.int32)[:, :, 1].fill_(-1)
                for row in range(rows):
                    count = max(0, valid - row * 5)
                    slots = torch.randperm(width, device=device, generator=generator)[:count]
                    ids = torch.randperm(width, device=device, generator=generator)[:count]
                    scores = (torch.randint(-3, 4, (count,), device=device, generator=generator).float() if ties
                              else torch.randn(count, device=device, generator=generator))
                    packed[row, slots, 0] = scores
                    packed.view(torch.int32)[row, slots, 1] = ids.int()
                results = []
                for flag, (operation, output) in operations.items():
                    operation.scratch.fill_(255)
                    output.fill_(-777)
                    graphs[flag].replay()
                    torch.cuda.synchronize()
                    results.append(output.clone())
                assert torch.equal(results[0], results[1])
        for graph in graphs.values():
            graph.reset()
