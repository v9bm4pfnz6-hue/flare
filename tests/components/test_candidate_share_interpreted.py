# SPDX-License-Identifier: Apache-2.0

"""Score exchange against pair exchange for a distributed top-k, in Triton's CPU interpreter.

The Triton programs of both forms run on CPU tensors; the b12x row top-k, a CuTe kernel, is replaced
by torch.topk, which returns the same K greatest values. For every row, every rank's share must be
the pair form's selection restricted to that rank. The interpreter is chosen when a Triton program
is declared, so the cases run in a child process that sets TRITON_INTERPRET before importing them.
"""
import os
import subprocess
import sys

import pytest
import torch

INNER = os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("FLARE_CANDIDATE_SHARE_CASES") == "1"
WORLD, TOPK, ROWS = 9, 512, 2
KINDS = ("random", "ties", "short", "exact-k", "one-owner", "all-equal", "minus-inf")


def test_every_rank_share_equals_the_pair_selection_restricted_to_the_rank():
    if INNER:
        pytest.skip("this process runs the cases")
    environment = dict(os.environ, TRITON_INTERPRET="1", FLARE_CANDIDATE_SHARE_CASES="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__, "-k", "case"],
                            env=environment, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0 and f"{len(KINDS) * 2} passed" in result.stdout, (
        result.stdout[-3000:] + result.stderr[-2000:])


def _pair_form(indices, scores):
    import triton

    from ignition.components import candidate_merge as merge, distributed_topk

    width = WORLD * TOPK
    packed = torch.zeros((ROWS, width, 2), dtype=torch.float32)
    for rank in range(WORLD):
        local = torch.zeros((ROWS, TOPK, 2), dtype=torch.float32)
        distributed_topk._pack_candidates[(ROWS, triton.cdiv(TOPK, 1024))](
            indices[rank], scores[rank], local.view(torch.int32), torch.tensor([rank]), TOPK,
            TOPK=TOPK, WORLD=WORLD, BLOCK=1024)
        packed[:, rank * TOPK:(rank + 1) * TOPK] = local
    tiles = triton.cdiv(width, 512)
    values, ids = torch.zeros((ROWS, width)), torch.full((ROWS, width), -7, dtype=torch.int32)
    lengths, counts = torch.zeros(ROWS, dtype=torch.int32), torch.zeros((ROWS, tiles), dtype=torch.int32)
    top, cutoffs = torch.zeros((ROWS, TOPK)), torch.zeros(ROWS)
    ties, output = torch.zeros((ROWS, tiles), dtype=torch.int32), torch.full((ROWS, TOPK), -777, dtype=torch.int32)
    grid = (ROWS, tiles, 1)
    merge._count[grid](packed, counts, width, tiles, 512)
    merge._compact[grid](packed, counts, values, ids, lengths, width, tiles, 512, triton.next_power_of_2(tiles))
    for row in range(ROWS):
        length = int(lengths[row])
        if length > TOPK:
            top[row] = torch.topk(values[row, :length], TOPK).values
        else:
            # With at most K candidates the row top-k returns every candidate and -1 padding.
            output[row] = -1
            output[row, :length] = ids[row, :length]
    merge._threshold_counts[grid](top, values, lengths, cutoffs, counts, ties, output, width, TOPK, tiles, 512,
                                  triton.next_power_of_2(TOPK))
    merge._select[grid](values, ids, lengths, cutoffs, counts, ties, output, width, TOPK, tiles, 512,
                        triton.next_power_of_2(tiles))
    return output


def _score_form(indices, scores):
    import triton

    from ignition.components import candidate_share as share, distributed_topk

    gathered = torch.zeros((ROWS, WORLD * TOPK), dtype=torch.float32)
    for rank in range(WORLD):
        keys = torch.full((ROWS, TOPK), 3.0)
        distributed_topk._pack_keys[(ROWS, triton.cdiv(TOPK, 1024))](
            indices[rank], scores[rank], keys, TOPK, TOPK=TOPK, BLOCK=1024)
        gathered[:, rank * TOPK:(rank + 1) * TOPK] = keys
    assert not torch.isnan(gathered).any()
    top = torch.topk(gathered, TOPK, dim=1).values.clone()
    for row in range(ROWS):
        # A row with at most K candidates must not depend on what the row top-k returns for it.
        if int((gathered[row] > -torch.inf).sum()) <= TOPK:
            top[row] = torch.nan
    shares = torch.full((WORLD, ROWS, TOPK), -777, dtype=torch.int32)
    for rank in range(WORLD):
        share._select_share[(ROWS, 1, 1)](top, gathered, indices[rank], scores[rank], torch.tensor([rank]),
                                          shares[rank], TOPK, TOPK, WORLD, triton.next_power_of_2(TOPK))
    return shares


def _candidates(kind, seed, generator):
    indices = torch.full((WORLD, ROWS, TOPK), -1, dtype=torch.int32)
    scores = torch.zeros((WORLD, ROWS, TOPK))
    for rank in range(WORLD):
        for row in range(ROWS):
            if kind == "short":
                count = int(torch.randint(0, TOPK // WORLD, (1,), generator=generator))
            elif kind == "exact-k":
                count = TOPK // WORLD + (TOPK % WORLD if rank == 0 else 0)
            elif kind == "one-owner":
                count = TOPK if rank == 4 else int(torch.randint(0, TOPK // 4, (1,), generator=generator))
            else:
                count = int(torch.randint(TOPK // 2, TOPK + 1, (1,), generator=generator))
            slots = torch.randperm(TOPK, generator=generator)[:count]
            indices[rank, row, slots] = torch.randperm(4 * TOPK, generator=generator)[:count].int()
            if kind == "ties":
                values = torch.randint(-2, 3, (count,), generator=generator).float()
            elif kind == "all-equal":
                values = torch.full((count,), 1.5)
            else:
                values = torch.randn(count, generator=generator) + (100.0 if kind == "one-owner" and rank == 4 else 0.0)
            if kind == "minus-inf" and count:
                # Half of the candidates, or all but a few, carry a score of -inf.
                lost = count // 2 if seed == 0 else count - count // 20
                values[torch.randperm(count, generator=generator)[:lost]] = -torch.inf
            scores[rank, row, slots] = values
            # A slot without a candidate carries an arbitrary score: its index alone marks it.
            empty = indices[rank, row] < 0
            scores[rank, row, empty] = torch.randn(int(empty.sum()), generator=generator) * 50
    return indices, scores


@pytest.mark.skipif(not INNER, reason="runs in the interpreter child process")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", (0, 1))
def test_case(kind, seed):
    generator = torch.Generator().manual_seed(1000 + 17 * seed + KINDS.index(kind))
    indices, scores = _candidates(kind, seed, generator)
    selected, shares = _pair_form(indices, scores), _score_form(indices, scores)
    for row in range(ROWS):
        reference = sorted(int(value) for value in selected[row] if value >= 0)
        total = int((indices[:, row] >= 0).sum())
        assert len(reference) == min(total, TOPK) and len(set(reference)) == len(reference)
        if kind == "exact-k":
            assert total == TOPK
        # A candidate with a score of -inf is no candidate of the score exchange; the pair form
        # keeps such candidates only while fewer than K finite ones exist.
        finite = {int(indices[rank, row, slot]) * WORLD + rank
                  for rank in range(WORLD) for slot in range(TOPK)
                  if indices[rank, row, slot] >= 0 and scores[rank, row, slot] > -torch.inf}
        assert kind == "minus-inf" or len(finite) == total
        for rank in range(WORLD):
            own = sorted(int(value) for value in shares[rank, row] if value >= 0)
            assert own == [value for value in reference if value % WORLD == rank and value in finite], (row, rank)
            assert int((shares[rank, row] < -1).sum()) == 0, (row, rank)
