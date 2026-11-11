# SPDX-License-Identifier: Apache-2.0

"""CPU-interpreted exactness tests for Kimi K3.1 extreme prefill Triton kernels and planning."""

import subprocess
import sys

import pytest
import torch


def _run_interpreted(code: str) -> None:
    import os

    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise AssertionError(f"TRITON_INTERPRET subprocess failed (rc={res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")


def test_expert_reduce_norm_matches_canonical_rms_norm_bitwise():
    _run_interpreted("""
import torch
from ignition.recipes.kimi3.pointwise import (
    _distributed_rms_norm,
    _pack_expert_columns,
    _partial_sum_sq,
    _rms_norm,
)

gen = torch.Generator().manual_seed(20261111)
world, rows, width, local_width, group_rows = 9, 288, 3584, 416, 32
eps = 1e-6
partials = [(torch.randn((rows, width), generator=gen) * 2.5).to(torch.bfloat16) for _ in range(world)]
full_weight = (torch.randn((width,), generator=gen) * 0.5 + 1.0).to(torch.bfloat16)

# Reference ordered ring sum rounded once to BF16 per chunk
chunks = []
flat_partials = [p.reshape(world, -1) for p in partials]
for c in range(world):
    acc = flat_partials[c][c].float().clone()
    for step in range(1, world):
        acc = acc + flat_partials[(c + step) % world][c].float()
    chunks.append(acc.to(torch.bfloat16))
latent_sum = torch.stack(chunks, dim=0).reshape(rows, width)

# Pack each rank's partial into [world, rows, local_width] and reduce-scatter to owner r
packed_all = []
for r in range(world):
    pk = torch.zeros((world, rows, local_width), dtype=torch.bfloat16)
    _pack_expert_columns[(rows, world)](partials[r], pk, partials[r].stride(0), pk.stride(0), pk.stride(1),
                                        WIDTH=width, LOCAL_WIDTH=local_width, BLOCK=512)
    packed_all.append(pk)

reduced_shards = []
local_stats = []
for r in range(world):
    # Ring order for reduce_scatter_only: starts at (r + 1) % world .. r
    acc = packed_all[(r + 1) % world][r].float().clone()
    for step in range(2, world + 1):
        acc = acc + packed_all[(r + step) % world][r].float()
    shard_bf16 = acc.to(torch.bfloat16)
    reduced_shards.append(shard_bf16)
    st = torch.zeros((rows,), dtype=torch.float32)
    _partial_sum_sq[(rows,)](shard_bf16, st, shard_bf16.stride(0), LOCAL_WIDTH=local_width, BLOCK=512)
    local_stats.append(st.view(rows // group_rows, group_rows))

# Gathered layout [batches, world, group_rows]
gathered_stats = torch.stack(local_stats, dim=1).contiguous()

for r in range(world):
    begin_col = r * local_width
    valid_cols = max(0, min(local_width, width - begin_col))
    rank_weight = torch.zeros((local_width,), dtype=torch.bfloat16)
    if valid_cols > 0:
        rank_weight[:valid_cols] = full_weight[begin_col : begin_col + valid_cols]

    ref_out = torch.zeros((rows, local_width), dtype=torch.bfloat16)
    begin_tensor = torch.tensor([begin_col], dtype=torch.int32)
    _rms_norm[(rows,)](latent_sum, rank_weight, ref_out, begin_tensor,
                       latent_sum.stride(0), ref_out.stride(0), eps,
                       WIDTH=width, OUTPUT_WIDTH=local_width, OFFSET=True, BLOCK=4096)

    got_out = torch.zeros((rows, local_width), dtype=torch.bfloat16)
    _distributed_rms_norm[(rows,)](reduced_shards[r], gathered_stats.view(-1), rank_weight, got_out,
                                   reduced_shards[r].stride(0), got_out.stride(0), eps,
                                   FULL_WIDTH=width, LOCAL_WIDTH=local_width, WORLD=world,
                                   GROUP_ROWS=group_rows, BLOCK=512)
    assert torch.equal(got_out[:, :valid_cols].view(torch.int16), ref_out[:, :valid_cols].view(torch.int16)), f"mismatch on rank {r}"
    if valid_cols < local_width:
        assert torch.all(got_out[:, valid_cols:] == 0)
""")


def test_dsa_prefill_fused_sort_and_quantize_match_eager_reference_bitwise():
    _run_interpreted("""
import torch, triton
from ignition.recipes.kimi3.dsa_prefill import _filter_sort_selected, _quantize_prefill_queries

gen = torch.Generator().manual_seed(991)
world, local_rows, topk = 9, 4, 64
step = local_rows * world  # 36
total_rows = 60  # two slices: 36 and 24 (tail!)
num_slices = 2
first_rank = 3

# Build ring slots [world, num_slices * local_rows, topk]
raw_by_owner = torch.randint(-5, 80, (world, num_slices * local_rows, topk), generator=gen, dtype=torch.int32)
# Ensure distinct valid candidates per row so sort order is unique
for o in range(world):
    for lr in range(num_slices * local_rows):
        perm = torch.randperm(120, generator=gen)[:topk].to(torch.int32) - 10
        raw_by_owner[o, lr] = perm

slots = torch.empty_like(raw_by_owner)
for s in range(world):
    owner = (first_rank + s) % world
    slots[s] = raw_by_owner[owner]

positions = torch.arange(total_rows, dtype=torch.int64) + 10
heads, rec_bytes = 3, 64
total_cols = heads * rec_bytes
queries = (torch.randn((total_rows, total_cols), generator=gen) * 600.0).to(torch.bfloat16)

for ring_mode in (True, False):
    for slice_idx, begin in enumerate(range(0, total_rows, step)):
        count = min(step, total_rows - begin)
        # Build flat gathered [local_rows, world, topk] -> view [step, topk]
        flat = torch.stack([raw_by_owner[o, slice_idx * local_rows : (slice_idx + 1) * local_rows] for o in range(world)], dim=1).reshape(step, topk).contiguous()

        # Eager reference from original dsa_prefill.py
        global_ids = torch.arange(step, dtype=torch.int64)
        pos_slice = positions[(begin + global_ids).clamp_max(total_rows - 1)]
        live = (flat >= 0) & (flat <= pos_slice[:, None]) & (global_ids[:, None] < count)
        ordered = torch.sort(torch.where(live, flat, torch.iinfo(torch.int32).max), dim=1).values
        ref_phys = torch.where(ordered != torch.iinfo(torch.int32).max, ordered, -1).to(torch.int32)
        ref_cnt = live.sum(1).to(torch.int32)

        got_phys = torch.full((step, topk), -99, dtype=torch.int32)
        got_cnt = torch.full((step,), -99, dtype=torch.int32)
        src = slots if ring_mode else flat
        _filter_sort_selected[(step,)](
            src, got_phys, got_cnt, positions, slots.stride(0), begin, count, total_rows,
            FIRST_RANK=first_rank if ring_mode else 0, LOCAL_ROWS=local_rows,
            RING=ring_mode, WORLD=world, TOPK=topk,
        )
        assert torch.equal(got_cnt, ref_cnt), (ring_mode, begin)
        assert torch.equal(got_phys, ref_phys), (ring_mode, begin)

        # Quantize clamp + tail zeroing check on BF16 (Triton CPU interpreter lacks exact hardware e4m3 RN,
        # whereas SM120 PTX emits hardware cvt.rn.satfinite.e4m3x2.f32)
        ref_q = torch.zeros((step, total_cols), dtype=torch.bfloat16)
        ref_q[:count].copy_(queries[begin : begin + count].float().clamp(-448.0, 448.0).to(torch.bfloat16))
        got_q = torch.full((step, total_cols), 7.0, dtype=torch.bfloat16)
        _quantize_prefill_queries[(step, triton.cdiv(total_cols, 64))](
            queries, got_q, begin, count, queries.stride(0), got_q.stride(0),
            TOTAL_COLS=total_cols, BLOCK=64,
        )
        assert torch.equal(got_q.view(torch.int16), ref_q.view(torch.int16)), begin
""")


def test_bounded_pull_context_and_index_context_match_reference():
    _run_interpreted("""
import torch
from ignition.recipes.kimi3.dcp import _pull_context, _pull_index_context

gen = torch.Generator().manual_seed(777)
world, page_tokens, words = 9, 128, 144
num_pages = 6
caches = [torch.randint(-100000, 100000, (num_pages * page_tokens, words), generator=gen, dtype=torch.int32) for _ in range(world)]
addrs = torch.tensor([c.data_ptr() for c in caches], dtype=torch.int64)
table = torch.tensor([2, 0, 4, 1], dtype=torch.int32)

# 1. Empty cold window [0, 0] does nothing
slab = torch.full((3000, words), -7, dtype=torch.int32)
win_zero = torch.tensor([0, 0], dtype=torch.int64)
_pull_context[(18,)](addrs, table, win_zero, slab, table.numel(), WORLD=world, PAGE=page_tokens, TOKENS=16, WORDS=words, WORD_BLOCK=256)
assert torch.all(slab == -7)

# 2. Arbitrary unaligned window [17, 1503]
begin, end = 17, 1503
win = torch.tensor([begin, end], dtype=torch.int64)
slab = torch.full((3000, words), -7, dtype=torch.int32)
_pull_context[(18,)](addrs, table, win, slab, table.numel(), WORLD=world, PAGE=page_tokens, TOKENS=16, WORDS=words, WORD_BLOCK=256)
for t in range(begin, end):
    owner = t % world
    local = t // world
    pg = int(table[min(local // page_tokens, table.numel() - 1)].item())
    expected = caches[owner][pg * page_tokens + local % page_tokens]
    assert torch.equal(slab[t - begin], expected), f"token {t} mismatch"
assert torch.all(slab[end - begin :] == -7)

# 3. _pull_index_context for End=0 and End=700
idx_caches = [torch.randint(-100000, 100000, (num_pages * 2 * 64 * 33,), generator=gen, dtype=torch.int32) for _ in range(world)]
idx_addrs = torch.tensor([c.data_ptr() for c in idx_caches], dtype=torch.int64)
dest = torch.full((32 * 64 * 33,), -5, dtype=torch.int32)
_pull_index_context[(16,)](idx_addrs, table, torch.tensor([0], dtype=torch.int64), dest, table.numel(), 32 * 64, WORLD=world, PAGE=page_tokens, ROWS=8)
assert torch.all(dest == -5)

end_idx = 700
_pull_index_context[(16,)](idx_addrs, table, torch.tensor([end_idx], dtype=torch.int64), dest, table.numel(), 32 * 64, WORLD=world, PAGE=page_tokens, ROWS=8)
for t in range(end_idx):
    owner = t % world
    local = t // world
    pg = int(table[local // page_tokens].item())
    ipg = pg * 2 + (local % page_tokens) // 64
    src_k = ipg * (64 * 33) + (local % 64) * 32
    dst_k = (t // 64) * (64 * 33) + (t % 64) * 32
    assert torch.equal(dest[dst_k : dst_k + 32], idx_caches[owner][src_k : src_k + 32]), f"idx key {t}"
    assert dest[(t // 64) * (64 * 33) + 64 * 32 + t % 64] == idx_caches[owner][ipg * (64 * 33) + 64 * 32 + local % 64], f"idx scale {t}"
""")
