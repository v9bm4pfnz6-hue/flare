# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 single-kernel SiTU, RMSNorm and gating steps between projections.

Each kernel replaces a chain of small eager and generated kernels with one launch while keeping the
canonical rounding points: SiTU and gating compute in FP32 and round once; RMSNorm rounds the
normalized value to BF16 before applying its weight.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _situ(Source, Output, source_stride, output_stride, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0).to(tl.int64), tl.program_id(1)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    valid = columns < WIDTH
    gate = tl.load(Source + row * source_stride + columns, valid, other=0.).to(tl.float32)
    up = tl.load(Source + row * source_stride + WIDTH + columns, valid, other=0.).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-gate))
    value = 4. * libdevice.tanh(gate / 4.) * sigmoid * (25. * libdevice.tanh(up / 25.))
    tl.store(Output + row * output_stride + columns, value.to(Output.dtype.element_ty), valid)


def situ(gate_up, out):
    """``out = situ(gate_up)`` for gate/up halves of each row, matching ``model.situ``."""
    rows, width = out.shape
    if gate_up.shape != (rows, 2 * width) or gate_up.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("SiTU takes [rows, 2 * width] gate/up rows and writes [rows, width]")
    block = min(1024, triton.next_power_of_2(width))
    torch.library.wrap_triton(_situ)[(rows, triton.cdiv(width, block))](
        gate_up, out, gate_up.stride(0), out.stride(0), width, block, num_warps=4)
    return out


@triton.jit
def _rms_norm(Source, Weight, Output, Begin, source_stride, output_stride, eps, WIDTH: tl.constexpr,
              OUTPUT_WIDTH: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    valid = columns < WIDTH
    value = tl.load(Source + row * source_stride + columns, valid, other=0.).to(tl.float32)
    normalized = (value * tl.rsqrt(tl.sum(value * value) / WIDTH + eps)).to(Output.dtype.element_ty)
    begin = tl.load(Begin) if OFFSET else 0
    keep = valid & (columns >= begin) & (columns < begin + OUTPUT_WIDTH)
    weight = tl.load(Weight + columns - begin, keep, other=0.).to(tl.float32)
    tl.store(Output + row * output_stride + columns - begin, (weight * normalized.to(tl.float32)).to(Output.dtype.element_ty), keep)


def rms_norm_columns(source, weight, out, eps, begin=None):
    """Canonical RMSNorm over each full row of ``source``, writing ``out.shape[1]`` columns from ``begin``.

    ``weight`` holds the weights of the written columns only. ``begin`` is a one-element integer tensor, so
    tensor-parallel ranks share one compiled forward; ``None`` writes the leading columns.
    """
    rows, width = source.shape
    if out.shape[0] != rows or not 0 < out.shape[1] <= width or weight.shape != (out.shape[1],):
        raise ValueError("RMSNorm writes a column range of its rows")
    if source.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("RMSNorm rows must be contiguous")
    if begin is not None and (begin.shape != (1,) or begin.dtype not in (torch.int32, torch.int64)):
        raise ValueError("RMSNorm column offsets are one-element integer tensors")
    block = triton.next_power_of_2(width)
    torch.library.wrap_triton(_rms_norm)[(rows,)](
        source, weight, out, source if begin is None else begin, source.stride(0), out.stride(0), float(eps),
        width, out.shape[1], begin is not None, block, num_warps=4 if block <= 2048 else 8)
    return out


@triton.jit
def _sigmoid_gate(Value, Gate, value_stride, gate_stride, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0).to(tl.int64), tl.program_id(1)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    valid = columns < WIDTH
    value = tl.load(Value + row * value_stride + columns, valid, other=0.).to(tl.float32)
    gate = tl.load(Gate + row * gate_stride + columns, valid, other=0.).to(tl.float32)
    gate = (1. / (1. + tl.exp(-gate))).to(Value.dtype.element_ty).to(tl.float32)
    tl.store(Value + row * value_stride + columns, (value * gate).to(Value.dtype.element_ty), valid)


def sigmoid_gate(value, gate):
    """``value.mul_(gate.sigmoid())`` with the sigmoid rounded to BF16 first, as the eager pair does."""
    rows, width = value.shape
    if gate.shape != value.shape or value.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("sigmoid gating takes matching contiguous rows")
    block = min(1024, triton.next_power_of_2(width))
    torch.library.wrap_triton(_sigmoid_gate)[(rows, triton.cdiv(width, block))](
        value, gate, value.stride(0), gate.stride(0), width, block, num_warps=4)
    return value


@triton.jit
def _pack_expert_columns(Source, Packed, source_stride, packed_rank_stride, packed_row_stride,
                         WIDTH: tl.constexpr, LOCAL_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    rank = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    src_col = rank * LOCAL_WIDTH + cols
    valid_in = (cols < LOCAL_WIDTH) & (src_col < WIDTH)
    val = tl.load(Source + row * source_stride + src_col, mask=valid_in, other=0.0)
    tl.store(Packed + rank * packed_rank_stride + row * packed_row_stride + cols, val, mask=cols < LOCAL_WIDTH)


@triton.jit
def _partial_sum_sq(Shard, Stats, shard_stride, LOCAL_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    val = tl.load(Shard + row * shard_stride + cols, mask=cols < LOCAL_WIDTH, other=0.0).to(tl.float32)
    sq = tl.sum(val * val, axis=0)
    tl.store(Stats + row, sq)


@triton.jit
def _distributed_rms_norm(Shard, GatheredStats, Weight, Output, shard_stride, output_stride, eps,
                          FULL_WIDTH: tl.constexpr, LOCAL_WIDTH: tl.constexpr, WORLD: tl.constexpr,
                          GROUP_ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    batch = row // GROUP_ROWS
    lane = row % GROUP_ROWS
    # GatheredStats has layout [rows // GROUP_ROWS, WORLD, GROUP_ROWS] in float32.
    total_sq = tl.zeros((), dtype=tl.float32)
    for r in tl.static_range(0, WORLD):
        total_sq += tl.load(GatheredStats + (batch * WORLD + r) * GROUP_ROWS + lane)
    inv_rms = tl.rsqrt(total_sq / FULL_WIDTH + eps)
    cols = tl.arange(0, BLOCK)
    keep = cols < LOCAL_WIDTH
    val = tl.load(Shard + row * shard_stride + cols, mask=keep, other=0.0).to(tl.float32)
    normalized = (val * inv_rms).to(Output.dtype.element_ty)
    weight = tl.load(Weight + cols, mask=keep, other=0.0).to(tl.float32)
    tl.store(Output + row * output_stride + cols,
             (weight * normalized.to(tl.float32)).to(Output.dtype.element_ty), mask=keep)


def pack_expert_columns(source, packed):
    rows, width = source.shape
    world, p_rows, local_width = packed.shape
    if p_rows != rows or source.stride(-1) != 1 or packed.stride(-1) != 1:
        raise ValueError("pack_expert_columns requires contiguous [rows, width] and [world, rows, local_width]")
    block = triton.next_power_of_2(local_width)
    torch.library.wrap_triton(_pack_expert_columns)[(rows, world)](
        source, packed, source.stride(0), packed.stride(0), packed.stride(1),
        width, local_width, block, num_warps=4)
    return packed


def partial_sum_sq(shard, stats):
    rows, local_width = shard.shape
    block = triton.next_power_of_2(local_width)
    torch.library.wrap_triton(_partial_sum_sq)[(rows,)](
        shard, stats, shard.stride(0), local_width, block, num_warps=4)
    return stats


def distributed_rms_norm(shard, gathered_stats, weight, out, *, full_width, world, group_rows, eps):
    rows, local_width = shard.shape
    block = triton.next_power_of_2(local_width)
    torch.library.wrap_triton(_distributed_rms_norm)[(rows,)](
        shard, gathered_stats, weight, out, shard.stride(0), out.stride(0), float(eps),
        full_width, local_width, world, group_rows, block, num_warps=4)
    return out
