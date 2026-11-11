# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 attention residuals fused with the residual update and the following RMSNorm.

One program per row updates the running block sum, optionally commits it as a residual block,
mixes the committed blocks and the running sum by their softmax scores, and normalizes the
mixture for the next projection. The arithmetic follows the canonical composition: BF16 residual
additions, FP32 scores and mixing rounded to BF16, then RMSNorm with its BF16 rounding.
"""

import torch
import triton
import triton.language as tl

KEEP, ADD, REPLACE = 0, 1, 2
# Source tile and warps by committed block count, measured for one decode row.
TILES = ((2, 4), (2, 4), (4, 8), (4, 8), (8, 8), (8, 8), (8, 8), (8, 8), (4, 8))
# Decode geometry: a stream of at most SPLIT_ROWS rows mixes with a grid of rows x (hidden / SPLIT_CHUNK)
# programs of SPLIT_WARPS warps instead of one program per row, which keeps one SM per row busy over the
# whole width (5.2 to 5.8 us per call over the 187 calls of a round against 3.4 to 3.7 us split, measured
# by the change's author at 1 to 16 rows). Full-row FP32 sums are then added per chunk in chunk order: the
# mixture stays within one BF16 unit of the row programs' and the normalized output within three.
SPLIT_ROWS, SPLIT_CHUNK, SPLIT_WARPS = 16, 1024, 4
# Per-row partial slots: [squares | dots] x SPLIT_CHUNKS chunks x SPLIT_SOURCES padded sources, then one
# mixture square sum per chunk.
SPLIT_SOURCES, SPLIT_CHUNKS = 16, 16
SPLIT_STRIDE = 2 * SPLIT_CHUNKS * SPLIT_SOURCES + SPLIT_CHUNKS


@triton.jit
def _mix(Prefix, Delta, Blocks, Score, Norm, Output, Mixture, MixtureCount, prefix_stride, delta_stride, block_row_stride,
         block_stride, output_stride, mixture_stride, eps, output_eps, HIDDEN: tl.constexpr, SOURCES: tl.constexpr,
         WRITE: tl.constexpr, DELTA: tl.constexpr, MIXTURE: tl.constexpr, COMPACT_MIXTURE: tl.constexpr,
         ROWS: tl.constexpr, MIXTURE_ROWS: tl.constexpr, MIXTURE_WINDOW: tl.constexpr, MIXTURE_ALIGNMENT: tl.constexpr,
         BLOCK: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    valid = columns < HIDDEN
    current = tl.load(Prefix + row * prefix_stride + columns, valid, other=0.)
    if DELTA == 1:
        delta = tl.load(Delta + row * delta_stride + columns, valid, other=0.)
        current = (current.to(tl.float32) + delta.to(tl.float32)).to(Prefix.dtype.element_ty)
    elif DELTA == 2:
        current = tl.load(Delta + row * delta_stride + columns, valid, other=0.)
    if DELTA != 0:
        tl.store(Prefix + row * prefix_stride + columns, current, valid)
    if WRITE >= 0:
        tl.store(Blocks + row * block_row_stride + WRITE * block_stride + columns, current, valid)
    if SOURCES == 0:
        mixed = current.to(tl.float32)
    else:
        # The running sum is the last source; it is reread after its store is visible to the whole program.
        tl.debug_barrier()
        weight = tl.load(Score + columns, valid, other=0.)
        maximum = tl.full((), float("-inf"), tl.float32)
        total = tl.zeros((), tl.float32)
        accumulated = tl.zeros((BLOCK,), tl.float32)
        for tile in tl.static_range((SOURCES + TILE) // TILE):
            sources = tile * TILE + tl.arange(0, TILE)
            present = sources <= SOURCES
            pointers = tl.where((sources == SOURCES)[:, None], Prefix + row * prefix_stride + columns[None, :],
                                Blocks + row * block_row_stride + sources[:, None] * block_stride + columns[None, :])
            values = tl.load(pointers, present[:, None] & valid[None, :], other=0.).to(tl.float32)
            scale = tl.rsqrt(tl.sum(values * values, axis=1) / HIDDEN + eps)
            scores = tl.where(present, tl.sum(values * weight[None, :], axis=1) * scale, float("-inf"))
            peak = tl.maximum(maximum, tl.max(scores, axis=0))
            rescale = tl.exp(maximum - peak)
            exponentials = tl.exp(scores - peak)
            total = total * rescale + tl.sum(exponentials, axis=0)
            accumulated = accumulated * rescale + tl.sum(exponentials[:, None] * values, axis=0)
            maximum = peak
        mixed = (accumulated / total).to(Prefix.dtype.element_ty).to(tl.float32)
    if MIXTURE:
        if COMPACT_MIXTURE:
            count = tl.load(MixtureCount).to(tl.int64)
            origin = tl.maximum(count - MIXTURE_WINDOW, 0) // MIXTURE_ALIGNMENT * MIXTURE_ALIGNMENT
            destination = row - origin
            retained = (destination >= 0) & (destination < MIXTURE_ROWS)
            values = tl.where(row < count, mixed, 0).to(Mixture.dtype.element_ty)
            tl.store(Mixture + destination * mixture_stride + columns, values, valid & retained)
            padding = (row < MIXTURE_ROWS) & (origin + row >= ROWS)
            tl.store(Mixture + row * mixture_stride + columns, 0., valid & padding)
        else:
            tl.store(Mixture + row * mixture_stride + columns, mixed.to(Mixture.dtype.element_ty), valid)
    normalized = (mixed * tl.rsqrt(tl.sum(mixed * mixed) / HIDDEN + output_eps)).to(Prefix.dtype.element_ty)
    output = tl.load(Norm + columns, valid, other=0.).to(tl.float32) * normalized.to(tl.float32)
    tl.store(Output + row * output_stride + columns, output.to(Output.dtype.element_ty), valid)


@triton.jit
def _publish(Slots, values, tag, mask):
    # A partial and its call tag share one 64-bit word, so a reader that sees the tag sees the value.
    bits = values.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    tl.store(Slots, bits | (tag.to(tl.int64) << 32), mask)


@triton.jit
def _gather(Slots, tag, mask):
    # Waits until every slot carries this call's tag; slots of earlier calls carry earlier tags.
    words = tl.load(Slots, mask, other=0, volatile=True)
    while tl.min(tl.where(mask, ((words >> 32).to(tl.int32) == tag).to(tl.int32), 1)) == 0:
        words = tl.load(Slots, mask, other=0, volatile=True)
    return tl.where(mask, (words & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True), 0.)


@triton.jit
def _mix_split(Prefix, Delta, Blocks, Score, Norm, Output, Mixture, MixtureCount, Partial, Epoch, prefix_stride,
               delta_stride, block_row_stride, block_stride, output_stride, mixture_stride, eps, output_eps,
               HIDDEN: tl.constexpr, SOURCES: tl.constexpr, WRITE: tl.constexpr, DELTA: tl.constexpr,
               MIXTURE: tl.constexpr, COMPACT_MIXTURE: tl.constexpr, ROWS: tl.constexpr, MIXTURE_ROWS: tl.constexpr,
               MIXTURE_WINDOW: tl.constexpr, MIXTURE_ALIGNMENT: tl.constexpr, CHUNK: tl.constexpr,
               CHUNKS: tl.constexpr, TILE: tl.constexpr, PSOURCES: tl.constexpr, PCHUNKS: tl.constexpr,
               PSTRIDE: tl.constexpr):
    """``_mix`` for few rows: each program owns CHUNK columns of one row and loads all of its sources at once.
    Full-row FP32 sums are per-chunk partials, exchanged through tagged slots and added in chunk order, so all
    programs of a row see identical scores and norms; elementwise steps and BF16 rounding points are ``_mix``'s.
    A row's epoch advances once per call (after every program of the row has read it)."""
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    columns = chunk * CHUNK + tl.arange(0, CHUNK)
    valid = columns < HIDDEN
    partial = Partial + row * PSTRIDE
    chunks = tl.arange(0, PCHUNKS)
    owned = chunks < CHUNKS
    epoch = tl.load(Epoch + row, volatile=True)
    current = tl.load(Prefix + row * prefix_stride + columns, valid, other=0.)
    if DELTA != 0:
        delta = tl.load(Delta + row * delta_stride + columns, valid, other=0.)
    if SOURCES > 0:
        sources = tl.arange(0, PSOURCES)
        present = sources <= SOURCES
        weight = tl.load(Score + columns, valid, other=0.)
        history = tl.load(Blocks + row * block_row_stride + sources[:, None] * block_stride + columns[None, :],
                          (sources < SOURCES)[:, None] & valid[None, :], other=0.)
    norm = tl.load(Norm + columns, valid, other=0.)
    if DELTA == 1:
        current = (current.to(tl.float32) + delta.to(tl.float32)).to(Prefix.dtype.element_ty)
    elif DELTA == 2:
        current = delta
    if DELTA != 0:
        tl.store(Prefix + row * prefix_stride + columns, current, valid)
    if WRITE >= 0:
        tl.store(Blocks + row * block_row_stride + WRITE * block_stride + columns, current, valid)
    if SOURCES == 0:
        mixed = current.to(tl.float32)
    else:
        # The running sum is the last source.
        values = tl.where((sources == SOURCES)[:, None], current.to(tl.float32)[None, :], history.to(tl.float32))
        _publish(partial + chunk * PSOURCES + sources, tl.sum(values * values, axis=1), 2 * epoch + 1, present)
        _publish(partial + (PCHUNKS + chunk) * PSOURCES + sources, tl.sum(values * weight[None, :], axis=1),
                 2 * epoch + 1, present)
        slots = owned[:, None] & present[None, :]
        squares = tl.sum(_gather(partial + chunks[:, None] * PSOURCES + sources[None, :], 2 * epoch + 1, slots), axis=0)
        dots = tl.sum(_gather(partial + (PCHUNKS + chunks[:, None]) * PSOURCES + sources[None, :], 2 * epoch + 1,
                              slots), axis=0)
        scale = tl.rsqrt(squares / HIDDEN + eps)
        scores = tl.where(present, dots * scale, float("-inf"))
        peak = tl.max(scores, axis=0)
        exponentials = tl.exp(scores - peak)
        total = tl.sum(exponentials, axis=0)
        accumulated = tl.sum(exponentials[:, None] * values, axis=0)
        mixed = (accumulated / total).to(Prefix.dtype.element_ty).to(tl.float32)
    if MIXTURE:
        if COMPACT_MIXTURE:
            count = tl.load(MixtureCount).to(tl.int64)
            origin = tl.maximum(count - MIXTURE_WINDOW, 0) // MIXTURE_ALIGNMENT * MIXTURE_ALIGNMENT
            destination = row - origin
            retained = (destination >= 0) & (destination < MIXTURE_ROWS)
            mixture = tl.where(row < count, mixed, 0).to(Mixture.dtype.element_ty)
            tl.store(Mixture + destination * mixture_stride + columns, mixture, valid & retained)
            padding = (row < MIXTURE_ROWS) & (origin + row >= ROWS)
            tl.store(Mixture + row * mixture_stride + columns, 0., valid & padding)
        else:
            tl.store(Mixture + row * mixture_stride + columns, mixed.to(Mixture.dtype.element_ty), valid)
    own = chunks == chunk
    _publish(partial + 2 * PCHUNKS * PSOURCES + chunks, tl.where(own, tl.sum(mixed * mixed), 0.), 2 * epoch + 2, own)
    squares = tl.sum(_gather(partial + 2 * PCHUNKS * PSOURCES + chunks, 2 * epoch + 2, owned))
    normalized = (mixed * tl.rsqrt(squares / HIDDEN + output_eps)).to(Prefix.dtype.element_ty)
    output = norm.to(tl.float32) * normalized.to(tl.float32)
    tl.store(Output + row * output_stride + columns, output.to(Output.dtype.element_ty), valid)
    if chunk == 0:
        tl.store(Epoch + row, epoch + 1)


def score_weight(norm_weight, projection):
    """Fold the residual score's RMSNorm weight into its projection, as the canonical score does in FP32."""
    return (norm_weight.float() * projection.float().flatten()).contiguous()


class ResidualStream:
    """Running block sum, committed residual blocks, and the normalized input of the next projection."""

    def __init__(self, prefix, blocks, output, *, split_rows=SPLIT_ROWS):
        tokens, hidden = prefix.shape
        if (blocks.shape[0], blocks.shape[2]) != (tokens, hidden) or output.shape != prefix.shape:
            raise ValueError("residual blocks and output must match the running sum's rows and width")
        if any(t.stride(-1) != 1 or t.dtype != torch.bfloat16 for t in (prefix, blocks, output)):
            raise ValueError("residual stream tensors must be BF16 with contiguous rows")
        self.prefix, self.blocks, self.output = prefix, blocks, output
        self.split = None
        if 0 < tokens <= split_rows:
            chunk, warps = SPLIT_CHUNK, SPLIT_WARPS
            if chunk & (chunk - 1) or not 0 < triton.cdiv(hidden, chunk) <= SPLIT_CHUNKS:
                raise ValueError(f"the residual split chunk must be a power of two splitting the width into at "
                                 f"most {SPLIT_CHUNKS} chunks")
            # Tagged partial-sum slots and a per-row call epoch; allocated once, never reset.
            self.split = (torch.zeros((tokens, SPLIT_STRIDE), dtype=torch.int64, device=prefix.device),
                          torch.zeros((tokens,), dtype=torch.int32, device=prefix.device), chunk, warps)

    def mix(self, delta, mode, *, sources, write, score, norm, eps, output_eps, mixture=None,
            mixture_count=None, mixture_window=None, mixture_alignment=1, output=None):
        """Apply ``mode`` with ``delta``, commit block ``write`` (or none for -1), and normalize the mixture
        of the first ``sources`` committed blocks and the running sum into ``output`` (by default the stream's);
        ``mixture`` also receives it unnormalized. A scalar ``mixture_count`` retains the last
        ``mixture_window`` live rows of a single prefill, with the origin rounded down to
        ``mixture_alignment``. Extra retained rows preserve downstream GEMM row placement."""
        if not 0 <= sources <= self.blocks.shape[1] or not -1 <= write < self.blocks.shape[1] or write >= 0 and write < sources:
            raise ValueError("residual mixing reads only committed blocks and never overwrites one")
        if delta is not None and (delta.shape != self.prefix.shape or delta.stride(-1) != 1):
            raise ValueError("residual delta must match the running sum")
        if mixture_count is not None:
            if (mixture is None or mixture_count.numel() != 1 or mixture_count.dtype not in (torch.int32, torch.int64)
                    or mixture_count.device != self.prefix.device):
                raise ValueError("compact residual mixture requires one device integer live count")
            mixture_window = mixture.shape[0] if mixture_window is None else mixture_window
            if (type(mixture_window) is not int or type(mixture_alignment) is not int
                    or mixture_alignment <= 0 or mixture_window <= 0
                    or mixture_window + mixture_alignment - 1 > mixture.shape[0]):
                raise ValueError("compact residual capacity must include the window and alignment padding")
        if mixture is not None:
            if (mixture.ndim != 2 or mixture.shape[1] != self.prefix.shape[1] or mixture.stride(-1) != 1
                    or mixture.device != self.prefix.device
                    or not 0 < mixture.shape[0] <= self.prefix.shape[0]
                    or mixture_count is None and mixture.shape != self.prefix.shape):
                raise ValueError("residual mixture output must match the width and declared retained row capacity")
        output = self.output if output is None else output
        if output.shape != self.prefix.shape or output.stride(-1) != 1 or output.dtype != torch.bfloat16:
            raise ValueError("residual output must be BF16 rows matching the running sum")
        rows, hidden = self.prefix.shape
        delta = self.prefix if delta is None else delta
        tile, warps = TILES[min(sources, len(TILES) - 1)]
        if self.split is not None and sources < SPLIT_SOURCES:
            partial, epoch, chunk, split_warps = self.split
            torch.library.wrap_triton(_mix_split)[(rows, triton.cdiv(hidden, chunk))](
                self.prefix, delta, self.blocks, score, norm, output, output if mixture is None else mixture,
                self.prefix if mixture_count is None else mixture_count, partial, epoch,
                self.prefix.stride(0), delta.stride(0), self.blocks.stride(0), self.blocks.stride(1),
                output.stride(0), (output if mixture is None else mixture).stride(0), float(eps),
                float(output_eps), hidden, sources, write, mode, mixture is not None, mixture_count is not None,
                rows, 0 if mixture is None else mixture.shape[0], mixture_window or 0, mixture_alignment,
                chunk, triton.cdiv(hidden, chunk), tile, triton.next_power_of_2(sources + 1), SPLIT_CHUNKS, SPLIT_STRIDE,
                num_warps=split_warps)
            return output
        torch.library.wrap_triton(_mix)[(rows,)](
            self.prefix, delta, self.blocks, score, norm, output, output if mixture is None else mixture,
            self.prefix if mixture_count is None else mixture_count,
            self.prefix.stride(0), delta.stride(0), self.blocks.stride(0), self.blocks.stride(1),
            output.stride(0), (output if mixture is None else mixture).stride(0), float(eps),
            float(output_eps), hidden, sources, write, mode, mixture is not None, mixture_count is not None,
            rows, 0 if mixture is None else mixture.shape[0], mixture_window or 0, mixture_alignment,
            triton.next_power_of_2(hidden),
            tile, num_warps=warps)
        return output
