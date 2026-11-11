# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured chain acceptance and request-state commit.

Model verification produces one row for the pending input and each draft.
Commit emits a clipped prefix and consumes the equally long prefix of those
inputs. Model checkpoints and MTP history are selected by the caller using the
returned counts; these operations never reconstruct model state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.components.flashinfer_sampling import filter_probabilities_out, sample_probabilities, probability_filter_workspace, verify_chain

from ignition.components.capacity import BATCH_CAPACITIES
from ignition.components.completion import VECTOR_TOKENS_FIELD, validate_completion_buffers
from ignition.components.logprobs import PreparedLogprobs
from ignition.components.sampling import (
    DUPLICATE_SLOT,
    ERROR,
    FINISHED,
    INVALID_DISTRIBUTION,
    RUNNING,
)

from ignition.components.sampling import DeviceRequestState
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.scratch import bind_scratch

INVALID_SPECULATION = 6
TARGET_RNG_DOMAIN = 0x4D54505F56455249
DRAFT_RNG_DOMAIN = 0x4D54505F44524146
# Vocabulary-wide checks reduce each row in slices first, so compiled forwards spread
# them over many programs instead of one serial program per row.
VOCABULARY_SLICES = 128


@triton.jit
def _prepare_speculative_meta_kernel(Safe, Live, Candidates, TopK, Params, GrammarEnabled, GrammarCounts,
                                     PackedTopK, PackedTopP, RowFlags, Maximum,
                                     cand_stride, param_stride,
                                     WIDTH: tl.constexpr, VOCAB: tl.constexpr, PREFIX_OFFSET: tl.constexpr,
                                     INDICES: tl.constexpr, MASK_ROWS: tl.constexpr, COND_GRAMMAR: tl.constexpr):
    idx = tl.program_id(0)
    b = idx // WIDTH
    w = idx % WIDTH
    s = tl.load(Safe + b).to(tl.int64)
    lv = tl.load(Live + b).to(tl.int1) != 0
    pref = w + PREFIX_OFFSET
    p_valid = True
    for j in tl.static_range(0, INDICES):
        if j < pref:
            cid = tl.load(Candidates + b * cand_stride + j)
            p_valid = p_valid & (cid >= 0) & (cid < VOCAB)
    g_en = tl.load(GrammarEnabled + s).to(tl.int32) != 0
    if COND_GRAMMAR:
        g_cnt = tl.load(GrammarCounts + s).to(tl.int32)
        g_ok = (~g_en) | ((pref < g_cnt) & (g_cnt <= MASK_ROWS))
    elif WIDTH > 1:
        g_ok = (~g_en) | (pref == 0)
    else:
        g_ok = True
    pre_ok = lv & p_valid & g_ok
    tk = tl.load(TopK + s).to(tl.int32)
    tk_out = tl.where(tk > 0, tk, VOCAB)
    tp_out = tl.load(Params + b * param_stride + 1).to(tl.float32)
    tl.store(PackedTopK + idx, tk_out)
    tl.store(PackedTopP + idx, tp_out)
    tl.store(RowFlags + idx * 3, tl.where(pre_ok, 1, 0))
    tl.store(RowFlags + idx * 3 + 1, 0)
    tl.store(RowFlags + idx * 3 + 2, 0)
    tl.store(Maximum + idx, float('-inf'))


@triton.jit
def _prepare_speculative_scores_kernel(Logits, Histogram, Candidates, Safe, Params, GrammarEnabled, GrammarMasks,
                                       RowFlags, ScoresOut, Maximum,
                                       log_b_stride, log_w_stride, hist_stride, cand_stride, param_stride,
                                       gm_s_stride, gm_r_stride,
                                       WIDTH: tl.constexpr, VOCAB: tl.constexpr, PREFIX_OFFSET: tl.constexpr,
                                       INDICES: tl.constexpr, COND_GRAMMAR: tl.constexpr,
                                       WRITE_DUMMY: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < VOCAB
    b = idx // WIDTH
    w = idx % WIDTH
    pre_ok = tl.load(RowFlags + idx * 3) != 0
    if WRITE_DUMMY and (not pre_ok):
        dummy = tl.where(cols == 0, 0.0, float('-inf'))
        tl.store(ScoresOut + (b * WIDTH + w) * VOCAB + cols, dummy, mask=mask)
        return
    s = tl.load(Safe + b).to(tl.int64)
    pref = w + PREFIX_OFFSET
    cnt = tl.load(Histogram + s * hist_stride + cols.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
    for j in tl.static_range(0, INDICES):
        if j < pref:
            cid = tl.load(Candidates + b * cand_stride + j)
            if (cid >= 0) & (cid < VOCAB):
                cnt = cnt + tl.where(cols.to(tl.int64) == cid, 1.0, 0.0)
    p_base = Params + b * param_stride
    temp = tl.load(p_base + 0).to(tl.float32)
    penalty = tl.load(p_base + 3).to(tl.float32)
    p_pres = tl.load(p_base + 4).to(tl.float32)
    p_freq = tl.load(p_base + 5).to(tl.float32)
    sc = tl.load(Logits + b * log_b_stride + w * log_w_stride + cols.to(tl.int64), mask=mask, other=float('-inf')).to(tl.float32)
    # IEEE division and, with FMA contraction disabled at the launch, the unfused order of operations
    # reproduce the ATen score chain bit for bit, as the draft score program does.
    adj = tl.where(sc > 0.0, tl.math.div_rn(sc, penalty), sc * penalty)
    has_cnt = cnt > 0.0
    sc = tl.where(has_cnt, adj, sc)
    sc = sc - tl.where(has_cnt, 1.0, 0.0) * p_pres - cnt * p_freq
    sc = tl.math.div_rn(sc, tl.where(temp > 0.0, temp, 1.0))
    g_en = tl.load(GrammarEnabled + s).to(tl.int32) != 0
    if g_en:
        g_row = pref if COND_GRAMMAR else 0
        words = tl.load(GrammarMasks + s * gm_s_stride + g_row * gm_r_stride + (cols >> 5), mask=mask, other=0)
        allowed = ((words >> (cols & 31)) & 1) != 0
        sc = tl.where(allowed, sc, float('-inf'))
    tl.store(ScoresOut + (b * WIDTH + w) * VOCAB + cols, sc, mask=mask)
    valid_sc = tl.where(mask, sc, float('-inf'))
    bmax = tl.max(valid_sc, axis=0)
    tl.atomic_max(Maximum + idx, bmax)
    if bmax > float('-inf'):
        tl.atomic_or(RowFlags + idx * 3 + 1, 1)
    # The block maximum drops NaN scores; amax returns NaN for a row that holds one.
    nan = tl.sum(tl.where(mask & (sc != sc), 1, 0), axis=0)
    if nan > 0:
        tl.atomic_or(RowFlags + idx * 3 + 2, 1)


@triton.jit
def _finalize_speculative_valid_kernel(RowFlags, ValidOut, ScoresOut, Maximum,
                                       VOCAB: tl.constexpr, WRITE_DUMMY: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0)
    pre_ok = tl.load(RowFlags + idx * 3) != 0
    has_fin = tl.load(RowFlags + idx * 3 + 1) != 0
    has_nan = tl.load(RowFlags + idx * 3 + 2) != 0
    maximum = tl.load(Maximum + idx)
    if has_nan:
        tl.store(Maximum + idx, float('nan'))
    # A finite row maximum: a score above -inf, no NaN and no +inf.
    is_valid = pre_ok & has_fin & (~has_nan) & (maximum < float('inf'))
    tl.store(ValidOut + idx, is_valid)
    if WRITE_DUMMY and pre_ok and (not is_valid):
        c = 0
        while c < VOCAB:
            cols = c + tl.arange(0, BLOCK)
            m = cols < VOCAB
            tl.store(ScoresOut + idx * VOCAB + cols, tl.where(cols == 0, 0.0, float('-inf')), mask=m)
            c += BLOCK


def _sliced(values: torch.Tensor) -> torch.Tensor:
    slices = VOCABULARY_SLICES if values.shape[-1] % VOCABULARY_SLICES == 0 else 1
    return values.unflatten(-1, (slices, values.shape[-1] // slices))


def _tensor(value, shape, dtype, device, name):
    if value.shape != torch.Size(shape) or value.dtype != dtype or value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} requires contiguous {dtype} {tuple(shape)} on {device}")


def _geometry(batch, width):
    if type(batch) is not int or batch not in BATCH_CAPACITIES or type(width) is not int or not 1 <= width < 1 << 31:
        raise ValueError("speculation requires power-of-two batch capacity through 64 and positive Int32 verification width")


def _metadata(state, slots, generations, live_rows, batch, device):
    if state.generation.device != device:
        raise ValueError("request state and speculative operands must share a device")
    _tensor(slots, (batch,), torch.int32, device, "slots")
    _tensor(generations, (batch,), torch.int64, device, "generations")
    _tensor(live_rows, (1,), torch.int32, device, "live_rows")


def _live(state, slots, generations, live_rows, rows):
    safe = slots.long().clamp(0, state.capacity - 1)
    selected = (rows < live_rows[0]) & (live_rows[0] >= 0) & (live_rows[0] <= slots.numel())
    selected &= (slots >= 0) & (slots < state.capacity) & (state.generation[safe] == generations)
    duplicates = ((slots[:, None] == slots[None, :]) & selected[None, :]).sum(-1) > 1
    return safe, selected & ~duplicates & (state.status[safe] == RUNNING)


@dataclass
class SpeculativeAcceptance:
    token_ids: torch.Tensor
    proposed_count: torch.Tensor
    accepted_drafts: torch.Tensor
    offered_drafts: torch.Tensor
    valid: torch.Tensor
    rng_increment: torch.Tensor
    survival: torch.Tensor

    @classmethod
    def storage_layout(cls, batch, width):
        _geometry(batch, width)
        return dict(token_ids=((batch, width), torch.int64),
                    proposed_count=((batch,), torch.int32), accepted_drafts=((batch,), torch.int32),
                    offered_drafts=((batch,), torch.int32), valid=((batch,), torch.bool),
                    rng_increment=((batch,), torch.int64), survival=((batch, width - 1), torch.float32))

    @classmethod
    def broadcast_layout(cls, batch, width):
        return cls.storage_layout(batch, width)

    @classmethod
    def allocate(cls, batch, width, device):
        return cls(**{name: torch.full(shape, -1 if name in ("token_ids", "survival") else 0, dtype=dtype, device=device)
                      for name, (shape, dtype) in cls.storage_layout(batch, width).items()})

    def validate(self, batch, width, device):
        for name, (shape, dtype) in self.storage_layout(batch, width).items():
            _tensor(getattr(self, name), shape, dtype, device, name)


@dataclass
class SpeculativeRngState:
    """Physical-slot target-verification stream, independent of proposal draws."""

    generation: torch.Tensor
    target_offset: torch.Tensor

    @classmethod
    def allocate(cls, device, *, capacity=4):
        return cls(torch.zeros(capacity, dtype=torch.int64, device=device),
                   torch.zeros(capacity, dtype=torch.int64, device=device))


class PreparedSpeculativeDistributions:
    """Conditional request filtering using the ordinary FlashInfer policy.

    Row j includes candidate IDs before j+prefix_offset in its penalty counts.
    The committed histogram already contains the incoming pending token.
    Optional physical-slot grammar masks describe each conditional prefix.
    """

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return dict(scores=((batch, width, vocab), torch.float32),
            probabilities=((batch, width, vocab), torch.float32),
            maximum=((batch, width), torch.float32), valid=((batch, width), torch.bool),
            packed_top_k=((batch * width,), torch.int32), packed_top_p=((batch * width,), torch.float32),
            row_flags=((batch * width * 3,), torch.int32))

    def __init__(self, logits, requests, slots, generations, live_rows, candidate_prefix_ids,
                 *, prefix_offset=0, grammar_masks=None, grammar_mask_count=None, workspace=None):
        batch, width, vocab = logits.shape
        _geometry(batch, width)
        device = logits.device
        if logits.dtype not in (torch.float32, torch.bfloat16) or not logits.is_contiguous() or vocab != requests.vocab_size:
            raise ValueError("conditional logits require contiguous FP32/BF16 [B,W,vocabulary]")
        _metadata(requests, slots, generations, live_rows, batch, device)
        if candidate_prefix_ids.ndim != 2 or candidate_prefix_ids.shape[0] != batch:
            raise ValueError("candidate_prefix_ids requires [B,K]")
        _tensor(candidate_prefix_ids, candidate_prefix_ids.shape, torch.int64, device, "candidate_prefix_ids")
        if type(prefix_offset) is not int or prefix_offset < 0 or width - 1 + prefix_offset > candidate_prefix_ids.shape[1]:
            raise ValueError("prefix_offset must keep every conditional prefix within candidate IDs")
        self.logits, self.state, self.slots, self.generations, self.live_rows = logits, requests, slots, generations, live_rows
        self.candidates, self.prefix_offset = candidate_prefix_ids, prefix_offset
        if (grammar_masks is None) != (grammar_mask_count is None):
            raise ValueError("conditional grammar masks and counts must be provided together")
        if grammar_masks is not None:
            if grammar_masks.ndim != 3 or not 1 <= grammar_masks.shape[1] < 1 << 31:
                raise ValueError("conditional grammar requires a positive Int32 mask-row capacity")
            _tensor(grammar_masks, (requests.capacity, grammar_masks.shape[1], (vocab + 31) // 32), torch.int32, device, "grammar_masks")
            _tensor(grammar_mask_count, (requests.capacity,), torch.int32, device, "grammar_mask_count")
            if width + prefix_offset > grammar_masks.shape[1]:
                raise ValueError("conditional grammar prefix exceeds prepared mask rows")
        self.grammar_masks, self.grammar_mask_count = grammar_masks, grammar_mask_count
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(width, device=device)
        self.vocab_ids = torch.arange(vocab, device=device)
        for tensor in (self.rows, self.columns, self.vocab_ids):
            record_numeric_tensor(tensor)
        for name, tensor in bind_scratch(self.scratch_layout(batch, width, vocab),
                                        device=device, workspace=workspace).items():
            setattr(self, name, tensor)
        from ignition.components.probability_normalization import PreparedProbabilityNormalization
        self.normalization = PreparedProbabilityNormalization(self.probabilities, self.valid)
        self._prepared = False

    def prepare_scores(self, *, write_dummy_invalid: bool = False):
        state = self.state
        batch, width, vocab = self.logits.shape
        safe, live = _live(state, self.slots, self.generations, self.live_rows, self.rows)
        params = state.parameters[safe]
        indices = width - 1 + self.prefix_offset
        has_cond_grammar = self.grammar_masks is not None
        g_masks = self.grammar_masks if has_cond_grammar else state.grammar_mask[:, None, :]
        g_counts = self.grammar_mask_count if has_cond_grammar else self.slots
        n_rows = batch * width
        torch.library.wrap_triton(_prepare_speculative_meta_kernel)[(n_rows,)](
            safe, live, self.candidates, state.top_k, params, state.grammar_enabled, g_counts,
            self.packed_top_k, self.packed_top_p, self.row_flags, self.maximum,
            self.candidates.stride(0), params.stride(0),
            width, vocab, self.prefix_offset, indices, g_masks.shape[1], has_cond_grammar,
            num_warps=1)
        block = 1024
        torch.library.wrap_triton(_prepare_speculative_scores_kernel)[(n_rows, triton.cdiv(vocab, block))](
            self.logits, state.histogram, self.candidates, safe, params, state.grammar_enabled, g_masks,
            self.row_flags, self.scores, self.maximum,
            self.logits.stride(0), self.logits.stride(1), state.histogram.stride(0),
            self.candidates.stride(0), params.stride(0), g_masks.stride(0), g_masks.stride(1),
            width, vocab, self.prefix_offset, indices, has_cond_grammar, write_dummy_invalid, block,
            num_warps=4, enable_fp_fusion=False)
        torch.library.wrap_triton(_finalize_speculative_valid_kernel)[(n_rows,)](
            self.row_flags, self.valid, self.scores, self.maximum,
            vocab, write_dummy_invalid, block, num_warps=4)
        return safe, params

    def prepare(self):
        if self.logits.device.type != "cuda":
            raise ValueError("FlashInfer probability filtering requires CUDA")
        from flashinfer import sampling
        self._flashinfer = sampling
        self.library_scratch = (
            sampling._get_cache_buf(f"top_k_mask_logits_row_states_{self.logits.device}", 1024 * 1024, self.logits.device, zero_init=True),
            sampling._get_cache_buf("softmax_workspace", 1024 * 1024, self.logits.device),
        )
        self.top_p_workspace = probability_filter_workspace(
            self.packed_top_p.numel(), self.state.vocab_size, self.logits.device)
        self._prepared = True
        self()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("probability filtering requires prepare() before capture")
        _, params = self.prepare_scores(write_dummy_invalid=True)
        flat = self.scores.view(-1, self.state.vocab_size)
        filter_probabilities_out(flat, self.packed_top_k, self.packed_top_p,
            *self.library_scratch, self.probabilities.view_as(flat), self.top_p_workspace)
        return self.normalization(self.probabilities, params)


class PreparedStochasticVerification:
    """Fixed-width one-row FlashInfer calls with device-owned RNG offsets.

    K+2 bounds scalar Philox draws, including the independent-acceptance metric.
    Reservations are rounded to four draws. Functional library outputs remain
    graph-private and are copied into the persistent acceptance buffers.
    """

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return dict(safe_draft_probs=((batch, width - 1, vocab), torch.float32),
            safe_target_probs=((batch, width, vocab), torch.float32),
            safe_ids=((batch, width - 1), torch.int32), seed_bits=((batch,), torch.int64),
            offset=((batch,), torch.int64), independent_accepted=((batch,), torch.int32),
            emitted_drafts=((batch,), torch.int32))

    def __init__(self, draft_probs, draft_token_ids, target_probs, requests, slots, generations,
                 live_rows, *, rng, draft_valid=None, target_valid=None, output=None, token_limits=None,
                 workspace=None):
        batch, width, vocab = target_probs.shape
        _geometry(batch, width)
        device = target_probs.device
        for value, shape, dtype, name in ((draft_probs, (batch, width - 1, vocab), torch.float32, "draft_probs"),
                (draft_token_ids, (batch, width - 1), torch.int64, "draft_token_ids"),
                (target_probs, (batch, width, vocab), torch.float32, "target_probs")):
            _tensor(value, shape, dtype, device, name)
        _metadata(requests, slots, generations, live_rows, batch, device)
        if vocab != requests.vocab_size:
            raise ValueError("verification vocabulary must match request state")
        for name in ("generation", "target_offset"):
            _tensor(getattr(rng, name), (requests.capacity,), torch.int64, device, f"rng.{name}")
        self.draft_probs, self.drafts, self.target_probs = draft_probs, draft_token_ids, target_probs
        self.state, self.slots, self.generations, self.live_rows, self.rng = requests, slots, generations, live_rows, rng
        self.draft_valid = torch.ones((batch, width - 1), dtype=torch.bool, device=device) if draft_valid is None else draft_valid
        self.target_valid = torch.ones((batch, width), dtype=torch.bool, device=device) if target_valid is None else target_valid
        _tensor(self.draft_valid, (batch, width - 1), torch.bool, device, "draft_valid")
        _tensor(self.target_valid, (batch, width), torch.bool, device, "target_valid")
        self.output = SpeculativeAcceptance.allocate(batch, width, device) if output is None else output
        self.output.validate(batch, width, device)
        self.rows, self.columns = torch.arange(batch, device=device), torch.arange(width, device=device)
        self.dummy = (torch.arange(vocab, device=device) == 0).float()
        for name, tensor in bind_scratch(self.scratch_layout(batch, width, vocab),
                                        device=device, workspace=workspace).items():
            setattr(self, name, tensor)
        self.draw_reservation = 4 * ((width + 1 + 3) // 4)
        if token_limits is not None:
            _tensor(token_limits, (batch,), torch.int32, device, "token_limits")
        self.token_limits = token_limits
        self._prepared = False
        self.broadcast = None
        from .probability_mask import PreparedProbabilityMask

        outputs = (self.safe_draft_probs, self.safe_target_probs)
        self.probability_masks = (
            PreparedProbabilityMask(self.draft_probs, self.safe_draft_probs, self.output.valid,
                token_limits, self.dummy, limit_offset=1, outputs=outputs),
            PreparedProbabilityMask(self.target_probs, self.safe_target_probs, self.output.valid,
                token_limits, self.dummy, limit_offset=0, outputs=outputs),
        )

    def prepare_broadcast(self, **kwargs):
        from .collectives import PreparedBroadcast

        self.broadcast = PreparedBroadcast(tuple(vars(self.output).values()), **kwargs)

    def prepare(self):
        if self.target_probs.device.type != "cuda":
            raise ValueError("FlashInfer chain verification requires CUDA")
        from flashinfer import sampling
        self._flashinfer = sampling
        for mask in self.probability_masks:
            mask.prepare()
        self._prepared = True
        self()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("chain verification requires prepare() before capture")
        out = self.output
        safe, live = _live(self.state, self.slots, self.generations, self.live_rows, self.rows)
        if self.token_limits is None:
            valid = live & self.draft_valid.all(-1) & self.target_valid.all(-1)
            valid &= ((self.drafts >= 0) & (self.drafts < self.state.vocab_size)).all(-1)
            draft_probs, target_probs, ids = self.draft_probs, self.target_probs, self.drafts
        else:
            target_used = self.columns[None, :] < self.token_limits[:, None]
            draft_used = self.columns[None, :-1] < self.token_limits[:, None] - 1
            valid = live & (self.draft_valid | ~draft_used).all(-1) & (self.target_valid | ~target_used).all(-1)
            valid &= (((self.drafts >= 0) & (self.drafts < self.state.vocab_size)) | ~draft_used).all(-1)
            valid &= (self.token_limits >= 1) & (self.token_limits <= self.target_probs.shape[1])
            draft_probs = torch.where(draft_used[:, :, None], self.draft_probs, self.dummy)
            target_probs = torch.where(target_used[:, :, None], self.target_probs, self.dummy)
            ids = torch.where(draft_used, self.drafts, 0)
        for probabilities in (draft_probs, target_probs):
            slices = _sliced(probabilities)
            entries = (torch.isfinite(slices) & (slices >= 0)).all(-1).all(-1)
            mass = slices.sum(-1).sum(-1)
            valid &= (entries & torch.isfinite(mass) & ((mass - 1).abs() <= 1e-4)).all(-1)
        out.valid.copy_(valid)
        if self.token_limits is not None:
            for mask in self.probability_masks:
                mask.stage()
        for mask in self.probability_masks:
            if self.token_limits is None:
                mask.stage()
            mask()
        self.safe_ids.copy_(torch.where(valid[:, None], ids, 0))
        self.seed_bits.copy_(self.state.seed.view(torch.int64)[safe] ^ TARGET_RNG_DOMAIN)
        self.offset.copy_(torch.where(self.rng.generation[safe] == self.generations, self.rng.target_offset[safe], 0))
        self._verify(valid)
        if self.broadcast is not None:
            self.broadcast()
        return out

    def _verify(self, valid):
        out = self.output
        self.independent_accepted.zero_()
        self.emitted_drafts.zero_()
        tokens, _, _ = verify_chain(self.safe_draft_probs, self.safe_ids, self.safe_target_probs,
            self.independent_accepted, self.emitted_drafts, seed=self.seed_bits, offset=self.offset)
        accepted = self.emitted_drafts
        if self.token_limits is None:
            out.offered_drafts.fill_(self.drafts.shape[1])
        else:
            # The fixed-width library may accept dummy columns beyond a short
            # tail. Only the real prefix belongs to the commit contract.
            out.offered_drafts.copy_((self.token_limits - 1).clamp(0, self.drafts.shape[1]))
            accepted = torch.minimum(accepted, out.offered_drafts)
        out.token_ids.copy_(tokens)
        out.accepted_drafts.copy_(torch.where(valid, accepted, 0))
        out.proposed_count.copy_(torch.where(valid, accepted + 1, 0))
        out.rng_increment.copy_(torch.where(valid, self.draw_reservation, 0))
        out.token_ids.masked_fill_(~valid[:, None], -1)
        if self.drafts.shape[1]:
            indices = self.safe_ids.long().unsqueeze(-1)
            draft = self.safe_draft_probs.gather(-1, indices).squeeze(-1)
            target = self.safe_target_probs[:, :-1].gather(-1, indices).squeeze(-1)
            ratio = torch.where(draft > 0, (target / draft).clamp(max=1.), (target > 0).float())
            # Serial products preserve monotonicity despite FP32 rounding at unit ratios.
            prefixes = [ratio[:, 0]]
            for depth in range(1, self.drafts.shape[1]):
                prefixes.append(prefixes[-1] * ratio[:, depth])
            out.survival.fill_(-1.)
            out.survival[:, :self.drafts.shape[1]].copy_(torch.where(valid[:, None], torch.stack(prefixes, -1), -1.))


class PreparedStochasticBlockVerification(PreparedStochasticVerification):
    """Block rejection sampling over full-vocabulary probabilities on one rank.

    r_0=1 and r_(i+1)=min(r_i*p_i(x_i)/q_i(x_i),1) in the log domain, so a zero
    prefix stays zero. Correction mass d_i(x)=max(r_i*p_i(x)-q_i(x),0). The
    decision after draft i uses h_i=sum(d_(i+1))/(sum(d_(i+1))+1-r_(i+1)) when
    another offered draft follows and h_i=r_(i+1) at the last one. Decisions are
    all drawn: the last success selects the prefix, and stopping at the first
    failure would implement a different algorithm. The correction comes from
    d_n, or from p at the bonus position. Zero d_n can only arise from FP32 mass
    rounding and falls back to p.

    Every draw is a request-domain one-row library call, so tokens do not depend
    on the batch slot: decision i is subsequence i at the round's offset and the
    correction is subsequence zero at offset+1, inside the base reservation.
    Published survival is the running minimum of r, a non-increasing lower bound
    of P(prefix accepted | drafts) that the scheduler's forecast can validate.
    """

    def _verify(self, valid):
        out, depth = self.output, self.drafts.shape[1]
        target, draft = self.safe_target_probs, self.safe_draft_probs
        batch, vocab = target.shape[0], target.shape[2]
        if self.token_limits is None:
            out.offered_drafts.fill_(depth)
        else:
            out.offered_drafts.copy_((self.token_limits - 1).clamp(0, depth))
        offered = out.offered_drafts.long()
        if depth:
            ids = self.safe_ids.long()
            q = draft.gather(-1, ids[:, :, None]).squeeze(-1)
            p = target[:, :-1].gather(-1, ids[:, :, None]).squeeze(-1)
            steps = torch.where(q > 0, p.log() - q.log(), torch.where(p > 0, 0., -torch.inf))
            logs = [torch.zeros_like(steps[:, 0])]
            for column in range(depth):
                logs.append((logs[-1] + steps[:, column]).clamp(max=0.))
            scales = torch.stack(logs, -1)
            excess = (scales[:, :-1, None].exp() * target[:, :-1] - draft).clamp(min=0.)
            mass = excess.sum(-1, dtype=torch.float64)
            following = torch.cat((mass[:, 1:], torch.zeros_like(mass[:, :1])), -1)
            total = following - scales[:, 1:].double().expm1()
            block = torch.where(total > 0, following / total, 1.).float()
            positions = self.columns[None, :-1] + 1
            successes = torch.where(positions < offered[:, None], block,
                                    torch.where(positions == offered[:, None], scales[:, 1:].exp(), 0.)).clamp(0., 1.)
            outcomes = torch.stack((1. - successes, successes), -1)
            accepted = torch.stack([sample_probabilities(outcomes[row], self.seed_bits[row].repeat(depth),
                                                         self.offset[row].repeat(depth))
                                    for row in range(batch)]).long()
            accepted = (accepted * positions).amax(-1)
            drafted = excess.gather(1, accepted.clamp(max=depth - 1)[:, None, None].expand(-1, 1, vocab)).squeeze(1)
            drafted_mass = mass.gather(1, accepted.clamp(max=depth - 1)[:, None]).float()
            correct = (accepted < offered) & (drafted_mass[:, 0] > 0)
        else:
            accepted = offered
        bonus = target.gather(1, accepted[:, None, None].expand(-1, 1, vocab)).squeeze(1)
        bonus = bonus / bonus.sum(-1, keepdim=True)
        residual = torch.where(correct[:, None], drafted / drafted_mass.clamp(min=torch.finfo(torch.float32).tiny),
                               bonus) if depth else bonus
        correction = torch.cat([sample_probabilities(residual[row:row + 1], self.seed_bits[row:row + 1],
                                                     self.offset[row:row + 1] + 1)
                                for row in range(batch)]).long()
        tokens = correction[:, None].expand(-1, self.columns.numel())
        if depth:
            tokens = torch.where(self.columns[None] < accepted[:, None],
                                 torch.cat((self.safe_ids.long(), correction[:, None]), -1), tokens)
        out.token_ids.copy_(torch.where(self.columns[None] <= accepted[:, None], tokens, -1))
        out.accepted_drafts.copy_(torch.where(valid, accepted, 0))
        out.proposed_count.copy_(torch.where(valid, accepted + 1, 0))
        out.rng_increment.copy_(torch.where(valid, self.draw_reservation, 0))
        out.token_ids.masked_fill_(~valid[:, None], -1)
        if depth:
            prefixes = [scales[:, 1].exp()]
            for column in range(1, depth):
                prefixes.append(torch.minimum(prefixes[-1], scales[:, column + 1].exp()))
            out.survival.fill_(-1.)
            out.survival[:, :depth].copy_(torch.where(valid[:, None], torch.stack(prefixes, -1), -1.))


class PreparedSpeculativeStateCommit:
    """Commit a verified chain prefix, optionally publishing native completion vectors.

    A round emitting t tokens commits verification input t-1, leaving emitted
    token t-1 pending. Exact snapshot maintenance selects the committed state
    column after the request's model submissions have drained.
    """

    def __init__(self, requests: DeviceRequestState, acceptance: SpeculativeAcceptance,
                 input_ids, slots, generations, live_rows, completion, frame_index,
                 submission_sequence, *, logits=None, logprobs=None, rng=None, copy_offered_mask=None, draft_context_lag=1, token_limits=None, buffers=None):
        records = None if completion is None else completion.records
        batch, width = input_ids.shape
        tokens = width if completion is None else completion.token_capacity
        if width > tokens:
            raise ValueError("native speculative width exceeds completion token capacity")
        self.token_capacity = tokens
        record_fields = VECTOR_TOKENS_FIELD + tokens
        _geometry(batch, width)
        device = input_ids.device
        if draft_context_lag is not None and (type(draft_context_lag) is not int or draft_context_lag < 0):
            raise ValueError("draft_context_lag must be a nonnegative token count or None")
        self.draft_context_lag = draft_context_lag
        self.history = getattr(requests, "ple_history", None)
        if token_limits is not None:
            _tensor(token_limits, (batch,), torch.int32, device, "token_limits")
        self.token_limits = token_limits
        if completion is None:
            if logits is not None or logprobs is not None:
                raise ValueError("private state commit does not publish logprobs")
        elif logprobs is None:
            if logits is None or logits.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("target_logits requires BF16 or FP32 storage")
            _tensor(logits, (batch, width, requests.vocab_size), logits.dtype, device, "target_logits")
        elif logits is not None:
            raise ValueError("raw target logits and a prepared logprob reduction are mutually exclusive")
        elif not callable(logprobs) or not callable(getattr(logprobs, "prepare", None)):
            raise TypeError("the logprob reduction requires prepare() and captured execution")
        _tensor(input_ids, (batch, width), torch.int64, device, "input_ids")
        _metadata(requests, slots, generations, live_rows, batch, device)
        acceptance.validate(batch, width, device)
        if completion is not None:
            validate_completion_buffers(completion.buffers(), requests.capacity, tokens)
            _tensor(records, (2, requests.capacity, record_fields), torch.int64, device, "completion_records")
            _tensor(frame_index, (1,), torch.int32, device, "frame_index")
            _tensor(submission_sequence, (1,), torch.int64, device, "submission_sequence")
        self.copy_offered_mask = torch.zeros(batch, dtype=torch.int64, device=device) if copy_offered_mask is None else copy_offered_mask
        _tensor(self.copy_offered_mask, (batch,), torch.int64, device, "copy_offered_mask")
        if rng is not None:
            for name in ("generation", "target_offset"):
                _tensor(getattr(rng, name), (requests.capacity,), torch.int64, device, f"rng.{name}")
        self.logprobs = None if completion is None else (PreparedLogprobs(logits, completion, slots, generations, live_rows, frame_index)
                        if logprobs is None else logprobs)
        self.state, self.acceptance, self.inputs, self.slots = requests, acceptance, input_ids, slots
        self.generations, self.live, self.records, self.frame, self.sequence = generations, live_rows, records, frame_index, submission_sequence
        self.rng = rng
        self.rows, self.columns = torch.arange(batch, device=device), torch.arange(width, device=device)
        self.eos_columns = torch.arange(requests.eos_token_ids.shape[1], device=device)
        self.history_columns = torch.arange(2, device=device)
        def zeros(shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.emitted_ids = zeros(input_ids.shape).fill_(-1)
        self.committed_input_ids = zeros(input_ids.shape).fill_(-1)
        self.emitted_count = zeros((batch,), torch.int32)
        self.accepted_count = zeros((batch,), torch.int32)
        self.accepted_drafts = zeros((batch,), torch.int32)
        self.commit_column = zeros((batch,), torch.int32).fill_(-1)
        self.consumed_before = zeros((batch,))
        self.generated_before = zeros((batch,))
        self.history_before = zeros((batch, 2))
        # Pool-backed request state is staged here, so compiled commits write the pool without copying it.
        self.consumed_after = zeros((requests.capacity,))
        self.history_after = zeros((requests.capacity, 2))
        # One extra entry receives the rows that own no slot.
        self.owners = zeros((requests.capacity + 1,))
        self._records = zeros((requests.capacity, record_fields))
        self.survival = None if completion is None else completion.speculative_survival
        self._survival = zeros((requests.capacity, tokens - 1), torch.float32) if self.survival is not None else None
        self.survival_columns = torch.arange(tokens - 1, device=device)
        self.idle_record = torch.zeros(record_fields, dtype=torch.int64, device=device)
        self.idle_record[[4, 12, *range(VECTOR_TOKENS_FIELD, record_fields)]] = -1
        if draft_context_lag is None:
            self.idle_record[13] = -1

    def prepare(self):
        if self.logprobs is not None:
            self.logprobs.prepare()

    def __call__(self):
        state, acceptance, columns = self.state, self.acceptance, self.columns
        width = columns.numel()
        safe = self.slots.long().clamp(0, state.capacity - 1)
        selected = ((self.rows < self.live[0]) & (self.live[0] >= 0) & (self.live[0] <= self.rows.numel())
            & (self.slots >= 0) & (self.slots < state.capacity) & (state.generation[safe] == self.generations))
        self.consumed_before.copy_(state.consumed_tokens[safe])
        self.generated_before.copy_(state.generated_tokens[safe])
        if self.history is not None:
            self.history_before.copy_(self.history[safe])

        # Every row commits from its slot's last selected row, which alone writes the slot's state.
        # Rows are evaluated together from the state before the round; each slot then takes its
        # owner's values in one update, so no slot is written twice whatever the package capacity.
        matches = selected[:, None] & selected[None, :] & (self.slots[:, None] == self.slots[None, :])
        count = matches.sum(1)
        row = torch.where(matches, self.rows[None, :], 0).amax(1)
        leader = selected & (row == self.rows)
        ids = acceptance.token_ids[row]
        inputs = self.inputs[row]
        proposed = acceptance.proposed_count[row].long()
        accepted = acceptance.accepted_drafts[row].long()
        offered = acceptance.offered_drafts[row].long()
        valid = acceptance.valid[row]
        increment = acceptance.rng_increment[row]
        status_before, error_before = state.status[safe], state.error_code[safe]
        generated_before, pending_before = state.generated_tokens[safe], state.pending_token[safe]
        max_new_tokens = state.max_new_tokens[safe]
        remaining = (max_new_tokens - generated_before).clamp_min(0)
        enabled = (count == 1) & (status_before == RUNNING) & (remaining > 0)
        proposal = columns[None, :] < proposed[:, None]
        shape_valid = ((proposed >= 1) & (proposed <= width) & (accepted == proposed - 1)
            & (offered >= accepted) & (offered <= width - 1))
        shape_valid &= (((ids >= 0) & (ids < state.vocab_size)) | ~proposal).all(-1)
        shape_valid &= (((inputs >= 0) & (inputs < state.vocab_size)) | ~proposal).all(-1)
        shape_valid &= (pending_before >= 0) & (inputs[:, 0] == pending_before)
        shape_valid &= ((ids[:, :-1] == inputs[:, 1:]) | ~(columns[None, :-1] < accepted[:, None])).all(-1)
        if self.rng is None:
            shape_valid &= increment == 0
        else:
            shape_valid &= increment >= 0
        valid = valid & shape_valid
        eos = ((ids[:, :, None] == state.eos_token_ids[safe][:, None, :])
            & (self.eos_columns[None, None, :] < state.eos_count[safe][:, None, None])).any(-1)
        eos_position = torch.where(eos & proposal, columns[None, :] + 1, width).amin(-1)
        emitted = torch.minimum(torch.minimum(proposed, remaining), eos_position)
        if self.token_limits is not None:
            emitted = torch.minimum(emitted, self.token_limits[row])
        emitted = torch.where(enabled & valid, emitted, 0)
        emitted_drafts = torch.minimum(accepted.clamp_min(0), emitted)
        take = columns[None, :] < emitted[:, None]
        pending = ids.gather(1, (emitted - 1).clamp(0, width - 1)[:, None])[:, 0]
        consumed = state.consumed_tokens[safe] + emitted
        generated = generated_before + emitted
        terminal = (generated >= max_new_tokens) | (eos & take).any(-1)
        failed = enabled & ~valid
        duplicate = count > 1
        status = torch.where(duplicate | failed, ERROR,
            torch.where((count == 1) & (status_before == RUNNING) & (terminal | (remaining == 0)), FINISHED, status_before))
        error = torch.where(duplicate, DUPLICATE_SLOT,
            torch.where(failed, torch.where(shape_valid, INVALID_DISTRIBUTION, INVALID_SPECULATION), error_before))
        generation = state.generation[safe]
        counter = state.rng_counter.view(torch.int64)
        counters = counter[safe] + emitted.view(-1, *([1] * (counter.dim() - 1)))
        if self.history is not None:
            history = self.history[safe]
            tail = torch.cat((history, inputs), 1).gather(1, emitted[:, None] + self.history_columns[None, :])
            history = torch.where((emitted > 0)[:, None], tail, history)
        if self.rng is not None:
            offset = self.rng.target_offset[safe]
            base = torch.where(self.rng.generation[safe] == generation, offset, 0)
            advance = (emitted > 0) & (increment > 0)
            rng_offset = torch.where(advance, base + increment, offset)
            rng_generation = torch.where(advance, generation, self.rng.generation[safe])
        live = enabled & valid
        if self.records is not None:
            record = torch.cat((torch.stack((generation, self.sequence.expand_as(generation), consumed, generated,
                torch.where(emitted > 0, pending, -1), status.long(), error.long(), torch.zeros_like(emitted), emitted,
                torch.where(live, offered, 0), emitted_drafts, torch.where(live, self.copy_offered_mask[row], 0),
                torch.full_like(emitted, -1),
                ((consumed - self.draft_context_lag).clamp_min(0) if self.draft_context_lag is not None
                 else torch.full_like(emitted, -1)),
                torch.zeros_like(emitted), torch.full_like(emitted, width)), 1),
                torch.where(take, ids, -1), torch.full((emitted.numel(), self.token_capacity - width), -1, dtype=torch.int64, device=ids.device)), 1)

        # Records and per-row outputs precede request-state mutation: compiled graphs must not
        # re-derive enabled/valid fields from the status and token counts this round rewrites.
        self.emitted_ids.copy_(torch.where(selected[:, None] & take, ids, -1))
        self.committed_input_ids.copy_(torch.where(selected[:, None] & take, inputs, -1))
        self.emitted_count.copy_(torch.where(selected, emitted, 0))
        self.accepted_count.copy_(torch.where(selected, emitted, 0))
        self.accepted_drafts.copy_(torch.where(selected, emitted_drafts, 0))
        self.commit_column.copy_(torch.where(selected, emitted - 1, -1))
        self.owners.fill_(-1)
        self.owners.scatter_(0, torch.where(leader, safe, state.capacity), self.rows)
        owner = self.owners[:state.capacity]
        owned, source = owner >= 0, owner.clamp_min(0)
        if self._survival is not None:
            logical = torch.minimum(offered, (remaining - 1).clamp_min(0))
            if self.token_limits is not None:
                logical = torch.minimum(logical, (self.token_limits[row] - 1).clamp_min(0))
            extent = torch.minimum(logical, torch.where(emitted > emitted_drafts, logical, emitted_drafts))
            observed = owned[:, None] & live[source, None] & (self.survival_columns < extent[source, None])
            values = torch.cat((acceptance.survival[row][source],
                torch.full((state.capacity, self.token_capacity - width), -1., device=ids.device)), 1)
            self._survival.copy_(torch.where(observed, values, -1.))
            self.survival.index_copy_(0, self.frame.long(), self._survival.unsqueeze(0))

        def updated(tensor, values):
            mask = owned.view(-1, *([1] * (tensor.dim() - 1)))
            return torch.where(mask, values[source].to(tensor.dtype), tensor)

        def commit(tensor, values):
            tensor.copy_(updated(tensor, values))

        # Consumed tokens and history may be views of the state pool: both are read before
        # either is written, so compiled commits never materialize an updated pool.
        self.consumed_after.copy_(updated(state.consumed_tokens, consumed))
        if self.history is not None:
            self.history_after.copy_(updated(self.history, history))
        if self.records is not None:
            self._records.copy_(torch.where(owned[:, None], record[source], self.idle_record))
        state.histogram.view(-1).scatter_add_(0, (safe[:, None] * state.vocab_size
            + ids.clamp(0, state.vocab_size - 1)).view(-1), (take & leader[:, None]).to(torch.int32).view(-1))
        commit(state.generated_tokens, generated)
        commit(state.pending_token, torch.where(emitted > 0, pending, pending_before))
        commit(counter, counters)
        commit(state.status, status)
        commit(state.error_code, error)
        if self.rng is not None:
            commit(self.rng.target_offset, rng_offset)
            commit(self.rng.generation, rng_generation)
        state.consumed_tokens.copy_(self.consumed_after)
        if self.history is not None:
            self.history.copy_(self.history_after)
        if self.records is not None:
            self.records.index_copy_(0, self.frame.long(), self._records.unsqueeze(0))
            self.logprobs()
        return self


class PreparedSpeculativeCommit(PreparedSpeculativeStateCommit):
    """Commit a verified prefix and publish the prepared completion token vector."""

    def __init__(self, requests, acceptance, input_ids, slots, generations, live_rows,
                 completion, frame_index, submission_sequence, **kwargs):
        if completion is None:
            raise ValueError("native speculative commit requires completion storage")
        super().__init__(requests, acceptance, input_ids, slots, generations, live_rows,
                         completion, frame_index, submission_sequence, **kwargs)
