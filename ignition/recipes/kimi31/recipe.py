# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 Pro: gated sparse attention and full-context MLA drafting."""

from dataclasses import replace
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, SamplingDefaults, SamplingRange, SpeculativeMethod
from ignition.recipes.kimi3.config import LATENT_DSPARK_REPOSITORY, LATENT_DSPARK_REVISION
from ignition.recipes.kimi3.recipe import RECIPE as KIMI3, prepare_variant


def prepare(snapshot, plan, **kwargs):
    from . import bootstrap

    if plan.speculation is not None and plan.speculation.method == "dflash2":
        from .dflash_config import REPOSITORY, REVISION
        return prepare_variant(snapshot, plan, bootstrap=bootstrap, draft_repository=REPOSITORY,
            draft_revision=REVISION, draft_directory="dflash2", **kwargs)
    if plan.speculation is not None and plan.speculation.draft_head_quantization is not None:
        raise ValueError("Kimi draft head conversion requires --speculative-method dflash2")
    return prepare_variant(snapshot, plan, bootstrap=bootstrap, draft_repository=LATENT_DSPARK_REPOSITORY,
                           draft_revision=LATENT_DSPARK_REVISION, **kwargs)


RECIPE = replace(KIMI3, name="kimi31", served_model_name="Kimi-K3.1-Pro", prepare=prepare,
    context=1048576,
    # https://huggingface.co/moonshotai/Kimi-K3.1-Pro#3-evaluation-results
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
                                      ranges={"top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(alias="kimi-k3.1-pro", repository="moonshotai/Kimi-K3.1-Pro",
        revision="6d0a0202956796543364d666ab19c6ad1a494ee1", architecture="KimiK31ForConditionalGeneration",
        stored_bytes=1_561_217_377_280, device_bytes=1_496_756_986_630, draft_bytes=5_373_504_512),),
    online_mxfp8=True, language_model_only=True, draft_head_quantizations=("mxfp8", "nvfp4"),
    speculation=replace(KIMI3.speculation, method="mtp"),
    speculative_methods=MappingProxyType({"dflash2": SpeculativeMethod(MappingProxyType({
        8: RankCosts(state_row_bytes=65_761_536, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128*8),
        9: RankCosts(state_row_bytes=60_709_632, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128*9),
        16: RankCosts(state_row_bytes=35_450_112, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128*16),
    }), draft_bytes=5_569_622_016)}),
    # Gather threads are tuned independently; the attention components declare LSE's 512-thread geometry.
    # The TP9 gather reaches peak at the kernel's 64-block ceiling: measured 28.87 ms against 31.96 ms
    # at the inherited eight-block limit for 4608 rows, and 6.41 ms against 7.06 ms at 1024 rows.
    # The EXL3 expert kernel takes the measured prefill launch shapes: at TP9 the 128-row graph runs in
    # 0.232 s against 0.299 s, the 256-row graph in 0.351 s against 0.464 s and the 512-row graph in
    # 0.558 s against 0.659 s; launches below 128 rows and the 2,048-row graph keep the heuristic plan.
    # Decode launches of that kernel use the stripe split-K schedule: at TP9 a speculative round at equal
    # offered drafts takes 0.85 ms less of about 33 ms. Its finalize adds FP32 partial sums in lock order,
    # so decode rounds are not run-to-run reproducible; prefill launches keep whole-tile scheduling.
    # Collectives of a nine-rank deployment on four leaf pairs and one single rank (ranks (0,1), (2,3),
    # (4,5), (6,7) and 8): the paired route exchange pushes one block per row and selects the experts in
    # the same launch; head gathers cross once between ranks 0-3,8 and ranks 4-7, which share a single
    # link, and are forwarded in two more phases (b12x side_entry_gather_roles), with acquire-flag
    # barriers; the LSE reduce-scatter pushes each destination's heads, and between ranks 0-3 and ranks 4-7
    # a rank sends the rank across one LSE-weighted FP32 partial of its quad instead of four rows
    # (b12x lse_quad_roles; FP32 sums group differently). The other transports are byte for byte.
    # Expert launches of fewer than 128 tokens stage their table with cp.async, prefetch the rotation
    # side tables, pack routes inside the kernel and replace two grid barriers with per-block counters
    # (same arithmetic), and use the 128,128,64,256 tile (FP32 sums over K group differently).
    # The 2,048-row prefill graph takes the direct decode table with 48-row blocks (regroups FP32 sums).
    b12x_environment=MappingProxyType({**KIMI3.b12x_environment, "B12X_PCIE_DCP_THREADS": "0",
                                       "B12X_PCIE_DCP_BLOCK_LIMIT": "64", "B12X_W4A16_PREFILL_SHAPES": "1",
                                       "B12X_W4A16_SMALL_M_SPLITK": "decode",
                                       "B12X_PCIE_DCP_PAIR_PUSH": "1", "B12X_PCIE_DCP_GATHER_ACQUIRE": "1",
                                       "B12X_PCIE_DCP_GATHER_ISLAND": "2", "B12X_PCIE_DCP_LSE_PUSH_TP9": "2",
                                       "B12X_TRELLIS_SWITCH_SCOPE": "prefill_safe", "B12X_TRELLIS_LUT_CP_ASYNC": "2",
                                       "B12X_TRELLIS_PREFETCH_SIDES": "1", "B12X_TRELLIS_INKERNEL_ROUTE_PACK": "2",
                                       "B12X_TRELLIS_FLOW_FLAGS": "1", "B12X_TRELLIS_TILE_CONFIG": "128,128,64,256",
                                       "B12X_W4A16_PREFILL_TABLE": "2"}),
    ranks=MappingProxyType({8: RankCosts(state_row_bytes=63_181_056, page_bytes=2_543_616,
                                       graph_bytes=9 << 29, page_tokens=128 * 8),
                           9: RankCosts(state_row_bytes=58_129_152, page_bytes=2_543_616,
                                       graph_bytes=9 << 29, page_tokens=128 * 9),
                           16: RankCosts(state_row_bytes=32_869_632, page_bytes=2_543_616,
                                        graph_bytes=9 << 29, page_tokens=128 * 16)}),
    plain_ranks=MappingProxyType({8: RankCosts(state_row_bytes=58_361_088, page_bytes=2_174_976,
                                             graph_bytes=9 << 29, page_tokens=128 * 8),
                                 9: RankCosts(state_row_bytes=53_680_128, page_bytes=2_174_976,
                                             graph_bytes=9 << 29, page_tokens=128 * 9),
                                 16: RankCosts(state_row_bytes=30_275_328, page_bytes=2_174_976,
                                              graph_bytes=9 << 29, page_tokens=128 * 16)}))
