# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi prefill reduction transport declaration."""

import pytest

from ignition.preparation.collectives import create_dma_all_reduce_runtime
from ignition.recipes.kimi3.bootstrap import PreparationConfig


def test_prefill_reduction_defaults_to_the_lossless_fp32_owner_packet():
    assert PreparationConfig().prefill_reduction == "fp32-owner-packet"


def test_prefill_reduction_accepts_the_bf16_gather_declaration():
    config = PreparationConfig(prefill_reduction="bf16-gather")
    assert config.prefill_reduction == "bf16-gather"


@pytest.mark.parametrize("value", ["bf16", "fp8-owner-packet", "bf16-gather ", ""])
def test_prefill_reduction_rejects_unknown_transports(value):
    with pytest.raises(ValueError, match="prefill reduction must be"):
        PreparationConfig(prefill_reduction=value)


def test_bf16_gather_requires_a_two_byte_activation_declaration():
    with pytest.raises(ValueError, match="two-byte activation"):
        create_dma_all_reduce_runtime(None, 0, max_tokens=1, hidden_size=1,
            element_bytes=4, bf16_gather=True)
    with pytest.raises(ValueError, match="two-byte activation"):
        create_dma_all_reduce_runtime(None, 0, max_tokens=1, hidden_size=1,
            bf16_gather=1)
