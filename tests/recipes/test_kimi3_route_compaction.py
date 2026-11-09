# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the rank-major route compaction of the paired gather.

The decode route exchange writes rank-major padded rows into graph-owned
buffers, and the recipe compacts each gathered row into the caller-owned
latent and router rows. These tests compare the single-copy compaction with
a per-rank reference copy for the serving geometry and for worlds where the
last rank's block is clipped, using bit comparisons so NaN and infinite
payloads are covered.
"""

import pytest
import torch

from ignition.recipes.kimi3.routing import PreparedRouteGather


def _owner(latent, gathered_latent, logits, gathered_logits):
    """Bind only the attributes used by compact() to avoid CUDA preparation."""

    owner = PreparedRouteGather.__new__(PreparedRouteGather)
    owner.latent = latent
    owner.gathered_latent = gathered_latent
    owner.logits = logits
    owner.gathered_logits = gathered_logits
    return owner


def _reference_compact(owner, down_width, router_width):
    """Per-rank copy sequence that the production path previously issued."""

    world = owner.gathered_latent.shape[1] // down_width
    for rank in range(world):
        down_begin = rank * down_width
        down_count = min(down_width, owner.latent.shape[1] - down_begin)
        if down_count > 0:
            owner.latent[:, down_begin:down_begin + down_count].copy_(
                owner.gathered_latent[:, down_begin:down_begin + down_count])
        router_begin = rank * router_width
        router_count = min(router_width, owner.logits.shape[1] - router_begin)
        if router_count > 0:
            owner.logits[:, router_begin:router_begin + router_count].copy_(
                owner.gathered_logits[:, router_begin:router_begin + router_count])


# world, rows, logical latent width, gathered latent width, logical router
# width, gathered router width. Worlds 2/4/8 divide both extents; worlds
# 9/12 and 16 exercise a padded exchange where the logical width is shorter
# than world * local width (the last rank's block is clipped).
CASES = (
    (2, 6, 3584, 3584, 896, 896),
    (4, 32, 3584, 3584, 896, 896),
    (8, 6, 3584, 3584, 896, 896),
    (9, 1, 3584, 3744, 896, 936),
    (9, 6, 3584, 3744, 896, 936),
    (9, 32, 3584, 3744, 896, 936),
    (12, 6, 3584, 3840, 896, 960),
    (16, 6, 3584, 3584, 896, 896),
)


@pytest.mark.parametrize(
    ("world", "rows", "latent_width", "gathered_latent_width", "router_width", "gathered_router_width"),
    CASES,
)
def test_single_copy_compaction_equals_per_rank_copies(
    world, rows, latent_width, gathered_latent_width, router_width, gathered_router_width
):
    generator = torch.Generator().manual_seed(world * 1009 + rows)
    gathered_latent = torch.randn(rows, gathered_latent_width, generator=generator).to(torch.bfloat16)
    gathered_logits = torch.randn(rows, gathered_router_width, generator=generator)
    # Inactive-row style poison: a NaN latent and an infinite router value must
    # travel unchanged, and the comparison must stay bit-exact.
    gathered_latent[0, 0] = float("nan")
    gathered_logits[-1, -1] = float("-inf")
    gathered_latent_before = gathered_latent.clone()
    gathered_logits_before = gathered_logits.clone()

    latent = torch.full((rows, latent_width), float("nan"), dtype=torch.bfloat16)
    logits = torch.full((rows, router_width), float("nan"))
    reference_latent = latent.clone()
    reference_logits = logits.clone()

    _owner(latent, gathered_latent, logits, gathered_logits).compact()
    _reference_compact(
        _owner(reference_latent, gathered_latent, reference_logits, gathered_logits),
        gathered_latent_width // world,
        gathered_router_width // world,
    )

    assert torch.equal(latent.view(torch.int16), reference_latent.view(torch.int16))
    assert torch.equal(logits.view(torch.int32), reference_logits.view(torch.int32))
    assert torch.equal(gathered_latent.view(torch.int16), gathered_latent_before.view(torch.int16))
    assert torch.equal(gathered_logits.view(torch.int32), gathered_logits_before.view(torch.int32))


def test_compaction_copies_the_logical_prefix_only():
    world, rows = 9, 6
    gathered_latent = torch.arange(rows * 3744, dtype=torch.float32).reshape(rows, 3744).to(torch.bfloat16)
    gathered_logits = torch.arange(rows * 936, dtype=torch.float32).reshape(rows, 936)
    latent = torch.full((rows, 3584), float("nan"), dtype=torch.bfloat16)
    logits = torch.full((rows, 896), float("nan"))

    _owner(latent, gathered_latent, logits, gathered_logits).compact()

    assert torch.equal(latent.view(torch.int16), gathered_latent[:, :3584].view(torch.int16))
    assert torch.equal(logits.view(torch.int32), gathered_logits[:, :896].view(torch.int32))
