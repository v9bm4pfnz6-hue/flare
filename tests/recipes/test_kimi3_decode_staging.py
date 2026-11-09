# SPDX-License-Identifier: Apache-2.0

"""Byte-equal staging forms for the decode metadata and sample rows.

The decoder stages scheduler metadata and the sampled hidden rows at the top of
every forward. These tests pin the two in-place forms against the expressions
they replace so a future edit cannot silently change the staged bytes.
"""

import pytest
import torch


DTYPES = (torch.int32, torch.int64)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", (1, 2, 6, 7, 8))
def test_fresh_flag_in_place_matches_compare_then_copy(dtype, rows):
    starts = torch.tensor([0, 1, 0, 5, 0, 2, 0, 0], dtype=dtype)[:rows]
    fresh = torch.zeros(rows, dtype=torch.bool)
    expected = torch.zeros(rows, dtype=torch.bool)
    torch.eq(starts, 0, out=fresh)
    expected.copy_(starts == 0)
    assert torch.equal(fresh, expected)


def test_hidden_index_select_in_place_matches_advanced_index_copy():
    hidden = torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5).to(torch.bfloat16)
    indices = torch.tensor([7, 0, 3, 3, 5], dtype=torch.int64)
    staged = torch.zeros(5, 5, dtype=torch.bfloat16)
    expected = torch.zeros(5, 5, dtype=torch.bfloat16)
    torch.index_select(hidden, 0, indices, out=staged)
    expected.copy_(hidden[indices])
    assert torch.equal(staged, expected)


def test_hidden_index_select_in_place_preserves_unwritten_rows():
    hidden = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16)
    staged = torch.zeros(6, 3, dtype=torch.bfloat16)
    torch.index_select(hidden, 0, torch.tensor([2, 0], dtype=torch.int64), out=staged[3:5])
    assert torch.equal(staged[3:5], hidden[torch.tensor([2, 0], dtype=torch.int64)])
    untouched = torch.cat((staged[:3].flatten(), staged[5:].flatten()))
    assert torch.equal(untouched, torch.zeros_like(untouched))
