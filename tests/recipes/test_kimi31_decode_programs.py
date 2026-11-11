# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused decode programs of the Kimi MLA and DSA steps against the launches they replace.

Each case runs a fused Triton program and the sequence it replaces in Triton's interpreter
(tests/triton_interpreter.py) and requires identical bits, which covers the programs' indexing,
masks and order of operations. Whether a reduction keeps its thread layout on the device, and
the device's rounding to FP8, are properties of the lowered programs that these cases do not see.
"""
import pytest
import torch

from tests.triton_interpreter import interpreted, run_interpreted


def test_cases_in_triton_interpreter():
    run_interpreted(__file__, 6, "test_fused_latent_unpacking_matches", "test_fused_key_norm", "test_clamp_cast")


@interpreted
@pytest.mark.parametrize("rows,world,width", [(6, 9, 240), (1, 9, 240), (5, 2, 1056), (3, 1, 2112)])
def test_fused_latent_unpacking_matches_copies_and_row_norms(rows, world, width):
    from ignition.recipes.kimi3.pointwise import rms_norm_columns, unpack_and_rms_norm_qkv

    query, record = 1536, 576
    local_q, local_kv = -(-query // world), -(-record // world)
    generator = torch.Generator().manual_seed(rows * 31 + world)
    gathered = (torch.randn((rows, world, width), generator=generator) * 3).to(torch.bfloat16)
    q_weight = torch.randn(query, generator=generator).to(torch.bfloat16)
    kv_weight = torch.randn(512, generator=generator).to(torch.bfloat16)
    # The replaced sequence: rank-major copies, the query and KV row norms and the tail copy.
    query_input = torch.zeros((rows, world * local_q), dtype=torch.bfloat16)
    kv_input = torch.zeros((rows, world * local_kv), dtype=torch.bfloat16)
    query_input.view(rows, world, local_q).copy_(gathered[:, :, :local_q])
    kv_input.view(rows, world, local_kv).copy_(gathered[:, :, local_q:local_q + local_kv])
    expected_query = torch.zeros((rows, query), dtype=torch.bfloat16)
    expected_latent = torch.zeros((rows, record), dtype=torch.bfloat16)
    rms_norm_columns(query_input[:, :query], q_weight, expected_query, 1e-6)
    rms_norm_columns(kv_input[:, :512], kv_weight, expected_latent[:, :512], 1e-6)
    expected_latent[:, 512:].copy_(kv_input[:, 512:record])
    actual_query = torch.full((rows, query), float("nan"), dtype=torch.bfloat16)
    actual_latent = torch.full((rows, record), float("nan"), dtype=torch.bfloat16)
    unpack_and_rms_norm_qkv(gathered, local_q, local_kv, q_weight, kv_weight, actual_query, actual_latent, 1e-6)
    assert torch.equal(actual_query.view(torch.int16), expected_query.view(torch.int16))
    assert torch.equal(actual_latent.view(torch.int16), expected_latent.view(torch.int16))


@pytest.mark.parametrize("change", ["narrow", "tail", "short"])
def test_fused_latent_unpacking_rejects_geometry_its_programs_do_not_cover(change):
    from ignition.recipes.kimi3.pointwise import unpack_and_rms_norm_qkv

    world, local_q, local_kv, width, record = 9, 171, 64, 240, 576
    if change == "narrow":
        width = 200                               # a shard narrower than its query and KV parts
    elif change == "tail":
        local_kv, width, record = 72, 256, 648    # 136 columns beyond the normalized ones
    else:
        local_q = 170                             # nine shards hold fewer than 1,536 query columns
    with pytest.raises(ValueError, match="fused latent unpacking"):
        unpack_and_rms_norm_qkv(torch.zeros((2, world, width), dtype=torch.bfloat16), local_q, local_kv,
            torch.ones(1536, dtype=torch.bfloat16), torch.ones(512, dtype=torch.bfloat16),
            torch.zeros((2, 1536), dtype=torch.bfloat16), torch.zeros((2, record), dtype=torch.bfloat16), 1e-6)


@interpreted
def test_fused_key_norm_writes_the_keys_of_the_prefill_programs():
    from ignition.components.fp8_indexer import _write_keys
    from ignition.recipes.kimi3.dsa import _layernorm128_write_keys
    from ignition.recipes.kimi3.dsa_prefill import _layer_norm_128_bf16

    rows, pages, eps = 12, 4, 1e-5
    generator = torch.Generator().manual_seed(77)
    # Key inputs are 128 columns of a wider projection row; a negative slot writes no record.
    source = (torch.randn((rows, 160), generator=generator) * 2).to(torch.bfloat16)
    key_input = source[:, 16:144]
    weight = (1 + torch.randn(128, generator=generator) / 8).to(torch.bfloat16)
    bias = (torch.randn(128, generator=generator) / 8).to(torch.bfloat16)
    slots = torch.tensor([0, 5, 63, 64, 130, 255, -1, 7, 100, 191, 192, 1], dtype=torch.int64)

    def cache():
        return torch.randint(0, 256, (pages, 64 * 132), generator=torch.Generator().manual_seed(5),
                             dtype=torch.uint8)

    expected_keys, expected_cache = torch.zeros((rows, 128), dtype=torch.bfloat16), cache()
    _layer_norm_128_bf16[(rows,)](key_input, weight, bias, expected_keys, key_input.stride(0), 128, eps,
                                  num_warps=4)
    _write_keys[(rows,)](expected_keys, expected_cache.view(torch.float8_e4m3fn),
                         expected_cache.view(torch.float32), slots, expected_cache.stride(0), False, num_warps=4)
    actual_keys, actual_cache = torch.zeros((rows, 128), dtype=torch.bfloat16), cache()
    _layernorm128_write_keys[(rows,)](key_input, weight, bias, actual_keys,
        actual_cache.view(torch.float8_e4m3fn), actual_cache.view(torch.float32), slots, key_input.stride(0),
        128, actual_cache.stride(0), eps, False, num_warps=4)
    assert torch.equal(actual_keys.view(torch.int16), expected_keys.view(torch.int16))
    assert torch.equal(actual_cache, expected_cache) and not torch.equal(actual_cache, cache())
    # ATen's layer norm computes its moments in another order: the keys agree within one BF16 unit.
    reference = torch.nn.functional.layer_norm(key_input.float(), (128,), weight.float(), bias.float(), eps)
    assert bool(((actual_keys.float() - reference).abs() <= reference.abs() * 2 ** -7 + 1e-6).all())


@interpreted
def test_clamp_cast_writes_e4m3_values_and_saturates_out_of_range_inputs():
    from ignition.recipes.kimi3.pointwise import clamp_cast_fp8_e4m3

    values = torch.arange(256, dtype=torch.int16).to(torch.uint8).view(torch.float8_e4m3fn).float()
    values = values[torch.isfinite(values)]       # every E4M3 value, each exact in BF16
    extremes = torch.tensor([449., 1000., 3e38, float("inf"), -449., -1000., -3e38, -float("inf")])
    row = torch.cat((values, extremes))
    # Rows of two column blocks with a masked tail, read through a row stride.
    width = 5 * row.numel()
    source = torch.zeros((3, width + 6), dtype=torch.bfloat16)
    for index in range(3):
        source[index, :width] = row.roll(index * 11).repeat(5)
    destination = torch.full((3, width), 0x55, dtype=torch.uint8).view(torch.float8_e4m3fn)
    clamp_cast_fp8_e4m3(source[:, :width], destination)
    expected = source[:, :width].float().clamp(-448., 448.).to(torch.float8_e4m3fn)
    assert torch.equal(destination.view(torch.uint8), expected.view(torch.uint8))
