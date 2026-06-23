# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# NOTE: This file is dependent on the success of test_nvfp4_quantize_exact.py
# and also the test_nvfp4_rht_quantize_exact.py.
# Separate to make sure all the functionalities are working as expected.
# Otherwise reference implementation will get messy.

# Due to the structure of NVFP4Quantizer, we need to test the RHT functionality
# together with the quantization functionality.

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import (
    NVFP4QuantizerRef,
)
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.quantization import NVFP4BlockScalingRecipeState, QuantizerRole
from transformer_engine.common.recipe import NVFP4BlockScaling

import pytest
import torch
import random
import math

from nvfp4_utils import (
    get_nvfp4_scale_shape_no_padding,
    generate_split_sections,
    assert_same_shape_and_dtype,
    reference_group_quantize,
    swizzle_nvfp4_scale,
)

# Reuse the 4over6 exact-test infrastructure (PyTorch reference comparison matrix,
# the err-fast-math env toggle, and the FP4 nibble unpacker) so the grouped tests
# verify against the same NVFP4QuantizerRef contract as test_nvfp4_quantize_exact.
from test_nvfp4_quantize_exact import (
    NVFP4_4OVER6_CONFIGS,
    nvfp4_4over6_err_fast_math,
    unpack_fp4,
)

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _roundup(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _valid_rowwise_scale_bytes(grouped, total_rows: int, hidden: int) -> torch.Tensor:
    """Slice written bytes out of the padded compact E4M3 scale buffer."""
    sinv0 = _roundup(total_rows, 128)
    sinv1 = _roundup(hidden // 16, 4)
    flat = grouped.scale_inv.contiguous().view(torch.uint8).reshape(-1)
    assert flat.numel() == sinv0 * sinv1, (flat.numel(), sinv0, sinv1)
    block = flat.reshape(sinv0, sinv1)
    return block[:total_rows, : hidden // 16].reshape(-1).clone()


def _make_4over6_quantizer(
    err_mode: str,
    nvfp4_e4m3_max: int,
    *,
    rowwise: bool = True,
    columnwise: bool = False,
    row_scaled: bool = False,
) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=rowwise,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=row_scaled,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=err_mode,
    )


def _assert_rowwise_matches_per_tensor(
    grouped,
    refs: list,
    expected_amax: torch.Tensor,
    splits: list[int],
    rows: int,
    hidden: int,
) -> None:
    packed_offset = 0
    scale_rows = _roundup(rows, 128)
    scale_cols = _roundup(hidden // 16, 4)
    grouped_scales = grouped.scale_inv.contiguous().view(torch.uint8).reshape(
        scale_rows, scale_cols
    )
    row_offset = 0
    for split, ref in zip(splits, refs):
        packed_elems = split * hidden // 2
        torch.testing.assert_close(
            grouped.rowwise_data[packed_offset : packed_offset + packed_elems],
            ref._rowwise_data.view(dtype=torch.uint8).reshape(-1),
            atol=0,
            rtol=0,
        )
        valid_ref_scales = ref._rowwise_scale_inv.view(torch.uint8).reshape(
            _roundup(split, 128), scale_cols
        )[:split, : hidden // 16]
        torch.testing.assert_close(
            grouped_scales[row_offset : row_offset + split, : hidden // 16].reshape(-1),
            valid_ref_scales.reshape(-1),
            atol=0,
            rtol=0,
        )
        packed_offset += packed_elems
        row_offset += split

    torch.testing.assert_close(grouped.amax, expected_amax, atol=0, rtol=0)

    grouped_dequantized = torch.cat(
        [tex.dequantize(t, te.DType.kBFloat16) for t in grouped.split_into_quantized_tensors()],
        dim=0,
    )
    ref_dequantized = torch.cat([tex.dequantize(ref, te.DType.kBFloat16) for ref in refs], dim=0)
    torch.testing.assert_close(
        grouped_dequantized,
        ref_dequantized,
        atol=0,
        rtol=0,
    )


def _assert_columnwise_matches_per_tensor(
    grouped,
    refs: list,
    expected_amax: torch.Tensor,
    splits: list[int],
    hidden: int,
) -> None:
    packed_offset = 0
    scale_offset = 0
    for split, ref in zip(splits, refs):
        packed_elems = split * hidden // 2
        torch.testing.assert_close(
            grouped.columnwise_data[packed_offset : packed_offset + packed_elems],
            ref._columnwise_data.view(dtype=torch.uint8).reshape(-1),
            atol=0,
            rtol=0,
        )
        scale_rows = _roundup(hidden, 128)
        scale_cols = _roundup(split // 16, 4)
        scale_elems = scale_rows * scale_cols
        grouped_scales = grouped.columnwise_scale_inv[
            scale_offset : scale_offset + scale_elems
        ].view(torch.uint8).reshape(scale_rows, scale_cols)
        valid_ref_scales = ref._columnwise_scale_inv.view(torch.uint8).reshape(
            scale_rows, scale_cols
        )[:hidden, : split // 16]
        torch.testing.assert_close(
            grouped_scales[:hidden, : split // 16].reshape(-1),
            valid_ref_scales.reshape(-1),
            atol=0,
            rtol=0,
        )
        packed_offset += packed_elems
        scale_offset += scale_elems

    torch.testing.assert_close(grouped.columnwise_amax, expected_amax, atol=0, rtol=0)
    split_tensors = grouped.split_into_quantized_tensors()
    for split_tensor, ref in zip(split_tensors, refs):
        torch.testing.assert_close(
            split_tensor._columnwise_data.view(dtype=torch.uint8).reshape(-1),
            ref._columnwise_data.view(dtype=torch.uint8).reshape(-1),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            split_tensor._columnwise_scale_inv.view(torch.uint8),
            ref._columnwise_scale_inv.view(torch.uint8),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            split_tensor._amax_columnwise,
            ref._amax_columnwise,
            atol=0,
            rtol=0,
        )


# Only the 4over6 configs are relevant: row-scaled grouped quant requires 4over6.
_ROW_SCALED_4OVER6_CONFIGS = [cfg for cfg in NVFP4_4OVER6_CONFIGS if cfg.use_4over6]
_REGULAR_4OVER6_CONFIGS = [cfg for cfg in NVFP4_4OVER6_CONFIGS if cfg.use_4over6]


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("err_use_fast_math", [False, True], ids=["exact_err", "fast_err"])
def test_regular_4over6_group_quantize_from_recipe_state(err_use_fast_math: bool) -> None:
    """4over6 recipe-state quantizers should route through native group quantization."""

    device = "cuda"
    torch.manual_seed(4321)
    torch.cuda.manual_seed(4321)

    rows, hidden = 256, 512
    splits = [128, 128]
    first_dims = torch.tensor(splits, dtype=torch.int64, device=device)
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device=device)
    x_chunks = torch.split(x, splits)

    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        nvfp4_4over6="all",
        nvfp4_4over6_e4m3_use_256="all",
        nvfp4_4over6_err_mode="MSE",
    )
    quantizer = NVFP4BlockScalingRecipeState(
        recipe,
        mode="forward",
        num_quantizers=1,
        roles=[QuantizerRole(module_type="grouped_linear", tensor_type="input")],
    ).make_quantizers()[0]

    assert quantizer.nvfp4_use_4over6
    assert quantizer.nvfp4_e4m3_max == 256
    assert quantizer.nvfp4_4over6_err_mode == "MSE"
    assert not quantizer.with_rht
    assert not quantizer.stochastic_rounding
    assert not quantizer.with_2d_quantization

    expected_amax = torch.stack([chunk.abs().amax().float() for chunk in x_chunks])

    with nvfp4_4over6_err_fast_math(err_use_fast_math):
        grouped = tex.group_quantize(x, quantizer, len(splits), first_dims)
        refs = [quantizer(chunk) for chunk in x_chunks]

    _assert_rowwise_matches_per_tensor(grouped, refs, expected_amax, splits, rows, hidden)
    _assert_columnwise_matches_per_tensor(grouped, refs, expected_amax, splits, hidden)


# Fused row-scaled 4over6 grouped path.
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_row_scaled_4over6_grouped_split_dequantize_matches_flat() -> None:
    """Split grouped row-scaled NVFP4 tensors should match the flat quant/dequant path."""

    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    rows, hidden = 256, 512
    splits = [128, 128]
    split_sections = torch.tensor(splits, dtype=torch.int64, device=device)
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device=device)
    quantizer = _make_4over6_quantizer("MAE", 256, row_scaled=True)

    grouped = tex.group_quantize(x, quantizer, len(splits), split_sections)
    flat = quantizer(x)
    grouped_dequantized = torch.cat(
        [tex.dequantize(t, te.DType.kBFloat16) for t in grouped.split_into_quantized_tensors()],
        dim=0,
    )
    torch.testing.assert_close(
        grouped_dequantized,
        tex.dequantize(flat, te.DType.kBFloat16),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "rows, hidden, splits",
    [
        (256, 512, [128, 128]),  # fully 128-aligned, no scale padding
        (384, 320, [128, 128, 128]),  # hidden 320 -> 20 scale cols, 3 splits
        (304, 512, [96, 208]),  # rows 304 -> padded scale rows, uneven splits
        (640, 2064, [208, 432]),  # hidden 2064 -> padded scale cols, uneven splits
        (1024, 7168, [512, 512]),  # larger, realistic shape
    ],
)
@pytest.mark.parametrize(
    "cfg", _ROW_SCALED_4OVER6_CONFIGS, ids=[cfg.id for cfg in _ROW_SCALED_4OVER6_CONFIGS]
)
def test_row_scaled_4over6_group_quantize_matches_reference(
    rows: int, hidden: int, splits: list[int], cfg
) -> None:
    """Grouped row-scaled 4over6 should match the PyTorch reference."""
    assert sum(splits) == rows and rows % 16 == 0 and hidden % 16 == 0

    device = "cuda"
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device=device)
    first_dims = torch.tensor(splits, dtype=torch.int64, device=device)
    quantizer = _make_4over6_quantizer(cfg.err_mode, cfg.e4m3_max, row_scaled=True)

    with nvfp4_4over6_err_fast_math(cfg.err_use_fast_math):
        grouped = tex.group_quantize(x, quantizer, len(splits), first_dims)

    ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=False,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        row_scaled_nvfp4=True,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=cfg.e4m3_max,
        nvfp4_4over6_err_mode=cfg.err_mode,
        nvfp4_4over6_err_use_fast_math=cfg.err_use_fast_math,
    )
    x_ref = ref_quantizer.quantize(x)

    qx = unpack_fp4(grouped.rowwise_data.view(dtype=torch.uint8).reshape(rows, hidden // 2))
    qx_ref = unpack_fp4(x_ref.data.view(dtype=torch.uint8))
    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    sx_ref = x_ref.scale.view(dtype=torch.uint8)
    torch.testing.assert_close(
        _valid_rowwise_scale_bytes(grouped, rows, hidden),
        sx_ref.reshape(-1),
        atol=0.0,
        rtol=0.0,
    )

    torch.testing.assert_close(
        grouped.amax.reshape(-1), x_ref.global_amax_row.reshape(-1), atol=0.0, rtol=0.0
    )


# Regular 4over6 grouped quant path.
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "compute_amax, quantize_mode, precomputed_offsets",
    [
        (False, "rowwise_only", False),
        (True, "rowwise_only", False),
        (False, "columnwise_only", False),
        (False, "both_directions", False),
        (True, "both_directions", False),
        (False, "both_directions", True),
    ],
    ids=[
        "with_amax_rowwise_only",
        "compute_amax_rowwise_only",
        "with_amax_columnwise_only",
        "with_amax_both_directions",
        "compute_amax_both_directions",
        "precomputed_offsets",
    ],
)
@pytest.mark.parametrize(
    "rows, hidden, splits",
    [
        (384, 512, [128, 256]),
        (768, 384, [128, 256, 384]),  # uneven 128-aligned splits, padded scale columns
        (128, 512, [128]),  # single-tensor grouped path
    ],
)
@pytest.mark.parametrize(
    "cfg", _REGULAR_4OVER6_CONFIGS, ids=[cfg.id for cfg in _REGULAR_4OVER6_CONFIGS]
)
def test_regular_4over6_group_quantize_matches_per_tensor(
    compute_amax: bool,
    quantize_mode: str,
    precomputed_offsets: bool,
    rows: int,
    hidden: int,
    splits: list[int],
    cfg,
) -> None:
    """Grouped regular 4over6 quantization should match per-tensor quantization."""

    device = "cuda"
    torch.manual_seed(9876)
    torch.cuda.manual_seed(9876)

    assert sum(splits) == rows and rows % 16 == 0 and hidden % 16 == 0
    first_dims = torch.tensor(splits, dtype=torch.int64, device=device)
    tensor_offsets = None
    if precomputed_offsets:
        first_dims, (tensor_offsets,) = tex.splits_to_offsets_multi(
            first_dims,
            torch.device(device),
            strides=[hidden],
            include_leading_zero=[True],
            dtypes=[torch.int64],
        )
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device=device)
    x_chunks = torch.split(x, splits)
    return_rowwise = quantize_mode != "columnwise_only"
    return_columnwise = quantize_mode != "rowwise_only"
    quantizer = _make_4over6_quantizer(
        cfg.err_mode,
        cfg.e4m3_max,
        rowwise=return_rowwise,
        columnwise=return_columnwise,
    )

    expected_amax = torch.stack([chunk.abs().amax().float() for chunk in x_chunks])

    with nvfp4_4over6_err_fast_math(cfg.err_use_fast_math):
        if compute_amax:
            grouped = tex.group_quantize(x, quantizer, len(splits), first_dims)
        else:
            grouped = tex.nvfp4_group_quantize_with_amax(
                x,
                quantizer,
                len(splits),
                first_dims,
                expected_amax,
                expected_amax,
                tensor_offsets=tensor_offsets,
            )
        refs = [quantizer(chunk) for chunk in x_chunks]

    if tensor_offsets is not None:
        assert grouped.tensor_offsets.data_ptr() == tensor_offsets.data_ptr()
    if return_rowwise:
        _assert_rowwise_matches_per_tensor(
            grouped, refs, expected_amax, splits, rows, hidden
        )
    if return_columnwise:
        _assert_columnwise_matches_per_tensor(
            grouped, refs, expected_amax, splits, hidden
        )


def check_group_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
    split_sections: list[int],
    with_rht: bool = True,
    with_post_rht_amax: bool = True,
    with_random_sign_mask: bool = True,
    optimize_for_gemm: bool = False,
) -> None:

    te_dtype = te.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    num_chunks = len(split_sections)

    x_splits = torch.split(x, split_sections)

    # Reference quantizers (compact SF, default optimize_for_gemm=False).
    quantizers = [
        NVFP4Quantizer(
            fp4_dtype=te_dtype,
            rowwise=return_rowwise,
            columnwise=return_transpose,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=with_rht,
            with_post_rht_amax=with_post_rht_amax,
            with_random_sign_mask=with_random_sign_mask,
        )
        for _ in range(len(split_sections))
    ]
    x_qx_ref, x_sx_ref, x_amax_rowwise_ref, x_qx_t_ref, x_sx_t_ref, x_amax_colwise_ref = (
        reference_group_quantize(x, quantizers, split_sections, return_rowwise, return_transpose)
    )

    # SUT quantizers: same as reference, but with optimize_for_gemm toggled to
    # request direct swizzled SF emission from the RHT cast-fusion kernel.
    sut_quantizers = [q.copy() for q in quantizers]
    for q in sut_quantizers:
        q.optimize_for_gemm = optimize_for_gemm

    split_quantize_outputs = tex.split_quantize(x, split_sections, sut_quantizers)

    if return_rowwise:
        x_qx = [output._rowwise_data.view(dtype=torch.uint8) for output in split_quantize_outputs]
        x_sx = [output._rowwise_scale_inv for output in split_quantize_outputs]
        x_amax_rowwise = [output._amax_rowwise for output in split_quantize_outputs]

        for i in range(len(x_qx)):
            if split_sections[i] == 0:
                # then just assert the same shape and dtype because the buffer won't be zero out
                assert_same_shape_and_dtype(x_amax_rowwise[i], x_amax_rowwise_ref[i])
                assert_same_shape_and_dtype(x_qx[i], x_qx_ref[i])
                assert_same_shape_and_dtype(x_sx[i], x_sx_ref[i])
            else:
                torch.testing.assert_close(
                    x_amax_rowwise[i], x_amax_rowwise_ref[i], atol=0.0, rtol=0.0
                )
                torch.testing.assert_close(x_qx[i], x_qx_ref[i], atol=0.0, rtol=0.0)
                valid_scale_shape = get_nvfp4_scale_shape_no_padding(x_splits[i].shape, False)
                x_sx_valid = x_sx[i][: valid_scale_shape[0], : valid_scale_shape[1]]
                x_sx_ref_valid = x_sx_ref[i][: valid_scale_shape[0], : valid_scale_shape[1]]
                if optimize_for_gemm:
                    # SUT emits SF in the GEMM-swizzled layout directly; swizzle
                    # the reference compact SF for byte-equal comparison.
                    x_sx_ref_valid = swizzle_nvfp4_scale(
                        split_sections[i], N, x_sx_ref_valid, columnwise=False
                    )
                torch.testing.assert_close(x_sx_valid, x_sx_ref_valid, atol=0.0, rtol=0.0)

    if return_transpose:
        x_qx_t = [
            output._columnwise_data.view(dtype=torch.uint8) for output in split_quantize_outputs
        ]
        x_sx_t = [output._columnwise_scale_inv for output in split_quantize_outputs]
        x_amax_colwise = [output._amax_columnwise for output in split_quantize_outputs]
        # assert with zero tolerance
        for i in range(len(x_qx_t)):
            if split_sections[i] == 0:
                # then just assert the same shape and dtype because the buffer won't be zero out
                assert_same_shape_and_dtype(x_amax_colwise[i], x_amax_colwise_ref[i])
                assert_same_shape_and_dtype(x_qx_t[i], x_qx_t_ref[i])
                assert_same_shape_and_dtype(x_sx_t[i], x_sx_t_ref[i])
            else:
                torch.testing.assert_close(
                    x_amax_colwise[i], x_amax_colwise_ref[i], atol=0.0, rtol=0.0
                )
                torch.testing.assert_close(x_qx_t[i], x_qx_t_ref[i], atol=0.0, rtol=0.0)
                valid_scale_shape = get_nvfp4_scale_shape_no_padding(x_splits[i].shape, True)
                x_sx_t_valid = x_sx_t[i][: valid_scale_shape[0], : valid_scale_shape[1]]
                x_sx_t_ref_valid = x_sx_t_ref[i][: valid_scale_shape[0], : valid_scale_shape[1]]
                if optimize_for_gemm:
                    x_sx_t_ref_valid = swizzle_nvfp4_scale(
                        split_sections[i], N, x_sx_t_ref_valid, columnwise=True
                    )
                torch.testing.assert_close(x_sx_t_valid, x_sx_t_ref_valid, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # edge case, zero tokens for all
        (0, 512),
        # edge case, not 128 multiple hidden dimension
        (1024, 320),
        # full tile cases
        (256, 1024),
        (1024, 256),
        # larger sizes
        (8192, 1024),
        (16384, 8192),
        (16384, 16384),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "edge_cases",
    [
        "regular",
        "zero_tokens_front",
        "zero_tokens_end",
        "zero_tokens_middle",
        "random_uneven_split",
    ],
)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
@pytest.mark.parametrize(
    "with_random_sign_mask", [True, False], ids=["with_random_sign_mask", "no_random_sign_mask"]
)
@pytest.mark.parametrize("with_rht", [True, False], ids=["with_rht", "no_rht"])
@pytest.mark.parametrize(
    "optimize_for_gemm",
    [False, True],
    ids=["compact_sf", "swizzled_sf"],
)
def test_rht_with_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
    quantize_mode: str,
    with_random_sign_mask: bool,
    with_rht: bool,
    optimize_for_gemm: bool,
) -> None:

    # The "quantize writes swizzled SF" fast-path is gated in the C++ framework
    # on ``optimize_for_gemm && with_rht`` (see NVFP4Quantizer::create_tensor and
    # bulk_allocate_nvfp4_tensors in transformer_engine/pytorch/csrc). Without
    # ``with_rht=True`` the flag is silently dropped, so the swizzled_sf row
    # would just duplicate the compact_sf row — skip it instead of flooding the
    # matrix with redundant cases.
    if optimize_for_gemm and not with_rht:
        pytest.skip("optimize_for_gemm requires with_rht=True (framework gate)")

    # The grouped RHT cast-fusion kernel that honors with_gemm_swizzled_scales
    # (group_row_cast_col_hadamard_transform_cast_fusion.cu) is only dispatched
    # when:
    #   - cols are a 128 multiple (RHT cast-fusion eligibility), AND
    #   - every split section is a 128 multiple (all_aligned_token_dim path in
    #     split_quantize_nvfp4_impl_with_rht_helper).
    # For other shapes the C++ side falls back to the unfused row/col split,
    # which does NOT (yet) emit swizzled SF; we'd hit either an NVTE_CHECK or
    # silent SF-layout corruption. Restrict the swizzled coverage to the fused
    # path; the unfused fallback is covered by the optimize_for_gemm=False
    # baseline already exercised above.
    if optimize_for_gemm and N % 128 != 0:
        pytest.skip("RHT cast-fusion requires N % 128 == 0")

    # generate_split_sections hard-codes num_chunks=4 and requires every chunk
    # to be a least_multiple multiple. When optimize_for_gemm forces
    # least_multiple from 64 to 128, the test needs M >= 4*128 = 512 (and a
    # multiple of 512 for the regular/zero_tokens patterns, which the existing
    # M shapes already satisfy: 0, 1024, 8192, 16384). The small M=256 shape
    # cannot satisfy this and is exercised by the optimize_for_gemm=False rows.
    if optimize_for_gemm and 0 < M < 4 * 128:
        pytest.skip("optimize_for_gemm requires M==0 or M>=512 for 4-chunk 128-aligned split")

    least_multiple = 128 if optimize_for_gemm else 64
    split_sections = generate_split_sections(M, N, edge_cases, least_multiple=least_multiple)

    # currently disable pre-RHT amax
    with_post_rht_amax = with_rht

    if quantize_mode == "rowwise_only":
        return_rowwise = True
        return_transpose = False
    elif quantize_mode == "both_directions":
        return_rowwise = True
        return_transpose = True
    elif quantize_mode == "columnwise_only":
        return_rowwise = False
        return_transpose = True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    check_group_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
        split_sections=split_sections,
        with_rht=with_rht,
        with_post_rht_amax=with_post_rht_amax,
        with_random_sign_mask=with_random_sign_mask,
        optimize_for_gemm=optimize_for_gemm,
    )
