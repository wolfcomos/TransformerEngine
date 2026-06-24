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
from transformer_engine.pytorch.quantization import NVFP4BlockScalingRecipeState, QuantizerRole
from transformer_engine.common.recipe import NVFP4BlockScaling

import pytest
import torch

from nvfp4_utils import (
    get_nvfp4_scale_shape_no_padding,
    generate_split_sections,
    assert_same_shape_and_dtype,
    reference_group_quantize,
    swizzle_nvfp4_scale,
)

# Reuse the 4over6 exact-test config matrix and err-fast-math env toggle so
# grouped quantization is checked against the same recipe combinations as the
# flat NVFP4 exact tests.
from test_nvfp4_quantize_exact import (
    NVFP4_4OVER6_CONFIGS,
    nvfp4_4over6_err_fast_math,
)

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "kind": "recipe_state",
                "rows": 256,
                "hidden": 512,
                "splits": [128, 128],
                "compute_amax": True,
                "quantize_mode": "both_directions",
                "with_2d_quantization": False,
                "precomputed_offsets": False,
                "err_use_fast_math": err_use_fast_math,
            },
            id=f"recipe_state-{'fast_err' if err_use_fast_math else 'exact_err'}",
        )
        for err_use_fast_math in (False, True)
    ]
    + [
        pytest.param(
            {
                "kind": "row_scaled_1d",
                "cfg": cfg,
                "rows": rows,
                "hidden": hidden,
                "splits": splits,
                "compute_amax": True,
                "quantize_mode": "rowwise_only",
                "with_2d_quantization": False,
                "precomputed_offsets": False,
            },
            id=f"row_scaled_1d-{shape_id}-{cfg.id}",
        )
        for cfg in NVFP4_4OVER6_CONFIGS
        if cfg.use_4over6
        for rows, hidden, splits, shape_id in [
            (256, 512, [128, 128], "aligned_no_padding"),
            (384, 320, [128, 128, 128], "hidden_padding"),
            (304, 512, [96, 208], "uneven_splits"),
            (640, 2064, [208, 432], "scale_col_padding"),
            (1024, 7168, [512, 512], "large"),
        ]
    ]
    + [
        pytest.param(
            {
                "kind": "tensor_scaled_1d",
                "cfg": cfg,
                "rows": rows,
                "hidden": hidden,
                "splits": splits,
                "compute_amax": compute_amax,
                "quantize_mode": quantize_mode,
                "with_2d_quantization": False,
                "precomputed_offsets": precomputed_offsets,
            },
            id=f"tensor_scaled_1d-{path_id}-{quantize_mode}-{shape_id}-{cfg.id}",
        )
        for cfg in NVFP4_4OVER6_CONFIGS
        if cfg.use_4over6
        for compute_amax, quantize_mode, precomputed_offsets, path_id in [
            (False, "rowwise_only", False, "with_amax"),
            (True, "rowwise_only", False, "compute_amax"),
            (False, "columnwise_only", False, "with_amax"),
            (False, "both_directions", False, "with_amax"),
            (True, "both_directions", False, "compute_amax"),
            (False, "both_directions", True, "precomputed_offsets"),
        ]
        for rows, hidden, splits, shape_id in [
            (384, 512, [128, 256], "regular"),
            (768, 384, [128, 256, 384], "uneven_128_aligned"),
            (128, 512, [128], "single_tensor"),
        ]
    ]
    + [
        pytest.param(
            {
                "kind": "tensor_scaled_2d",
                "cfg": cfg,
                "rows": rows,
                "hidden": hidden,
                "splits": splits,
                "compute_amax": compute_amax,
                "quantize_mode": quantize_mode,
                "with_2d_quantization": True,
                "precomputed_offsets": False,
            },
            id=f"tensor_scaled_2d-{path_id}-{quantize_mode}-{shape_id}-{cfg.id}",
        )
        for cfg in NVFP4_4OVER6_CONFIGS
        if cfg.use_4over6
        for compute_amax, quantize_mode, path_id in [
            (False, "rowwise_only", "with_amax"),
            (True, "rowwise_only", "compute_amax"),
            (False, "both_directions", "with_amax"),
            (True, "both_directions", "compute_amax"),
        ]
        for rows, hidden, splits, shape_id in [
            (384, 512, [128, 256], "regular"),
            (768, 384, [128, 256, 384], "uneven_128_aligned"),
        ]
    ],
)
def test_grouped_nvfp4_4over6_quantize_matches_split_nvfp4_4over6(case: dict) -> None:
    """Grouped NVFP4 4over6 quantization should bitwise match per-split NVFP4 4over6."""

    device = "cuda"
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    kind = case["kind"]
    cfg = case.get("cfg")
    rows = case["rows"]
    hidden = case["hidden"]
    splits = case["splits"]
    compute_amax = case["compute_amax"]
    quantize_mode = case["quantize_mode"]
    with_2d_quantization = case["with_2d_quantization"]
    precomputed_offsets = case["precomputed_offsets"]
    return_rowwise = quantize_mode != "columnwise_only"
    return_columnwise = quantize_mode != "rowwise_only"

    assert sum(splits) == rows and rows % 16 == 0 and hidden % 16 == 0
    assert all(split % 16 == 0 for split in splits)

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

    if kind == "recipe_state":
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
        err_use_fast_math = case["err_use_fast_math"]
    else:
        quantizer = NVFP4Quantizer(
            fp4_dtype=te.DType.kFloat4E2M1,
            rowwise=return_rowwise,
            columnwise=return_columnwise,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=with_2d_quantization,
            stochastic_rounding=False,
            row_scaled_nvfp4=kind == "row_scaled_1d",
            nvfp4_use_4over6=True,
            nvfp4_e4m3_max=cfg.e4m3_max,
            nvfp4_4over6_err_mode=cfg.err_mode,
        )
        err_use_fast_math = cfg.err_use_fast_math

    if kind == "row_scaled_1d":
        expected_amax = torch.cat([chunk.abs().amax(dim=1).float() for chunk in x_chunks])
    else:
        expected_amax = torch.stack([chunk.abs().amax().float() for chunk in x_chunks])

    with nvfp4_4over6_err_fast_math(err_use_fast_math):
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
        packed_offset = 0
        row_offset = 0
        scale_rows = ((rows + 127) // 128) * 128
        scale_cols = (((hidden // 16) + 3) // 4) * 4
        grouped_scales = grouped.scale_inv.contiguous().view(torch.uint8).reshape(
            scale_rows, scale_cols
        )
        for split, ref in zip(splits, refs):
            packed_elems = split * hidden // 2
            torch.testing.assert_close(
                grouped.rowwise_data[packed_offset : packed_offset + packed_elems],
                ref._rowwise_data.view(dtype=torch.uint8).reshape(-1),
                atol=0,
                rtol=0,
            )
            split_scale_rows = ((split + 127) // 128) * 128
            valid_ref_scales = ref._rowwise_scale_inv.view(torch.uint8).reshape(
                split_scale_rows, scale_cols
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
            [
                tex.dequantize(t, te.DType.kBFloat16)
                for t in grouped.split_into_quantized_tensors()
            ],
            dim=0,
        )
        ref_dequantized = torch.cat(
            [tex.dequantize(ref, te.DType.kBFloat16) for ref in refs],
            dim=0,
        )
        torch.testing.assert_close(
            grouped_dequantized,
            ref_dequantized,
            atol=0,
            rtol=0,
        )

    if return_columnwise:
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
            scale_rows = ((hidden + 127) // 128) * 128
            scale_cols = (((split // 16) + 3) // 4) * 4
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
