/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_dequantize_4over6_nvfp4.cuh
 *  \brief Grouped NVFP4 dequantization for tensors produced by the 4over6 recipe.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_4OVER6_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../utils.cuh"
#include "group_common_4over6_nvfp4.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif  // FP4_TYPE_SUPPORTED

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

#if FP4_TYPE_SUPPORTED

namespace group_dequantize_4over6_kernel {

constexpr int kThreads = 256;
constexpr int kFP4BlockSize = 16;

template <typename OType, bool ROW_SCALED_NVFP4, int E4M3_MAX>
__global__ void __launch_bounds__(kThreads)
    group_dequantize_fp4_kernel(const void *const input, OType *output,
                                const fp8e4m3 *const scales, const float *const amax,
                                const int64_t *const __restrict__ offsets, const size_t rows,
                                const size_t cols, const size_t num_tensors,
                                const size_t scale_stride, const bool has_first_dims) {
  const size_t mread = cols / kFP4BlockSize;
  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t x = thread_idx % mread;
  const size_t y = thread_idx / mread;

  if (y >= rows) {
    return;
  }

  union fp4vec {
    uint64_t vec;
    fp4e2m1x4 small_vec[4];
  };
  using OVec = Vec<OType, 4>;

  const auto *const input_vectorized = reinterpret_cast<const uint64_t *>(input);
  OVec *output_vec = reinterpret_cast<OVec *>(output);

  const size_t tensor_id = ROW_SCALED_NVFP4
                               ? 0
                               : group_4over6::tensor_id_from_row(y, rows, cols, num_tensors,
                                                                  has_first_dims, offsets);
  const size_t amax_idx = ROW_SCALED_NVFP4 ? y : tensor_id;

  const size_t my_index = x + y * mread;
  const size_t my_scale_index = x + y * scale_stride;
  const size_t my_output_index = my_index * 4;

  fp4vec value;
  value.vec = input_vectorized[my_index];
  const fp8e4m3 scale = scales[my_scale_index];
  const float tensor_amax = amax[amax_idx];
  static_assert(E4M3_MAX == 448 || E4M3_MAX == 256, "Unsupported NVFP4 E4M3 max.");
  constexpr float factor_inv = 1.0f / (6.0f * static_cast<float>(E4M3_MAX));
  const float final_scale = static_cast<float>(scale) * tensor_amax * factor_inv;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 current = static_cast<float4>(value.small_vec[i]);
    OVec out;
    out.data.elt[0] = static_cast<OType>(current.x * final_scale);
    out.data.elt[1] = static_cast<OType>(current.y * final_scale);
    out.data.elt[2] = static_cast<OType>(current.z * final_scale);
    out.data.elt[3] = static_cast<OType>(current.w * final_scale);
    output_vec[my_output_index + i] = out;
  }
}

template <typename OType, bool ROW_SCALED_NVFP4, int E4M3_MAX>
void launch_group_dequantize(const GroupedTensor *input, GroupedTensor *output,
                             cudaStream_t stream) {
  const size_t rows = input->logical_shape.data[0];
  const size_t cols = input->logical_shape.data[1];
  if (rows == 0 || cols == 0) {
    return;
  }

  const size_t mread = cols / kFP4BlockSize;
  const size_t total = rows * mread;
  const size_t blocks = DIVUP(total, static_cast<size_t>(kThreads));
  const size_t scale_stride = group_4over6::rowwise_scale_cols(cols);

  group_dequantize_fp4_kernel<OType, ROW_SCALED_NVFP4, E4M3_MAX>
      <<<blocks, kThreads, 0, stream>>>(
          input->data.dptr, reinterpret_cast<OType *>(output->data.dptr),
          reinterpret_cast<fp8e4m3 *>(input->scale_inv.dptr),
          reinterpret_cast<float *>(input->amax.dptr),
          reinterpret_cast<const int64_t *>(input->tensor_offsets.dptr), rows, cols,
          input->num_tensors, scale_stride, input->first_dims.dptr != nullptr);
}

}  // namespace group_dequantize_4over6_kernel

#endif  // FP4_TYPE_SUPPORTED

inline void group_dequantize_4over6(const GroupedTensor *input, GroupedTensor *output,
                                    cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace group_dequantize_4over6_kernel;

  checkCuDriverContext(stream);
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Grouped NVFP4 4over6 dequantize requires rowwise input data.");
  NVTE_CHECK(output->has_data(), "Grouped NVFP4 4over6 dequantize requires rowwise output data.");
  NVTE_CHECK(!input->has_columnwise_data() && !output->has_columnwise_data(),
             "Grouped NVFP4 4over6 dequantize currently supports rowwise data only.");
  NVTE_CHECK(!input->with_gemm_swizzled_scales,
             "Grouped NVFP4 4over6 dequantize requires compact scale layout.");
  NVTE_CHECK(is_fp4_dtype(input->data.dtype), "Grouped NVFP4 4over6 input must have FP4 type.");
  NVTE_CHECK(is_high_precision_dtype(output->data.dtype),
             "Grouped NVFP4 4over6 output must be in higher precision.");

  const auto logical_shape = group_4over6::logical_shape_2d(*input, "Grouped dequantize input");
  NVTE_CHECK(output->logical_shape.ndim == 2 &&
                 output->logical_shape.data[0] == logical_shape[0] &&
                 output->logical_shape.data[1] == logical_shape[1],
             "Grouped NVFP4 4over6 input and output logical shapes must match.");
  const size_t rows = logical_shape[0];
  const size_t cols = logical_shape[1];
  const auto scale_shape = group_4over6::rowwise_scale_shape(rows, cols);

  NVTE_CHECK(input->data.numel() == rows * cols / 2,
             "Grouped NVFP4 4over6 input rowwise data has wrong packed size.");
  NVTE_CHECK(output->data.numel() == rows * cols,
             "Grouped NVFP4 4over6 output rowwise data has wrong size.");
  NVTE_CHECK(input->scale_inv.dptr != nullptr,
             "Grouped NVFP4 4over6 dequantize requires rowwise scale_inv.");
  NVTE_CHECK(input->scale_inv.dtype == DType::kFloat8E4M3,
             "Grouped NVFP4 4over6 scale_inv must have Float8E4M3 dtype.");
  NVTE_CHECK(input->scale_inv.numel() == product(scale_shape),
             "Grouped NVFP4 4over6 scale_inv has wrong size.");
  NVTE_CHECK(input->amax.dptr != nullptr, "Grouped NVFP4 4over6 dequantize requires amax.");

  const ShapeRepresentation shape_rep = group_4over6::shape_representation(*input);
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM,
             "Grouped NVFP4 4over6 dequantize currently requires a constant last dimension.");
  NVTE_CHECK(shape_rep != ShapeRepresentation::SAME_BOTH_DIMS || rows % input->num_tensors == 0,
             "Grouped NVFP4 4over6 dequantize requires rows divisible by num_tensors when "
             "first_dims are not provided.");
  NVTE_CHECK(shape_rep != ShapeRepresentation::VARYING_FIRST_DIM || input->tensor_offsets.dptr,
             "Grouped NVFP4 4over6 dequantize requires tensor_offsets for varying first dims.");

  const bool row_scaled_nvfp4 = input->row_scaled_nvfp4;
  if (row_scaled_nvfp4) {
    NVTE_CHECK(input->amax.numel() == rows,
               "Row-scaled grouped NVFP4 4over6 dequantize requires one amax per row.");
  } else {
    NVTE_CHECK(input->amax.numel() == input->num_tensors,
               "Grouped NVFP4 4over6 dequantize requires one amax per tensor.");
  }

  const int e4m3_max = input->nvfp4_e4m3_max;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      output->data.dtype, OType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          row_scaled_nvfp4, ROW_SCALED_NVFP4,
          if (e4m3_max == 256) {
            launch_group_dequantize<OType, ROW_SCALED_NVFP4, 256>(input, output, stream);
          } else {
            NVTE_CHECK(e4m3_max == 448, "Unsupported NVFP4 E4M3 max (got ", e4m3_max, ")");
            launch_group_dequantize<OType, ROW_SCALED_NVFP4, 448>(input, output, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_4OVER6_NVFP4_CUH_
