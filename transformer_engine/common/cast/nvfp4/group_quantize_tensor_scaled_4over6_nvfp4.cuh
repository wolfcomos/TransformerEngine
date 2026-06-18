/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_tensor_scaled_4over6_nvfp4.cuh
 *  \brief Grouped tensor-scaled NVFP4 4over6 quantization.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_TENSOR_SCALED_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_TENSOR_SCALED_4OVER6_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>

#include "../../common.h"
#include "../../utils.cuh"
#include "group_common_4over6_nvfp4.cuh"
#include "quantize_4over6_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

#if FP4_TYPE_SUPPORTED

namespace group_quantize_tensor_scaled_4over6_kernel {

using namespace quantize_4over6_kernel;

constexpr int kThreads = 256;

template <typename IType>
__device__ __forceinline__ void load_global_row_group(const IType *input, const size_t row,
                                                      const size_t cols, const size_t col,
                                                      float (&x0)[8], float (&x1)[8],
                                                      float *amax) {
  Vec<IType, kElementsPerHalfGroup> x0_vec;
  Vec<IType, kElementsPerHalfGroup> x1_vec;
  const IType *base = input + row * cols + col;
  x0_vec.load_from(base);
  x1_vec.load_from(base + kElementsPerHalfGroup);

  *amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kElementsPerHalfGroup; ++i) {
    const float v0 = static_cast<float>(x0_vec.data.elt[i]);
    const float v1 = static_cast<float>(x1_vec.data.elt[i]);
    x0[i] = v0;
    x1[i] = v1;
    *amax = fmaxf(*amax, fabsf(v0));
    *amax = fmaxf(*amax, fabsf(v1));
  }
}

template <typename Cfg, int E4M3_MAX, typename IType>
__global__ void __launch_bounds__(kThreads)
    group_quantize_tensor_scaled_4over6_kernel(
        const IType *__restrict__ input, fp4e2m1x2 *__restrict__ output,
        nvfp4_scale_t *__restrict__ scales, const float *__restrict__ amax,
        const int64_t *__restrict__ offsets, const size_t rows, const size_t cols,
        const size_t num_tensors, const size_t scale_stride, const bool has_first_dims,
        const float *__restrict__ noop) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t groups_per_row = cols / kGroupSize;
  const size_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row = group_idx / groups_per_row;
  const size_t col_group = group_idx - row * groups_per_row;
  if (row >= rows) {
    return;
  }
  const size_t col = col_group * kGroupSize;
  const size_t tensor_id =
      group_4over6::tensor_id_from_row(row, rows, cols, num_tensors, has_first_dims, offsets);
  const float global_amax = amax[tensor_id];

  float x0[kElementsPerHalfGroup];
  float x1[kElementsPerHalfGroup];
  float block_amax = 0.0f;
  load_global_row_group(input, row, cols, col, x0, x1, &block_amax);

  const ScalePair scale_pair = compute_scale_pair<E4M3_MAX>(block_amax, global_amax);
  const CandidatePair candidates = make_candidates<Cfg, E4M3_MAX>(x0, x1, scale_pair, global_amax);

  const bool pick_map4 = candidates.map4.err < candidates.map6.err;
  const nvfp4_scale_t selected_scale = select_scale(scale_pair, pick_map4);
  const uint32_t *selected = select_packed(candidates, pick_map4);

  scales[row * scale_stride + col_group] = selected_scale;
  store_packed_group(selected, &output[(row * cols + col) / 2]);
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <typename Cfg, int E4M3_MAX, typename IType>
void launch_group_quantize_tensor_scaled_4over6(const GroupedTensor *input, GroupedTensor *output,
                                                const Tensor *noop, cudaStream_t stream) {
  const size_t rows = input->logical_shape.data[0];
  const size_t cols = input->logical_shape.data[1];
  if (rows == 0 || cols == 0) {
    return;
  }

  const size_t groups = rows * (cols / kGroupSize);
  const dim3 grid(static_cast<unsigned int>(DIVUP(groups, static_cast<size_t>(kThreads))));
  const dim3 block(kThreads);
  const auto *input_ptr = reinterpret_cast<const IType *>(input->data.dptr);
  auto *output_ptr = reinterpret_cast<fp4e2m1x2 *>(output->data.dptr);
  auto *scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  const auto *amax_ptr = reinterpret_cast<const float *>(output->amax.dptr);
  const auto *offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const auto *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const size_t scale_stride = group_4over6::rowwise_scale_cols(cols);

  group_quantize_tensor_scaled_4over6_kernel<Cfg, E4M3_MAX, IType>
      <<<grid, block, 0, stream>>>(
      input_ptr, output_ptr, scales_ptr, amax_ptr, offsets_ptr, rows, cols, output->num_tensors,
      scale_stride, output->first_dims.dptr != nullptr, noop_ptr);
}

}  // namespace group_quantize_tensor_scaled_4over6_kernel

#endif  // FP4_TYPE_SUPPORTED

inline void group_quantize_tensor_scaled_4over6(const GroupedTensor *input, GroupedTensor *output,
                                                const QuantizationConfig *quant_config,
                                                cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace group_quantize_tensor_scaled_4over6_kernel;
  using namespace quantize_4over6_kernel;

  checkCuDriverContext(stream);
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(quant_config != nullptr &&
                 quant_config->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled,
             "Grouped NVFP4 4over6 quantization requires a non-disabled 4over6 mode.");
  NVTE_CHECK(!quant_config->nvfp4_2d_quantization,
             "Grouped NVFP4 4over6 quantization currently supports 1D rowwise quantization only.");
  NVTE_CHECK(!quant_config->stochastic_rounding,
             "Grouped NVFP4 4over6 quantization does not support stochastic rounding.");
  NVTE_CHECK(!output->row_scaled_nvfp4,
             "Use group_quantize_row_scaled_4over6 for row-scaled grouped 4over6.");
  NVTE_CHECK(input->has_data(), "Grouped NVFP4 4over6 quantization requires rowwise input data.");
  NVTE_CHECK(output->has_data(), "Grouped NVFP4 4over6 quantization requires rowwise output data.");
  NVTE_CHECK(!output->has_columnwise_data(),
             "Grouped NVFP4 4over6 quantization currently does not support columnwise output.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "Grouped NVFP4 4over6 quantization requires compact scale layout.");

  const auto logical_shape = group_4over6::logical_shape_2d(*input, "Grouped quantize input");
  NVTE_CHECK(output->logical_shape.ndim == 2 &&
                 output->logical_shape.data[0] == logical_shape[0] &&
                 output->logical_shape.data[1] == logical_shape[1],
             "Grouped NVFP4 4over6 input and output logical shapes must match.");
  const size_t rows = logical_shape[0];
  const size_t cols = logical_shape[1];
  const auto scale_shape = group_4over6::rowwise_scale_shape(rows, cols);

  NVTE_CHECK(input->data.numel() == rows * cols,
             "Grouped NVFP4 4over6 input rowwise data has wrong size.");
  NVTE_CHECK(output->data.numel() == rows * cols / 2,
             "Grouped NVFP4 4over6 output rowwise data has wrong packed size.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr,
             "Grouped NVFP4 4over6 quantization requires rowwise scale_inv.");
  NVTE_CHECK(output->scale_inv.dtype == DType::kFloat8E4M3,
             "Grouped NVFP4 4over6 scale_inv must have Float8E4M3 dtype.");
  NVTE_CHECK(output->scale_inv.numel() == product(scale_shape),
             "Grouped NVFP4 4over6 scale_inv has wrong size.");
  NVTE_CHECK(output->amax.dptr != nullptr,
             "Grouped NVFP4 4over6 quantization requires per-tensor amax.");
  NVTE_CHECK(output->amax.numel() == output->num_tensors,
             "Grouped NVFP4 4over6 quantization requires one amax per tensor.");

  const ShapeRepresentation shape_rep = group_4over6::shape_representation(*output);
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM,
             "Grouped NVFP4 4over6 quantization currently requires a constant last dimension.");
  NVTE_CHECK(shape_rep != ShapeRepresentation::SAME_BOTH_DIMS || rows % output->num_tensors == 0,
             "Grouped NVFP4 4over6 quantization requires rows divisible by num_tensors when "
             "first_dims are not provided.");
  NVTE_CHECK(shape_rep != ShapeRepresentation::VARYING_FIRST_DIM || output->tensor_offsets.dptr,
             "Grouped NVFP4 4over6 quantization requires tensor_offsets for varying first dims.");

  Tensor dummy_tensor;
  const Tensor *noop = &dummy_tensor;
  if (quant_config->noop_tensor != nullptr) {
    noop = convertNVTETensorCheck(quant_config->noop_tensor);
  }

  TRANSFORMER_ENGINE_NVFP4_4OVER6_E4M3_MAX_SWITCH(
      output->nvfp4_e4m3_max, E4M3_MAX,
      TRANSFORMER_ENGINE_NVFP4_4OVER6_MODE_SWITCH(
          quant_config->nvfp4_4over6_mode, MODE,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              quant_config->nvfp4_4over6_err_use_fast_math, ERR_USE_FAST_MATH, {
                using Cfg = quantize_4over6_kernel::Config<MODE, ERR_USE_FAST_MATH>;
                TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
                    input->dtype(), IType,
                    launch_group_quantize_tensor_scaled_4over6<Cfg, E4M3_MAX, IType>(
                        input, output, noop, stream););
              });););

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_TENSOR_SCALED_4OVER6_NVFP4_CUH_
