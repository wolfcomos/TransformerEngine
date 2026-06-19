/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_4over6_nvfp4.cuh
 *  \brief Grouped NVFP4 4over6 quantization, including row-scaled fused quantization.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_4OVER6_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/math.h"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"
#include "group_common_4over6_nvfp4.cuh"
#include "quantize_4over6_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

#if FP4_TYPE_SUPPORTED

namespace group_quantize_4over6_kernel {

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
    group_quantize_4over6_kernel(
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
void launch_group_quantize_4over6(const GroupedTensor *input, GroupedTensor *output,
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

  group_quantize_4over6_kernel<Cfg, E4M3_MAX, IType><<<grid, block, 0, stream>>>(
      input_ptr, output_ptr, scales_ptr, amax_ptr, offsets_ptr, rows, cols, output->num_tensors,
      scale_stride, output->first_dims.dptr != nullptr, noop_ptr);
}

// A group of WARPS_PER_ROW warps cooperates on one row; a fixed 256-thread CTA
// (kFusedBlockWarps warps) therefore holds kFusedBlockWarps / WARPS_PER_ROW
// rows. Per-row work (amax reduction + per-1x16-group 4over6 quantization)
// stays warp-local, so threads are fully utilized for any hidden size, and
// WARPS_PER_ROW is chosen at launch so the total warp count (rows *
// WARPS_PER_ROW) fills the GPU even for small row counts -- the regime where
// occupancy, not per-row work, is the bottleneck. WARPS_PER_ROW == 1 needs no
// cross-warp coordination (the fast path for large row counts); larger values
// add one tiny shared-memory amax combine per row.
constexpr int kFusedBlockWarps = 8;
constexpr int kFusedThreads = kFusedBlockWarps * kWarpThreads;

// Each row is processed by WARPS_PER_ROW cooperating warps (kRowThreads
// threads). Pass 1 reduces the per-row amax across those threads; pass 2
// quantizes every 1x16 group with 4over6 candidate selection. The input row is
// read twice from global memory, which still replaces two kernel launches
// (compute_rowwise_amax + quantize_4over6) with one. max-of-abs is associative
// and commutative and BF16/FP16 -> FP32 is exact, so the result is bit-identical
// to the two-launch path regardless of how the reduction is split across warps.
template <int WARPS_PER_ROW, typename Cfg, int E4M3_MAX, typename IType>
__global__ void __launch_bounds__(kFusedThreads)
    fused_row_scaled_4over6_kernel(const IType *__restrict__ input,
                                   fp4e2m1x2 *__restrict__ output,
                                   nvfp4_scale_t *__restrict__ scales,
                                   float *__restrict__ amax_out, const size_t rows,
                                   const size_t cols, const size_t scale_stride,
                                   const float *__restrict__ noop) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  constexpr int kRowsPerBlock = kFusedBlockWarps / WARPS_PER_ROW;
  constexpr int kRowThreads = WARPS_PER_ROW * kWarpThreads;

  const int lane = threadIdx.x % kWarpThreads;
  const int warp_id = threadIdx.x / kWarpThreads;
  const int warp_in_row = warp_id % WARPS_PER_ROW;
  const int row_in_block = warp_id / WARPS_PER_ROW;
  const int row_thread = warp_in_row * kWarpThreads + lane;
  const size_t row = static_cast<size_t>(blockIdx.x) * kRowsPerBlock + row_in_block;
  const bool active = row < rows;
  const IType *row_in = input + (active ? row : 0) * cols;

  constexpr int kVecElems = 16 / sizeof(IType);
  const size_t num_vecs = cols / kVecElems;
  float thread_amax = 0.0f;
  if (active) {
    for (size_t v = row_thread; v < num_vecs; v += kRowThreads) {
      Vec<IType, kVecElems> vec;
      vec.load_from(row_in + v * kVecElems);
#pragma unroll
      for (int e = 0; e < kVecElems; ++e) {
        thread_amax = fmaxf(thread_amax, fabsf(static_cast<float>(vec.data.elt[e])));
      }
    }
  }
  float row_amax = warp_reduce_max_broadcast(thread_amax);
  if constexpr (WARPS_PER_ROW > 1) {
    __shared__ float row_amax_smem[kRowsPerBlock][WARPS_PER_ROW];
    if (lane == 0) {
      row_amax_smem[row_in_block][warp_in_row] = row_amax;
    }
    __syncthreads();
    float combined = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS_PER_ROW; ++w) {
      combined = fmaxf(combined, row_amax_smem[row_in_block][w]);
    }
    row_amax = combined;
  }
  if (active && row_thread == 0) {
    amax_out[row] = row_amax;
  }
  if (!active) {
    return;
  }

  const size_t num_groups = cols / kGroupSize;
  for (size_t g = row_thread; g < num_groups; g += kRowThreads) {
    const size_t col = g * kGroupSize;

    Vec<IType, kElementsPerHalfGroup> x0_vec;
    Vec<IType, kElementsPerHalfGroup> x1_vec;
    x0_vec.load_from(row_in + col);
    x1_vec.load_from(row_in + col + kElementsPerHalfGroup);

    float x0[kElementsPerHalfGroup];
    float x1[kElementsPerHalfGroup];
    float block_amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kElementsPerHalfGroup; ++i) {
      const float v0 = static_cast<float>(x0_vec.data.elt[i]);
      const float v1 = static_cast<float>(x1_vec.data.elt[i]);
      x0[i] = v0;
      x1[i] = v1;
      block_amax = fmaxf(block_amax, fabsf(v0));
      block_amax = fmaxf(block_amax, fabsf(v1));
    }

    const ScalePair scale_pair = compute_scale_pair<E4M3_MAX>(block_amax, row_amax);
    const CandidatePair candidates = make_candidates<Cfg, E4M3_MAX>(x0, x1, scale_pair, row_amax);

    const bool pick_map4 = candidates.map4.err < candidates.map6.err;
    const nvfp4_scale_t selected_scale = select_scale(scale_pair, pick_map4);
    const uint32_t *selected = select_packed(candidates, pick_map4);

    scales[row * scale_stride + g] = selected_scale;
    store_packed_group(selected, &output[(row * cols + col) / 2]);
  }
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <int WARPS_PER_ROW, typename Cfg, int E4M3_MAX, typename IType>
void launch_fused_row_scaled_4over6_one(const IType *input, fp4e2m1x2 *output,
                                        nvfp4_scale_t *scales, float *amax, const float *noop,
                                        const size_t rows, const size_t cols,
                                        const size_t scale_stride, cudaStream_t stream) {
  constexpr int kRowsPerBlock = kFusedBlockWarps / WARPS_PER_ROW;
  const dim3 grid(static_cast<unsigned int>(DIVUP(rows, static_cast<size_t>(kRowsPerBlock))));
  const dim3 block(kFusedThreads);
  fused_row_scaled_4over6_kernel<WARPS_PER_ROW, Cfg, E4M3_MAX, IType>
      <<<grid, block, 0, stream>>>(input, output, scales, amax, rows, cols, scale_stride, noop);
}

template <typename Cfg, int E4M3_MAX, typename IType>
void launch_fused_row_scaled_4over6(const IType *input, fp4e2m1x2 *output, nvfp4_scale_t *scales,
                                    float *amax, const float *noop, const size_t rows,
                                    const size_t cols, const size_t scale_stride,
                                    cudaStream_t stream) {
  if (rows == 0 || cols == 0) {
    return;
  }
  const size_t num_groups = cols / kGroupSize;
  const size_t goal_warps = static_cast<size_t>(transformer_engine::cuda::sm_count()) * 64;
  int warps_per_row = 1;
  while (warps_per_row < kFusedBlockWarps &&
         rows * static_cast<size_t>(warps_per_row) < goal_warps &&
         static_cast<size_t>(warps_per_row) * 2 * kWarpThreads <= num_groups) {
    warps_per_row *= 2;
  }
  switch (warps_per_row) {
    case 8:
      launch_fused_row_scaled_4over6_one<8, Cfg, E4M3_MAX, IType>(
          input, output, scales, amax, noop, rows, cols, scale_stride, stream);
      break;
    case 4:
      launch_fused_row_scaled_4over6_one<4, Cfg, E4M3_MAX, IType>(
          input, output, scales, amax, noop, rows, cols, scale_stride, stream);
      break;
    case 2:
      launch_fused_row_scaled_4over6_one<2, Cfg, E4M3_MAX, IType>(
          input, output, scales, amax, noop, rows, cols, scale_stride, stream);
      break;
    default:
      launch_fused_row_scaled_4over6_one<1, Cfg, E4M3_MAX, IType>(
          input, output, scales, amax, noop, rows, cols, scale_stride, stream);
      break;
  }
}

template <typename Cfg, int E4M3_MAX, typename IType>
void launch_fused_row_scaled_4over6(const Tensor &input, const Tensor *noop, Tensor *output,
                                    cudaStream_t stream) {
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  if (rows == 0 || cols == 0) {
    return;
  }

  const auto *input_ptr = reinterpret_cast<const IType *>(input.data.dptr);
  auto *output_ptr = reinterpret_cast<fp4e2m1x2 *>(output->data.dptr);
  auto *scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  auto *amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  const auto *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const size_t scale_stride = output->scale_inv.shape[1];

  launch_fused_row_scaled_4over6<Cfg, E4M3_MAX, IType>(input_ptr, output_ptr, scales_ptr, amax_ptr,
                                                       noop_ptr, rows, cols, scale_stride, stream);
}

}  // namespace group_quantize_4over6_kernel

#endif  // FP4_TYPE_SUPPORTED

// Fused row-scaled NVFP4 4over6 grouped quantization. The grouped storage is
// expressed as a flat [total_rows, hidden] tensor; the kernel is group-agnostic
// because row-scaled 4over6 quantization has no cross-group dependency.
inline void group_quantize_row_scaled_4over6(const Tensor &input, Tensor *output,
                                             const QuantizationConfig *quant_config,
                                             cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_4over6_kernel;
  using namespace group_quantize_4over6_kernel;

  checkCuDriverContext(stream);
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(quant_config != nullptr, "Fused grouped 4over6 quantization requires a config.");
  NVTE_CHECK(output->row_scaled_nvfp4,
             "Fused grouped 4over6 quantization requires a row-scaled NVFP4 output.");
  NVTE_CHECK(output->has_data(), "Fused grouped 4over6 quantization requires rowwise output data.");
  NVTE_CHECK(!output->has_columnwise_data(),
             "Fused grouped 4over6 quantization does not support columnwise output.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "Fused grouped 4over6 quantization requires compact scale layout.");
  NVTE_CHECK(quant_config->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled,
             "Fused grouped 4over6 quantization requires a non-disabled 4over6 mode.");
  NVTE_CHECK(!quant_config->nvfp4_2d_quantization,
             "Fused grouped 4over6 quantization does not support 2D quantization.");
  NVTE_CHECK(!quant_config->stochastic_rounding,
             "Fused grouped 4over6 quantization does not support stochastic rounding.");
  NVTE_CHECK(input.flat_last_dim() % kGroupSize == 0,
             "Fused grouped 4over6 quantization requires last dim divisible by ", kGroupSize, ".");

  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  NVTE_CHECK(output->amax.dptr != nullptr, "Rowwise amax tensor must be allocated.");
  NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  const size_t rows = input.flat_first_dim();
  NVTE_CHECK(output->amax.numel() == rows, "Row-scaled rowwise amax must have ", rows,
             " entries, got ", output->amax.shape, ".");

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
                    input.dtype(), IType,
                    launch_fused_row_scaled_4over6<Cfg, E4M3_MAX, IType>(
                        input, noop, output, stream););
              });););

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

inline void group_quantize_4over6(const GroupedTensor *input, GroupedTensor *output,
                                  const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace group_quantize_4over6_kernel;
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
                    launch_group_quantize_4over6<Cfg, E4M3_MAX, IType>(
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

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_4OVER6_NVFP4_CUH_
