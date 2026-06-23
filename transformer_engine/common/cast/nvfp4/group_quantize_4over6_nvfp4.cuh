/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_4OVER6_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <vector>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/math.h"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"
#include "quantize_4over6_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace group_4over6 {

constexpr size_t kGroupSize = 16;

inline std::vector<size_t> logical_shape_2d(const GroupedTensor &tensor, const char *name) {
  NVTE_CHECK(tensor.num_tensors > 0, name, " must contain at least one tensor.");
  NVTE_CHECK(tensor.logical_shape.ndim == 2, name, " must have 2D logical shape.");
  NVTE_CHECK(tensor.logical_shape.data[1] > 0, name, " must have positive last dimension.");
  return std::vector<size_t>{tensor.logical_shape.data[0], tensor.logical_shape.data[1]};
}

inline size_t rowwise_scale_cols(const size_t cols) {
  NVTE_CHECK(cols % kGroupSize == 0,
             "Grouped NVFP4 4over6 requires last dim divisible by ", kGroupSize, ".");
  return DIVUP_TO_MULTIPLE(cols / kGroupSize, static_cast<size_t>(4));
}

inline std::vector<size_t> rowwise_scale_shape(const size_t rows, const size_t cols) {
  return std::vector<size_t>{DIVUP_TO_MULTIPLE(rows, static_cast<size_t>(128)),
                             rowwise_scale_cols(cols)};
}

inline size_t columnwise_scale_cols(const size_t rows) {
  NVTE_CHECK(rows % kGroupSize == 0,
             "Grouped NVFP4 4over6 columnwise requires first dim divisible by ", kGroupSize, ".");
  return DIVUP_TO_MULTIPLE(rows / kGroupSize, static_cast<size_t>(4));
}

inline std::vector<size_t> columnwise_scale_shape(const size_t rows, const size_t cols) {
  return std::vector<size_t>{DIVUP_TO_MULTIPLE(cols, static_cast<size_t>(128)),
                             columnwise_scale_cols(rows)};
}

inline Tensor make_grouped_input_tensor_view(const GroupedTensor &grouped_input,
                                             const char *name) {
  const auto logical_shape = logical_shape_2d(grouped_input, name);
  NVTE_CHECK(grouped_input.data.dptr != nullptr, name, " rowwise data must be allocated.");
  NVTE_CHECK(grouped_input.data.numel() == logical_shape[0] * logical_shape[1], name,
             " rowwise data must have ", logical_shape[0] * logical_shape[1],
             " entries for logical shape ", logical_shape, ", got ", grouped_input.data.shape,
             ".");

  Tensor input_view;
  input_view.scaling_mode = grouped_input.scaling_mode;
  input_view.data = SimpleTensor(grouped_input.data.dptr, logical_shape, grouped_input.data.dtype);
  return input_view;
}

inline Tensor make_row_scaled_grouped_output_tensor_view(const GroupedTensor &grouped_output,
                                                         const char *name) {
  const auto logical_shape = logical_shape_2d(grouped_output, name);
  const size_t rows = logical_shape[0];
  const size_t cols = logical_shape[1];
  const auto scale_shape = rowwise_scale_shape(rows, cols);

  NVTE_CHECK(grouped_output.data.dptr != nullptr, name, " rowwise data must be allocated.");
  NVTE_CHECK(is_fp4_dtype(grouped_output.data.dtype), name, " rowwise data must have FP4 type.");
  NVTE_CHECK(grouped_output.data.numel() == rows * cols / 2, name, " rowwise data must have ",
             rows * cols / 2, " packed entries for logical shape ", logical_shape, ", got ",
             grouped_output.data.shape, ".");
  NVTE_CHECK(grouped_output.scale_inv.dptr != nullptr, name,
             " rowwise scale_inv must be allocated.");
  NVTE_CHECK(grouped_output.scale_inv.dtype == DType::kFloat8E4M3, name,
             " rowwise scale_inv must have Float8E4M3 dtype.");
  NVTE_CHECK(grouped_output.scale_inv.numel() == product(scale_shape), name,
             " rowwise scale_inv must have ", product(scale_shape),
             " entries for logical shape ", logical_shape, ", got ", grouped_output.scale_inv.shape,
             ".");
  NVTE_CHECK(grouped_output.amax.dptr != nullptr, name, " rowwise amax must be allocated.");
  NVTE_CHECK(grouped_output.amax.dtype == DType::kFloat32, name,
             " rowwise amax must have Float32 dtype.");
  NVTE_CHECK(grouped_output.amax.numel() == rows, name, " row-scaled amax must have ", rows,
             " entries, got ", grouped_output.amax.shape, ".");

  Tensor output_view;
  output_view.scaling_mode = grouped_output.scaling_mode;
  output_view.data =
      SimpleTensor(grouped_output.data.dptr, logical_shape, grouped_output.data.dtype);
  output_view.scale_inv =
      SimpleTensor(grouped_output.scale_inv.dptr, scale_shape, grouped_output.scale_inv.dtype);
  output_view.with_gemm_swizzled_scales = grouped_output.with_gemm_swizzled_scales;
  output_view.row_scaled_nvfp4 = grouped_output.row_scaled_nvfp4;
  output_view.nvfp4_e4m3_max = grouped_output.nvfp4_e4m3_max;
  output_view.amax =
      SimpleTensor(grouped_output.amax.dptr, std::vector<size_t>{rows}, grouped_output.amax.dtype);
  return output_view;
}

inline ShapeRepresentation shape_representation(const GroupedTensor &tensor) {
  if (tensor.all_same_shape()) return ShapeRepresentation::SAME_BOTH_DIMS;
  if (tensor.all_same_first_dim()) return ShapeRepresentation::VARYING_LAST_DIM;
  if (tensor.all_same_last_dim()) return ShapeRepresentation::VARYING_FIRST_DIM;
  if (tensor.varying_both_dims()) return ShapeRepresentation::VARYING_BOTH_DIMS;
  NVTE_ERROR("Invalid grouped tensor shape representation.");
}

__device__ __forceinline__ size_t tensor_id_from_row(
    const size_t row, const size_t rows, const size_t cols, const size_t num_tensors,
    const bool has_first_dims, const int64_t *const __restrict__ offsets) {
  if (!has_first_dims) {
    return row / (rows / num_tensors);
  }

  const size_t row_offset = row * cols;
  size_t low = 1;
  size_t hi = num_tensors;
  while (low < hi) {
    const size_t mid = low + (hi - low) / 2;
    if (static_cast<size_t>(offsets[mid]) <= row_offset) {
      low = mid + 1;
    } else {
      hi = mid;
    }
  }
  return low - 1;
}

__device__ __forceinline__ size_t tensor_start_row_from_id(
    const size_t tensor_id, const size_t rows, const size_t cols, const size_t num_tensors,
    const bool has_first_dims, const int64_t *const __restrict__ offsets) {
  return has_first_dims ? static_cast<size_t>(offsets[tensor_id]) / cols
                        : tensor_id * (rows / num_tensors);
}

__device__ __forceinline__ size_t tensor_rows_from_id(
    const size_t tensor_id, const size_t rows, const size_t cols, const size_t num_tensors,
    const bool has_first_dims, const int64_t *const __restrict__ offsets) {
  if (!has_first_dims) {
    return rows / num_tensors;
  }
  return (static_cast<size_t>(offsets[tensor_id + 1]) - static_cast<size_t>(offsets[tensor_id])) /
         cols;
}

__device__ __forceinline__ size_t columnwise_scale_cols_device(const size_t rows) {
  return (((rows / kGroupSize) + 3) / 4) * 4;
}

__device__ __forceinline__ size_t columnwise_scale_offset_from_id(
    const size_t tensor_id, const size_t rows, const size_t cols, const size_t num_tensors,
    const bool has_first_dims, const int64_t *const __restrict__ offsets) {
  const size_t scale_rows = ((cols + 127) / 128) * 128;
  if (!has_first_dims) {
    const size_t tensor_rows = rows / num_tensors;
    return tensor_id * scale_rows * columnwise_scale_cols_device(tensor_rows);
  }

  size_t offset = 0;
  for (size_t i = 0; i < tensor_id; ++i) {
    const size_t tensor_rows =
        (static_cast<size_t>(offsets[i + 1]) - static_cast<size_t>(offsets[i])) / cols;
    offset += scale_rows * columnwise_scale_cols_device(tensor_rows);
  }
  return offset;
}

}  // namespace group_4over6

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

template <typename IType>
__device__ __forceinline__ void load_global_col_group(const IType *input,
                                                      const size_t row_start, const size_t cols,
                                                      const size_t col, float (&x0)[8],
                                                      float (&x1)[8], float *amax) {
  *amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kElementsPerHalfGroup; ++i) {
    const float v0 = static_cast<float>(input[(row_start + i) * cols + col]);
    const float v1 =
        static_cast<float>(input[(row_start + kElementsPerHalfGroup + i) * cols + col]);
    x0[i] = v0;
    x1[i] = v1;
    *amax = fmaxf(*amax, fabsf(v0));
    *amax = fmaxf(*amax, fabsf(v1));
  }
}

template <typename Cfg, int E4M3_MAX>
__device__ __forceinline__ void store_quantized_group(float (&x0)[kElementsPerHalfGroup],
                                                      float (&x1)[kElementsPerHalfGroup],
                                                      const float block_amax,
                                                      const float global_amax,
                                                      nvfp4_scale_t *scale_out,
                                                      fp4e2m1x2 *output) {
  const ScalePair scale_pair = compute_scale_pair<E4M3_MAX>(block_amax, global_amax);
  const CandidatePair candidates = make_candidates<Cfg, E4M3_MAX>(x0, x1, scale_pair, global_amax);
  const bool pick_map4 = candidates.map4.err < candidates.map6.err;

  *scale_out = select_scale(scale_pair, pick_map4);
  store_packed_group(select_packed(candidates, pick_map4), output);
}

template <typename Cfg, int E4M3_MAX, typename IType>
__device__ __forceinline__ void quantize_group_rowwise(
    const IType *__restrict__ input, fp4e2m1x2 *__restrict__ output,
    nvfp4_scale_t *__restrict__ scales, const float *__restrict__ amax,
    const int64_t *__restrict__ offsets, const size_t rows, const size_t cols,
    const size_t num_tensors, const size_t scale_stride, const bool has_first_dims,
    const size_t group_idx) {
  const size_t groups_per_row = cols / kGroupSize;
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

  store_quantized_group<Cfg, E4M3_MAX>(x0, x1, block_amax, global_amax,
                                       &scales[row * scale_stride + col_group],
                                       &output[(row * cols + col) / 2]);
}

template <typename Cfg, int E4M3_MAX, typename IType>
__device__ __forceinline__ void quantize_group_colwise(
    const IType *__restrict__ input, fp4e2m1x2 *__restrict__ output_t,
    nvfp4_scale_t *__restrict__ scales_t, const float *__restrict__ amax,
    const int64_t *__restrict__ offsets, const size_t rows, const size_t cols,
    const size_t num_tensors, const bool has_first_dims, const size_t group_idx) {
  const size_t row_groups = rows / kGroupSize;
  const size_t col = group_idx / row_groups;
  const size_t row_group = group_idx - col * row_groups;
  if (col >= cols) {
    return;
  }
  const size_t row_start = row_group * kGroupSize;
  const size_t tensor_id =
      group_4over6::tensor_id_from_row(row_start, rows, cols, num_tensors, has_first_dims, offsets);
  const size_t tensor_row_start = group_4over6::tensor_start_row_from_id(
      tensor_id, rows, cols, num_tensors, has_first_dims, offsets);
  const size_t tensor_rows =
      group_4over6::tensor_rows_from_id(tensor_id, rows, cols, num_tensors, has_first_dims, offsets);
  const size_t local_row = row_start - tensor_row_start;
  if (local_row + kGroupSize > tensor_rows) {
    return;
  }
  const float global_amax = amax[tensor_id];

  float x0[kElementsPerHalfGroup];
  float x1[kElementsPerHalfGroup];
  float block_amax = 0.0f;
  load_global_col_group(input, row_start, cols, col, x0, x1, &block_amax);

  const size_t tensor_scale_offset = group_4over6::columnwise_scale_offset_from_id(
      tensor_id, rows, cols, num_tensors, has_first_dims, offsets);
  const size_t tensor_scale_stride = group_4over6::columnwise_scale_cols_device(tensor_rows);
  const size_t tensor_packed_offset = (tensor_row_start * cols) / 2;
  const size_t tensor_colwise_offset = tensor_packed_offset + (col * tensor_rows + local_row) / 2;
  store_quantized_group<Cfg, E4M3_MAX>(
      x0, x1, block_amax, global_amax,
      &scales_t[tensor_scale_offset + col * tensor_scale_stride + local_row / kGroupSize],
      &output_t[tensor_colwise_offset]);
}

template <bool RETURN_ROWWISE, bool RETURN_COLUMNWISE, typename Cfg, int E4M3_MAX, typename IType>
__global__ void __launch_bounds__(kThreads)
    group_quantize_4over6_kernel(
        const IType *__restrict__ input, fp4e2m1x2 *__restrict__ output,
        fp4e2m1x2 *__restrict__ output_t, nvfp4_scale_t *__restrict__ scales,
        nvfp4_scale_t *__restrict__ scales_t, const float *__restrict__ amax_rowwise,
        const float *__restrict__ amax_colwise, const int64_t *__restrict__ offsets,
        const size_t rows, const size_t cols, const size_t num_tensors,
        const size_t scale_stride, const bool has_first_dims, const float *__restrict__ noop) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (RETURN_ROWWISE) {
    quantize_group_rowwise<Cfg, E4M3_MAX>(input, output, scales, amax_rowwise, offsets, rows, cols,
                                          num_tensors, scale_stride, has_first_dims, group_idx);
  }

  if constexpr (RETURN_COLUMNWISE) {
    const float *columnwise_amax = amax_colwise;
    if (columnwise_amax == nullptr) {
      columnwise_amax = amax_rowwise;
    }
    quantize_group_colwise<Cfg, E4M3_MAX>(input, output_t, scales_t, columnwise_amax, offsets, rows,
                                          cols, num_tensors, has_first_dims, group_idx);
  }
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <bool RETURN_ROWWISE, bool RETURN_COLUMNWISE, typename Cfg, int E4M3_MAX, typename IType>
void launch_group_quantize_4over6(const GroupedTensor *input, GroupedTensor *output,
                                  const Tensor *noop, cudaStream_t stream) {
  const size_t rows = input->logical_shape.data[0];
  const size_t cols = input->logical_shape.data[1];
  if (rows == 0 || cols == 0) {
    return;
  }

  const size_t rowwise_groups = RETURN_ROWWISE ? rows * (cols / kGroupSize) : 0;
  const size_t columnwise_groups = RETURN_COLUMNWISE ? cols * (rows / kGroupSize) : 0;
  const size_t groups = rowwise_groups > columnwise_groups ? rowwise_groups : columnwise_groups;
  const dim3 grid(static_cast<unsigned int>(DIVUP(groups, static_cast<size_t>(kThreads))));
  const dim3 block(kThreads);
  const auto *input_ptr = reinterpret_cast<const IType *>(input->data.dptr);
  auto *output_ptr = reinterpret_cast<fp4e2m1x2 *>(output->data.dptr);
  auto *output_t_ptr = reinterpret_cast<fp4e2m1x2 *>(output->columnwise_data.dptr);
  auto *scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  auto *scales_t_ptr = reinterpret_cast<nvfp4_scale_t *>(output->columnwise_scale_inv.dptr);
  const auto *amax_rowwise_ptr = reinterpret_cast<const float *>(output->amax.dptr);
  const auto *amax_colwise_ptr = reinterpret_cast<const float *>(output->columnwise_amax.dptr);
  const auto *offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const auto *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const size_t scale_stride = RETURN_ROWWISE ? group_4over6::rowwise_scale_cols(cols) : 0;

  group_quantize_4over6_kernel<RETURN_ROWWISE, RETURN_COLUMNWISE, Cfg, E4M3_MAX, IType>
      <<<grid, block, 0, stream>>>(input_ptr, output_ptr, output_t_ptr, scales_ptr, scales_t_ptr,
                                   amax_rowwise_ptr, amax_colwise_ptr, offsets_ptr, rows, cols,
                                   output->num_tensors, scale_stride,
                                   output->first_dims.dptr != nullptr, noop_ptr);
}

constexpr int kFusedBlockWarps = 8;
constexpr int kFusedThreads = kFusedBlockWarps * kWarpThreads;
constexpr int kFusedWarpsPerRow = 1;

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

    store_quantized_group<Cfg, E4M3_MAX>(x0, x1, block_amax, row_amax,
                                         &scales[row * scale_stride + g],
                                         &output[(row * cols + col) / 2]);
  }
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <typename Cfg, int E4M3_MAX, typename IType>
void launch_fused_row_scaled_4over6(const IType *input, fp4e2m1x2 *output, nvfp4_scale_t *scales,
                                    float *amax, const float *noop, const size_t rows,
                                    const size_t cols, const size_t scale_stride,
                                    cudaStream_t stream) {
  if (rows == 0 || cols == 0) {
    return;
  }
  constexpr int kRowsPerBlock = kFusedBlockWarps / kFusedWarpsPerRow;
  const dim3 grid(static_cast<unsigned int>(DIVUP(rows, static_cast<size_t>(kRowsPerBlock))));
  const dim3 block(kFusedThreads);
  fused_row_scaled_4over6_kernel<kFusedWarpsPerRow, Cfg, E4M3_MAX, IType>
      <<<grid, block, 0, stream>>>(input, output, scales, amax, rows, cols, scale_stride, noop);
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
             "Grouped NVFP4 4over6 quantization currently supports 1D quantization only.");
  NVTE_CHECK(!quant_config->stochastic_rounding,
             "Grouped NVFP4 4over6 quantization does not support stochastic rounding.");
  NVTE_CHECK(!output->row_scaled_nvfp4,
             "Use group_quantize_row_scaled_4over6 for row-scaled grouped 4over6.");
  NVTE_CHECK(input->has_data(), "Grouped NVFP4 4over6 quantization requires rowwise input data.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Grouped NVFP4 4over6 quantization requires rowwise or columnwise output data.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "Grouped NVFP4 4over6 quantization requires compact scale layout.");

  const auto logical_shape = group_4over6::logical_shape_2d(*input, "Grouped quantize input");
  NVTE_CHECK(output->logical_shape.ndim == 2 &&
                 output->logical_shape.data[0] == logical_shape[0] &&
                 output->logical_shape.data[1] == logical_shape[1],
             "Grouped NVFP4 4over6 input and output logical shapes must match.");
  const size_t rows = logical_shape[0];
  const size_t cols = logical_shape[1];
  const bool return_rowwise = output->has_data();
  const bool return_columnwise = output->has_columnwise_data();

  NVTE_CHECK(input->data.numel() == rows * cols,
             "Grouped NVFP4 4over6 input rowwise data has wrong size.");
  if (return_rowwise) {
    NVTE_CHECK(output->data.numel() == rows * cols / 2,
               "Grouped NVFP4 4over6 output rowwise data has wrong packed size.");
    NVTE_CHECK(output->scale_inv.dptr != nullptr,
               "Grouped NVFP4 4over6 quantization requires rowwise scale_inv.");
    NVTE_CHECK(output->scale_inv.dtype == DType::kFloat8E4M3,
               "Grouped NVFP4 4over6 rowwise scale_inv must have Float8E4M3 dtype.");
    NVTE_CHECK(output->scale_inv.numel() == product(group_4over6::rowwise_scale_shape(rows, cols)),
               "Grouped NVFP4 4over6 rowwise scale_inv has wrong size.");
    NVTE_CHECK(is_fp4_dtype(output->data.dtype),
               "Grouped NVFP4 4over6 rowwise output must have FP4 type.");
    NVTE_CHECK(output->amax.dptr != nullptr,
               "Grouped NVFP4 4over6 rowwise quantization requires per-tensor amax.");
    NVTE_CHECK(output->amax.numel() == output->num_tensors,
               "Grouped NVFP4 4over6 rowwise quantization requires one amax per tensor.");
  }
  if (return_columnwise) {
    NVTE_CHECK(rows % kGroupSize == 0,
               "Grouped NVFP4 4over6 columnwise quantization requires first dim divisible by ",
               kGroupSize, ".");
    NVTE_CHECK(output->columnwise_data.numel() == rows * cols / 2,
               "Grouped NVFP4 4over6 output columnwise data has wrong packed size.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Grouped NVFP4 4over6 quantization requires columnwise scale_inv.");
    NVTE_CHECK(output->columnwise_scale_inv.dtype == DType::kFloat8E4M3,
               "Grouped NVFP4 4over6 columnwise scale_inv must have Float8E4M3 dtype.");
    NVTE_CHECK(output->columnwise_scale_inv.numel() ==
                   product(group_4over6::columnwise_scale_shape(rows, cols)),
               "Grouped NVFP4 4over6 columnwise scale_inv has wrong size.");
    NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
               "Grouped NVFP4 4over6 columnwise output must have FP4 type.");
    NVTE_CHECK(output->columnwise_amax.dptr != nullptr || output->amax.dptr != nullptr,
               "Grouped NVFP4 4over6 columnwise quantization requires columnwise amax or "
               "rowwise amax.");
    if (output->columnwise_amax.dptr != nullptr) {
      NVTE_CHECK(output->columnwise_amax.numel() == output->num_tensors,
                 "Grouped NVFP4 4over6 columnwise quantization requires one amax per tensor.");
    }
  }

  const ShapeRepresentation shape_rep = group_4over6::shape_representation(*output);
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM,
             "Grouped NVFP4 4over6 quantization currently requires a constant last dimension.");
  NVTE_CHECK(shape_rep != ShapeRepresentation::SAME_BOTH_DIMS || rows % output->num_tensors == 0,
             "Grouped NVFP4 4over6 quantization requires rows divisible by num_tensors when "
             "first_dims are not provided.");
  NVTE_CHECK(shape_rep != ShapeRepresentation::SAME_BOTH_DIMS || !return_columnwise ||
                 (rows / output->num_tensors) % kGroupSize == 0,
             "Grouped NVFP4 4over6 columnwise quantization requires each tensor first dim "
             "divisible by ",
             kGroupSize, " when first_dims are not provided.");
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
                    input->dtype(), IType, {
                      TRANSFORMER_ENGINE_SWITCH_CONDITION(return_rowwise, RETURN_ROWWISE, {
                        TRANSFORMER_ENGINE_SWITCH_CONDITION(return_columnwise, RETURN_COLUMNWISE, {
                          launch_group_quantize_4over6<RETURN_ROWWISE, RETURN_COLUMNWISE, Cfg,
                                                       E4M3_MAX, IType>(input, output, noop,
                                                                        stream);
                        });
                      });
                    });
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
