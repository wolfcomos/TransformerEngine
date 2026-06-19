/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_common_4over6_nvfp4.cuh
 *  \brief Shared helpers for grouped NVFP4 4over6 recipe paths.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_COMMON_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_COMMON_4OVER6_NVFP4_CUH_

#include <transformer_engine/transformer_engine.h>

#include <vector>

#include "../../common.h"
#include "../../utils.cuh"

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

inline size_t rowwise_scale_rows(const size_t rows) {
  return DIVUP_TO_MULTIPLE(rows, static_cast<size_t>(128));
}

inline size_t rowwise_scale_cols(const size_t cols) {
  NVTE_CHECK(cols % kGroupSize == 0,
             "Grouped NVFP4 4over6 requires last dim divisible by ", kGroupSize, ".");
  return DIVUP_TO_MULTIPLE(cols / kGroupSize, static_cast<size_t>(4));
}

inline std::vector<size_t> rowwise_scale_shape(const size_t rows, const size_t cols) {
  return std::vector<size_t>{rowwise_scale_rows(rows), rowwise_scale_cols(cols)};
}

inline size_t columnwise_scale_rows(const size_t cols) {
  return rowwise_scale_rows(cols);
}

inline size_t columnwise_scale_cols(const size_t rows) {
  NVTE_CHECK(rows % kGroupSize == 0,
             "Grouped NVFP4 4over6 columnwise requires first dim divisible by ", kGroupSize, ".");
  return DIVUP_TO_MULTIPLE(rows / kGroupSize, static_cast<size_t>(4));
}

inline std::vector<size_t> columnwise_scale_shape(const size_t rows, const size_t cols) {
  return std::vector<size_t>{columnwise_scale_rows(cols), columnwise_scale_cols(rows)};
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

inline Tensor make_grouped_rowwise_output_tensor_view(const GroupedTensor &grouped_output,
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

  Tensor output_view;
  output_view.scaling_mode = grouped_output.scaling_mode;
  output_view.data =
      SimpleTensor(grouped_output.data.dptr, logical_shape, grouped_output.data.dtype);
  output_view.scale_inv =
      SimpleTensor(grouped_output.scale_inv.dptr, scale_shape, grouped_output.scale_inv.dtype);
  output_view.with_gemm_swizzled_scales = grouped_output.with_gemm_swizzled_scales;
  output_view.row_scaled_nvfp4 = grouped_output.row_scaled_nvfp4;
  output_view.nvfp4_e4m3_max = grouped_output.nvfp4_e4m3_max;
  return output_view;
}

inline Tensor make_row_scaled_grouped_output_tensor_view(const GroupedTensor &grouped_output,
                                                         const char *name) {
  Tensor output_view = make_grouped_rowwise_output_tensor_view(grouped_output, name);
  const size_t rows = output_view.flat_first_dim();
  NVTE_CHECK(grouped_output.amax.numel() == rows, name, " row-scaled amax must have ", rows,
             " entries, got ", grouped_output.amax.shape, ".");
  output_view.amax =
      SimpleTensor(grouped_output.amax.dptr, std::vector<size_t>{rows}, grouped_output.amax.dtype);
  return output_view;
}

inline ShapeRepresentation shape_representation(const GroupedTensor &tensor) {
  if (tensor.all_same_shape()) {
    return ShapeRepresentation::SAME_BOTH_DIMS;
  }
  if (tensor.all_same_first_dim()) {
    return ShapeRepresentation::VARYING_LAST_DIM;
  }
  if (tensor.all_same_last_dim()) {
    return ShapeRepresentation::VARYING_FIRST_DIM;
  }
  if (tensor.varying_both_dims()) {
    return ShapeRepresentation::VARYING_BOTH_DIMS;
  }
  NVTE_ERROR("Invalid grouped tensor shape representation.");
}

__device__ __forceinline__ size_t tensor_id_from_row(
    const size_t row, const size_t rows, const size_t cols, const size_t num_tensors,
    const bool has_first_dims, const int64_t *const __restrict__ offsets) {
  if (!has_first_dims) {
    const size_t rows_per_tensor = rows / num_tensors;
    return row / rows_per_tensor;
  }

  const size_t row_offset = row * cols;
  size_t low = 1;
  size_t hi = num_tensors;
  while (low < hi) {
    const size_t mid = low + (hi - low) / 2;
    const size_t mid_offset = static_cast<size_t>(offsets[mid]);
    if (mid_offset <= row_offset) {
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
  if (!has_first_dims) {
    return tensor_id * (rows / num_tensors);
  }
  return static_cast<size_t>(offsets[tensor_id]) / cols;
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

__device__ __forceinline__ size_t columnwise_scale_rows_device(const size_t cols) {
  return ((cols + 127) / 128) * 128;
}

__device__ __forceinline__ size_t columnwise_scale_cols_device(const size_t rows) {
  return (((rows / kGroupSize) + 3) / 4) * 4;
}

__device__ __forceinline__ size_t columnwise_scale_offset_from_id(
    const size_t tensor_id, const size_t rows, const size_t cols, const size_t num_tensors,
    const bool has_first_dims, const int64_t *const __restrict__ offsets) {
  const size_t scale_rows = columnwise_scale_rows_device(cols);
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
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_COMMON_4OVER6_NVFP4_CUH_
