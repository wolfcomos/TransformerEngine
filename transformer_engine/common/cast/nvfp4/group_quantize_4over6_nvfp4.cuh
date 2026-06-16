/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_4over6_nvfp4.cuh
 *  \brief Fused row-scaled NVFP4 4over6 grouped quantization.
 *
 *  The grouped row-scaled NVFP4 path stores all groups contiguously as a flat
 *  [total_rows, hidden] tensor and uses one global amax per row. The reference
 *  path runs two kernels: compute_rowwise_amax (one FP32 amax per row) followed
 *  by quantize_4over6 (per-1x16-block map-to-4 / map-to-6 candidate selection).
 *
 *  This header fuses both into a single launch with one CTA per row. The CTA
 *  first reduces the row amax, then evaluates the 4over6 candidates for every
 *  1x16 group using the exact same registers-only helpers as quantize_4over6.
 *  Because the per-row amax is a max of absolute values, it is order
 *  independent and bit-identical to compute_rowwise_amax, so the fused output
 *  matches the two-launch path exactly for rowwise data, scale_inv and amax.
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
#include "../../util/math.h"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"
#include "quantize_4over6_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

#if FP4_TYPE_SUPPORTED

namespace group_quantize_4over6_kernel {

// Reuse the registers-only 4over6 candidate helpers (compute_scale_pair,
// make_candidates, select_scale, select_packed, store_packed_group, Config,
// kGroupSize, kWarpThreads, kElementsPerHalfGroup, ScalePair, CandidatePair).
using namespace quantize_4over6_kernel;

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

// Cached SM count, used to size WARPS_PER_ROW so a launch fills the device.
inline int fused_multiprocessor_count() {
  static const int count = [] {
    int device = 0;
    int value = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
      cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device);
    }
    return value > 0 ? value : 132;
  }();
  return count;
}

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
  // Out-of-range rows in the last block stay resident (no early return) so the
  // whole block still reaches the shared-memory barrier below; they merely skip
  // every global load/store.
  const bool active = row < rows;
  const IType *row_in = input + (active ? row : 0) * cols;

  // Pass 1: per-row amax (max of absolute values) across the row's threads.
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
    // Combine the per-warp maxima of each row through shared memory.
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

  // Pass 2: 4over6 quantize each 1x16 group of the row using row_amax as the
  // global amax (row-scaled NVFP4) and the per-group amax as the block amax.
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

// Raw-pointer launch entry point. Chooses WARPS_PER_ROW so the launch fills the
// GPU (rows * WARPS_PER_ROW warps) without spawning pass-2-idle threads (capped
// by the number of 1x16 groups), then dispatches the matching instantiation.
template <typename Cfg, int E4M3_MAX, typename IType>
void launch_fused_row_scaled_4over6(const IType *input, fp4e2m1x2 *output, nvfp4_scale_t *scales,
                                    float *amax, const float *noop, const size_t rows,
                                    const size_t cols, const size_t scale_stride,
                                    cudaStream_t stream) {
  if (rows == 0 || cols == 0) {
    return;
  }
  const size_t num_groups = cols / kGroupSize;
  const size_t goal_warps = static_cast<size_t>(fused_multiprocessor_count()) * 64;
  int warps_per_row = 1;
  // Grow warps/row (power of two, up to kFusedBlockWarps) while it both helps
  // fill the device and still has at least one group of work per added thread.
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

// Tensor-API overload used by the production dispatch path.
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
inline void group_quantize_4over6_row_scaled(const Tensor &input, Tensor *output,
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
                    group_quantize_4over6_kernel::launch_fused_row_scaled_4over6<Cfg, E4M3_MAX,
                                                                                 IType>(
                        input, noop, output, stream););
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
