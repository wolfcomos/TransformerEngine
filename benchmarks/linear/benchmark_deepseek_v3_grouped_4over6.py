#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepSeek V3 GroupedLinear benchmark for NVFP4 4over6 group quantization.

Timing granularity: end-to-end GroupedLinear forward + backward, including
quantization and GEMMs.  This is not an isolated quantize-kernel benchmark.

This is a benchmark-only adapter.  In TE today, backward_override="high_precision"
disables the native grouped-tensor module path, so this script wraps
tex.split_quantize.  The "split" path keeps the legacy split path when possible
and uses a Python loop over flat quantizers for 2D activation quantization.  The
"group" path routes activation split quantization through tex.group_quantize and
then splits the GroupedTensor back to the per-expert tensor list expected by the
high-precision path.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import statistics
import time
from typing import Iterable

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe
from transformer_engine.pytorch.quantization import FP8GlobalStateManager, autocast
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
import transformer_engine_torch as tex


FLAT_4OVER6_KERNEL = "quantize_4over6_kernel"
TENSOR_SCALED_GROUP_KERNEL = "group_quantize_4over6_kernel"
ROWSCALED_GROUP_KERNEL = "group_quantize_row_scaled_4over6_kernel"


@dataclasses.dataclass
class QuantPathStats:
    split_quantize_calls: int = 0
    loop_quantize_calls: int = 0
    group_quantize_calls: int = 0


def _parse_splits(raw: str | None, total_m: int, num_gemms: int) -> list[int]:
    if raw:
        splits = [int(v) for v in raw.split(",") if v]
        if sum(splits) != total_m:
            raise ValueError(f"--splits sums to {sum(splits)}, expected --m={total_m}")
        return splits
    if total_m % num_gemms != 0:
        raise ValueError("--m must be divisible by --num-gemms when --splits is omitted")
    return [total_m // num_gemms] * num_gemms


def _split_sections_to_list(split_sections: Iterable[int] | torch.Tensor) -> list[int]:
    if isinstance(split_sections, torch.Tensor):
        return [int(v) for v in split_sections.detach().cpu().tolist()]
    return [int(v) for v in split_sections]


def _make_recipe(recipe_name: str) -> te_recipe.NVFP4BlockScaling:
    common = dict(
        disable_rht=True,
        disable_stochastic_rounding=True,
        nvfp4_4over6="all",
        backward_override="high_precision",
    )

    if recipe_name == "tensor_scaled_2d":
        recipe = te_recipe.NVFP4BlockScaling(**common)
        recipe.fp4_quant_fwd_inp = te_recipe.QParams(fp4_2d_quantization=True)
        recipe.fp4_quant_fwd_weight = te_recipe.QParams(fp4_2d_quantization=True)
        recipe.fp4_quant_bwd_grad = te_recipe.QParams()
        return recipe

    if recipe_name == "row_scaled_1d":
        recipe = te_recipe.NVFP4BlockScaling(
            **common,
            disable_2d_quantization=True,
            row_scaled_activation=True,
        )
        recipe.fp4_quant_fwd_inp = te_recipe.QParams()
        recipe.fp4_quant_fwd_weight = te_recipe.QParams()
        recipe.fp4_quant_bwd_grad = te_recipe.QParams()
        return recipe

    raise ValueError(f"Unknown recipe: {recipe_name}")


def _is_row_scaled_quantizer(quantizer: NVFP4Quantizer) -> bool:
    return bool(getattr(quantizer, "row_scaled_nvfp4", False))


def _is_2d_quantizer(quantizer: NVFP4Quantizer) -> bool:
    return bool(getattr(quantizer, "with_2d_quantization", False))


def _check_group_quant_supported(
    tensor: torch.Tensor,
    split_sections: list[int],
    quantizers: list[object],
) -> NVFP4Quantizer:
    if not quantizers:
        raise RuntimeError("split_quantize was called without quantizers")
    if not all(isinstance(q, NVFP4Quantizer) for q in quantizers):
        names = [type(q).__name__ for q in quantizers]
        raise RuntimeError(f"group path only supports NVFP4Quantizer, got {names}")

    quantizer = quantizers[0]
    if _is_row_scaled_quantizer(quantizer):
        return quantizer

    if tensor.shape[-1] % 128 != 0 or any(split % 128 != 0 for split in split_sections):
        raise RuntimeError(
            "Tensor-scaled grouped NVFP4 4over6 compute-amax currently requires hidden and every "
            "split section to be 128-aligned. Use DeepSeek-like aligned splits or benchmark "
            "the split path for this shape."
        )
    return quantizer


def _loop_flat_quantize_2d(
    tensor: torch.Tensor,
    split_sections: list[int],
    quantizers: Iterable[NVFP4Quantizer],
) -> list[object]:
    chunks = torch.split(tensor, split_sections)
    return [quantizer(chunk) for quantizer, chunk in zip(quantizers, chunks)]


@contextlib.contextmanager
def _quant_path(path: str, stats: QuantPathStats):
    original_split_quantize = tex.split_quantize

    def split_path(tensor, split_sections, quantizers, *args, **kwargs):
        stats.split_quantize_calls += 1
        split_list = _split_sections_to_list(split_sections)
        quantizer_list = list(quantizers)
        if all(isinstance(q, NVFP4Quantizer) and _is_2d_quantizer(q) for q in quantizer_list):
            stats.loop_quantize_calls += 1
            return _loop_flat_quantize_2d(tensor, split_list, quantizer_list)
        return original_split_quantize(tensor, split_sections, quantizers, *args, **kwargs)

    def group_path(tensor, split_sections, quantizers, *args, **kwargs):
        stats.split_quantize_calls += 1
        split_list = _split_sections_to_list(split_sections)
        quantizer = _check_group_quant_supported(tensor, split_list, list(quantizers))
        split_tensor = torch.tensor(split_list, dtype=torch.int64, device=tensor.device)
        grouped = tex.group_quantize(tensor, quantizer, len(split_list), split_tensor)
        stats.group_quantize_calls += 1
        return grouped.split_into_quantized_tensors()

    tex.split_quantize = split_path if path == "split" else group_path
    try:
        yield
    finally:
        tex.split_quantize = original_split_quantize


def _make_layer(k: int, n: int, num_gemms: int, seed: int) -> te.GroupedLinear:
    torch.manual_seed(seed)
    return te.GroupedLinear(
        num_gemms,
        k,
        n,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()


def _make_tensors(m: int, k: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    dy = torch.randn((m, n), dtype=torch.bfloat16, device="cuda", generator=generator)
    x.requires_grad_(True)
    return x, dy


def _run_step(
    layer: te.GroupedLinear,
    x: torch.Tensor,
    dy: torch.Tensor,
    m_splits: list[int],
    recipe: te_recipe.NVFP4BlockScaling,
    fwd_only: bool,
) -> None:
    layer.zero_grad(set_to_none=True)
    x.grad = None
    with autocast(enabled=True, recipe=recipe):
        y = layer(x, m_splits, is_first_microbatch=True)
        if not fwd_only:
            y.backward(dy)


def _profile_once(
    step,
    path: str,
    recipe_name: str,
    stats: QuantPathStats,
    assert_kernels: bool,
) -> dict[str, str]:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
    torch.cuda.synchronize()

    names = sorted({evt.key for evt in prof.key_averages()})
    joined = "\n".join(names)
    expected = FLAT_4OVER6_KERNEL
    if path == "group":
        expected = (
            ROWSCALED_GROUP_KERNEL if recipe_name == "row_scaled_1d" else TENSOR_SCALED_GROUP_KERNEL
        )

    found_expected = expected in joined
    if assert_kernels and not found_expected:
        interesting = "\n".join(name for name in names if "4over6" in name or "quant" in name)
        raise RuntimeError(
            f"Did not find expected profiler entry containing {expected!r}.\n"
            f"Observed quant/4over6 entries:\n{interesting}"
        )
    if assert_kernels and path == "split" and stats.group_quantize_calls != 0:
        raise RuntimeError("split path unexpectedly called tex.group_quantize")
    if assert_kernels and path == "group" and stats.group_quantize_calls == 0:
        raise RuntimeError("group path did not call tex.group_quantize")
    if assert_kernels and stats.split_quantize_calls == 0:
        raise RuntimeError("GroupedLinear did not call tex.split_quantize")

    hits = [name for name in names if expected in name]
    return {
        "expected_kernel": expected,
        "kernel_check": "found" if found_expected else "missing",
        "matched_kernel": hits[0] if hits else "",
    }


def _bench_case(args, recipe_name: str, path: str, splits: list[int]) -> dict[str, object]:
    recipe = _make_recipe(recipe_name)
    layer = _make_layer(args.k, args.n, len(splits), args.seed)
    x, dy = _make_tensors(args.m, args.k, args.n, args.seed + 17)
    stats = QuantPathStats()

    def step():
        _run_step(layer, x, dy, splits, recipe, args.fwd_only)

    FP8GlobalStateManager.reset()
    with _quant_path(path, stats):
        for _ in range(args.warmup):
            step()
        sanity = _profile_once(step, path, recipe_name, stats, not args.no_assert_kernels)

        torch.cuda.synchronize()
        times_ms = []
        start_wall = time.perf_counter()
        while len(times_ms) < args.iters or time.perf_counter() - start_wall < args.min_run_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            end.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
            if len(times_ms) >= args.max_iters:
                break
        torch.cuda.synchronize()

    FP8GlobalStateManager.reset()
    return {
        "recipe": recipe_name,
        "path": path,
        "mode": "fwd_only" if args.fwd_only else "fwd_bwd",
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "num_gemms": len(splits),
        "splits": ",".join(str(v) for v in splits),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "iters": len(times_ms),
        "split_quantize_calls": stats.split_quantize_calls,
        "loop_quantize_calls": stats.loop_quantize_calls,
        "group_quantize_calls": stats.group_quantize_calls,
        **sanity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=65536)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--num-gemms", type=int, default=8)
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument(
        "--recipe", choices=["tensor_scaled_2d", "row_scaled_1d", "all"], default="all"
    )
    parser.add_argument("--path", choices=["split", "group", "all"], default="all")
    parser.add_argument("--fwd-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--min-run-time", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--no-assert-kernels", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        raise RuntimeError(f"NVFP4 is not available: {reason}")

    os.environ.setdefault("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "1")
    splits = _parse_splits(args.splits, args.m, args.num_gemms)
    recipes = ["tensor_scaled_2d", "row_scaled_1d"] if args.recipe == "all" else [args.recipe]
    paths = ["split", "group"] if args.path == "all" else [args.path]

    rows = []
    for recipe_name in recipes:
        for path in paths:
            print(f"\n=== recipe={recipe_name} path={path} splits={splits} ===", flush=True)
            rows.append(_bench_case(args, recipe_name, path, splits))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
