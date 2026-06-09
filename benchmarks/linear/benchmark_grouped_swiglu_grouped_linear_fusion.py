#!/usr/bin/env python
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped activation+quantize fallback for GroupedLinear blocks.

This benchmark targets the ``te_ops.Sequential`` fallback path for:
    GroupedLinear -> SwiGLU -> GroupedLinear

and prints fuser metadata so we can verify whether grouped-MLP fused kernels
are being used.

Examples
--------
Forward only (MXFP8):
    NVTE_DEBUG_2988=1 python -m benchmarks.linear.benchmark_grouped_swiglu_grouped_linear_fusion --fwd-only

Forward+backward (MXFP8):
    python -m benchmarks.linear.benchmark_grouped_swiglu_grouped_linear_fusion

BF16 control:
    python -m benchmarks.linear.benchmark_grouped_swiglu_grouped_linear_fusion --recipe bf16 --fwd-only
"""

# Keep these toggles aligned with the issue repro guidance before importing TE.
import os

os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")
os.environ.setdefault("NVTE_GROUPED_SWIGLU_GROUPED_LINEAR_FUSION", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
os.environ.setdefault("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")

import argparse
from contextlib import nullcontext

import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


MXFP8_AVAILABLE, REASON_FOR_NO_MXFP8 = FP8GlobalStateManager.is_mxfp8_available()


def _uniform_splits(total_tokens: int, num_groups: int) -> list[int]:
    if total_tokens % num_groups != 0:
        raise ValueError(
            "Expected total_tokens to be divisible by num_groups, got "
            f"total_tokens={total_tokens}, num_groups={num_groups}"
        )
    return [total_tokens // num_groups] * num_groups


def _build_module(
    *,
    num_groups: int,
    hidden_dim: int,
    intermediate_dim: int,
    output_dim: int,
    dtype: torch.dtype,
    enable_quantized_weights: bool,
    recipe: MXFP8BlockScaling | None,
) -> te_ops.Sequential:
    init_ctx = (
        te.quantized_model_init(enabled=True, recipe=recipe)
        if enable_quantized_weights
        else nullcontext()
    )
    with init_ctx:
        return te_ops.Sequential(
            te_ops.GroupedLinear(
                num_groups,
                hidden_dim,
                intermediate_dim * 2,
                bias=False,
                device="cuda",
                dtype=dtype,
            ),
            te_ops.SwiGLU(),
            te_ops.GroupedLinear(
                num_groups,
                intermediate_dim,
                output_dim,
                bias=False,
                device="cuda",
                dtype=dtype,
            ),
        )


def _run_steps(
    module: torch.nn.Module,
    x: torch.Tensor,
    split_sizes: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    quantized_compute: bool,
    recipe: MXFP8BlockScaling | None,
    fwd_only: bool,
    num_steps: int,
) -> torch.Tensor:
    quant_ctx = te.autocast(enabled=quantized_compute, recipe=recipe) if quantized_compute else nullcontext()

    if fwd_only:
        with torch.no_grad(), quant_ctx:
            for _ in range(num_steps):
                out = module(x, split_sizes, split_sizes)
        return out

    module.zero_grad(set_to_none=True)
    x.grad = None
    with quant_ctx:
        for step in range(num_steps):
            torch.cuda.nvtx.range_push(f"step_{step}")
            out = module(x, split_sizes, split_sizes)
            out.backward(grad_output)
            torch.cuda.nvtx.range_pop()
    return out


def _extract_fuser_names(module: te_ops.Sequential) -> tuple[str, list[str], list[str]]:
    module_group = module._module_groups[0]  # pylint: disable=protected-access
    group_type = type(module_group).__name__
    forward_names = [type(op).__name__ for op, _ in module_group._forward_ops]  # pylint: disable=protected-access
    backward_names = [type(op).__name__ for op, _ in module_group._backward_ops]  # pylint: disable=protected-access
    return group_type, forward_names, backward_names


def _ensure_unfused_grouped_mlp(forward_names: list[str], backward_names: list[str]) -> None:
    bad = [name for name in forward_names + backward_names if "GroupedMLP" in name]
    if bad:
        raise RuntimeError(
            "Expected pure unfused grouped MLP path, but found grouped-MLP fused ops: "
            f"{bad}. Check NVTE_CUTEDSL_FUSED_GROUPED_MLP."
        )


def benchmark_case(args: argparse.Namespace) -> float:
    quantized_compute = args.recipe == "mxfp8"
    recipe = MXFP8BlockScaling() if quantized_compute else None
    os.environ["NVTE_GROUPED_SWIGLU_GROUPED_LINEAR_FUSION"] = (
        "0" if args.disable_grouped_activation_quantize_fusion else "1"
    )

    split_sizes_list = _uniform_splits(args.tokens, args.num_groups)
    split_sizes = torch.tensor(
        split_sizes_list,
        dtype=torch.int64,
        device=args.split_device,
    )
    if split_sizes.device.type != "cuda":
        raise RuntimeError("This benchmark expects CUDA split_sizes. Use --split-device cuda.")

    x = torch.randn(
        (args.tokens, args.hidden_dim),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=not args.fwd_only,
    )
    grad_output = torch.ones(
        (args.tokens, args.output_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )

    module = _build_module(
        num_groups=args.num_groups,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        output_dim=args.output_dim,
        dtype=torch.bfloat16,
        enable_quantized_weights=quantized_compute,
        recipe=recipe,
    )

    # Warmup and force fuser planning.
    _run_steps(
        module,
        x,
        split_sizes,
        grad_output,
        quantized_compute=quantized_compute,
        recipe=recipe,
        fwd_only=args.fwd_only,
        num_steps=args.warmup_steps,
    )
    torch.cuda.synchronize()

    group_type, forward_names, backward_names = _extract_fuser_names(module)
    _ensure_unfused_grouped_mlp(forward_names, backward_names)

    print(f"module group type: {group_type}")
    print(f"forward fused op names: {forward_names}")
    print(f"backward fused op names: {backward_names}")
    print(f"m_splits: {split_sizes_list}")

    timing_ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True) if args.profile else nullcontext()
    with timing_ctx:
        torch.cuda.nvtx.range_push("grouped_swiglu_grouped_linear_fusion_benchmark")
        timing = benchmark.Timer(
            stmt=(
                "_run_steps(module, x, split_sizes, grad_output, "
                "quantized_compute=quantized_compute, recipe=recipe, "
                "fwd_only=fwd_only, num_steps=num_microbatches)"
            ),
            globals={
                "_run_steps": _run_steps,
                "module": module,
                "x": x,
                "split_sizes": split_sizes,
                "grad_output": grad_output,
                "quantized_compute": quantized_compute,
                "recipe": recipe,
                "fwd_only": args.fwd_only,
                "num_microbatches": args.num_microbatches,
            },
            num_threads=1,
        ).blocked_autorange(min_run_time=args.min_run_time)
        torch.cuda.nvtx.range_pop()

    per_mb_ms = timing.median * 1000.0 / args.num_microbatches
    print(f"per-microbatch time ms: {per_mb_ms:.6f}")
    print(f"timer details: {timing}")
    return per_mb_ms


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable NVTX profiling context")
    parser.add_argument("--fwd-only", action="store_true", help="Benchmark forward only")
    parser.add_argument(
        "--disable-grouped-activation-quantize-fusion",
        action="store_true",
        help="Disable SwiGLU->GroupedLinear grouped activation+quantize fallback fusion",
    )
    parser.add_argument("--recipe", type=str, default="mxfp8", choices=("mxfp8", "bf16"))
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--intermediate-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=64)
    parser.add_argument("--split-device", type=str, default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--num-microbatches", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--min-run-time", type=float, default=5.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.recipe == "mxfp8" and not MXFP8_AVAILABLE:
        raise RuntimeError(f"MXFP8 is not available: {REASON_FOR_NO_MXFP8}")

    print("Environment toggles:")
    for env_name in (
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        "NVTE_GROUPED_SWIGLU_GROUPED_LINEAR_FUSION",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
        "CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL",
        "NVTE_DEBUG_2988",
    ):
        print(f"  {env_name}={os.environ.get(env_name)}")
    print(
        "case:",
        f"recipe={args.recipe}",
        f"fwd_only={args.fwd_only}",
        f"tokens={args.tokens}",
        f"hidden={args.hidden_dim}",
        f"intermediate={args.intermediate_dim}",
        f"output={args.output_dim}",
        f"num_groups={args.num_groups}",
        f"split_device={args.split_device}",
    )

    benchmark_case(args)


if __name__ == "__main__":
    main()
