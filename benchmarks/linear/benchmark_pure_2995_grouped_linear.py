# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Standalone benchmark for NVIDIA/TransformerEngine#2995:
#   "Reduce CPU overheads in te Sequential GroupedLinear Op"
#
# Run this from a TransformerEngine source checkout with TE installed from source.

"""Benchmark pure te_ops.Sequential(GroupedLinear) CPU overhead.

This intentionally avoids ScaledSwiGLU / grouped MLP patterns so the fuser
should exercise the GroupedLinear op path from #2923 instead of the grouped-MLP
CuTeGEMM fusion path from #2897/#3075.

Examples:

    python benchmark_pure_2995_grouped_linear.py

    nsys profile \
    --output=./pure_2995_grouped_linear \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas,osrt,python-gil \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    python benchmark_pure_2995_grouped_linear.py --profile --num-microbatches 8
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os

# Match the grouped-linear / graph-safe performance-sensitive settings before
# importing Transformer Engine.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
os.environ.setdefault("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
os.environ.setdefault("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")

# Avoid accidentally enabling the separate grouped-MLP fusion lane while using
# this script as a #2995 harness.
os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def make_uniform_splits(total_tokens: int, num_groups: int) -> list[int]:
    if total_tokens % num_groups != 0:
        raise ValueError(
            "Uniform split requires total_tokens divisible by num_groups, "
            f"got total_tokens={total_tokens}, num_groups={num_groups}"
        )
    return [total_tokens // num_groups] * num_groups


def make_split_tensor(
    split_sizes: list[int],
    *,
    split_device: str,
) -> torch.Tensor:
    device = "cuda" if split_device == "cuda" else "cpu"
    pin_memory = split_device == "cpu_pinned"
    return torch.tensor(
        split_sizes,
        dtype=torch.int64,
        device=device,
        pin_memory=pin_memory,
    )


def init_main_grads(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            if param is None:
                continue
            if getattr(param, "main_grad", None) is None:
                param.main_grad = torch.empty(
                    param.size(),
                    dtype=torch.float32,
                    device=param.device,
                )
            param.main_grad.zero_()


def build_sequential_grouped_linear(
    *,
    num_groups: int,
    hidden_dim: int,
    output_dim: int,
    dtype: torch.dtype,
    recipe_name: str,
    single_grouped_weight: bool,
    accumulate_into_main_grad: bool,
    bias: bool,
) -> te_ops.Sequential:
    recipe = MXFP8BlockScaling() if recipe_name == "mxfp8" else None
    init_context = (
        te.quantized_model_init(enabled=True, recipe=recipe)
        if recipe is not None
        else nullcontext()
    )
    with init_context:
        layer = te_ops.GroupedLinear(
            num_groups,
            hidden_dim,
            output_dim,
            bias=bias,
            device="cuda",
            dtype=dtype,
            single_grouped_weight=single_grouped_weight,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )
    return te_ops.Sequential(layer)


def describe_fuser(module: te_ops.Sequential) -> None:
    if module._module_groups is None:
        return
    group = module._module_groups[0]
    print("module group:", type(group).__name__)
    forward_ops = getattr(group, "_forward_ops", None)
    backward_ops = getattr(group, "_backward_ops", None)
    if forward_ops is not None:
        print(
            "forward fused op:",
            type(forward_ops[0][0]).__name__ if forward_ops else "none",
        )
    if backward_ops is not None:
        print(
            "backward fused op:",
            type(backward_ops[0][0]).__name__ if backward_ops else "none",
        )


def zero_grads(module: torch.nn.Module, x: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    x.grad = None


def run_grouped_linear_steps(
    module: torch.nn.Module,
    x: torch.Tensor,
    split_sizes: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    recipe_name: str,
    fwd_only: bool,
    num_steps: int,
    accumulate_into_main_grad: bool,
) -> torch.Tensor:
    recipe = MXFP8BlockScaling() if recipe_name == "mxfp8" else None
    quantization_context = (
        te.autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()
    )

    if fwd_only:
        with torch.no_grad(), quantization_context:
            for step in range(num_steps):
                torch.cuda.nvtx.range_push(f"step_{step}")
                out = module(x, split_sizes)
                torch.cuda.nvtx.range_pop()
        return out

    zero_grads(module, x)
    if accumulate_into_main_grad:
        init_main_grads(module)

    with quantization_context:
        for step in range(num_steps):
            torch.cuda.nvtx.range_push(f"step_{step}")
            out = module(x, split_sizes)
            out.backward(grad_output)
            torch.cuda.nvtx.range_pop()
    return out


def benchmark_case(
    *,
    total_tokens: int,
    hidden_dim: int,
    output_dim: int,
    num_groups: int,
    split_sizes_list: list[int],
    split_device: str,
    recipe_name: str,
    dtype: torch.dtype,
    fwd_only: bool,
    single_grouped_weight: bool,
    accumulate_into_main_grad: bool,
    bias: bool,
    num_microbatches: int,
    min_run_time: float,
    profile: bool,
) -> float:
    split_sizes = make_split_tensor(split_sizes_list, split_device=split_device)
    x = torch.randn(
        (total_tokens, hidden_dim),
        dtype=dtype,
        device="cuda",
        requires_grad=not fwd_only,
    )
    grad_output = torch.ones(
        (total_tokens, output_dim),
        dtype=dtype,
        device="cuda",
    )

    module = build_sequential_grouped_linear(
        num_groups=num_groups,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dtype=dtype,
        recipe_name=recipe_name,
        single_grouped_weight=single_grouped_weight,
        accumulate_into_main_grad=accumulate_into_main_grad,
        bias=bias,
    )

    print(
        "case:",
        f"tokens={total_tokens}",
        f"hidden={hidden_dim}",
        f"output={output_dim}",
        f"num_groups={num_groups}",
        f"recipe={recipe_name}",
        f"fwd_only={fwd_only}",
        f"split_device={split_device}",
        f"single_grouped_weight={single_grouped_weight}",
        f"accumulate_into_main_grad={accumulate_into_main_grad}",
        f"bias={bias}",
    )
    print(f"m_splits: {split_sizes_list}")

    # Warmup also materializes OperationFuser internals.
    run_grouped_linear_steps(
        module,
        x,
        split_sizes,
        grad_output,
        recipe_name=recipe_name,
        fwd_only=fwd_only,
        num_steps=128,
        accumulate_into_main_grad=accumulate_into_main_grad,
    )
    torch.cuda.synchronize()
    describe_fuser(module)

    label = "pure_2995_sequential_grouped_linear"
    timing_context = (
        torch.autograd.profiler.emit_nvtx(record_shapes=True)
        if profile
        else nullcontext()
    )
    with timing_context:
        torch.cuda.nvtx.range_push(label)
        timing = benchmark.Timer(
            stmt=(
                "run_grouped_linear_steps("
                "module, x, split_sizes, grad_output, "
                "recipe_name=recipe_name, fwd_only=fwd_only, "
                "num_steps=num_microbatches, "
                "accumulate_into_main_grad=accumulate_into_main_grad)"
            ),
            globals={
                "run_grouped_linear_steps": run_grouped_linear_steps,
                "module": module,
                "x": x,
                "split_sizes": split_sizes,
                "grad_output": grad_output,
                "recipe_name": recipe_name,
                "fwd_only": fwd_only,
                "num_microbatches": num_microbatches,
                "accumulate_into_main_grad": accumulate_into_main_grad,
            },
            num_threads=1,
        ).blocked_autorange(min_run_time=min_run_time)
        torch.cuda.nvtx.range_pop()

    print(f"{recipe_name}_grouped_linear: {timing}\n")
    return timing.median * 1000 / num_microbatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable NVTX annotations")
    parser.add_argument("--fwd-only", action="store_true", help="Benchmark forward only")
    parser.add_argument("--recipe", choices=("bf16", "mxfp8"), default="mxfp8")
    parser.add_argument("--num-groups", type=str, default="64")
    parser.add_argument("--token-dims", type=str, default="8192")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--num-microbatches", type=int, default=32)
    parser.add_argument("--min-run-time", type=float, default=10.0)
    parser.add_argument(
        "--split-device",
        choices=("cuda", "cpu", "cpu_pinned"),
        default="cuda",
        help="Where split_sizes starts before GroupedLinear sees it.",
    )
    parser.add_argument(
        "--jagged-input",
        type=str,
        default=None,
        help="Comma-separated split sizes. Overrides --token-dims and --num-groups.",
    )
    parser.add_argument("--single-grouped-weight", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument(
        "--no-accumulate-into-main-grad",
        dest="accumulate_into_main_grad",
        action="store_false",
    )
    parser.set_defaults(accumulate_into_main_grad=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.recipe == "mxfp8":
        available, reason = FP8GlobalStateManager.is_mxfp8_available()
        if not available:
            raise RuntimeError(f"MXFP8 is not available: {reason}")

    dtype = torch.bfloat16
    print("Environment toggles:")
    for name in (
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
        "CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL",
        "NVTE_GROUPED_LINEAR_SINGLE_PARAM",
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
    ):
        print(f"  {name}={os.environ.get(name)}")
    print()

    if args.jagged_input:
        split_sizes_list = parse_int_list(args.jagged_input)
        num_groups_list = [len(split_sizes_list)]
        token_dims = [sum(split_sizes_list)]
    else:
        num_groups_list = parse_int_list(args.num_groups)
        token_dims = parse_int_list(args.token_dims)
        split_sizes_list = []

    data = []
    for num_groups in num_groups_list:
        for total_tokens in token_dims:
            splits = (
                split_sizes_list
                if split_sizes_list
                else make_uniform_splits(total_tokens, num_groups)
            )
            timing_ms = benchmark_case(
                total_tokens=total_tokens,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
                num_groups=num_groups,
                split_sizes_list=splits,
                split_device=args.split_device,
                recipe_name=args.recipe,
                dtype=dtype,
                fwd_only=args.fwd_only,
                single_grouped_weight=args.single_grouped_weight,
                accumulate_into_main_grad=args.accumulate_into_main_grad,
                bias=args.bias,
                num_microbatches=args.num_microbatches,
                min_run_time=args.min_run_time,
                profile=args.profile,
            )
            data.append(
                [
                    total_tokens,
                    args.hidden_dim,
                    args.output_dim,
                    num_groups,
                    args.recipe,
                    args.split_device,
                    args.single_grouped_weight,
                    args.accumulate_into_main_grad,
                    args.bias,
                    "fwd" if args.fwd_only else "fwd_bwd",
                    timing_ms,
                ]
            )

    df = pd.DataFrame(
        data=data,
        columns=[
            "tokens",
            "hidden_dim",
            "output_dim",
            "num_groups",
            "recipe",
            "split_device",
            "single_grouped_weight",
            "accumulate_into_main_grad",
            "bias",
            "mode",
            "time_per_microbatch_ms",
        ],
    )
    print(df)


if __name__ == "__main__":
    main()
