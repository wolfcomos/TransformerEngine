# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fallback fused forward path for SwiGLU + GroupedLinear."""

from __future__ import annotations
from collections.abc import Iterable
import os
from typing import Any, Optional

import torch

import transformer_engine_torch as tex

from ...constants import DType
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...quantization import Recipe
from ...tensor import Float8CurrentScalingQuantizer, Quantizer
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from .._common import maybe_dequantize, _grouped_swiglu_quantize_for_grouped_linear
from ..basic import GroupedLinear, SwiGLU
from ..op import FusedOperation, FusibleOperation, OperationContext


class ForwardGroupedSwiGLUGroupedLinear(FusedOperation):
    """Fused forward fallback for ``SwiGLU -> GroupedLinear``."""

    def __init__(
        self,
        *,
        activation: SwiGLU,
        fc2: GroupedLinear,
    ) -> None:
        super().__init__((activation, fc2))

    def _run_unfused_fallback(
        self,
        *,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        """Run original unfused ops when fused preconditions are not met."""
        activation_op, fc2_op = self.basic_ops
        activation_ctx, fc2_ctx = basic_op_ctxs
        activation_out = activation_op.op_forward(
            activation_ctx,
            input_,
            prev_op_grad_output_quantizer,
            # GroupedLinear owns grouped quantization of its input.
            None,
        )
        fc2_out, fc2_extra = fc2_op.fuser_forward(
            [fc2_ctx],
            activation_out,
            basic_op_extra_inputs=[basic_op_extra_inputs[1]],
            prev_op_grad_output_quantizer=activation_op.get_grad_output_quantizer(),
            next_op_input_quantizer=next_op_input_quantizer,
            basic_op_kwargs=[basic_op_kwargs[1]],
        )
        return fc2_out, [(), *fc2_extra]

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        activation_op, fc2_op = self.basic_ops
        activation_ctx, fc2_ctx = basic_op_ctxs

        if not hasattr(tex, "grouped_swiglu_quantize"):
            return self._run_unfused_fallback(
                basic_op_ctxs=basic_op_ctxs,
                input_=input_,
                basic_op_extra_inputs=basic_op_extra_inputs,
                prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
                next_op_input_quantizer=next_op_input_quantizer,
                basic_op_kwargs=basic_op_kwargs,
            )

        if basic_op_kwargs[0]:
            raise ValueError("SwiGLU forward does not expect keyword arguments")

        if not basic_op_extra_inputs[1]:
            raise ValueError("GroupedLinear forward expects split_sizes extra input")
        split_sizes = basic_op_extra_inputs[1][0]

        # Use GroupedLinear's standard path if fused preconditions are not met.
        if (
            not isinstance(fc2_op.get_input_quantizer(), MXFP8Quantizer)
            or input_.device.type != "cuda"
        ):
            return self._run_unfused_fallback(
                basic_op_ctxs=basic_op_ctxs,
                input_=input_,
                basic_op_extra_inputs=basic_op_extra_inputs,
                prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
                next_op_input_quantizer=next_op_input_quantizer,
                basic_op_kwargs=basic_op_kwargs,
            )

        if split_sizes.dtype != torch.int64:
            split_sizes = split_sizes.to(dtype=torch.int64)
        if split_sizes.device != input_.device:
            split_sizes = split_sizes.to(device=input_.device)

        # Compute dtype and activation input exactly as SwiGLU.forward does.
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = input_.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")

        activation_input = maybe_dequantize(input_.contiguous(), dtype)
        swiglu_input = activation_input
        if activation_op.glu_interleave_size is not None:
            shape = swiglu_input.size()
            swiglu_input = swiglu_input.reshape(
                -1,
                shape[-1] // (2 * activation_op.glu_interleave_size),
                2,
                activation_op.glu_interleave_size,
            )
            swiglu_input = swiglu_input.transpose(1, 2).contiguous()
            swiglu_input = swiglu_input.view(shape)

        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        weight_requires_grad = fc2_ctx.requires_grad and fc2_weight_param.requires_grad
        fc2_input_quantizer = fc2_op.get_input_quantizer()
        fc2_input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
        fc2_input_quantizer.optimize_for_gemm = True

        grouped_fc2_input = _grouped_swiglu_quantize_for_grouped_linear(
            swiglu_input,
            fc2_input_quantizer,
            fc2_op.num_groups,
            split_sizes,
        )

        # Match SwiGLU context save behavior for backward.
        if activation_op.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(
                DType.kFloat8E4M3,
                activation_input.device,
            )
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            activation_input = input_quantizer(activation_input)

        if activation_ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(activation_input)
            activation_ctx.save_for_backward(activation_input)
            activation_ctx.dtype = dtype
            activation_ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        fc2_out, fc2_extra = fc2_op.fuser_forward(
            [fc2_ctx],
            grouped_fc2_input,
            basic_op_extra_inputs=[basic_op_extra_inputs[1]],
            prev_op_grad_output_quantizer=activation_op.get_grad_output_quantizer(),
            next_op_input_quantizer=next_op_input_quantizer,
            basic_op_kwargs=[basic_op_kwargs[1]],
        )
        return fc2_out, [(), *fc2_extra]

    @staticmethod
    def fuse_forward_ops(
        ops: list[FusibleOperation],
        *,
        recipe: Optional[Recipe] = None,
        **unused,  # pylint: disable=unused-argument
    ) -> list[FusibleOperation]:
        """Apply fallback fusion for ``SwiGLU -> GroupedLinear``."""

        if int(os.environ.get("NVTE_GROUPED_SWIGLU_GROUPED_LINEAR_FUSION", "1")) <= 0:
            return ops
        if not hasattr(tex, "grouped_swiglu_quantize"):
            return ops
        if recipe is None or not recipe.mxfp8():
            return ops

        out: list[FusibleOperation] = []
        window, ops = ops[:2], ops[2:]
        while len(window) == 2:
            matches_pattern = isinstance(window[0], SwiGLU) and isinstance(window[1], GroupedLinear)
            if matches_pattern:
                fc2_op: GroupedLinear = window[1]
                weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
                input_quantizers = [
                    fc2_op.get_quantizer("forward", 2 * group_idx)
                    for group_idx in range(fc2_op.num_groups)
                ]
                matches_pattern = fc2_op._is_graph_safe_path_supported(  # pylint: disable=protected-access
                    with_quantized_compute=True,
                    input_quantizers=input_quantizers,
                    dtype=weight_param.dtype,
                )

            if matches_pattern:
                window = [
                    ForwardGroupedSwiGLUGroupedLinear(
                        activation=window[0],
                        fc2=window[1],
                    )
                ]
            else:
                out.extend(window[:-1])
                window = window[-1:]

            out.extend(window[:-2])
            window = window[-2:]
            while ops and len(window) < 2:
                window.append(ops[0])
                ops = ops[1:]

        out.extend(window)
        return out
