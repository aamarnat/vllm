# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from contextlib import suppress

import torch
import torch._inductor.pattern_matcher as pm
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)

FP8_DTYPE = current_platform.fp8_dtype()

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")


def _has_aiter_op(op_name: str) -> bool:
    """Check whether a specific aiter custom op is registered."""
    return hasattr(torch.ops, "aiter") and hasattr(torch.ops.aiter, op_name)


logger = init_logger(__name__)

if hasattr(torch.ops._C, "scaled_fp4_quant"):
    STATIC_FP4_QUANT_OP = torch.ops._C.scaled_fp4_quant.default


def _flashinfer_scaled_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale_result: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> None:
    # Import lazily to avoid a circular import during module initialization
    # when docs or other tooling import the pass without FlashInfer.
    from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm_out

    assert bias is None, "FlashInfer symm_mem adapter does not support bias"
    assert scale_result is None, (
        "FlashInfer symm_mem adapter does not support result scaling"
    )
    assert not use_fast_accum, (
        "FlashInfer symm_mem adapter does not support use_fast_accum"
    )
    assert A.ndim == 2 and B.ndim == 2 and out.ndim == 2, (
        "FlashInfer symm_mem adapter expects 2D inputs and output"
    )
    assert scale_a.numel() == 1 and scale_b.numel() == 1, (
        "FlashInfer symm_mem adapter only supports tensor-wise FP8 scales"
    )

    flashinfer_scaled_fp8_mm_out(
        A,
        B,
        scale_a,
        scale_b,
        out=out,
        out_dtype=out_dtype or out.dtype,
    )


def _flashinfer_fp4_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> None:
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm_out

    assert A.ndim == 2 and B.ndim == 2 and out.ndim == 2, (
        "FlashInfer FP4 symm_mem adapter expects 2D inputs and output"
    )
    flashinfer_scaled_fp4_mm_out(
        A,
        B,
        scale_a,
        scale_b,
        alpha,
        out=out,
        out_dtype=out_dtype or out.dtype,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend=backend,
    )


def fused_flashinfer_scaled_matmul_reduce_scatter_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    result_shape = list(output_shape)
    result_shape[orig_scatter_dim] //= world_size
    return torch.empty(
        result_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A.device,
    )


def fused_flashinfer_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert orig_scatter_dim == 0 and scatter_dim_after_maybe_reshape == 0, (
        "FlashInfer symm_mem adapter currently only supports scatter_dim=0"
    )
    world_size = c10d._resolve_process_group(group_name).size()
    assert A.ndim == 2 and B.ndim == 2, "FlashInfer symm_mem adapter expects 2D inputs"
    assert A.is_contiguous(), "FlashInfer symm_mem adapter expects contiguous A"
    assert A_scale.numel() == 1 and B_scale.numel() == 1, (
        "FlashInfer symm_mem adapter only supports tensor-wise FP8 scales"
    )
    assert A.shape[0] % world_size == 0, (
        "FlashInfer symm_mem adapter expects M divisible by world size"
    )

    kwargs = {
        "scale_b": B_scale,
        "bias": None,
        "scale_result": None,
        "out_dtype": out_dtype,
        "use_fast_accum": False,
    }
    return torch.distributed._symmetric_memory._fused_scaled_matmul_reduce_scatter_impl(
        mm_out_op=_flashinfer_scaled_mm_out,
        A=A,
        B=B,
        A_scale=A_scale,
        kwargs=kwargs,
        out_dtype=out_dtype,
        reduce_op=reduce_op,
        orig_scatter_dim=orig_scatter_dim,
        scatter_dim_after_maybe_reshape=scatter_dim_after_maybe_reshape,
        group_name=group_name,
        output_shape=output_shape,
    )


def fused_all_gather_flashinfer_scaled_matmul_fake(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    output_shape = list(A_shard.shape)
    output_shape[gather_dim] *= world_size
    output_shape[-1] = B.shape[1]
    return torch.empty(
        output_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A_shard.device,
    )


def fused_all_gather_flashinfer_scaled_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert gather_dim == 0, (
        "FlashInfer symm_mem adapter currently only supports gather_dim=0"
    )
    _, outputs = torch.distributed._symmetric_memory._fused_all_gather_matmul_impl(
        mm_out_op=_flashinfer_scaled_mm_out,
        A_shard=A_shard,
        Bs=[B],
        A_scale=A_scale,
        kwargs_list=[
            {
                "scale_b": B_scale,
                "bias": None,
                "scale_result": None,
                "out_dtype": out_dtype,
                "use_fast_accum": False,
            }
        ],
        out_dtypes=[out_dtype],
        gather_dim=gather_dim,
        group_name=group_name,
        return_A=False,
    )
    return outputs[0]


def fused_all_gather_flashinfer_fp4_matmul_fake(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale_shard: torch.Tensor,
    B_scale: torch.Tensor,
    alpha: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
    view_a_scale_as_fp8: bool = False,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    output_shape = list(A_shard.shape)
    output_shape[gather_dim] *= world_size
    output_shape[-1] = B.shape[1]
    return torch.empty(
        output_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A_shard.device,
    )


def fused_all_gather_flashinfer_fp4_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale_shard: torch.Tensor,
    B_scale: torch.Tensor,
    alpha: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
    view_a_scale_as_fp8: bool = False,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> torch.Tensor:
    assert gather_dim == 0, (
        "FlashInfer FP4 symm_mem adapter currently only supports gather_dim=0"
    )
    assert A_shard.ndim == 2 and A_scale_shard.ndim == 2 and B.ndim == 2, (
        "FlashInfer FP4 symm_mem adapter expects 2D inputs"
    )
    if view_a_scale_as_fp8:
        A_scale_shard = A_scale_shard.view(torch.float8_e4m3fn)

    group = c10d._resolve_process_group(group_name)
    world_size = group.size()
    output = A_shard.new_empty(
        A_shard.shape[0] * world_size,
        B.shape[1],
        dtype=out_dtype or torch.bfloat16,
    )
    output_shards = output.chunk(world_size)

    A = A_shard.new_empty(A_shard.shape[0] * world_size, A_shard.shape[1])
    A_scale = A_scale_shard.new_empty(
        A_scale_shard.shape[0] * world_size,
        A_scale_shard.shape[1],
    )

    def fp4_shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
        _flashinfer_fp4_mm_out(
            shards[0],
            B,
            scale_a=shards[1],
            scale_b=B_scale,
            alpha=alpha,
            out=output_shards[rank],
            out_dtype=out_dtype,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )

    torch.distributed._symmetric_memory._pipelined_multi_all_gather_and_consume(
        [A_shard, A_scale_shard],
        fp4_shard_consumer,
        [A, A_scale],
        group_name,
        False,
    )
    return output


direct_register_custom_op(
    op_name="fused_flashinfer_scaled_matmul_reduce_scatter",
    op_func=fused_flashinfer_scaled_matmul_reduce_scatter,
    fake_impl=fused_flashinfer_scaled_matmul_reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="fused_all_gather_flashinfer_scaled_matmul",
    op_func=fused_all_gather_flashinfer_scaled_matmul,
    fake_impl=fused_all_gather_flashinfer_scaled_matmul_fake,
)

direct_register_custom_op(
    op_name="fused_all_gather_flashinfer_fp4_matmul",
    op_func=fused_all_gather_flashinfer_fp4_matmul,
    fake_impl=fused_all_gather_flashinfer_fp4_matmul_fake,
)


class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class GEMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "sum",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherGEMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            # Prefer the aiter kernel on ROCm, but fall back to symm_mem when it
            # is unavailable so ROCm keeps the upstream fusion either way.
            if current_platform.is_rocm() and _has_aiter_op(
                "fused_all_gather_gemm_m_shard"
            ):
                ag_output, mm_outputs = torch.ops.aiter.fused_all_gather_gemm_m_shard(
                    x,
                    [weight],
                    gather_dim=0,
                    group_name=self.tp.device_group.group_name,
                )
            else:
                ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                    x,
                    [weight],
                    gather_dim=0,
                    group_name=self.tp.device_group.group_name,
                )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherGEMMPatternKShard(BasePattern):
    """
    Pattern for all-gather + GEMM fusion with K-dimension sharding.

    This pattern handles the case where the input is sharded along the K dimension:
    - Input: (M, K_local) where K is sharded across ranks
    - All-gather on dim=1 produces (M, K) where K = K_local * world_size
    - Then performs GEMM: (M, K) @ (K, N) = (M, N)
    """

    def get_inputs(self):
        # Input is sharded on K dimension: (M, K_local)
        # Use dimensions that won't cause shape issues during tracing
        M = 8  # Batch dimension
        K_local = 4  # Local K dimension
        K_total = K_local * self.tp_size  # Total K after all-gather
        N = 8  # Output dimension

        x = torch.empty([M, K_local], device=self.device, dtype=self.dtype)
        # Weight needs full K dimension after all-gather: (K, N)
        weight = torch.empty([K_total, N], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=1,  # K-dimension sharding
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(
            x: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Only use aiter.fused_all_gather_gemm_k_shard for K-sharding
            # Note: symm_mem.fused_all_gather_matmul doesn't support
            # K-dimension sharding
            ag_output, mm_outputs = torch.ops.aiter.fused_all_gather_gemm_k_shard(
                x,
                [weight],
                gather_dim=1,  # K-dimension
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)
        return [input, mm_weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            scaled_mm = torch.ops.aten._scaled_mm.default(
                input,
                mat2=mat2,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                scaled_mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            return torch.ops.aten._scaled_mm.default(
                all_gather,
                mat2=weight,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class CutlassScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        cutlass_mm_output = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        return [input, mm_weight, scale_a, scale_b, cutlass_mm_output]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=cutlass_mm_output,
                a=input,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )

            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                cutlass_scaled_mm[1],
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherCutlassScaledMMPattern(BasePattern):
    def get_inputs(self):
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        s2 = weight.shape[1]
        output = torch.empty([s1, s2], device=self.device, dtype=self.dtype)

        return [x, weight, scale_a, scale_b, output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=output,
                a=all_gather,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )
            return cutlass_scaled_mm[1]

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class FlashInferBMMFP8ReduceScatterPattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def get_inputs(self) -> list[torch.Tensor]:
        a_2d = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        b_2d = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        a_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a_2d, b_2d, a_scale, b_scale]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            bmm = torch.ops.vllm.bmm_fp8.default(
                torch.ops.aten.unsqueeze.default(a_2d, 0),
                torch.ops.aten.unsqueeze.default(b_2d, 0),
                a_scale,
                b_scale,
                self.dtype,
                "auto",
            )
            output = torch.ops.aten.reshape.default(bmm, list(bmm.shape[1:]))
            return torch.ops.vllm.reduce_scatter.default(
                output,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default(
                a_2d,
                b_2d,
                a_scale,
                b_scale,
                "sum",
                0,
                0,
                self.tp.device_group.group_name,
                [a_2d.shape[0], b_2d.shape[1]],
                self.dtype,
            )

        return _replacement


class FlashInferAllGatherBMMFP8Pattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def get_inputs(self) -> list[torch.Tensor]:
        a_shard_2d = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        b_2d = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        a_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a_shard_2d, b_2d, a_scale, b_scale]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                a_shard_2d,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return torch.ops.vllm.bmm_fp8.default(
                torch.ops.aten.unsqueeze.default(all_gather, 0),
                torch.ops.aten.unsqueeze.default(b_2d, 0),
                a_scale,
                b_scale,
                self.dtype,
                "auto",
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            fused = torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default(
                a_shard_2d,
                b_2d,
                a_scale,
                b_scale,
                0,
                self.tp.device_group.group_name,
                self.dtype,
            )
            return torch.ops.aten.unsqueeze.default(fused, 0)

        return _replacement


class FlashInferAllGatherFP4Pattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def __init__(
        self,
        dtype: torch.dtype,
        device: str | None,
        backend: str,
        use_8x4_sf_layout: bool,
        a_scale_view: str,
    ) -> None:
        super().__init__(dtype, device)
        self.backend = backend
        self.use_8x4_sf_layout = use_8x4_sf_layout
        self.a_scale_view = a_scale_view

    def get_inputs(self) -> list[torch.Tensor]:
        a_shard_2d = torch.empty([8, 8], device=self.device, dtype=torch.uint8)
        b_2d = torch.empty([8, 16], device=self.device, dtype=torch.uint8)
        a_scale_shard = torch.empty([128, 4], device=self.device, dtype=torch.int32)
        b_scale = torch.empty([4, 128], device=self.device, dtype=torch.uint8)
        alpha = torch.empty([], device=self.device, dtype=torch.float32)
        return [
            a_shard_2d,
            b_2d,
            a_scale_shard,
            b_scale,
            alpha,
        ]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale_shard: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            all_gather_a = torch.ops.vllm.all_gather.default(
                a_shard_2d,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            all_gather_a_scale = torch.ops.vllm.all_gather.default(
                a_scale_shard,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            a_scale = all_gather_a_scale
            if self.a_scale_view in ("float8", "float8_uint8"):
                a_scale = torch.ops.aten.view.dtype(a_scale, torch.float8_e4m3fn)
            if self.a_scale_view in ("uint8", "float8_uint8"):
                a_scale = torch.ops.aten.view.dtype(a_scale, torch.uint8)
            return torch.ops.vllm.flashinfer_mm_fp4.default(
                all_gather_a,
                b_2d,
                a_scale,
                b_scale,
                alpha,
                self.dtype,
                self.use_8x4_sf_layout,
                self.backend,
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale_shard: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_all_gather_flashinfer_fp4_matmul.default(
                a_shard_2d,
                b_2d,
                a_scale_shard,
                b_scale,
                alpha,
                0,
                self.tp.device_group.group_name,
                self.dtype,
                self.a_scale_view in ("float8", "float8_uint8"),
                self.use_8x4_sf_layout,
                self.backend,
            )

        return _replacement


class GEMMAllReducePattern(BasePattern):
    """
    Pattern for GEMM + all-reduce fusion with K-dimension sharding.

    Matches RowParallelLinear output projection:
    - Input: (M, K_local) where K is sharded across ranks
    - GEMM: (M, K_local) @ (K_local, N) = (M, N) partial sum
    - All-reduce sums partial results across ranks -> (M, N) final
    Replaced by a single fused_gemm_all_reduce_k_shard kernel.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            mm = torch.ops.aten.mm.default(x, weight)
            ar = torch.ops.vllm.all_reduce.default(
                mm,
                group_name=self.tp.unique_name,
            )
            return ar

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            c_local, c_global = torch.ops.aiter.fused_gemm_all_reduce_k_shard(
                x,
                [weight],
                0,
                self.tp.device_group.group_name,
            )
            return c_global

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ROCmGEMMAllReducePattern(BasePattern):
    """
    Pattern for ROCm unquantized GEMM + all-reduce fusion.

    On ROCm, UnquantizedLinearMethod dispatches through
    torch.ops.vllm.rocm_unquantized_gemm (which runtime-selects between
    aiter Triton GEMM and rocBLAS). This pattern matches:
        rocm_unquantized_gemm(x, weight, bias=None) -> all_reduce(result)
    and replaces with fused_gemm_all_reduce_k_shard.

    Only matches when bias is None (RowParallelLinear on ranks > 0).
    Weight is in (N, K_local) layout (F.linear convention), so it is
    transposed to (K_local, N) for the fused kernel.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.ops.vllm.rocm_unquantized_gemm.default(x, weight, None)
            ar = torch.ops.vllm.all_reduce.default(
                gemm,
                group_name=self.tp.unique_name,
            )
            return ar

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            c_local, c_global = torch.ops.aiter.fused_gemm_all_reduce_k_shard(
                x,
                [weight.t()],
                0,
                self.tp.device_group.group_name,
            )
            return c_global

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ROCmGEMMAllReduceWithBiasPattern(BasePattern):
    """
    Pattern for ROCm unquantized GEMM + all-reduce fusion when bias is present.

    Matches RowParallelLinear on rank 0 where bias is a real tensor:
        rocm_unquantized_gemm(x, weight, bias) -> all_reduce(result)
    Replaced with fused_gemm_all_reduce_k_shard + bias addition.

    Correctness: RowParallelLinear only adds bias on tp_rank==0; other ranks
    pass bias=None (matched by ROCmGEMMAllReducePattern). After all-reduce
    the bias appears exactly once in the final result.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        bias = torch.empty([4], device=self.device, dtype=self.dtype)
        return [x, weight, bias]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.ops.vllm.rocm_unquantized_gemm.default(x, weight, bias)
            ar = torch.ops.vllm.all_reduce.default(
                gemm,
                group_name=self.tp.unique_name,
            )
            return ar

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            c_local, c_global = torch.ops.aiter.fused_gemm_all_reduce_k_shard(
                x,
                [weight.t()],
                0,
                self.tp.device_group.group_name,
            )
            return c_global + bias

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ROCmGEMMReduceScatterPattern(BasePattern):
    """
    Pattern for ROCm unquantized GEMM + reduce-scatter fusion.

    On ROCm, UnquantizedLinearMethod dispatches through
    torch.ops.vllm.rocm_unquantized_gemm, so upstream's GEMMReduceScatterPattern
    (which matches aten.mm) never fires. This pattern matches:
        rocm_unquantized_gemm(x, weight, bias=None) -> reduce_scatter(result)
    and replaces it with the iris-backed gemm_reduce_scatter_m_shard.

    The reduce_scatter node is introduced by SequenceParallelismPass, so this
    pattern is only registerable from a pass that runs after it.

    Only matches when bias is None (RowParallelLinear on ranks > 0). The weight
    is in (N, K_local) layout (F.linear convention) and the fused kernel wants
    (K_local, N), hence weight.t().
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.ops.vllm.rocm_unquantized_gemm.default(x, weight, None)
            return torch.ops.vllm.reduce_scatter.default(
                gemm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.aiter.gemm_reduce_scatter_m_shard(
                x,
                [weight.t()],
                None,
                0,
                self.tp.device_group.group_name,
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ROCmGEMMReduceScatterWithBiasPattern(BasePattern):
    """
    Pattern for ROCm unquantized GEMM + reduce-scatter fusion when bias is present.

    Matches RowParallelLinear on rank 0, where bias is a real tensor:
        rocm_unquantized_gemm(x, weight, bias) -> reduce_scatter(result)

    Correctness: the bias is passed INTO the fused op, which folds it into the
    pre-reduction partial via addmm. Reduce-scatter sums across ranks per row,
    so rank 0's bias lands in every output row exactly once. Adding it to the
    op's output instead would bias only the row slice rank 0 owns -- unlike the
    all-reduce case, where every rank holds the full sum.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        bias = torch.empty([4], device=self.device, dtype=self.dtype)
        return [x, weight, bias]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            gemm = torch.ops.vllm.rocm_unquantized_gemm.default(x, weight, bias)
            return torch.ops.vllm.reduce_scatter.default(
                gemm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.aiter.gemm_reduce_scatter_m_shard(
                x,
                [weight.t()],
                bias,
                0,
                self.tp.device_group.group_name,
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ROCmReduceScatterPattern(BasePattern):
    """
    Pattern replacing a bare vllm.reduce_scatter with the iris reduce-scatter.

    Standalone fallback for deployments where the producer is not a fusable
    GEMM: the partial sum is copied into the symmetric-heap staging buffer and
    reduced by the same one-shot pull kernel, without touching the producer.

    This anchors on the same node as the GEMM+RS patterns and matches any
    producer, so it is registered only when the fused patterns are not.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        return [torch.empty([16, 4], device=self.device, dtype=self.dtype)]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.vllm.reduce_scatter.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        def replacement(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aiter.reduce_scatter_m_shard(
                x,
                0,
                self.tp.device_group.group_name,
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class GEMMAllReducePass(VllmPatternMatcherPass):
    """
    Inductor pass that fuses GEMM + all-reduce into a single iris-backed
    kernel. Must run BEFORE SequenceParallelismPass so that the all_reduce
    node is still present in the graph.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="gemm_all_reduce_pass"
        )

        # Every pattern in this pass lowers to the same aiter op, so without it
        # there is nothing to register and tracing a replacement would raise.
        if not _has_aiter_op("fused_gemm_all_reduce_k_shard"):
            raise RuntimeError(
                "fuse_gemm_all_reduce requires the aiter op "
                "torch.ops.aiter.fused_gemm_all_reduce_k_shard, which is not "
                "registered. Check that aiter is installed and that "
                "`import aiter.ops.triton.comms.fused` succeeds."
            )

        registered_patterns = ["GEMMAllReducePattern(aten.mm)"]
        GEMMAllReducePattern(self.model_dtype, self.device).register(self.patterns)

        if current_platform.is_rocm():
            try:
                _ = torch.ops.vllm.rocm_unquantized_gemm.default
                ROCmGEMMAllReducePattern(self.model_dtype, self.device).register(
                    self.patterns
                )
                ROCmGEMMAllReduceWithBiasPattern(
                    self.model_dtype, self.device
                ).register(self.patterns)
                registered_patterns.append("ROCmGEMMAllReducePattern(bias=None)")
                registered_patterns.append(
                    "ROCmGEMMAllReduceWithBiasPattern(bias=Tensor)"
                )
                logger.info(
                    "Registered ROCmGEMMAllReducePattern (bias=None + "
                    "bias=Tensor) for rocm_unquantized_gemm + all_reduce fusion"
                )
            except AttributeError:
                logger.warning(
                    "rocm_unquantized_gemm op not registered, "
                    "skipping ROCmGEMMAllReducePattern"
                )

        logger.debug(
            "GEMMAllReducePass registered %d pattern(s): %s",
            len(registered_patterns),
            ", ".join(registered_patterns),
        )
        self.dump_patterns(config, self.patterns)

    # Minimum M dimension for the fused kernel. The persistent Triton kernel
    # requires block_size_m to be a power of 2 and M >= block_size_m.
    # For M < 16, the overhead of fusion exceeds any benefit.
    MIN_M_FOR_FUSION = 16

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if (
            not self.compilation_config.splitting_ops
            or self.compilation_config.use_inductor_graph_partition
        ):
            return True
        tp_size = get_tensor_model_parallel_world_size()
        if not (compile_range.is_single_size() and compile_range.end % tp_size == 0):
            return False
        # Skip fusion for very small M — the fused Triton kernel requires
        # M >= block_size_m (min 16) and small batches don't benefit from fusion.
        return compile_range.end >= self.MIN_M_FOR_FUSION

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.info("GEMMAllReducePass replaced %s patterns", self.matched_count)


class AsyncTPPass(VllmFusionPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, pass_name="async_tp_pass")

        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        GEMMReduceScatterPattern(self.model_dtype, self.device).register(self.pm_pass)

        # Registered unconditionally, as upstream does: the replacement picks
        # aiter or symm_mem at trace time depending on availability.
        AllGatherGEMMPattern(self.model_dtype, self.device).register(self.pm_pass)

        # Find K-shard AGMM pattern. It has no symm_mem equivalent, so it is
        # registered only when the aiter op is available.
        if _has_aiter_op("fused_all_gather_gemm_k_shard"):
            AllGatherGEMMPatternKShard(self.model_dtype, self.device).register(
                self.pm_pass
            )
        elif current_platform.is_rocm():
            logger.warning(
                "aiter.fused_all_gather_gemm_k_shard op not registered, "
                "skipping AllGatherGEMMPatternKShard"
            )

        # These fusions are enabled only for bfloat16 models because
        # `scaled_mm` or `cutlass_scaled_mm` with per-token (row-wise) scaling
        # only supports bfloat16 as the output dtype.
        if self.model_dtype == torch.bfloat16:
            ScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.pm_pass
            )
            AllGatherScaledMMPattern(self.model_dtype, self.device).register(
                self.pm_pass
            )
            if hasattr(torch.ops._C, "cutlass_scaled_mm"):
                CutlassScaledMMReduceScatterPattern(
                    self.model_dtype, self.device
                ).register(self.pm_pass)
                AllGatherCutlassScaledMMPattern(self.model_dtype, self.device).register(
                    self.pm_pass
                )
            with suppress(ImportError):
                import vllm.utils.flashinfer  # noqa: F401
            if hasattr(torch.ops.vllm, "bmm_fp8"):
                self.register(
                    FlashInferAllGatherBMMFP8Pattern(self.model_dtype, self.device)
                )
                self.register(
                    FlashInferBMMFP8ReduceScatterPattern(self.model_dtype, self.device)
                )
            if hasattr(torch.ops.vllm, "flashinfer_mm_fp4"):
                for backend in ("cutlass", "cudnn"):
                    for a_scale_view in ("float8_uint8", "uint8"):
                        self.register(
                            FlashInferAllGatherFP4Pattern(
                                self.model_dtype,
                                self.device,
                                backend,
                                use_8x4_sf_layout=False,
                                a_scale_view=a_scale_view,
                            )
                        )
                for use_8x4_sf_layout in (False, True):
                    for a_scale_view in ("float8",):
                        self.register(
                            FlashInferAllGatherFP4Pattern(
                                self.model_dtype,
                                self.device,
                                "trtllm",
                                use_8x4_sf_layout=use_8x4_sf_layout,
                                a_scale_view=a_scale_view,
                            )
                        )
                self.register(
                    FlashInferAllGatherFP4Pattern(
                        self.model_dtype,
                        self.device,
                        "cute-dsl",
                        use_8x4_sf_layout=False,
                        a_scale_view="float8",
                    )
                )
                # NVFP4 reduce-scatter does not need scale communication: FP4
                # scales are consumed by the local GEMM and only BF16 partial
                # outputs are reduced. Keep this PR scoped to the all-gather
                # path; reduce-scatter needs a dedicated FP4 producer rather
                # than the existing FP8-style helper.

        # Registered last so that every upstream reduce-scatter pattern gets
        # first refusal on the reduce_scatter node they all anchor on.
        if current_platform.is_rocm():
            self._register_rocm_reduce_scatter(config)

        self.dump_patterns(config, self.pm_pass)

    def _register_rocm_reduce_scatter(self, config: VllmConfig) -> None:
        """Register the iris-backed reduce-scatter patterns on ROCm.

        Args:
            config: The full vLLM config, used to derive the warmup shapes.
        """
        mode = getattr(self.pass_config, "fuse_gemm_reduce_scatter", "off")
        if mode == "off":
            return
        if mode not in ("fused", "standalone"):
            logger.warning(
                "Unknown fuse_gemm_reduce_scatter=%r, expected one of "
                "'off', 'fused', 'standalone'; disabling the fusion",
                mode,
            )
            return

        if mode == "standalone":
            if not _has_aiter_op("reduce_scatter_m_shard"):
                logger.warning(
                    "aiter.reduce_scatter_m_shard op not registered, skipping "
                    "ROCmReduceScatterPattern"
                )
                return
            ROCmReduceScatterPattern(self.model_dtype, self.device).register(
                self.pm_pass
            )
        else:
            if not _has_aiter_op("gemm_reduce_scatter_m_shard"):
                logger.warning(
                    "aiter.gemm_reduce_scatter_m_shard op not registered, skipping "
                    "ROCmGEMMReduceScatterPattern"
                )
                return
            if not hasattr(torch.ops.vllm, "rocm_unquantized_gemm"):
                logger.warning(
                    "rocm_unquantized_gemm op not registered, skipping "
                    "ROCmGEMMReduceScatterPattern"
                )
                return
            ROCmGEMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.pm_pass
            )
            ROCmGEMMReduceScatterWithBiasPattern(
                self.model_dtype, self.device
            ).register(self.pm_pass)

        logger.info("Registered ROCm iris reduce-scatter patterns (mode=%s)", mode)
        self._register_reduce_scatter_warmup_shapes(config)

    def _register_reduce_scatter_warmup_shapes(self, config: VllmConfig) -> None:
        """Declare every (M, N, dtype) the aiter reduce-scatter cache must hold.

        A cache miss inside a graph capture region performs a collective
        symmetric-heap allocation and a Triton JIT compile, and deadlocks. The
        shapes are recorded here and materialized by a later collective
        `warmup_buffers()`, which must run after this pass is constructed and
        before capture.

        M is every cudagraph capture size and compile size divisible by the TP
        size; SequenceParallelismPass only reduce-scatters the residual stream,
        so N is the model hidden size.

        Args:
            config: The full vLLM config, holding the capture sizes and the
                model hidden size.
        """
        if config.model_config is None or self.model_dtype is None:
            logger.warning(
                "No model config available, cannot pre-declare fused "
                "reduce-scatter warmup shapes; the first captured forward will "
                "allocate collectively and hang"
            )
            return

        try:
            from aiter.ops.triton.comms.fused import fused_gemm_reduce_scatter
        except ImportError:
            logger.warning(
                "aiter.ops.triton.comms.fused.fused_gemm_reduce_scatter is not "
                "importable, cannot pre-declare reduce-scatter warmup shapes"
            )
            return

        n = config.model_config.get_hidden_size()
        compilation_config = config.compilation_config
        candidates = set(compilation_config.cudagraph_capture_sizes or ())
        candidates.update(
            m for m in (compilation_config.compile_sizes or ()) if isinstance(m, int)
        )
        tp_size = get_tensor_model_parallel_world_size()
        m_sizes = sorted(m for m in candidates if m > 0 and m % tp_size == 0)

        for m in m_sizes:
            fused_gemm_reduce_scatter.register_shape(m, n, self.model_dtype)

        logger.info(
            "Declared %d fused reduce-scatter warmup shape(s): M=%s, N=%d, dtype=%s",
            len(m_sizes),
            m_sizes,
            n,
            self.model_dtype,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass is applied on top of the sequence parallelism pass,
        # which is only supported in fullgraph compilation mode.
        assert (
            self.compilation_config.use_inductor_graph_partition
            or not self.compilation_config.splitting_ops
        ), "AsyncTPPass requires full-graph compilation"
        return True

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.pm_pass.apply(graph)
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        logger.debug("Replaced %s patterns", self.matched_count)
