# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from torch import fx as fx

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import set_env_var

from .ir.clone_elimination import UnsafeCloneEliminationPass
from .ir.lowering_pass import VllmIRLoweringPass
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

if rocm_aiter_ops.is_enabled():
    from .fusion.allreduce_rms_fusion import (
        RocmAiterAllReduceFusionPass,
    )
    from .fusion.rocm_aiter_fusion import (
        MLADualRMSNormFusionPass,
        RocmAiterRMSNormQuantFusionPass,
        RocmAiterSiluMulFp8GroupQuantFusionPass,
        RocmAiterTritonAddRMSNormPadFusionPass,
    )

if current_platform.is_cuda_alike() or current_platform.is_xpu():
    from .fusion.add_rms_fusion import (
        AddRMSNormFusionPass,
        RMSNormReshapeFusionPass,
    )
    from .fusion.collective_fusion import AsyncTPPass
    from .fusion.qk_norm_rope_fusion import QKNormRoPEFusionPass
    from .fusion.sequence_parallelism import SequenceParallelismPass
    from .utility.split_coalescing import SplitCoalescingPass

if current_platform.is_cuda_alike():
    from .fusion.act_quant_fusion import ActivationQuantFusionPass
    from .fusion.attn_quant_fusion import AttnQuantFusionPass
    from .fusion.mla_attn_quant_fusion import MLAAttnQuantFusionPass
    from .fusion.mla_rope_kvcache_cat_fusion import MLARoPEKVCacheCatFusionPass
    from .fusion.qk_norm_rope_kvcache_fusion import QkNormRopeKvCacheFusionPass
    from .fusion.rms_quant_fusion import RMSNormQuantFusionPass
    from .fusion.rope_kvcache_fusion import RopeKVCacheFusionPass
    from .utility.scatter_split_replace import ScatterSplitReplacementPass

if current_platform.is_cuda():
    from .fusion.allreduce_rms_fusion import AllReduceFusionPass

if current_platform.is_xpu():
    from .fusion.act_quant_fusion import ActivationQuantFusionPass
    from .fusion.rms_quant_fusion import RMSNormQuantFusionPass

from .inductor_pass import (
    CustomGraphPass,
    InductorPass,
    get_pass_context,
)
from .utility.fix_functionalization import FixFunctionalizationPass
from .utility.noop_elimination import NoOpEliminationPass

logger = init_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def with_pattern_match_debug(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Function decorator that turns on inductor pattern match debug
    for the duration of the call.
    Used to avoid logging builtin Inductor pattern matching.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if (debug_val := envs.VLLM_PATTERN_MATCH_DEBUG) is not None:
            # optionally check rank here
            with set_env_var("TORCHINDUCTOR_PATTERN_MATCH_DEBUG", debug_val):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


class PostGradPassManager(CustomGraphPass):  # type: ignore[misc]
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It supports uuid for the Inductor code cache. That includes torch<2.6
    support using pickling (in .inductor_pass.CustomGraphPass).

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (NoopEliminationPass, FusionPass)
    3. config["post_grad_custom_post_pass"] (if it exists)
    4. fix_functionalization
    This way, all passes operate on a functionalized graph.
    """

    def __init__(self) -> None:
        self.passes: list[InductorPass] = []

    @with_pattern_match_debug
    def __call__(self, graph: fx.Graph) -> None:
        VllmInductorPass.dump_prefix = 0  # reset dump index

        compile_range = get_pass_context().compile_range
        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
                VllmInductorPass.dump_prefix += 1
            else:
                logger.debug("Skipping %s with compile range %s", pass_, compile_range)

        # perform the first post-cleanup before IR lowering to clean up fusion artifacts
        # and make sure no dead IR ops are lowered.
        self.post_cleanup(graph)
        VllmInductorPass.dump_prefix += 1

        # lowering before cleanup so DCE can clean up lowered ops.
        # DCE handles mutating ops correctly as well.
        self.ir_lowering(graph)
        VllmInductorPass.dump_prefix += 1
        self.clone_elimination(graph)
        VllmInductorPass.dump_prefix += 1

        # clean up after lowering again
        self.post_cleanup(graph)
        VllmInductorPass.dump_prefix += 1

        # always run fix_functionalization last
        self.fix_functionalization(graph)
        VllmInductorPass.dump_prefix = None  # Cleanup index

        VllmPatternMatcherPass.log_match_summary()

    def _warmup_fused_reduce_scatter(self, config: VllmConfig) -> None:
        """Size the iris symmetric heap from this deployment and warm the ops.

        Must run after `AsyncTPPass` is constructed, since that is what declares
        the shapes. Anything left unwarmed performs a Triton JIT compile or an RCCL
        communicator creation on its first forward, both of which deadlock inside a
        graph capture region.

        The heap size is DERIVED here, from the scheduler's own token ceiling, and
        not left to the op's fallback. It used to be: `init_iris()` with no argument
        took a fixed 1 GiB default, the warmup covered only the cudagraph capture
        sizes (largest M=512), and the first prefill wider than that allocated on the
        heap mid-forward. The heap never frees, so a few such prefills exhausted it
        and killed the engine a handful of requests in. `max_num_batched_tokens` is
        the number that bounds what the scheduler can hand the op, so it is the
        number both the heap size and the staging slabs are computed from.

        Args:
            config: The full vLLM config, holding the capture sizes, the model
                dtype and hidden size, and the scheduler's token budget.
        """
        from vllm.distributed import get_tp_group
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )

        try:
            from aiter.ops.triton.comms.fused.fused_gemm_reduce_scatter import (
                HeapTooSmallError,
                heap_bytes_required,
                init_iris,
                warmup_buffers,
            )
        except ImportError as e:
            logger.warning("Failed to import iris fused GEMM+reduce_scatter: %s", e)
            return

        compilation_config = config.compilation_config
        compile_sizes = set(compilation_config.cudagraph_capture_sizes or ())
        compile_sizes.update(
            m for m in (compilation_config.compile_sizes or ()) if isinstance(m, int)
        )

        model_config = config.model_config
        weight_shapes = None
        dtype = None
        if model_config is not None:
            hidden_size = model_config.get_hidden_size()
            weight_shapes = [(hidden_size, hidden_size)]
            dtype = model_config.dtype

        tp_size = get_tensor_model_parallel_world_size()
        max_m = self._fused_reduce_scatter_max_m(config, compile_sizes, tp_size)

        heap_size = None
        if weight_shapes is not None and dtype is not None:
            heap_size = heap_bytes_required(
                max_m, [n for _, n in weight_shapes], [dtype], tp_size
            )
            logger.info(
                "Sizing the iris symmetric heap for fused GEMM+reduce_scatter: "
                "max_num_batched_tokens -> M_max=%d, N=%d, dtype=%s, TP=%d "
                "-> %d bytes (%.1f MiB)",
                max_m,
                hidden_size,
                dtype,
                tp_size,
                heap_size,
                heap_size / (1 << 20),
            )
        else:
            logger.warning(
                "No model config, so the iris symmetric heap cannot be sized from "
                "this deployment; falling back to the aiter default, which is not "
                "sized for any particular workload"
            )

        try:
            init_iris(heap_size=heap_size)
            warmup_buffers(
                compile_sizes=sorted(compile_sizes),
                weight_shapes=weight_shapes,
                dtype=dtype,
                group_name=get_tp_group().device_group.group_name,
                max_m=max_m,
            )
        except HeapTooSmallError:
            # Deliberately NOT swallowed. The broad handler below exists so that a
            # broken or absent iris degrades to the RCCL path instead of killing the
            # server -- but degrading is not on the table here. By the time this runs
            # the fusion pass has already rewritten the graph, so the fused op WILL
            # be called; a heap too small to stage it means the server starts and
            # then dies mid-request with an allocator OOM, which is precisely the
            # failure the capacity check exists to replace. Refuse at startup, where
            # the message names the numbers and the operator can act on it.
            raise
        except Exception as e:
            logger.warning(
                "Failed to initialize iris for fused GEMM+reduce_scatter: %s", e
            )

    @staticmethod
    def _fused_reduce_scatter_max_m(
        config: VllmConfig, compile_sizes: set[int], tp_size: int
    ) -> int:
        """Largest M the scheduler can hand the fused reduce-scatter ops.

        `max_num_batched_tokens` is the scheduler's ceiling on tokens per forward and
        so on the reduce-scatter's M. It is rounded UP to a multiple of the TP size
        because the op shards M across ranks and rejects an M that is not divisible.
        The captured sizes are folded in only as a floor: they are always <= the
        budget in practice, but a config that captured above it must still be warmed.

        Args:
            config: The full vLLM config.
            compile_sizes: Capture and compile sizes already collected.
            tp_size: Tensor-parallel world size.

        Returns:
            M_max in rows.
        """
        budget = 0
        scheduler_config = getattr(config, "scheduler_config", None)
        if scheduler_config is not None:
            budget = int(getattr(scheduler_config, "max_num_batched_tokens", 0) or 0)
        largest_captured = max(compile_sizes) if compile_sizes else 0
        max_m = max(budget, largest_captured, tp_size)
        return -(-max_m // tp_size) * tp_size

    def configure(self, config: VllmConfig) -> None:
        self.pass_config = config.compilation_config.pass_config
        model_config = config.model_config
        enable_transformers_norm_canonicalization = (
            (
                self.pass_config.fuse_act_padding
                or self.pass_config.fuse_allreduce_rms
                or self.pass_config.fuse_norm_quant
            )
            and model_config is not None
            and model_config.using_transformers_backend()
        )

        # Set the current vllm config to allow tracing CustomOp instances
        with set_current_vllm_config(config, check_compile=False):
            if self.pass_config.eliminate_noops:
                self.passes += [NoOpEliminationPass(config)]

            if self.pass_config.fuse_gemm_all_reduce:
                # Import aiter BEFORE constructing the pass: the aiter custom
                # ops are registered as a side effect of this import, and the
                # pass refuses to build if they are missing.
                if current_platform.is_rocm():
                    try:
                        from aiter.ops.triton.comms.fused.fused_gemm_all_reduce import (
                            init_iris,
                            warmup_buffers,
                        )

                        init_iris()
                        # Pre-cache device_context tensor and SM count so that
                        # these host-side operations don't happen during
                        # CUDAGraph capture. Per-shape buffer allocation
                        # happens during the warmup forward pass.
                        warmup_buffers()
                    except Exception as e:
                        logger.warning(
                            "Failed to initialize iris for fused GEMM+AR: %s", e
                        )

                from .fusion.collective_fusion import GEMMAllReducePass

                self.passes += [GEMMAllReducePass(config)]

            if self.pass_config.enable_sp:
                self.passes += [SequenceParallelismPass(config)]
                if self.pass_config.fuse_gemm_comms:
                    if current_platform.is_rocm():
                        # AsyncTPPass probes for aiter ops when deciding which
                        # patterns to register, so import aiter first.
                        with contextlib.suppress(Exception):
                            import aiter.ops.triton.comms.fused  # noqa: F401

                    self.passes += [AsyncTPPass(config)]

                    if current_platform.is_rocm() and (
                        self.pass_config.fuse_gemm_reduce_scatter not in (None, "off")
                    ):
                        self._warmup_fused_reduce_scatter(config)

            if enable_transformers_norm_canonicalization:
                self.passes += [AddRMSNormFusionPass(config)]

            if self.pass_config.fuse_act_padding and rocm_aiter_ops.is_enabled():
                # Run the more specific RMSNorm+router-pad fusion before
                # AR+RMS, since both consume fused_add_rms_norm.
                self.passes += [RocmAiterTritonAddRMSNormPadFusionPass(config)]

            if self.pass_config.fuse_allreduce_rms:
                if rocm_aiter_ops.is_enabled():
                    self.passes += [RocmAiterAllReduceFusionPass(config)]
                else:
                    self.passes += [AllReduceFusionPass(config)]

            if enable_transformers_norm_canonicalization:
                # Let AR+RMS match before moving output reshapes ahead of the
                # remaining RMSNorms, exposing them to RMS+Quant fusion.
                self.passes += [RMSNormReshapeFusionPass(config)]

            if self.pass_config.fuse_norm_quant:
                if rocm_aiter_ops.is_enabled():
                    self.passes += [
                        RocmAiterRMSNormQuantFusionPass(config),
                    ]
                self.passes += [RMSNormQuantFusionPass(config)]

            if self.pass_config.fuse_act_quant:
                self.passes += [ActivationQuantFusionPass(config)]
                if rocm_aiter_ops.is_enabled():
                    self.passes += [RocmAiterSiluMulFp8GroupQuantFusionPass(config)]

            if self.pass_config.fuse_qk_norm_rope_kvcache:
                self.passes += [SplitCoalescingPass(config)]
                self.passes += [ScatterSplitReplacementPass(config)]
                self.passes += [QkNormRopeKvCacheFusionPass(config)]

            if self.pass_config.fuse_mla_dual_rms_norm and rocm_aiter_ops.is_enabled():
                self.passes += [MLADualRMSNormFusionPass(config)]

            if self.pass_config.fuse_rope_kvcache:
                self.passes += [SplitCoalescingPass(config)]
                self.passes += [ScatterSplitReplacementPass(config)]
                self.passes += [RopeKVCacheFusionPass(config)]

            if self.pass_config.fuse_rope_kvcache_cat_mla:
                self.passes += [MLARoPEKVCacheCatFusionPass(config)]

            if self.pass_config.fuse_attn_quant:
                self.passes += [AttnQuantFusionPass(config)]
                self.passes += [MLAAttnQuantFusionPass(config)]

            if self.pass_config.enable_qk_norm_rope_fusion:
                self.passes += [SplitCoalescingPass(config)]
                self.passes += [QKNormRoPEFusionPass(config)]

            self.ir_lowering = VllmIRLoweringPass(config)
            self.clone_elimination = UnsafeCloneEliminationPass(config)
            self.post_cleanup = PostCleanupPass(config)
            self.fix_functionalization = FixFunctionalizationPass(config)

    def add(self, pass_: InductorPass) -> None:
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self) -> str:
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        passes = []

        state: dict[str, Any] = {"pass_config": self.pass_config.compute_hash()}
        for pass_ in self.passes:
            passes.append(pass_.uuid())

        passes.append(self.post_cleanup.uuid())
        passes.append(self.ir_lowering.uuid())
        passes.append(self.clone_elimination.uuid())
        passes.append(self.post_cleanup.uuid())
        passes.append(self.fix_functionalization.uuid())

        # Include the compile range in the uuid to ensure that inductor
        # recompiles the graph for the new dynamic compile range.
        state["compile_range"] = str(get_pass_context().compile_range)
        state["passes"] = passes
        return InductorPass.hash_dict(state)
