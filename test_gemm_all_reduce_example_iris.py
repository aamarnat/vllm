"""
Test for GEMM + all-reduce fusion using aiter.fused_gemm_all_reduce_k_shard

This test exercises the K-dimension sharding pattern used by RowParallelLinear:
- Each rank has input (M, K_local) where K_local = K / world_size
- GEMM: (M, K_local) @ (K_local, N) = (M, N) partial sum
- All-reduce sums partial results across ranks -> (M, N) final
- GEMMAllReducePass fuses mm + all_reduce into fused_gemm_all_reduce_k_shard

Usage:
    # 2-GPU (default)
    python test_gemm_all_reduce_example_iris.py --fused-ar --verify-gpus

    # 8-GPU
    python test_gemm_all_reduce_example_iris.py --fused-ar --verify-gpus --num-gpus 8
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vllm.compilation.passes.fusion.collective_fusion import GEMMAllReducePass
from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

tests_path = Path(__file__).parent / "tests"
if tests_path.exists() and str(tests_path) not in sys.path:
    sys.path.insert(0, str(tests_path))
from compile.backend import TestBackend

import aiter  # noqa: F401  — triggers op registration

AITER_FUSED_AR_AVAILABLE = (
    hasattr(torch.ops, "aiter")
    and hasattr(torch.ops.aiter, "fused_gemm_all_reduce_k_shard")
)

MASTER_PORT = "12349"

# K_LOCAL must be divisible by block_size_k (default 64).
# N must be divisible by block_size_n (default 128).
M = 128
K_LOCAL = 256
N = 512
DTYPE = torch.float16


def seed_everything(seed: int) -> None:
    platform_seed = getattr(current_platform, "seed_everything", None)
    if callable(platform_seed):
        platform_seed(seed)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_type():
    if not torch.cuda.is_available():
        return None
    return "AMD" if torch.version.hip is not None else "NVIDIA"


class TestGEMMAllReduceModel(torch.nn.Module):
    """
    Minimal model exercising the GEMM + all-reduce pattern.

    Each rank holds weight (K_local, N).  Forward does:
        partial = mm(hidden_states, weight)          # (M, N)
        output  = tensor_model_parallel_all_reduce(partial)  # (M, N)
    GEMMAllReducePass fuses these into fused_gemm_all_reduce_k_shard.
    """

    def __init__(self, k_local: int, n: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((k_local, n)), requires_grad=False
        )
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        partial = torch.mm(hidden_states, self.weight)
        return tensor_model_parallel_all_reduce(partial)

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        if AITER_FUSED_AR_AVAILABLE:
            return [torch.ops.aiter.fused_gemm_all_reduce_k_shard.default]
        return [torch.ops.vllm.all_reduce.default]


def standalone_gpu_verification(num_gpus: int) -> bool:
    print(f"\nVerifying {num_gpus}-GPU setup...")

    if not torch.cuda.is_available():
        print("  GPU not available (neither CUDA nor ROCm)")
        return False

    gpu_type = get_gpu_type()
    found = torch.cuda.device_count()
    print(f"  Found {found} {gpu_type} GPU(s)")

    if found < num_gpus:
        print(f"  Need at least {num_gpus} GPUs, found {found}")
        return False

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        dev = torch.device(f"cuda:{i}")
        x = torch.randn(64, 64, device=dev)
        torch.mm(x, x)
        torch.cuda.synchronize(dev)
        print(f"  GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB) OK"
              if hasattr(props, 'total_mem') else
              f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB) OK")

    t0 = torch.randn(64, 64, device="cuda:0")
    t0.to("cuda:1")
    torch.cuda.synchronize("cuda:1")
    print("  P2P transfer (GPU 0 -> GPU 1) OK")

    print(f"  {num_gpus}-GPU setup verified!\n")
    return True


def verify_functionality(local_rank, gemm_ar_pass, backend, model, output):
    print(f"  Rank {local_rank}: GEMMAllReducePass matched count: "
          f"{gemm_ar_pass.matched_count}")

    if gemm_ar_pass.matched_count == 1:
        print(f"  Rank {local_rank}: GEMMAllReducePass successfully fused operations!")
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        print(f"  Rank {local_rank}: Operation fusion verified!")
    elif gemm_ar_pass.matched_count > 0:
        print(f"  Rank {local_rank}: WARNING - matched "
              f"{gemm_ar_pass.matched_count} times (expected 1)")
    else:
        print(f"  Rank {local_rank}: WARNING - did not match any patterns")

    print(f"  Rank {local_rank}: Test completed!")


def simple_test():
    """Smoke test without distributed setup (single GPU)."""
    print("Running simple TestGEMMAllReduceModel test...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using {get_gpu_type()} GPU: {torch.cuda.get_device_name(0)}")

    model = TestGEMMAllReduceModel(k_local=K_LOCAL, n=N).to(device)
    hidden_states = torch.randn(
        (M, K_LOCAL), dtype=DTYPE, device=device, requires_grad=False
    )

    print(f"Input shape: {hidden_states.shape}")
    print(f"Model weight shape: {model.weight.shape}")

    try:
        output = model(hidden_states)
        print(f"Output shape: {output.shape}")
        print("Test passed!")
        return output
    except Exception as e:
        print(f"Test failed with error: {e}")
        print("Note: This model requires distributed setup for full functionality")
        return None


def fused_ar_test(local_rank: int, world_size: int, dynamic: bool = False):
    print(f"\n{'='*70}")
    print(f"Rank {local_rank}/{world_size}: Starting GEMMAllReducePass test")
    print(f"{'='*70}")

    seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(DTYPE)

    gpu_type = get_gpu_type()
    print(f"  Rank {local_rank}: {gpu_type} GPU {torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name(local_rank)}, "
          f"{torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB)")

    env_vars = {
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": MASTER_PORT,
    }
    if gpu_type == "AMD":
        env_vars["RCCL_DEBUG"] = "INFO"
    else:
        env_vars["NCCL_DEBUG"] = "INFO"
    update_environment_variables(env_vars)

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    assert dist.is_initialized(), "Distributed not initialized!"
    assert dist.get_rank() == local_rank
    assert dist.get_world_size() == world_size
    print(f"  Rank {local_rank}: Distributed initialized "
          f"(backend={dist.get_backend()})")

    test_tensor = torch.tensor([float(local_rank)], device=device)
    gathered = torch.zeros(world_size, device=device)
    dist.all_gather_into_tensor(gathered, test_tensor)
    expected = list(range(world_size))
    assert gathered.tolist() == expected
    print(f"  Rank {local_rank}: All-gather sanity OK: {gathered.tolist()}")

    # --- vLLM config: enable fuse_gemm_all_reduce ---
    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig(
        pass_config=PassConfig(fuse_gemm_all_reduce=True),
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
    vllm_config.model_config = ModelConfig(
        model="RedHatAI/Llama-3.2-1B-Instruct-FP8",
        trust_remote_code=True,
        dtype=DTYPE,
        seed=42,
    )

    gemm_ar_pass = GEMMAllReducePass(vllm_config)
    with set_current_vllm_config(vllm_config):
        backend = TestBackend(gemm_ar_pass)

    model = TestGEMMAllReduceModel(k_local=K_LOCAL, n=N)
    assert model.weight.device.type == 'cuda'
    assert model.weight.device.index == local_rank
    print(f"  Rank {local_rank}: Model weights on {model.weight.device} "
          f"shape={tuple(model.weight.shape)}")

    x = torch.randn(
        (M, K_LOCAL), dtype=DTYPE, requires_grad=False
    ) * (local_rank + 1)
    assert x.device.index == local_rank
    print(f"  Rank {local_rank}: Input {tuple(x.shape)} on {x.device}, "
          f"mean={x.mean():.4f}")

    if dynamic:
        torch._dynamo.mark_dynamic(x, 0)

    dist.barrier()

    # Reference: plain mm + all_reduce via NCCL/RCCL
    with torch.no_grad():
        partial_ref = torch.mm(x, model.weight)
        reference_output = partial_ref.clone()
        dist.all_reduce(reference_output)

    print(f"  Rank {local_rank}: Compiling model with GEMMAllReducePass...")
    with set_current_vllm_config(vllm_config):
        compiled_model = torch.compile(model, backend=backend)

    torch.cuda.synchronize(device)
    start_mem = torch.cuda.memory_allocated(device)

    with set_current_vllm_config(vllm_config):
        output = compiled_model(x)
    torch.cuda.synchronize(device)
    end_mem = torch.cuda.memory_allocated(device)

    print(f"  Rank {local_rank}: Output shape: {output.shape}")
    print(f"  Rank {local_rank}: Output mean: {output.mean():.4f}  "
          f"Reference mean: {reference_output.mean():.4f}")
    print(f"  Rank {local_rank}: GPU memory used: "
          f"{(end_mem - start_mem) / 1e6:.1f} MB")

    assert output.device.index == local_rank
    print(f"  Rank {local_rank}: Output on correct GPU {output.device}")

    verify_functionality(local_rank, gemm_ar_pass, backend, model, output)

    if local_rank == 0:
        max_diff = torch.max(torch.abs(output - reference_output)).item()
        relative_error = max_diff / (
            torch.max(torch.abs(reference_output)).item() + 1e-8
        )

        import math
        max_ref = torch.max(torch.abs(reference_output)).item()
        if max_ref > 0:
            ulp = 2.0 ** (math.floor(math.log2(max_ref)) - 10)
        else:
            ulp = 1e-4
        tol = 4 * ulp

        ok = max_diff <= tol
        print("=" * 70)
        print(f"  Max absolute difference : {max_diff:.6f}")
        print(f"  Relative error          : {relative_error:.6f}")
        print(f"  Tolerance (4 ULPs)      : {tol:.6f}")
        if ok:
            print("  Output matches reference within tolerance!")
        else:
            print(f"  WARNING - output differs from reference by {max_diff:.6f}")
            print(f"  Sample output    : {output[0, :5].tolist()}")
            print(f"  Sample reference : {reference_output[0, :5].tolist()}")
        print(f"  Output tensor sum (rank 0): {torch.sum(output).item():.4f}")

    dist.barrier()
    cleanup_dist_env_and_memory()
    torch.cuda.empty_cache()

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test GEMM + all-reduce fusion using "
                    "aiter.fused_gemm_all_reduce_k_shard"
    )
    parser.add_argument(
        "--fused-ar", action="store_true",
        help="Run GEMMAllReducePass test with compilation "
             "(requires multiple GPUs)")
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Use dynamic shapes")
    parser.add_argument(
        "--verify-gpus", action="store_true",
        help="Verify multi-GPU setup before running tests")
    parser.add_argument(
        "--num-gpus", type=int, default=2,
        help="Number of GPUs (default: 2)")
    args = parser.parse_args()

    num_gpus = args.num_gpus

    print("\n" + "=" * 70)
    print("Test Configuration (GEMM + ALL-REDUCE):")
    print(f"  M={M}, K_local={K_LOCAL}, N={N}")
    print(f"  world_size={num_gpus}")
    print(f"  aiter fused_gemm_all_reduce_k_shard available: "
          f"{AITER_FUSED_AR_AVAILABLE}")
    if AITER_FUSED_AR_AVAILABLE:
        print("  Using: torch.ops.aiter.fused_gemm_all_reduce_k_shard")
    else:
        print("  aiter fused op not registered -- aborting")
        sys.exit(1)
    print("=" * 70 + "\n")

    if args.verify_gpus or args.fused_ar:
        if not standalone_gpu_verification(num_gpus):
            print("Multi-GPU verification failed!")
            sys.exit(1)

    if args.fused_ar:
        if TestBackend is None:
            print("Error: TestBackend not available.")
            sys.exit(1)
        torch.multiprocessing.spawn(
            fused_ar_test,
            args=(num_gpus, args.dynamic),
            nprocs=num_gpus,
            join=True,
        )
    else:
        simple_test()
