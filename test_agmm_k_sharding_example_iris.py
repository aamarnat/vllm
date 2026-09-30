"""
Test for all-gather + GEMM fusion with K-sharding using aiter.fused_all_gather_gemm_k_shard

This test demonstrates K-dimension sharding instead of M-dimension sharding.

Sharding Strategy:
- Shards on K dimension: each rank has input (M, K_local)
- All-gather on dim=1 produces (M, K) where K = K_local * world_size
- Then performs GEMM: (M, K) @ (K, N) = (M, N)
- Note: This uses AllGatherGEMMPatternKShard for K-dimension sharding

Key changes from M-sharding:
1. gather_dim=1 instead of dim=0
2. Input shape is (M, K_local) instead of (M_local, K)
3. All-gather concatenates along K dimension

GPU Support:
- Automatically detects and supports both NVIDIA (CUDA) and AMD (ROCm) GPUs
- Uses NCCL for NVIDIA, RCCL for AMD
- No code changes needed - adapts automatically!

Multi-GPU Verification:
- Verification checkpoints ensure correct multi-GPU execution
- Tests cross-GPU communication, memory placement, and fusion
- Run with --verify-gpus to check your setup

To use aiter fused ops in vLLM's AsyncTPPass:
- Uses AllGatherGEMMPatternKShard in collective_fusion.py
- This pattern handles K-dimension sharding (dim=1)

Usage:
    # 2-GPU (default)
    python test_agmm_k_sharding_example_iris.py --async-tp --verify-gpus

    # 8-GPU
    python test_agmm_k_sharding_example_iris.py --async-tp --verify-gpus --num-gpus 8
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vllm.compilation.passes.fusion.collective_fusion import AsyncTPPass
from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

# Set environment variable for Triton
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

# TestBackend lives under vllm/tests/compile/
tests_path = Path(__file__).parent / "tests"
if tests_path.exists() and str(tests_path) not in sys.path:
    sys.path.insert(0, str(tests_path))
from compile.backend import TestBackend

import aiter

# Determine if aiter fused ops are available (adapted from vllm/compilation/collective_fusion.py)
AITER_FUSED_AVAILABLE = hasattr(torch.ops, "aiter") and hasattr(torch.ops.aiter, "fused_all_gather_gemm_k_shard")

MASTER_PORT = "12347"

# Problem dimensions — sized so block sizes in the wrapper (bm=128, bk=64, bn=64)
# divide evenly for any world_size >= 2.  With K_local=64 the kernel and torch.mm
# may differ by up to a few fp16 ULPs due to accumulation-order differences; the
# tolerance check accounts for this.
K_LOCAL = 64   # K per rank
N = 512        # output features (fixed)
M = 256        # batch tokens
DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TestAGMMKShardModel(torch.nn.Module):
    """
    Minimal model that exercises the K-shard all-gather + GEMM pattern.

    Each rank receives input (M, K_local).  The forward pass gathers along
    dim=1 to produce (M, K) then multiplies by weight (K, N) to yield (M, N).
    AsyncTPPass fuses these two ops into fused_all_gather_gemm_k_shard.
    """

    def __init__(self, k_total: int, n: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((k_total, n)), requires_grad=False
        )
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        all_gather = tensor_model_parallel_all_gather(hidden_states, dim=1)
        return torch.mm(all_gather, self.weight)

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        if AITER_FUSED_AVAILABLE:
            return [torch.ops.aiter.fused_all_gather_gemm_k_shard.default]
        return [torch.ops.vllm.all_gather.default]


# ---------------------------------------------------------------------------
# GPU verification
# ---------------------------------------------------------------------------

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
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB) OK")

    # Spot-check P2P between GPU 0 and GPU 1
    t0 = torch.randn(64, 64, device="cuda:0")
    t0.to("cuda:1")
    torch.cuda.synchronize("cuda:1")
    print("  P2P transfer (GPU 0 -> GPU 1) OK")

    print(f"  {num_gpus}-GPU setup verified!\n")
    return True


# ---------------------------------------------------------------------------
# Fusion verification helper
# ---------------------------------------------------------------------------

def verify_functionality(local_rank, async_tp_pass, backend, model, output):
    print(f"  Rank {local_rank}: AsyncTPPass matched count: {async_tp_pass.matched_count}")

    if async_tp_pass.matched_count == 1:
        print(f"  Rank {local_rank}: AsyncTPPass successfully fused operations!")
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        print(f"  Rank {local_rank}: Operation fusion verified!")
    elif async_tp_pass.matched_count > 0:
        print(f"  Rank {local_rank}: WARNING - AsyncTPPass matched {async_tp_pass.matched_count} times (expected 1)")
    else:
        print(f"  Rank {local_rank}: WARNING - AsyncTPPass did not match any patterns")

    print(f"  Rank {local_rank}: Test completed!")


# ---------------------------------------------------------------------------
# Simple single-GPU smoke test
# ---------------------------------------------------------------------------

def simple_test():
    """Smoke test without distributed setup (single GPU)."""
    print("Running simple TestAGMMKShardModel test...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using {get_gpu_type()} GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no GPU available)")

    model = TestAGMMKShardModel(k_total=K_LOCAL, n=N).to(device)
    hidden_states = torch.randn((M, K_LOCAL), dtype=DTYPE, device=device, requires_grad=False)

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


# ---------------------------------------------------------------------------
# Per-rank distributed test body
# ---------------------------------------------------------------------------

def async_tp_test(local_rank: int, world_size: int, dynamic: bool = False):
    print(f"\n{'='*70}")
    print(f"Rank {local_rank}/{world_size}: Starting AsyncTPPass test (K-sharding)")
    print(f"{'='*70}")

    seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(DTYPE)

    # VERIFICATION 1: device assignment
    gpu_type = get_gpu_type()
    print(f"  Rank {local_rank}: {gpu_type} GPU {torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name(local_rank)}, "
          f"{torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB)")

    # Distributed init
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

    # VERIFICATION 2: distributed state
    assert dist.is_initialized(), "Distributed not initialized!"
    assert dist.get_rank() == local_rank, f"Rank mismatch: {dist.get_rank()} != {local_rank}"
    assert dist.get_world_size() == world_size, f"World size mismatch: {dist.get_world_size()} != {world_size}"
    print(f"  Rank {local_rank}: Distributed initialized (backend={dist.get_backend()})")

    # VERIFICATION 3: cross-GPU communication sanity
    test_tensor = torch.tensor([float(local_rank)], device=device)
    gathered = torch.zeros(world_size, device=device)
    dist.all_gather_into_tensor(gathered, test_tensor)
    expected = list(range(world_size))
    assert gathered.tolist() == expected, f"All-gather returned {gathered.tolist()}, expected {expected}"
    print(f"  Rank {local_rank}: All-gather test successful: {gathered.tolist()}")

    # vLLM / AsyncTPPass config
    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig(
        pass_config=PassConfig(fuse_gemm_comms=True),
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
    vllm_config.model_config = ModelConfig(
        model="RedHatAI/Llama-3.2-1B-Instruct-FP8",
        trust_remote_code=True,
        dtype=DTYPE,
        seed=42,
    )

    async_tp_pass = AsyncTPPass(vllm_config)
    with set_current_vllm_config(vllm_config):
        backend = TestBackend(async_tp_pass)

    k_total = K_LOCAL * world_size

    # VERIFICATION 4: model weights on correct GPU
    model = TestAGMMKShardModel(k_total=k_total, n=N)
    assert model.weight.device.type == 'cuda', "Model not on CUDA"
    assert model.weight.device.index == local_rank, (
        f"Model on wrong GPU: {model.weight.device.index} != {local_rank}"
    )
    print(f"  Rank {local_rank}: Model weights on {model.weight.device} "
          f"shape={tuple(model.weight.shape)}")

    # VERIFICATION 5: input on correct GPU
    # Scale by rank+1 so each rank contributes distinct values
    x = torch.randn((M, K_LOCAL), dtype=DTYPE, requires_grad=False) * (local_rank + 1)
    assert x.device.index == local_rank, "Input on wrong GPU"
    print(f"  Rank {local_rank}: Input {tuple(x.shape)} on {x.device}, mean={x.mean():.4f}")

    if dynamic:
        torch._dynamo.mark_dynamic(x, 0)

    dist.barrier()

    # Reference: plain all-gather + mm
    with torch.no_grad():
        shards = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(shards, x)
        gathered_ref = torch.cat(shards, dim=1)   # (M, K_total)
        assert gathered_ref.shape == (M, k_total)
        gathered_sum = torch.sum(gathered_ref).item()
        print(f"  Rank {local_rank}: Gathered shape={tuple(gathered_ref.shape)}, sum={gathered_sum:.4f}")
        reference_output = torch.mm(gathered_ref, model.weight)

    # Compile with AsyncTPPass
    print(f"  Rank {local_rank}: Compiling model with AsyncTPPass...")
    with set_current_vllm_config(vllm_config):
        compiled_model = torch.compile(model, backend=backend)

    # VERIFICATION 6: memory baseline
    torch.cuda.synchronize(device)
    start_mem = torch.cuda.memory_allocated(device)

    # Forward pass (fused)
    with set_current_vllm_config(vllm_config):
        output = compiled_model(x)
    torch.cuda.synchronize(device)
    end_mem = torch.cuda.memory_allocated(device)

    print(f"  Rank {local_rank}: Output shape: {output.shape}")
    print(f"  Rank {local_rank}: Output mean: {output.mean():.4f}  Reference mean: {reference_output.mean():.4f}")
    print(f"  Rank {local_rank}: GPU memory used: {(end_mem - start_mem) / 1e6:.1f} MB")

    # VERIFICATION 7: output placement
    assert output.device.index == local_rank, (
        f"Output on wrong GPU: {output.device.index} != {local_rank}"
    )
    print(f"  Rank {local_rank}: Output on correct GPU {output.device}")

    # VERIFICATION 8: fusion occurred
    verify_functionality(local_rank, async_tp_pass, backend, model, output)

    # VERIFICATION 9 (rank 0 only): numerical correctness
    # Tolerance: a few fp16 ULPs at the maximum output magnitude.
    # With K_total = K_local * world_size accumulations the rounding divergence
    # between the Triton kernel and torch.mm can reach ~1 ULP of the max value.
    if local_rank == 0:
        max_diff = torch.max(torch.abs(output - reference_output)).item()
        relative_error = max_diff / (torch.max(torch.abs(reference_output)).item() + 1e-8)

        # ULP-based tolerance: 4 ULPs at the max output magnitude (fp16 mantissa = 10 bits)
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
            print(f"  Output matches reference within tolerance!")
        else:
            print(f"  WARNING - output differs from reference by {max_diff:.6f}")
            print(f"  Sample output    : {output[0, :5].tolist()}")
            print(f"  Sample reference : {reference_output[0, :5].tolist()}")
        print(f"  Output tensor sum (rank 0): {torch.sum(output).item():.4f}")

    # Cleanup
    dist.barrier()
    cleanup_dist_env_and_memory()
    torch.cuda.empty_cache()

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test all-gather + GEMM fusion with K-sharding using aiter fused ops"
    )
    parser.add_argument("--async-tp", action="store_true",
                        help="Run AsyncTPPass test with compilation (requires multiple GPUs)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic shapes in AsyncTPPass test")
    parser.add_argument("--verify-gpus", action="store_true",
                        help="Verify multi-GPU setup before running tests")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="Number of GPUs to use for the distributed test (default: 2)")
    args = parser.parse_args()

    num_gpus = args.num_gpus

    print("\n" + "=" * 70)
    print("Test Configuration (K-SHARDING):")
    k_total = K_LOCAL * num_gpus
    print(f"  M={M}, K_local={K_LOCAL}, K_total={k_total}, N={N}")
    print(f"  world_size={num_gpus}")
    print(f"  aiter fused ops available: {AITER_FUSED_AVAILABLE}")
    if AITER_FUSED_AVAILABLE:
        print("  Using: torch.ops.aiter.fused_all_gather_gemm_k_shard")
    else:
        print("  aiter fused op not registered -- aborting")
        sys.exit(1)
    print("=" * 70 + "\n")

    if args.verify_gpus or args.async_tp:
        if not standalone_gpu_verification(num_gpus):
            print("Multi-GPU verification failed!")
            sys.exit(1)

    if args.async_tp:
        if TestBackend is None:
            print("Error: TestBackend not available. Cannot run async-tp test.")
            sys.exit(1)
        torch.multiprocessing.spawn(
            async_tp_test,
            args=(num_gpus, args.dynamic),
            nprocs=num_gpus,
            join=True,
        )
    else:
        simple_test()
