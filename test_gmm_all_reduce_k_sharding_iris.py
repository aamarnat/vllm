"""
Test for GEMM + All-Reduce fusion with K-sharding using aiter.fused_gemm_all_reduce_k_shard

This test demonstrates K-dimension sharding for the GEMM+All-Reduce pattern,
the counterpart to the All-Gather+GEMM pattern.

Sharding Strategy:
- Each rank holds input (M, K_local) and weight shard (K_local, N)
- Each rank computes a partial GEMM: C_partial = x @ weight  shape (M, N)
- All-reduce sums partial results across ranks: C = sum_r(x_r @ w_r) = (M, N)
- Equivalent to: full_x @ full_weight where full_x = cat(x_r, dim=1)
                                              full_weight = block_diag(w_r)
  ... which simplifies to the standard matmul all-reduce in tensor parallelism.

Use case in tensor parallelism (column-parallel linear):
- Input is replicated across ranks: each rank has the same (M, K)
- Weight is sharded column-wise: rank r has weight[:, r*N_local:(r+1)*N_local]
- BUT here we model the K-shard variant where input is sharded on K.

Run:
    cd /workspace/vllm
    sudo HSA_OVERRIDE_GFX_VERSION=9.4.2 HSA_NO_SCRATCH_RECLAIM=1 \\
        /opt/venv/bin/python test_gmm_all_reduce_k_sharding_iris.py --async-tp --verify-gpus
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"

# TestBackend lives under vllm/tests/compile/
tests_path = Path(__file__).parent / "tests"
if tests_path.exists() and str(tests_path) not in sys.path:
    sys.path.insert(0, str(tests_path))

import aiter  # noqa: E402  (triggers custom-op registration)

MASTER_PORT = "12349"

# Problem dimensions — chosen so block sizes in the wrapper (bm=128, bk=64, bn=64) fit
M = 256          # batch tokens
K_LOCAL = 64     # K per rank (weight rows per rank)
N = 512          # output features

DTYPE = torch.float16

AITER_FUSED_AVAILABLE = (
    hasattr(torch.ops, "aiter")
    and hasattr(torch.ops.aiter, "fused_gemm_all_reduce_k_shard")
)


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
# GPU verification
# ---------------------------------------------------------------------------

def standalone_gpu_verification(num_gpus: int = 2) -> bool:
    print(f"\nVerifying {num_gpus}-GPU setup...")
    if not torch.cuda.is_available():
        print("  GPU not available")
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

    t0 = torch.randn(64, 64, device="cuda:0")
    t0.to("cuda:1")
    torch.cuda.synchronize("cuda:1")
    print("  P2P transfer (GPU 0 -> GPU 1) OK")

    print(f"  {num_gpus}-GPU setup verified\n")
    return True


# ---------------------------------------------------------------------------
# Per-rank test body
# ---------------------------------------------------------------------------

def async_tp_test(local_rank: int, world_size: int, dynamic: bool = False):
    print(f"\n{'='*70}")
    print(f"Rank {local_rank}/{world_size}: Starting GEMM+AllReduce K-shard test")
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
    assert dist.is_initialized(), "Distributed not initialized"
    assert dist.get_rank() == local_rank
    assert dist.get_world_size() == world_size
    print(f"  Rank {local_rank}: distributed OK (backend={dist.get_backend()}, "
          f"rank={dist.get_rank()}/{dist.get_world_size()})")

    # VERIFICATION 3: cross-GPU communication sanity
    probe = torch.tensor([float(local_rank)], device=device)
    buf = torch.zeros(world_size, device=device)
    dist.all_gather_into_tensor(buf, probe)
    expected = list(range(world_size))
    assert buf.tolist() == expected, f"All-gather sanity failed: {buf.tolist()}"
    print(f"  Rank {local_rank}: comm sanity OK {buf.tolist()}")

    # VERIFICATION 4: input/weight on correct GPU
    # Each rank has input (M, K_local) and weight shard (K_local, N)
    # Scale input by rank+1 so each rank contributes different values
    x = torch.randn((M, K_LOCAL), dtype=DTYPE, requires_grad=False) * (local_rank + 1)
    weight = torch.randn((K_LOCAL, N), dtype=DTYPE) * 0.02

    assert x.device.index == local_rank
    assert weight.device.index == local_rank
    print(f"  Rank {local_rank}: input {tuple(x.shape)}, weight {tuple(weight.shape)} "
          f"on {x.device}, input mean={x.mean():.4f}")

    dist.barrier()

    # Reference: local matmul then all-reduce sum across ranks
    # C_ref = sum_r(x_r @ w_r), each term shape (M, N)
    with torch.no_grad():
        local_partial = torch.mm(x, weight)       # (M, N)
        reference = local_partial.clone()
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)
        local_sum = torch.sum(local_partial).item()
        global_sum = torch.sum(reference).item()
        print(f"  Rank {local_rank}: local partial sum={local_sum:.4f}, "
              f"reference (all-reduced) sum={global_sum:.4f}")

    # VERIFICATION 5: memory baseline
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    # Run fused op
    _, C_global = torch.ops.aiter.fused_gemm_all_reduce_k_shard(
        x,
        [weight],
        reduce_dim=1,
        group_name="test",
    )
    torch.cuda.synchronize(device)
    mem_after = torch.cuda.memory_allocated(device)

    # VERIFICATION 6: output placement
    assert C_global.device.index == local_rank, (
        f"Output on GPU {C_global.device.index}, expected {local_rank}"
    )
    print(f"  Rank {local_rank}: output {tuple(C_global.shape)} on {C_global.device}, "
          f"mean={C_global.mean():.4f} (ref={reference.mean():.4f}), "
          f"mem_delta={(mem_after - mem_before) / 1e6:.1f} MB")

    # VERIFICATION 7 (rank 0): numerical correctness
    if local_rank == 0:
        max_diff = torch.max(torch.abs(C_global - reference)).item()
        rel_err = max_diff / (torch.max(torch.abs(reference)).item() + 1e-8)
        # fp16 tolerance: a few ULPs at the max output magnitude
        tol = max(1e-2, 10 * (2 ** (int(reference.abs().max().log2().item()) - 10)))
        ok = max_diff <= tol
        print("=" * 70)
        print(f"  Max absolute difference : {max_diff:.6f}")
        print(f"  Relative error          : {rel_err:.6f}")
        print(f"  Tolerance               : {tol:.6f}")
        if ok:
            print("  Output matches reference within tolerance!")
        else:
            print(f"  WARNING - output differs from reference by {max_diff:.6f}")
            print(f"  Sample output    : {C_global[0, :8].tolist()}")
            print(f"  Sample reference : {reference[0, :8].tolist()}")
        print(f"  Output sum (rank 0): {torch.sum(C_global).item():.4f}")

    # Cleanup
    dist.barrier()
    cleanup_dist_env_and_memory()
    torch.cuda.empty_cache()

    return C_global


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GEMM + All-Reduce K-shard fusion test via iris.ops.matmul_all_reduce"
    )
    parser.add_argument("--async-tp", action="store_true",
                        help="Run the distributed test (requires 2 GPUs)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Use dynamic shapes (currently unused, reserved)")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="Number of GPUs to use (default: 2)")
    parser.add_argument("--verify-gpus", action="store_true",
                        help="Verify GPU setup before running")
    args = parser.parse_args()

    num_gpus = args.num_gpus

    print("\n" + "=" * 70)
    print("Test Configuration (GEMM+ALL-REDUCE, K-SHARDING):")
    print(f"  M={M}, K_local={K_LOCAL}, K_total={K_LOCAL * num_gpus}, N={N}")
    print(f"  world_size={num_gpus}")
    print(f"  aiter fused ops available: {AITER_FUSED_AVAILABLE}")
    if AITER_FUSED_AVAILABLE:
        print("  Using: torch.ops.aiter.fused_gemm_all_reduce_k_shard")
    else:
        print("  aiter fused op not registered -- aborting")
        sys.exit(1)
    print("=" * 70 + "\n")

    if args.verify_gpus or args.async_tp:
        if not standalone_gpu_verification(num_gpus):
            print("GPU verification failed")
            sys.exit(1)

    if args.async_tp:
        torch.multiprocessing.spawn(
            async_tp_test,
            args=(num_gpus, args.dynamic),
            nprocs=num_gpus,
            join=True,
        )
    else:
        print("Pass --async-tp to run the distributed test.")
