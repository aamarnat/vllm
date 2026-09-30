"""
Profile test: o_proj GEMM + All-Reduce — Unfused vs Iris Fused

Compares two implementations of the output projection pattern from
GPT-OSS-120B with TP=8:

  1. Unfused:  torch.mm(x, weight) + dist.all_reduce(out, SUM)
  2. Fused:    torch.ops.aiter.fused_gemm_all_reduce_k_shard(x, [weight], ...)

Tests three phases matching real serving workloads:
  - Decode:  M=32   (one token per concurrent sequence)
  - Hybrid:  M=896  (mixed prefill+decode batch)
  - Prefill: M=2048 (full prefill chunk)

Shape per rank: (M, K_local) @ (K_local, N) -> (M, N), then all_reduce
  K_local = 512  (num_heads * head_dim / TP = 64*64/8)
  N = 2880       (hidden_size)

Usage:
    cd /workspace/vllm
    sudo env PATH="$PATH" PYTHONPATH="$PYTHONPATH" HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" \\
        python test_oproj_profile.py --num-gpus 8 --verify-gpus

    # Quick 2-GPU test
    sudo env PATH="$PATH" PYTHONPATH="$PYTHONPATH" HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" \\
        python test_oproj_profile.py --num-gpus 2 --verify-gpus
"""

import os
import sys
import math
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

import aiter  # noqa: E402 — triggers custom-op registration
import iris
from iris.ops.matmul_all_reduce import matmul_all_reduce
from iris.ops.config import FusedConfig
from iris.ops.workspace import FusedWorkspace

MASTER_PORT = "12550"

# o_proj dimensions for GPT-OSS-120B with TP=8
K_LOCAL = 512       # num_heads * head_dim / TP = 64*64/8
N = 2880            # hidden_size
DTYPE = torch.float16

# Three phases to profile
M_VALUES = [32, 896, 2048]
PHASE_NAMES = ["decode", "hybrid", "prefill"]

# Timing parameters
WARMUP_ITERS = 10
TIMED_ITERS = 50

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

def standalone_gpu_verification(num_gpus: int) -> bool:
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
    print("  P2P transfer OK")

    print(f"  {num_gpus}-GPU setup verified\n")
    return True


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, warmup, iters, device):
    """Time a callable using CUDA events. Returns average ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)

    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Per-rank profiling body
# ---------------------------------------------------------------------------

def profile_test(local_rank: int, world_size: int):
    print(f"\n{'='*70}")
    print(f"Rank {local_rank}/{world_size}: o_proj GEMM+AllReduce profile")
    print(f"{'='*70}")

    seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(DTYPE)

    gpu_type = get_gpu_type()
    print(f"  Rank {local_rank}: {gpu_type} GPU {torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name(local_rank)})")

    # Distributed init
    env_vars = {
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": MASTER_PORT,
    }
    if gpu_type == "AMD":
        env_vars["RCCL_DEBUG"] = "WARN"
    else:
        env_vars["NCCL_DEBUG"] = "WARN"
    update_environment_variables(env_vars)

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    assert dist.is_initialized()
    assert dist.get_rank() == local_rank
    assert dist.get_world_size() == world_size
    print(f"  Rank {local_rank}: distributed OK (backend={dist.get_backend()})")

    # Comm sanity check
    probe = torch.tensor([float(local_rank)], device=device)
    buf = torch.zeros(world_size, device=device)
    dist.all_gather_into_tensor(buf, probe)
    assert buf.tolist() == list(range(world_size))
    print(f"  Rank {local_rank}: comm sanity OK")

    # Create weight (same on all ranks — o_proj weight is replicated)
    weight = torch.randn((K_LOCAL, N), dtype=DTYPE, device=device) * 0.02
    dist.broadcast(weight, src=0)

    # Initialize iris once for all phases
    iris_inst = iris.iris()

    results = []

    for M, phase in zip(M_VALUES, PHASE_NAMES):
        if local_rank == 0:
            print(f"\n  --- {phase.upper()} (M={M}) ---")
            print(f"  GEMM: ({M}, {K_LOCAL}) @ ({K_LOCAL}, {N}) -> ({M}, {N})")

        # Input differs per rank (models different attention outputs)
        x = torch.randn((M, K_LOCAL), dtype=DTYPE, device=device) * (local_rank + 1)

        dist.barrier()

        # ---- Reference ----
        with torch.no_grad():
            ref_partial = torch.mm(x, weight)
            ref = ref_partial.clone()
            dist.all_reduce(ref, op=dist.ReduceOp.SUM)

        # ---- Time unfused: torch.mm + dist.all_reduce ----
        # Pre-allocate output buffer
        unfused_out = torch.empty((M, N), dtype=DTYPE, device=device)

        def unfused_fn():
            torch.mm(x, weight, out=unfused_out)
            dist.all_reduce(unfused_out, op=dist.ReduceOp.SUM)

        unfused_ms = time_fn(unfused_fn, WARMUP_ITERS, TIMED_ITERS, device)

        # Verify unfused correctness
        unfused_fn()
        torch.cuda.synchronize(device)

        # ---- Time iris fused: sweep variants and block sizes ----
        C_global = torch.zeros((M, N), dtype=DTYPE, device=device)
        C_global = iris_inst.as_symmetric(C_global)

        bk = 64
        while bk > 8 and bk > K_LOCAL:
            bk //= 2

        # Sweep block sizes and scheduling params.
        # Reuse a single workspace to avoid symmetric heap exhaustion.
        variant_results = {}
        best_key = None
        best_ms = float('inf')

        configs_to_test = [
            # (variant, bm, bn, gsm, xcds)
            ("two_shot", min(128, M), 128, 4, 8),
            ("two_shot", min(64, M), 128, 4, 8),
            ("two_shot", min(64, M), 64, 4, 8),
            ("two_shot", 32, 128, 4, 8),
            ("two_shot", 32, 64, 4, 8),
            ("two_shot", 32, 128, 8, 8),
            ("one_shot", min(64, M), 128, 4, 8),
            ("one_shot", 32, 128, 4, 8),
        ]

        # Pre-allocate workspace with max tile count across all configs
        # (shmem.zeros is collective and can't be called mid-sweep)
        max_tiles = 0
        for _, bm_c, bn_c, _, _ in configs_to_test:
            nt = math.ceil(M / bm_c) * math.ceil(N / bn_c)
            max_tiles = max(max_tiles, nt)

        workspace = FusedWorkspace()
        workspace.locks = iris_inst.zeros((max_tiles,), dtype=torch.int32)
        workspace.aux_buffer = iris_inst.zeros((M, N), dtype=DTYPE)
        workspace._barrier_tensor = torch.zeros(1, dtype=torch.int32, device=device)
        workspace.operation = "matmul_all_reduce"
        workspace.shape = (M, N, K_LOCAL)
        workspace.dtype = DTYPE
        workspace.world_size = world_size
        workspace.variant = "two_shot"
        workspace.prepared = True

        for variant, bm, bn, gsm, xcds in configs_to_test:
            if bn > N:
                continue
            key = (
                f"{variant}_bm{bm}_bn{bn}"
                f"_gsm{gsm}_xcd{xcds}"
            )
            config_v = FusedConfig(
                block_size_m=bm,
                block_size_n=bn,
                block_size_k=bk,
                group_size_m=gsm,
                num_xcds=xcds,
                all_reduce_variant=variant,
            )

            # Update workspace variant to match current config
            workspace.variant = variant

            # Reset versioned lock state: zero the lock array and reset
            # call_counter so the kernel doesn't spin on stale lock values
            # from the previous config (which may have had fewer tiles,
            # leaving extra lock entries at old call_counter values).
            workspace.locks.zero_()
            workspace.call_counter = 0

            def make_fn(c, ws):
                def fn():
                    matmul_all_reduce(
                        iris_inst, C_global,
                        x, weight,
                        config=c, workspace=ws,
                    )
                return fn

            # Sync ranks before each config
            torch.cuda.synchronize(device)
            dist.barrier()

            fn_v = make_fn(config_v, workspace)
            ms_v = time_fn(
                fn_v, WARMUP_ITERS,
                TIMED_ITERS, device,
            )
            variant_results[key] = ms_v * 1000

            if ms_v < best_ms:
                best_ms = ms_v
                best_key = key
                best_variant = variant
                best_bm = bm
                best_bn = bn
                best_gsm = gsm
                best_xcds = xcds

        fused_ms = best_ms
        config = FusedConfig(
            block_size_m=best_bm, block_size_n=best_bn,
            block_size_k=bk,
            group_size_m=best_gsm, num_xcds=best_xcds,
            all_reduce_variant=best_variant,
        )

        # Verify fused correctness with best config (fresh workspace)
        verify_ws = FusedWorkspace()
        C_verify = torch.zeros((M, N), dtype=DTYPE, device=device)
        C_verify = iris_inst.as_symmetric(C_verify)
        matmul_all_reduce(
            iris_inst, C_verify, x, weight,
            config=config, workspace=verify_ws,
        )
        torch.cuda.synchronize(device)
        fused_out = C_verify.clone()

        # Cross-rank consistency: every rank should have the same result
        # Gather rank 0's fused output and broadcast to compare
        rank0_fused = fused_out.clone()
        dist.broadcast(rank0_fused, src=0)
        cross_rank_diff = torch.max(
            torch.abs(fused_out - rank0_fused)
        ).item()
        cross_rank_ok = cross_rank_diff == 0.0

        # Per-rank numerical verification against reference
        max_ref = torch.max(torch.abs(ref)).item()
        if max_ref > 0:
            ulp = 2.0 ** (math.floor(math.log2(max_ref)) - 10)
        else:
            ulp = 1e-4
        tol = 10 * ulp

        unfused_diff = torch.max(
            torch.abs(unfused_out - ref)
        ).item()
        fused_diff = torch.max(torch.abs(fused_out - ref)).item()

        unfused_ok = unfused_diff <= tol
        fused_ok = fused_diff <= tol

        # Detailed error statistics
        fused_abs_err = torch.abs(fused_out - ref)
        fused_mean_err = fused_abs_err.mean().item()
        fused_median_err = fused_abs_err.median().item()
        nonzero_ref = torch.abs(ref).clamp(min=1e-7)
        fused_rel_err = (fused_abs_err / nonzero_ref).max().item()
        n_exact = (fused_abs_err == 0).sum().item()
        n_total = fused_abs_err.numel()

        # Gather per-rank fused_diff to check consistency
        all_diffs = torch.tensor(
            [fused_diff], device=device, dtype=torch.float32,
        )
        diff_list = [
            torch.zeros_like(all_diffs) for _ in range(world_size)
        ]
        dist.all_gather(diff_list, all_diffs)

        speedup = unfused_ms / fused_ms if fused_ms > 0 else float('inf')

        if local_rank == 0:
            per_rank_diffs = [d.item() for d in diff_list]
            ranks_ok = all(d <= tol for d in per_rank_diffs)

            print(f"  Unfused:  {unfused_ms * 1000:.1f} us  "
                  f"(max_diff={unfused_diff:.6f}, "
                  f"{'OK' if unfused_ok else 'FAIL'})")
            print(f"  Fused:    {fused_ms * 1000:.1f} us  "
                  f"(max_diff={fused_diff:.6f}, "
                  f"{'OK' if fused_ok else 'FAIL'})")
            print(f"    max_ref={max_ref:.2f}, "
                  f"ulp={ulp:.6f}, tol={tol:.4f}")
            print(f"    mean_err={fused_mean_err:.6f}, "
                  f"median_err={fused_median_err:.6f}, "
                  f"max_rel_err={fused_rel_err:.6f}")
            print(f"    exact_match: {n_exact}/{n_total} "
                  f"({100*n_exact/n_total:.1f}%)")
            print(f"    cross-rank consistency: "
                  f"{'OK' if cross_rank_ok else 'FAIL'} "
                  f"(max_diff={cross_rank_diff})")
            print(f"    per-rank max_diffs: "
                  f"{[f'{d:.4f}' for d in per_rank_diffs]}")
            all_ranks_ok = ranks_ok and cross_rank_ok

            print(f"  Best config: {best_key}")
            sorted_vr = sorted(
                variant_results.items(), key=lambda kv: kv[1],
            )
            for v, us in sorted_vr[:5]:
                marker = " <-- best" if v == best_key else ""
                print(f"    {v:>30}: {us:>8.1f} us{marker}")
            if len(sorted_vr) > 5:
                print(f"    ... ({len(sorted_vr) - 5} more)")
            print(f"  Speedup:  {speedup:.2f}x")

            results.append({
                "phase": phase,
                "M": M,
                "unfused_us": unfused_ms * 1000,
                "fused_us": fused_ms * 1000,
                "speedup": speedup,
                "unfused_ok": unfused_ok,
                "fused_ok": fused_ok and all_ranks_ok,
                "best_key": best_key,
            })

        dist.barrier()

    # ---- Summary table (rank 0) ----
    if local_rank == 0:
        print(f"\n{'='*70}")
        print(f"  o_proj Profile: Unfused vs Iris Fused GEMM+AllReduce")
        print(f"{'='*70}")
        print(f"  {'Phase':<10} {'M':>6}  {'Unfused (us)':>13}  {'Fused (us)':>11}  {'Speedup':>8}  {'Status':>8}")
        print(f"  {'-'*10} {'-'*6}  {'-'*13}  {'-'*11}  {'-'*8}  {'-'*8}")
        for r in results:
            status = "OK" if (r["unfused_ok"] and r["fused_ok"]) else "FAIL"
            print(f"  {r['phase']:<10} {r['M']:>6}  {r['unfused_us']:>13.1f}  {r['fused_us']:>11.1f}  {r['speedup']:>7.2f}x  {status:>8}")
        print(f"  {'-'*10} {'-'*6}  {'-'*13}  {'-'*11}  {'-'*8}  {'-'*8}")
        print(f"  Config: K_local={K_LOCAL}, N={N}, TP={world_size}, dtype={DTYPE}")
        print(f"  Variant: two_shot (versioned locks, zero pre-kernel overhead)")
        print(f"  Timing: warmup={WARMUP_ITERS}, timed={TIMED_ITERS} iterations")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*70}\n")

    # Cleanup
    dist.barrier()
    cleanup_dist_env_and_memory()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile o_proj GEMM+AllReduce: unfused vs iris fused"
    )
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="Number of GPUs (default: 2)")
    parser.add_argument("--verify-gpus", action="store_true",
                        help="Verify GPU setup before running")
    args = parser.parse_args()

    num_gpus = args.num_gpus

    print("\n" + "=" * 70)
    print("o_proj Profile Configuration:")
    print(f"  K_local={K_LOCAL}, N={N}, TP={num_gpus}")
    print(f"  M values: {M_VALUES} ({', '.join(PHASE_NAMES)})")
    print(f"  dtype={DTYPE}")
    print(f"  warmup={WARMUP_ITERS}, timed={TIMED_ITERS}")
    print(f"  aiter fused op available: {AITER_FUSED_AVAILABLE}")
    if not AITER_FUSED_AVAILABLE:
        print("  aiter fused op not registered -- aborting")
        sys.exit(1)
    print("=" * 70 + "\n")

    if args.verify_gpus:
        if not standalone_gpu_verification(num_gpus):
            print("GPU verification failed")
            sys.exit(1)

    torch.multiprocessing.spawn(
        profile_test,
        args=(num_gpus,),
        nprocs=num_gpus,
        join=True,
    )
