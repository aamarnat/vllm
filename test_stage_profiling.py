"""
Profile each optimization stage of the fused GEMM+AR kernel.

Tests configurations corresponding to the progression in OPTIMIZATION_SUMMARY.md Part 2:
  - atomic variant (baseline fused, pre-optimization)
  - two_shot with large blocks (bm=128, bn=64) — pre-adaptive
  - two_shot with adaptive blocks (bm=32/64, bn=128) — final
  - one_shot with adaptive blocks (for comparison)
  - Unfused baseline (torch.mm + dist.all_reduce)

Usage:
    python test_stage_profiling.py --num-gpus 2
"""

import os
import sys
import math

os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    cleanup_dist_env_and_memory,
)
from vllm.utils.system_utils import update_environment_variables

import aiter  # noqa
import iris
from iris.ops.matmul_all_reduce import matmul_all_reduce
from iris.ops.config import FusedConfig
from iris.ops.workspace import FusedWorkspace

MASTER_PORT = "12560"
K_LOCAL = 512
N = 2880
DTYPE = torch.float16
M_VALUES = [32, 896, 2048]
PHASE_NAMES = ["decode", "hybrid", "prefill"]
WARMUP_ITERS = 10
TIMED_ITERS = 50


def time_fn(fn, warmup, iters, device):
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


def profile_worker(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(DTYPE)

    update_environment_variables({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost", "MASTER_PORT": MASTER_PORT,
        "RCCL_DEBUG": "WARN",
    })
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    weight = torch.randn((K_LOCAL, N), dtype=DTYPE, device=device) * 0.02
    dist.broadcast(weight, src=0)

    iris_inst = iris.iris()

    # Configurations to test — each corresponds to an optimization stage
    stage_configs = [
        ("Unfused (mm+AR)", None, None),
        ("S0: atomic bm128 bn64", "atomic", (128, 64, 64, 1, 1)),
        ("S2: two_shot bm128 bn64", "two_shot", (128, 64, 64, 4, 8)),
        ("S3+4: two_shot bm32 bn128", "two_shot", (32, 128, 64, 4, 8)),
        ("S3+4: one_shot bm32 bn128", "one_shot", (32, 128, 64, 4, 8)),
        ("S3+4: two_shot bm64 bn128", "two_shot", (64, 128, 64, 4, 8)),
        ("S3+4: one_shot bm64 bn128", "one_shot", (64, 128, 64, 4, 8)),
    ]

    if rank == 0:
        print(f"\n{'='*90}")
        print(f"  Stage Profiling: K_LOCAL={K_LOCAL}, N={N}, TP={world_size}, dtype={DTYPE}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Warmup={WARMUP_ITERS}, Timed={TIMED_ITERS}")
        print(f"{'='*90}")

    for M, phase in zip(M_VALUES, PHASE_NAMES):
        x = torch.randn((M, K_LOCAL), dtype=DTYPE, device=device) * (rank + 1)

        # Reference
        with torch.no_grad():
            ref = torch.mm(x, weight)
            dist.all_reduce(ref, op=dist.ReduceOp.SUM)

        if rank == 0:
            print(f"\n  --- {phase.upper()} (M={M}) ---")
            print(f"  {'Config':<35} {'Latency (us)':>12} {'Speedup':>8} {'Correct':>8}")
            print(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*8}")

        unfused_us = None

        for label, variant, params in stage_configs:
            if variant is None:
                # Unfused baseline
                unfused_out = torch.empty((M, N), dtype=DTYPE, device=device)

                def unfused_fn():
                    torch.mm(x, weight, out=unfused_out)
                    dist.all_reduce(unfused_out, op=dist.ReduceOp.SUM)

                ms = time_fn(unfused_fn, WARMUP_ITERS, TIMED_ITERS, device)
                us = ms * 1000
                unfused_us = us

                unfused_fn()
                torch.cuda.synchronize(device)
                diff = torch.max(torch.abs(unfused_out - ref)).item()
                correct = "OK" if diff < 0.1 else f"FAIL({diff:.4f})"

                if rank == 0:
                    print(f"  {label:<35} {us:>12.1f} {'--':>8} {correct:>8}")
            else:
                bm, bn, bk, gsm, xcds = params
                # Clamp bm to M (and ensure power of 2)
                actual_bm = bm
                if actual_bm > M:
                    p = 1
                    while p * 2 <= M:
                        p *= 2
                    actual_bm = max(p, 16)
                if actual_bm > M:
                    if rank == 0:
                        print(f"  {label:<35} {'SKIP (M<bm)':>12} {'--':>8} {'--':>8}")
                    continue

                # Clamp bn to N
                actual_bn = bn
                while actual_bn > N:
                    actual_bn //= 2

                C_global = iris_inst.as_symmetric(torch.zeros(M, N, dtype=DTYPE, device=device))
                config = FusedConfig(
                    block_size_m=actual_bm,
                    block_size_n=actual_bn,
                    block_size_k=bk,
                    group_size_m=gsm,
                    num_xcds=xcds,
                    all_reduce_variant=variant,
                )
                ws = FusedWorkspace()

                def make_fn(c, w):
                    def fn():
                        matmul_all_reduce(iris_inst, C_global, x, weight, config=c, workspace=w)
                    return fn

                ws.locks = None
                ws.call_counter = 0
                torch.cuda.synchronize(device)
                dist.barrier()

                fn = make_fn(config, ws)
                ms = time_fn(fn, WARMUP_ITERS, TIMED_ITERS, device)
                us = ms * 1000
                torch.cuda.synchronize(device)

                # Verify
                ws2 = FusedWorkspace()
                C_verify = iris_inst.as_symmetric(torch.zeros(M, N, dtype=DTYPE, device=device))
                matmul_all_reduce(iris_inst, C_verify, x, weight, config=config, workspace=ws2)
                torch.cuda.synchronize(device)
                diff = torch.max(torch.abs(C_verify - ref)).item()
                correct = "OK" if diff < 0.1 else f"FAIL({diff:.4f})"

                speedup = unfused_us / us if unfused_us and us > 0 else 0
                if rank == 0:
                    print(f"  {label:<35} {us:>12.1f} {speedup:>7.2f}x {correct:>8}")

                # Reset for next config
                ws.locks = None
                ws.call_counter = 0
                torch.cuda.synchronize(device)
                dist.barrier()

    dist.barrier()
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=2)
    args = parser.parse_args()

    torch.multiprocessing.spawn(profile_worker, args=(args.num_gpus,), nprocs=args.num_gpus, join=True)
    print("\nDONE")
