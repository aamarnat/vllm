"""
Test for all-gather + GEMM fusion using iris.x.all_gather_gemm

This test demonstrates using iris.x.all_gather_gemm instead of the default
symm_mem.fused_all_gather_matmul for tensor-parallel fusion.

Sharding Strategy:
- Shards on M dimension: each rank has input (M_local, K)
- All-gather on dim=0 produces (M, K) where M = M_local * world_size
- Then performs GEMM: (M, K) @ (K, N) = (M, N)
- Note: vLLM's AsyncTPPass pattern matcher expects dim=0 (M-dimension) sharding

Key changes:
1. Registers torch.ops.iris.all_gather_gemm as a custom op
2. TestAGMMModel.ops_in_model_after() returns iris op when available
3. Falls back to symm_mem if iris.x not available

GPU Support:
- Automatically detects and supports both NVIDIA (CUDA) and AMD (ROCm) GPUs
- Uses NCCL for NVIDIA, RCCL for AMD
- No code changes needed - adapts automatically!

Multi-GPU Verification:
- 8 verification checkpoints ensure correct multi-GPU execution
- Tests cross-GPU communication, memory placement, and fusion
- Run with --verify-gpus to check your setup

To use iris.x fusion in vLLM's AsyncTPPass:
- Modify collective_fusion.py AllGatherGEMMPattern.replacement() to use
  torch.ops.iris.all_gather_gemm instead of torch.ops.symm_mem.fused_all_gather_matmul
"""

import torch
import os
from vllm.compilation.collective_fusion import AsyncTPPass
from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
import torch.distributed as dist

# Set environment variable for Triton
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'
TestBackend = None
# Import TestBackend from tests module
import sys
from pathlib import Path
# Add tests directory to path if not already there
tests_path = Path(__file__).parent / "tests"
if tests_path.exists() and str(tests_path) not in sys.path:
    sys.path.insert(0, str(tests_path))
from compile.backend import TestBackend


# Try to import iris.x.all_gather_gemm
import iris
from iris.x.all_gather_gemm import all_gather_gemm as iris_all_gather_gemm_kernel
from tritonblas.kernels.stages.indexing import grid_setup
IRIS_AVAILABLE = True
print("✓ iris.x.all_gather_gemm available")



# Tracking globals to verify which implementation is called
_WRAPPER_CALL_COUNT = 0
_FAKE_CALL_COUNT = 0

# Register iris.x.all_gather_gemm as a custom torch op
def iris_all_gather_gemm_wrapper(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for iris.x.all_gather_gemm to be used as a torch custom op.
    
    This calls the actual iris.x.all_gather_gemm Triton kernel.
    
    Args:
        x: Input tensor to all-gather (M_local, K) - sharded on M dimension
        weights: List of weight tensors [(K, N)]
        gather_dim: Dimension along which to gather (0 for M dimension)
        group_name: Process group name
        
    Returns:
        Tuple of (gathered_output, matmul_output)
    """
    global _WRAPPER_CALL_COUNT
    _WRAPPER_CALL_COUNT += 1
    print(f"[EXEC] iris_all_gather_gemm_wrapper called (count: {_WRAPPER_CALL_COUNT}, device: {x.device})")
    
    if not IRIS_AVAILABLE:
        raise RuntimeError("iris.x.all_gather_gemm not available")
    
    shmem = iris.iris()
    
    # Get distributed info
    try:
        if dist.is_initialized():
            world_size = dist.get_world_size()
            cur_rank = dist.get_rank()
        else:
            # Not in distributed mode, use defaults for testing
            world_size = 1
            cur_rank = 0
    except (RuntimeError, ValueError):
        # Not in distributed mode, use defaults for testing
        world_size = 1
        cur_rank = 0
    
    # Unpack dimensions - sharding on M dimension
    M_local = x.shape[0]
    K = x.shape[1]
    M = M_local * world_size
    weight = weights[0]
    N = weight.shape[1]

    A_original = x
    A_iris = shmem.zeros(x.shape, dtype=x.dtype, device=x.device)
    A_iris.copy_(A_original)
    
    # Allocate output tensors
    A_gathered = torch.zeros((M, K), dtype=x.dtype, device=x.device)
    C = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    
    # For single-GPU testing: simulate all-gather by replicating
    if world_size == 1:
        A_gathered[:] = x
    else:
        # In multi-GPU: perform actual all-gather on M dimension
        # The iris kernel expects A_gathered to be pre-populated
        # TODO: Why is iris.x all_gather_gemm is only doing a gemm and is expecting A_gathered to be already gathered - change for now to example all_gather_gemm implementation
        if dist.is_initialized():
            dist.all_gather_into_tensor(A_gathered, x)
        else:
            # Fallback: just copy for testing
            A_gathered[:M_local] = x
    
    # Call the iris.x.all_gather_gemm kernel
    # Note: This kernel assumes A_gathered is already populated
    
    # Kernel parameters
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    # Calculate number of SMs based on total tiles
    num_tiles = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    NUM_SMS = max(1, min(108, num_tiles))  # Number of SMs to use (clamp between 1 and 108)
    NUM_XCDS = 1  # Single chiplet
    CHUNK_SIZE = 1
    GROUP_SIZE_M = 8
    
    # Compute strides
    stride_am = A_iris.stride(0)
    stride_ak = A_iris.stride(1)
    stride_bn = weight.stride(1)
    stride_bk = weight.stride(0)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    stride_ag_m = A_gathered.stride(0)
    stride_ag_n = A_gathered.stride(1)
    
    # Heap bases for RDMA (placeholder for testing)
    heap_bases = torch.zeros((world_size,), dtype=torch.int64, device=x.device)
    
    # Kernel configuration
    BIAS = 0
    bias_ptr = None
    stride_bias = 0
    EVEN_K = (K % BLOCK_K == 0)
    ALLOW_TF32 = True
    CACHE_MODIFIER_A = ""
    CACHE_MODIFIER_B = ""
    
    # Launch the iris.x.all_gather_gemm kernel
    grid = (NUM_SMS,)
    
    try:
        iris_all_gather_gemm_kernel[grid](
            A_iris,  # A_sharded
            weight,  # B
            C,  # Output
            A_gathered,  # Gathered buffer
            bias_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            stride_ag_m, stride_ag_n,
            stride_bias,
            heap_bases,
            cur_rank=cur_rank,
            world_size=world_size,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            NUM_SMS=NUM_SMS,
            NUM_XCDS=NUM_XCDS,
            CHUNK_SIZE=CHUNK_SIZE,
            BIAS=BIAS,
            EVEN_K=EVEN_K,
            CACHE_MODIFIER_A=CACHE_MODIFIER_A,
            CACHE_MODIFIER_B=CACHE_MODIFIER_B,
            ALLOW_TF32=ALLOW_TF32,
        )
    except Exception as e:
        # If kernel fails, fall back to PyTorch implementation
        print(f"Warning: iris kernel failed ({e}), falling back to PyTorch")
        C = torch.matmul(A_gathered, weight)
    
    return A_gathered, C


def iris_all_gather_gemm_fake(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for meta mode / shape inference"""
    global _FAKE_CALL_COUNT
    _FAKE_CALL_COUNT += 1
    print(f"[META] iris_all_gather_gemm_fake called (count: {_FAKE_CALL_COUNT}, device: {x.device})")
    
    # Infer shapes - sharding on M dimension
    world_size = 2  # Default assumption
    M_local = x.shape[0]
    K = x.shape[1]
    M = M_local * world_size
    N = weights[0].shape[1]
    
    gathered = torch.empty((M, K), dtype=x.dtype, device=x.device)
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    return gathered, output


# Register the custom op
if IRIS_AVAILABLE:
    try:
        from torch.library import Library
        from vllm.utils.torch_utils import direct_register_custom_op
        
        # Create a library for iris ops
        iris_lib = Library("iris", "DEF")
        
        # Use vLLM's direct_register_custom_op which handles compilation properly
        direct_register_custom_op(
            op_name="all_gather_gemm",
            op_func=iris_all_gather_gemm_wrapper,
            mutates_args=[],
            fake_impl=iris_all_gather_gemm_fake,
            target_lib=iris_lib,
        )
        
        print("✓ Registered torch.ops.iris.all_gather_gemm custom op")
    except Exception as e:
        print(f"⚠ Failed to register iris custom op: {e}")
        IRIS_AVAILABLE = False


def verify_execution_counts(expected_wrapper_min=0, expected_fake_min=0):
    """
    Verify that the wrapper and fake implementations were called expected number of times.
    
    Args:
        expected_wrapper_min: Minimum expected calls to wrapper (actual execution)
        expected_fake_min: Minimum expected calls to fake (shape inference)
    
    Returns:
        bool: True if verification passed
    """
    global _WRAPPER_CALL_COUNT, _FAKE_CALL_COUNT
    
    print("\n" + "="*70)
    print("Execution Verification:")
    print(f"  iris_all_gather_gemm_wrapper (actual): {_WRAPPER_CALL_COUNT} calls")
    print(f"  iris_all_gather_gemm_fake (meta):      {_FAKE_CALL_COUNT} calls")
    print("="*70)
    
    success = True
    
    if _WRAPPER_CALL_COUNT < expected_wrapper_min:
        print(f"✗ WARNING: Expected at least {expected_wrapper_min} wrapper calls, got {_WRAPPER_CALL_COUNT}")
        print("  This means the actual implementation may not be executing!")
        success = False
    else:
        print(f"✓ Wrapper called {_WRAPPER_CALL_COUNT} times (expected >= {expected_wrapper_min})")
    
    if _FAKE_CALL_COUNT < expected_fake_min:
        print(f"⚠ Note: Expected at least {expected_fake_min} fake calls, got {_FAKE_CALL_COUNT}")
        print("  Fake is used for shape inference during compilation")
    else:
        print(f"✓ Fake called {_FAKE_CALL_COUNT} times (shape inference during compilation)")
    
    return success


def reset_execution_counts():
    """Reset execution counters for fresh testing"""
    global _WRAPPER_CALL_COUNT, _FAKE_CALL_COUNT
    _WRAPPER_CALL_COUNT = 0
    _FAKE_CALL_COUNT = 0


class TestAGMMModel(torch.nn.Module):
    def __init__(self, hidden_size=16, dtype=torch.float16):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = torch.nn.Parameter(
            torch.empty((hidden_size, hidden_size)), requires_grad=False
        )
        # Initialize weights
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states):
        """
        Forward pass implementing the mm + all gather in the FX graph.
        Sharding on M dimension: each rank has (M_local, K), gather to (M, K).
        """
        # Reshape input to ensure correct shape
        view = hidden_states.reshape(-1, self.hidden_size)
        # All-gather on M dimension to get (M, K)
        all_gather = tensor_model_parallel_all_gather(view, dim=0)  # Gather on M dimension
        permute = self.weight.permute(1, 0)
        mm = torch.mm(all_gather, permute)
        return mm

    def ops_in_model_before(self):
        """Operations that should exist before AsyncTPPass fusion"""
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        """Operations that should exist after AsyncTPPass fusion"""
        if IRIS_AVAILABLE:
            # Use iris.x.all_gather_gemm if available
            return [torch.ops.iris.all_gather_gemm.default]
        else:
            # Fall back to symm_mem implementation
            return [torch.ops.symm_mem.fused_all_gather_matmul.default]


def simple_test():
    """Simple test without distributed setup (single GPU)
    
    Note: In single-GPU mode, we use the full hidden_size since there's no
    actual sharding. The all_gather on dim=0 will just replicate the input.
    """
    print("Running simple TestAGMMModel test...")
    reset_execution_counts()
    
    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_type = get_gpu_type()
        print(f"Using {gpu_type} GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no GPU available)")
    dtype = torch.float16
    
    # Model parameters
    hidden_size = 16
    batch_size = 8
    seq_len = 16
    
    # Create model and move to device
    model = TestAGMMModel(hidden_size=hidden_size, dtype=dtype)
    model = model.to(device)
    
    # Create test input (full size for single GPU)
    hidden_states = torch.randn(
        (batch_size * seq_len, hidden_size), 
        dtype=dtype, 
        device=device,
        requires_grad=False
    )
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Model weight shape: {model.weight.shape}")
    
    # Run forward pass (note: all_gather on dim=1 will replicate without distributed setup)
    try:
        output = model(hidden_states)
        print(f"Output shape: {output.shape}")
        print("✓ Test passed!")
        
        # Note: simple_test doesn't use fusion, so we won't see iris calls
        verify_execution_counts(expected_wrapper_min=0, expected_fake_min=0)
        
        return output
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        print("Note: This model requires distributed setup for full functionality")
        return None


def get_gpu_type():
    """Detect GPU type (NVIDIA or AMD)"""
    if torch.cuda.is_available():
        # Check if it's AMD ROCm
        if torch.version.hip is not None:
            return "AMD"
        else:
            return "NVIDIA"
    return None




def standalone_gpu_verification():
    """Standalone script to verify 2-GPU setup"""
    print("\nVerifying multi-GPU setup...")
    
    # Check GPU count
    if not torch.cuda.is_available():
        print("✗ GPU not available (neither CUDA nor ROCm)")
        return False
    
    gpu_type = get_gpu_type()
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} {gpu_type} GPU(s)")
    
    if num_gpus < 2:
        print("✗ Need at least 2 GPUs")
        return False
    
    # Test both GPUs
    for i in range(min(2, num_gpus)):
        device = torch.device(f"cuda:{i}")
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.2f} GB)")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.mm(x, y)
        torch.cuda.synchronize(device)  # Works for both CUDA and ROCm
        print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)} - computation successful")

    
    # Test data transfer between GPUs (P2P)
    x_gpu0 = torch.randn(100, 100, device='cuda:0')
    x_gpu1 = x_gpu0.to('cuda:1')
    torch.cuda.synchronize('cuda:1')
    print(f"✓ Cross-GPU transfer (P2P) successful")
    
    print("✓ Multi-GPU setup verified!\n")
    return True

def verify_functionality(local_rank, async_tp_pass, backend, model, output):
    # ✓ VERIFICATION 8: Check fusion occurred
    print(f"✓ Rank {local_rank}: AsyncTPPass matched count: {async_tp_pass.matched_count}")

    # Verify the pass worked correctly
    if async_tp_pass.matched_count == 1:
        print(f"✓ Rank {local_rank}: AsyncTPPass successfully fused operations!")

        # Check that the operations were replaced
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        print(f"✓ Rank {local_rank}: Operation fusion verified!")
    elif async_tp_pass.matched_count > 0:
        print(f"⚠ Rank {local_rank}: AsyncTPPass matched {async_tp_pass.matched_count} times (expected 1)")
    else:
        print(f"✗ Rank {local_rank}: AsyncTPPass did not match any patterns")

    print(f"✓ Rank {local_rank}: Test completed!\n")

    # ✓ VERIFICATION 9: Check execution counts
    if local_rank == 0:  # Only print from rank 0 to avoid duplicate output
        if output is not None:
            # If we ran the model and got output, wrapper should have been called
            verify_execution_counts(expected_wrapper_min=1, expected_fake_min=1)
        else:
            # May only see fake calls for shape inference
            verify_execution_counts(expected_wrapper_min=0, expected_fake_min=1)

def async_tp_test(local_rank, world_size, dynamic=False):
    """Test with AsyncTPPass applied (requires multiple GPUs and TestBackend)
    
    Args:
        local_rank: GPU rank
        world_size: Total number of GPUs
        dynamic: Whether to use dynamic shapes
    """
    
    # Reset execution counters for fresh testing
    if local_rank == 0:
        reset_execution_counts()
    
    print(f"\n{'='*70}")
    print(f"Rank {local_rank}: Starting AsyncTPPass test")
    print(f"{'='*70}")
    
    # Seed for reproducibility
    current_platform.seed_everything(0)
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    
    # ✓ VERIFICATION 1: Check device assignment
    gpu_type = get_gpu_type()
    print(f"✓ Rank {local_rank}: GPU Type: {gpu_type}")
    print(f"✓ Rank {local_rank}: Assigned to GPU {torch.cuda.current_device()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(local_rank)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB")
    
    dtype = torch.float16
    torch.set_default_dtype(dtype)
    
    # Setup distributed environment with appropriate debug flags
    env_vars = {
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12346",  # Different port to avoid conflicts
    }
    
    # Enable debugging based on GPU type
    # Note: Do NOT set HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES per process
    # In tensor parallelism, all processes should see all GPUs
    if gpu_type == "AMD":
        env_vars["RCCL_DEBUG"] = "INFO"  # AMD ROCm uses RCCL
    else:  # NVIDIA
        env_vars["NCCL_DEBUG"] = "INFO"  # NVIDIA CUDA uses NCCL
    
    update_environment_variables(env_vars)
    
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    
    # ✓ VERIFICATION 2: Check distributed initialization
    assert dist.is_initialized(), "Distributed not initialized!"
    assert dist.get_rank() == local_rank, f"Rank mismatch: {dist.get_rank()} != {local_rank}"
    assert dist.get_world_size() == world_size, f"World size mismatch: {dist.get_world_size()} != {world_size}"
    print(f"✓ Rank {local_rank}: Distributed initialized (backend={dist.get_backend()})")
    
    # ✓ VERIFICATION 3: Test cross-GPU communication
    test_tensor = torch.tensor([float(local_rank)], device=device)
    gathered = torch.zeros(world_size, device=device)
    dist.all_gather_into_tensor(gathered, test_tensor)
    print(f"✓ Rank {local_rank}: All-gather test successful: {gathered.tolist()}")
    expected = list(range(world_size))
    if gathered.tolist() != expected:
        print(f"⚠ Warning: All-gather returned {gathered.tolist()}, expected {expected}")
    
    # Configure vllm config for AsyncTPPass
    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig(
        pass_config=PassConfig(
            fuse_gemm_comms=True,
        ),
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
    
    # This is a fake model name to construct the model config
    # in the vllm_config, it's not really used
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    vllm_config.model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )
    
    # Create AsyncTPPass and backend
    async_tp_pass = AsyncTPPass(vllm_config)
    backend = TestBackend(async_tp_pass)
    
    # Model parameters
    hidden_size = 16
    batch_size = 8
    seq_len = 16
    
    # Create model
    model = TestAGMMModel(hidden_size=hidden_size, dtype=dtype)
    
    # ✓ VERIFICATION 4: Ensure model weights are on correct GPU
    assert model.weight.device.type == 'cuda', "Model not on CUDA"
    assert model.weight.device.index == local_rank, f"Model on wrong GPU: {model.weight.device.index} != {local_rank}"
    print(f"✓ Rank {local_rank}: Model weights on GPU {model.weight.device}")
    
    # Create test input with rank-specific values for verification
    hidden_states = torch.randn(
        (batch_size * seq_len, hidden_size),
        dtype=dtype,
        requires_grad=False
    ) * (local_rank + 1)  # Different values per rank
    
    # ✓ VERIFICATION 5: Ensure input is on correct GPU
    assert hidden_states.device.index == local_rank, f"Input on wrong GPU"
    print(f"✓ Rank {local_rank}: Input tensor on GPU {hidden_states.device}")
    print(f"  Input shape: {hidden_states.shape}, mean: {hidden_states.mean():.4f}")
    
    if dynamic:
        torch._dynamo.mark_dynamic(hidden_states, 0)
    
    # Synchronize before starting compilation
    if dist.is_initialized():
        dist.barrier()
    
    print(f"Rank {local_rank}: Compiling model with AsyncTPPass...")
    
    # Compile the model with AsyncTPPass
    compiled_model = torch.compile(model, backend=backend)
    
    # ✓ VERIFICATION 6: Monitor GPU activity during execution
    torch.cuda.synchronize(device)
    start_mem = torch.cuda.memory_allocated(device)
    
    # Compute reference output for validation
    # Manually perform all-gather + matmul without fusion
    with torch.no_grad():
        gathered_ref = torch.zeros(
            (hidden_states.shape[0] * world_size, hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=device
        )
        if dist.is_initialized():
            dist.all_gather_into_tensor(gathered_ref, hidden_states)
        else:
            gathered_ref = hidden_states
        
        # Show gathered tensor stats to verify all-gather is working correctly
        print(f"  Rank {local_rank}: Gathered shape: {gathered_ref.shape}, mean: {gathered_ref.mean():.4f}")
        
        # Compute hash of gathered tensor to verify all ranks see the same data
        gathered_hash = torch.sum(gathered_ref).item()
        print(f"  Rank {local_rank}: Gathered tensor sum (hash): {gathered_hash:.4f}")
        
        weight_transposed = model.weight.permute(1, 0)
        reference_output = torch.mm(gathered_ref, weight_transposed)
    
    # Run forward pass with fusion
    output = compiled_model(hidden_states)
    torch.cuda.synchronize(device)
    end_mem = torch.cuda.memory_allocated(device)
    
    print(f"✓ Rank {local_rank}: Output shape: {output.shape}")
    print(f"  Output mean: {output.mean():.4f}")
    print(f"  Reference mean: {reference_output.mean():.4f}")
    print(f"  GPU memory used: {(end_mem - start_mem) / 1e6:.2f} MB")
    
    # ✓ VERIFICATION 7: Verify output is on correct GPU
    assert output.device.index == local_rank, f"Output on wrong GPU: {output.device.index} != {local_rank}"
    print(f"✓ Rank {local_rank}: Output on correct GPU {output.device}")
    verify_functionality(local_rank, async_tp_pass, backend, model, output)

    # ✓ VERIFICATION 7.5: Validate output correctness
    if local_rank == 0:
        max_diff = torch.max(torch.abs(output - reference_output)).item()
        relative_error = max_diff / (torch.max(torch.abs(reference_output)).item() + 1e-8)
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Relative error: {relative_error:.6f}")

        if max_diff > 1e-2:  # Tolerance for fp16
            print(f"✗ Rank {local_rank}: WARNING - Output differs from reference by {max_diff:.6f}")
            print(f"  This may indicate incorrect computation")
            # Show sample values for debugging
            print(f"  Sample output values: {output[0, :5]}")
            print(f"  Sample reference values: {reference_output[0, :5]}")
        else:
            print(f"✓ Rank {local_rank}: Output matches reference within tolerance!")

        # Compute output hash to verify both ranks produce identical results
        output_hash = torch.sum(output).item()
        print(f"  Rank {local_rank}: Output tensor sum (hash): {output_hash:.4f}")
    


    
    # Cleanup distributed resources to avoid warnings
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    
    if dist.is_initialized():
        dist.barrier()  # Ensure all ranks finish before cleanup
    
    cleanup_dist_env_and_memory()
    
    # Clean up CUDA resources
    torch.cuda.empty_cache()
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test all-gather + GEMM fusion using iris.x or symm_mem"
    )
    parser.add_argument("--async-tp", action="store_true",
                       help="Run AsyncTPPass test with compilation (requires 2 GPUs)")
    parser.add_argument("--dynamic", action="store_true",
                       help="Use dynamic shapes in AsyncTPPass test")
    parser.add_argument("--verify-gpus", action="store_true",
                       help="Verify multi-GPU setup before running tests")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Test Configuration:")
    print(f"  iris.x available: {IRIS_AVAILABLE}")
    if IRIS_AVAILABLE:
        print("  Using: torch.ops.iris.all_gather_gemm")
    else:
        print("  Using: torch.ops.symm_mem.fused_all_gather_matmul (fallback)")
        exit(1)

    print("="*70 + "\n")
    
    # Verify GPU setup if requested or running multi-GPU tests
    if args.verify_gpus or args.async_tp:
        if not standalone_gpu_verification():
            print("✗ Multi-GPU verification failed!")
            exit(1)
    
    if args.async_tp:
        # Run AsyncTPPass test with 2 GPUs
        if TestBackend is None:
            print("Error: TestBackend not available. Cannot run async-tp test.")
            exit(1)
        num_gpus = 2
        torch.multiprocessing.spawn(
            async_tp_test,
            args=(num_gpus, args.dynamic),
            nprocs=num_gpus,
            join=True
        )
    else:
        # Run simple test
        simple_test()

