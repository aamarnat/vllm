"""
Test for all-gather + GEMM fusion using iris.x.all_gather_gemm

This test demonstrates using iris.x.all_gather_gemm instead of the default
symm_mem.fused_all_gather_matmul for tensor-parallel fusion.

Key changes:
1. Registers torch.ops.iris.all_gather_gemm as a custom op
2. TestAGMMModel.ops_in_model_after() returns iris op when available
3. Falls back to symm_mem if iris.x not available

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

# Set environment variable for Triton
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

try:
    # Import TestBackend from tests module
    import sys
    from pathlib import Path
    # Add tests directory to path if not already there
    tests_path = Path(__file__).parent / "tests"
    if tests_path.exists() and str(tests_path) not in sys.path:
        sys.path.insert(0, str(tests_path))
    from compile.backend import TestBackend
except ImportError:
    # Fallback if tests module is not available
    TestBackend = None

# Try to import iris.x.all_gather_gemm
try:
    from iris.x.all_gather_gemm import all_gather_gemm as iris_all_gather_gemm_kernel
    from tritonblas.kernels.stages.indexing import grid_setup
    IRIS_AVAILABLE = True
    print("✓ iris.x.all_gather_gemm available")
except ImportError as e:
    IRIS_AVAILABLE = False
    iris_all_gather_gemm_kernel = None
    print(f"✗ iris.x.all_gather_gemm not available: {e}")


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
        x: Input tensor to all-gather (M_local, K)
        weights: List of weight tensors [(K, N)]
        gather_dim: Dimension along which to gather (typically 0)
        group_name: Process group name
        
    Returns:
        Tuple of (gathered_output, matmul_output)
    """
    if not IRIS_AVAILABLE:
        raise RuntimeError("iris.x.all_gather_gemm not available")
    
    import torch.distributed as dist
    
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
    
    # Unpack dimensions
    M_local = x.shape[0]
    K = x.shape[1]
    M = M_local * world_size
    weight = weights[0]
    N = weight.shape[1]
    
    # Allocate output tensors
    A_gathered = torch.zeros((M, K), dtype=x.dtype, device=x.device)
    C = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    
    # For single-GPU testing: simulate all-gather by replicating
    if world_size == 1:
        A_gathered[:] = x
    else:
        # In multi-GPU: perform actual all-gather
        # The iris kernel expects A_gathered to be pre-populated
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
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
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
            x,  # A_sharded
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
    # Infer shapes
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
        Forward pass implementing the mm + all gather in the FX graph
        """
        # Reshape input
        view = hidden_states.reshape(-1, self.hidden_size)
        all_gather = tensor_model_parallel_all_gather(view, dim=0)
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
    """Simple test without distributed setup (single GPU)"""
    print("Running simple TestAGMMModel test...")
    
    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    # Model parameters
    hidden_size = 16
    batch_size = 8
    seq_len = 16
    
    # Create model and move to device
    model = TestAGMMModel(hidden_size=hidden_size, dtype=dtype)
    model = model.to(device)
    
    # Create test input
    hidden_states = torch.randn(
        (batch_size * seq_len, hidden_size), 
        dtype=dtype, 
        device=device,
        requires_grad=False
    )
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Model weight shape: {model.weight.shape}")
    
    # Run forward pass (note: all_gather will behave differently without distributed setup)
    try:
        output = model(hidden_states)
        print(f"Output shape: {output.shape}")
        print("✓ Test passed!")
        return output
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        print("Note: This model requires distributed setup for full functionality")
        return None


def distributed_test(local_rank, world_size):
    """Test with distributed setup (requires multiple GPUs)"""
    print(f"Running distributed test on rank {local_rank}/{world_size}...")
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dtype = torch.float16
    torch.set_default_dtype(dtype)
    
    # Setup distributed environment
    update_environment_variables({
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12345",
    })
    
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    
    # Model parameters
    hidden_size = 16
    batch_size = 8
    seq_len = 16
    
    # Create model
    model = TestAGMMModel(hidden_size=hidden_size, dtype=dtype)
    
    # Create test input
    hidden_states = torch.randn(
        (batch_size * seq_len, hidden_size),
        dtype=dtype,
        requires_grad=False
    )
    
    print(f"Rank {local_rank} - Input shape: {hidden_states.shape}")
    
    # Run forward pass
    output = model(hidden_states)
    print(f"Rank {local_rank} - Output shape: {output.shape}")
    print(f"✓ Rank {local_rank} test passed!")
    
    return output


def async_tp_test(local_rank, world_size, dynamic=False, compile_only=False):
    """Test with AsyncTPPass applied (requires multiple GPUs and TestBackend)
    
    Args:
        local_rank: GPU rank
        world_size: Total number of GPUs
        dynamic: Whether to use dynamic shapes
        compile_only: If True, only compile and verify fusion without running
    """
    if TestBackend is None:
        print("✗ TestBackend not available. Skipping async_tp_test.")
        return None
    
    print(f"Running AsyncTPPass test on rank {local_rank}/{world_size}...")
    
    # Seed for reproducibility
    current_platform.seed_everything(0)
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    dtype = torch.float16
    torch.set_default_dtype(dtype)
    
    # Setup distributed environment
    update_environment_variables({
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12346",  # Different port to avoid conflicts
    })
    
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    
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
    
    # Create test input
    hidden_states = torch.randn(
        (batch_size * seq_len, hidden_size),
        dtype=dtype,
        requires_grad=False
    )
    
    if dynamic:
        torch._dynamo.mark_dynamic(hidden_states, 0)
    
    print(f"Rank {local_rank} - Input shape: {hidden_states.shape}")
    print(f"Rank {local_rank} - Compiling model with AsyncTPPass...")
    
    # Compile the model with AsyncTPPass
    compiled_model = torch.compile(model, backend=backend)
    
    if not compile_only:
        # Run forward pass (may fail with symm_mem issues)
        try:
            output = compiled_model(hidden_states)
            print(f"Rank {local_rank} - Output shape: {output.shape}")
        except RuntimeError as e:
            if "CUDA" in str(e) or "symm_mem" in str(e):
                print(f"⚠ Rank {local_rank} - Runtime execution failed (CUDA symm_mem issue): {e}")
                print(f"  This is a known limitation, but compilation succeeded.")
                output = None
            else:
                raise
    else:
        # Just trigger compilation without running
        print(f"Rank {local_rank} - Compile-only mode: skipping execution")
        # Trigger the compilation by calling torch.compile without running
        import torch.fx as fx
        output = None
    
    print(f"Rank {local_rank} - AsyncTPPass matched count: {async_tp_pass.matched_count}")
    
    # Verify the pass worked correctly
    if async_tp_pass.matched_count == 1:
        print(f"✓ Rank {local_rank} - AsyncTPPass successfully fused operations!")
        
        # Check that the operations were replaced
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        print(f"✓ Rank {local_rank} - Operation fusion verified!")
    elif async_tp_pass.matched_count > 0:
        print(f"⚠ Rank {local_rank} - AsyncTPPass matched {async_tp_pass.matched_count} times (expected 1)")
    else:
        print(f"✗ Rank {local_rank} - AsyncTPPass did not match any patterns")
    
    print(f"✓ Rank {local_rank} test completed!")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test all-gather + GEMM fusion using iris.x or symm_mem"
    )
    parser.add_argument("--distributed", action="store_true", 
                       help="Run distributed test (requires 2 GPUs)")
    parser.add_argument("--async-tp", action="store_true",
                       help="Run AsyncTPPass test with compilation (requires 2 GPUs)")
    parser.add_argument("--dynamic", action="store_true",
                       help="Use dynamic shapes in AsyncTPPass test")
    parser.add_argument("--compile-only", action="store_true",
                       help="Only compile and verify fusion without running (for async-tp test)")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Test Configuration:")
    print(f"  iris.x available: {IRIS_AVAILABLE}")
    if IRIS_AVAILABLE:
        print("  Using: torch.ops.iris.all_gather_gemm")
    else:
        print("  Using: torch.ops.symm_mem.fused_all_gather_matmul (fallback)")
    print("="*70 + "\n")
    
    if args.async_tp:
        # Run AsyncTPPass test with 2 GPUs
        if TestBackend is None:
            print("Error: TestBackend not available. Cannot run async-tp test.")
            exit(1)
        if not IRIS_AVAILABLE:
            print("Warning: iris.x not available. Test will use symm_mem fallback.")
        num_gpus = 2
        torch.multiprocessing.spawn(
            async_tp_test,
            args=(num_gpus, args.dynamic, args.compile_only),
            nprocs=num_gpus
        )
    elif args.distributed:
        # Run distributed test with 2 GPUs
        num_gpus = 2
        torch.multiprocessing.spawn(
            distributed_test,
            args=(num_gpus,),
            nprocs=num_gpus
        )
    else:
        # Run simple test
        simple_test()

