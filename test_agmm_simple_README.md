# TestAGMMModel - Simple Test Suite

A test suite for demonstrating and testing the All-Gather + Matrix Multiply (AGMM) pattern with vLLM's AsyncTPPass compilation optimization.

## Model Overview

`TestAGMMModel` implements a simple pattern:
1. Reshape input tensor
2. All-gather across tensor parallel ranks
3. Matrix multiplication

This pattern is common in distributed transformer models and can be optimized through operation fusion.

## Test Modes

### 1. Simple Test (Single GPU)
```bash
python test_agmm_simple.py
```
- Runs on a single GPU
- Will fail because all-gather requires distributed setup
- Useful for basic model structure verification

### 2. Distributed Test (2 GPUs)
```bash
python test_agmm_simple.py --distributed
```
- Requires 2 GPUs
- Runs standard distributed forward pass
- No compilation/optimization

**Expected Output:**
- Input shape: [128, 16]
- Output shape: [256, 16] (doubled due to all-gather from 2 ranks)

### 3. AsyncTPPass Test (2 GPUs + Compilation)
```bash
python test_agmm_simple.py --async-tp
```
- Requires 2 GPUs
- Applies AsyncTPPass compilation optimization
- Fuses all-gather + matmul into a single operation

**What it does:**
- Compiles the model with `torch.compile` and AsyncTPPass backend
- Transforms `vllm.all_gather` + `torch.mm` → `symm_mem.fused_all_gather_matmul`
- Verifies operation fusion was successful
- Attempts to run (may fail with CUDA symm_mem runtime issues on some systems)

**Optional flags:**
- `--dynamic`: Use dynamic shapes during compilation
- `--compile-only`: Skip execution (just verify compilation)

## Debug Configurations

The following VS Code debug configurations are available in `.vscode/launch.json`:

1. **vLLM: test_agmm_simple (Simple Mode)** - Single GPU test
2. **vLLM: test_agmm_simple (Distributed - 2 GPUs)** - Standard distributed test
3. **vLLM: test_agmm_simple (AsyncTPPass)** - AsyncTPPass compilation test
4. **vLLM: test_agmm_simple (AsyncTPPass + Dynamic)** - AsyncTPPass with dynamic shapes

## AsyncTPPass Details

### Before Fusion
```python
all_gather = tensor_model_parallel_all_gather(view, dim=0)
mm = torch.mm(all_gather, weight.T)
```
**Operations:** `vllm.all_gather` + `torch.mm`

### After Fusion
```python
output = fused_all_gather_matmul(view, weight.T, dim=0, group_name)
```
**Operation:** `symm_mem.fused_all_gather_matmul`

### Benefits
- Overlaps communication (all-gather) with computation (matmul)
- Reduces memory overhead
- Improves latency for distributed inference

## Known Limitations

The AsyncTPPass test successfully compiles and fuses operations, but may encounter runtime CUDA symmetric memory errors on certain configurations:
```
RuntimeError: handle_type_ != Expandable_Segments_Handle_Type::UNSPECIFIED
```

This is a PyTorch CUDA symm_mem limitation, not an issue with the compilation pass itself. The test will catch this error and report that compilation succeeded.

## Test Output Example

```
Running AsyncTPPass test on rank 0/2...
Rank 0 - Input shape: torch.Size([128, 16])
Rank 0 - Compiling model with AsyncTPPass...
Rank 0 - AsyncTPPass matched count: 1
✓ Rank 0 - AsyncTPPass successfully fused operations!
✓ Rank 0 - Operation fusion verified!
✓ Rank 0 test completed!
```

## Requirements

- 2 CUDA GPUs for distributed tests
- vLLM with compilation support
- PyTorch with distributed support

