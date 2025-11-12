# MX Format Tests - Comprehensive Walkthrough

This directory contains detailed execution traces for the MX (Microscaling) format tests in `test/prototype/mx_formats/`. Each document provides a frame-by-frame trace from user-facing APIs down to the lowest level of execution (CUDA kernels, Triton kernels, or PyTorch internals).

## Test Modules Covered

### Core Quantization & Tensors
1. **[test_mx_tensor.md](./test_mx_tensor.md)** - Core MX tensor quantization and operations
2. **[test_nvfp4_tensor.md](./test_nvfp4_tensor.md)** - NVIDIA FP4 tensor format
3. **[test_kernels.md](./test_kernels.md)** - Low-level kernel implementations

### Training & Inference
4. **[test_mx_linear.md](./test_mx_linear.md)** - MXLinear training integration, autograd, and torch.compile
5. **[test_mx_mm.md](./test_mx_mm.md)** - Hardware-accelerated GEMM (CUBLAS FP8, CUTLASS FP4)
6. **[test_mx_dtensor.md](./test_mx_dtensor.md)** - Distributed training with tensor parallelism

### Serialization
7. **[test_mx_serialization.md](./test_mx_serialization.md)** - Model serialization and deserialization

## What is MX Format?

**Microscaling (MX)** is a quantization format developed by Microsoft, AMD, Intel, Qualcomm, and ARM that uses block-based scaling with shared exponents. Key features:

- **Shared exponents**: Multiple data elements share a single E8M0 (exponent-only) scale
- **Block-based**: Typically 32 elements per block
- **Multiple precisions**: FP8 (E4M3, E5M2), FP6 (E2M3, E3M2), FP4 (E2M1)
- **Hardware-friendly**: Designed for efficient tensor core execution

### MX vs NVFP4

| Aspect | MX Format | NVFP4 |
|--------|-----------|-------|
| Origin | OCP MX Spec | NVIDIA |
| Block size | 32 (configurable) | 16 (fixed) |
| Scales | E8M0 (exponent-only) | E4M3 (FP8) |
| FP4 format | E2M1 | E2M1 |
| Scaling | Single-level | Two-level (per-tensor + blockwise) |
| Hardware | General purpose | Optimized for Blackwell (SM100+) |

## Documentation Structure

Each test module documentation follows this structure:

1. **Overview** - What the test module is testing
2. **Test-by-Test Breakdown** - Summary of each test function
3. **Execution Traces** - Frame-by-frame walkthrough of key APIs:
   - **Python layer** - User-facing API calls
   - **TorchAO layer** - Quantization logic, tensor operations
   - **PyTorch internals** - Core tensor operations, torch.compile
   - **Kernel layer** - Triton JIT, CUDA kernels, or CUTLASS

## How to Read the Traces

Each trace is organized as "literate code" with:

```
📦 FRAME N: [Layer Name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 Source: [file_path:line_number](file_path#LlineNumber)

**What happens**: Brief description

**Code snippet**:
```python
# Annotated code
```

**Key operations**:
- Operation 1
- Operation 2

**Next**: → Calls [next_function] in Frame N+1
```

## Key Files Reference

### Test Files
- [test_mx_tensor.py](../../../test/prototype/mx_formats/test_mx_tensor.py) - Core quantization tests
- [test_nvfp4_tensor.py](../../../test/prototype/mx_formats/test_nvfp4_tensor.py) - NVFP4 format tests
- [test_kernels.py](../../../test/prototype/mx_formats/test_kernels.py) - Low-level kernel tests
- [test_mx_linear.py](../../../test/prototype/mx_formats/test_mx_linear.py) - Training integration tests
- [test_mx_mm.py](../../../test/prototype/mx_formats/test_mx_mm.py) - Hardware GEMM tests
- [test_mx_dtensor.py](../../../test/prototype/mx_formats/test_mx_dtensor.py) - Distributed training tests
- [test_mx_serialization.py](../../../test/prototype/mx_formats/test_mx_serialization.py) - Serialization tests

### Implementation Files
- [mx_tensor.py](../../../torchao/prototype/mx_formats/mx_tensor.py) - MXTensor class
- [nvfp4_tensor.py](../../../torchao/prototype/mx_formats/nvfp4_tensor.py) - NVFP4Tensor class
- [kernels.py](../../../torchao/prototype/mx_formats/kernels.py) - Triton/PyTorch kernels
- [mx_linear.py](../../../torchao/prototype/mx_formats/mx_linear.py) - Training integration
- [inference_workflow.py](../../../torchao/prototype/mx_formats/inference_workflow.py) - Inference configs

### CUDA/C++ Files
- [mxfp8_extension.cpp](../../../torchao/csrc/cuda/mx_kernels/mxfp8_extension.cpp) - Python bindings
- [mxfp8_cuda.cu](../../../torchao/csrc/cuda/mx_kernels/mxfp8_cuda.cu) - CUDA bridge
- [mxfp8_quantize.cuh](../../../torchao/csrc/cuda/mx_kernels/mxfp8_quantize.cuh) - CUDA kernels
- [mx_fp_cutlass_kernels.cu](../../../torchao/csrc/cuda/mx_kernels/mx_fp_cutlass_kernels.cu) - CUTLASS GEMM

## Quick Navigation

### Getting Started
- **Want to understand basic MX quantization?** → Start with [test_mx_tensor.md § test_hello_world](./test_mx_tensor.md#test_hello_world-execution-trace)
- **Need to understand NVFP4?** → See [test_nvfp4_tensor.md § test_nvfp4_reconstruction](./test_nvfp4_tensor.md#test_nvfp4_reconstruction-execution-trace)

### Training & Inference
- **How does MXLinear work for training?** → Read [test_mx_linear.md § test_linear_eager_vs_hp](./test_mx_linear.md#test-1-test_linear_eager_vs_hp)
- **What are the three GEMMs in backward pass?** → See [test_mx_linear.md § mx_mm.backward()](./test_mx_linear.md#-frame-8-backward-pass---mx_mmbackward)
- **How does torch.compile optimize MX?** → Check [test_mx_linear.md § test_linear_compile](./test_mx_linear.md#test-3-test_linear_compile)

### Hardware Acceleration
- **How do hardware FP8 GEMMs work?** → Read [test_mx_mm.md § torch._scaled_mm](./test_mx_mm.md#-frame-3-hardware-fp8-gemm---torch_scaled_mm)
- **What is scale swizzling?** → See [test_mx_mm.md § Scale Swizzling](./test_mx_mm.md#-frame-2-data-preparation-for-hardware-gemm)
- **How do CUTLASS FP4 kernels work?** → Check [test_mx_mm.md § mx_fp4_bf16](./test_mx_mm.md#-frame-4-hardware-fp4-gemm---torchaoopsmx_fp4_bf16)

### Distributed Training
- **How does MX work with DTensor?** → Read [test_mx_dtensor.md § DTensor Quantization](./test_mx_dtensor.md#-frame-3-dtensor-quantization)
- **What is tensor parallelism?** → See [test_mx_dtensor.md § Tensor Parallelism Setup](./test_mx_dtensor.md#-frame-2-tensor-parallelism-setup)
- **Why use dim1 quantization?** → Check [test_mx_dtensor.md § Dim1 Kernel Variants](./test_mx_dtensor.md#test-3-dim1-kernel-variants)

### Low-Level Implementation
- **Curious about Triton kernels?** → Check [test_kernels.md § Triton Kernel Traces](./test_kernels.md#triton-kernel-traces)
- **Want to see CUDA implementation?** → Read [test_kernels.md § CUDA MXFP8 Trace](./test_kernels.md#cuda-mxfp8-quantization-trace)

## Visual Glossary

Throughout the documentation, we use these symbols:

- 📦 **Frame** - A layer in the execution stack
- 📍 **Source** - File location with line numbers
- 🔄 **Flow** - Data flow or control flow
- ⚡ **Kernel** - GPU kernel execution
- 🎯 **Key Point** - Important concept or optimization
- ⚠️ **Note** - Edge case or special consideration
- 🔍 **Deep Dive** - Extra detail for advanced readers

## Contributing

When adding new traces:
1. Follow the frame-by-frame structure
2. Include source code links with line numbers
3. Annotate code with inline comments
4. Explain *why*, not just *what*
5. Link to related traces for context
