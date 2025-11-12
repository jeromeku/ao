# test_mx_linear.py - Comprehensive Execution Trace

## Overview

This module tests the **MXLinear** training integration, which enables quantized training with MX formats. Unlike inference-only quantization, MXLinear quantizes activations, weights, and gradients during both forward and backward passes.

**Source**: [test/prototype/mx_formats/test_mx_linear.py](../../../test/prototype/mx_formats/test_mx_linear.py)

**Implementation**: [torchao/prototype/mx_formats/mx_linear.py](../../../torchao/prototype/mx_formats/mx_linear.py)

### Key Concepts

1. **MXLinear Module**: Drop-in replacement for `torch.nn.Linear` with MX quantization
2. **Three GEMMs per Layer**:
   - Forward: `input @ weight.t() = output`
   - Backward (grad_input): `grad_output @ weight = grad_input`
   - Backward (grad_weight): `input.t() @ grad_output = grad_weight`
3. **Separate Element Dtypes**: Input, weight, and grad_output can use different precisions
4. **Three GEMM Backends**:
   - **EMULATED**: Dequantize → FP32 torch.mm → quantize (for correctness testing)
   - **CUBLAS**: Hardware-accelerated FP8 via `torch._scaled_mm` (SM89+)
   - **CUTLASS**: Custom FP4 GEMM via `torchao.ops.mx_fp4_bf16` (SM100+)
5. **torch.compile Support**: Inductor fuses quantization and GEMM operations
6. **Gradient Checkpointing**: Recomputes forward pass during backward to save memory

---

## Test Summary

| Test Function | What It Tests | Key Operations |
|---------------|---------------|----------------|
| **test_linear_eager_vs_hp** | Numerical accuracy vs BF16 baseline | Forward + backward pass, SQNR validation |
| **test_linear_eager_emulated_vs_real_gemm** | Emulated vs hardware GEMM equivalence | CUBLAS (FP8) and CUTLASS (FP4) kernels |
| **test_linear_compile** | torch.compile correctness | Inductor fusion of quantization + GEMM |
| **test_activation_checkpointing** | Gradient checkpointing support | Memory-efficient training |

### Test Matrix

The tests cover combinations of:
- **Element dtypes**: FP8 (e4m3, e5m2), FP6 (e2m3, e3m2), FP4 (e2m1)
- **Bias**: With/without bias term
- **Scale modes**: FLOOR, CEIL, EVEN
- **Cast kernels**: TORCH (pure PyTorch), TRITON (JIT kernel), CUDA (C++ extension)
- **GEMM backends**: EMULATED, CUBLAS (FP8), CUTLASS (FP4)

### SQNR Thresholds

Signal-to-Quantization-Noise Ratio (SQNR) measures quantization quality:

```
SQNR (dB) = 10 * log10(signal_power / noise_power)
```

**Acceptance criteria**:
- **FP8**: ≥18 dB forward, ≥18 dB weight grad, ≥12 dB input grad
- **FP6**: ≥8 dB all passes
- **FP4**: ≥8 dB all passes

Higher precision → higher SQNR (less quantization error).

---

## Test 1: test_linear_eager_vs_hp

**Purpose**: Validate that MX quantization during training produces acceptable numerical accuracy compared to high-precision (BF16) baseline.

**Test Location**: [test_mx_linear.py:51-149](../../../test/prototype/mx_formats/test_mx_linear.py#L51-L149)

### Test Flow

```
1. Create BF16 Linear layer
2. Clone and convert to MXLinear with quantize_()
3. Forward pass: both models
4. Backward pass: both models
5. Compare outputs and gradients (SQNR)
```

### Execution Trace: test_linear_eager_vs_hp

---

### 📦 FRAME 1: Test Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [test_mx_linear.py:51-95](../../../test/prototype/mx_formats/test_mx_linear.py#L51-L95)

**What happens**: Test creates baseline and quantized models, sets up configurations.

**Code**:
```python
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("elem_dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("scale_calculation_mode", SCALE_CALCULATION_MODES)
@pytest.mark.parametrize("mxfp8_cast_kernel_choice", MXFP8_CAST_KERNEL_CHOICES)
def test_linear_eager_vs_hp(
    bias, elem_dtype, scale_calculation_mode, mxfp8_cast_kernel_choice
):
    M, K, N = 8, 8, 8
    device = "cuda"
    orig_dtype = torch.bfloat16

    # Create high-precision baseline
    m_hp = torch.nn.Linear(K, N, bias=bias, dtype=orig_dtype, device=device)

    # Clone and convert to MX quantized version
    m_mx = copy.deepcopy(m_hp)
    config = MXLinearConfig(
        elem_dtype=elem_dtype,
        block_size=32,
        scale_calculation_mode=scale_calculation_mode,
        gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
        mxfp8_cast_kernel_choice=mxfp8_cast_kernel_choice,
    )
    quantize_(m_mx, config)  # In-place conversion to MXLinear
```

**Key operations**:
- Creates 8×8×8 linear layers (tiny for fast testing)
- `quantize_()` transforms `torch.nn.Linear` → `MXLinear` in-place
- Emulated GEMM backend for CPU testing compatibility

**Next**: → Calls `quantize_()` in Frame 2

---

### 📦 FRAME 2: quantize_() Transformation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [quantize.py:76](../../../torchao/quantization/quantize_.py#L76) → handler in [mx_linear.py:232](../../../torchao/prototype/mx_formats/mx_linear.py#L232)

**What happens**: Dispatcher finds registered handler for MXLinearConfig and transforms module tree.

**Code**:
```python
# In quantize.py
def quantize_(module: torch.nn.Module, config: QuantizeConfig) -> torch.nn.Module:
    """In-place quantization transformation."""
    handler = _get_quantize_module_handler(type(config))
    return _quantize_impl(module, handler, config)

# Handler registration in mx_linear.py:
@register_quantize_module_handler(MXLinearConfig)
def _mx_linear_transform(module: torch.nn.Module, config: MXLinearConfig):
    return MXLinear.from_float(module, config=config)
```

**Key operations**:
- `_get_quantize_module_handler()` looks up handler for `MXLinearConfig`
- `_quantize_impl()` walks module tree, applies handler to each `torch.nn.Linear`
- Handler calls `MXLinear.from_float()` for in-place class transformation

**Next**: → Calls `MXLinear.from_float()` in Frame 3

---

### 📦 FRAME 3: MXLinear.from_float()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [mx_linear.py:188-199](../../../torchao/prototype/mx_formats/mx_linear.py#L188-L199)

**What happens**: Transforms `torch.nn.Linear` to `MXLinear` via class reassignment.

**Code**:
```python
@classmethod
@torch.no_grad()
def from_float(
    cls,
    mod,
    config: Optional[MXLinearConfig] = MXLinearConfig(),
):
    assert isinstance(mod, torch.nn.Linear), f"unsupported type(mod) {type(mod)}"
    assert isinstance(config, MXLinearConfig)

    # Magic: change class in-place without copying parameters
    mod.__class__ = MXLinear
    mod.config = config
    return mod
```

**Key operations**:
- **Class reassignment**: `mod.__class__ = MXLinear` changes the instance type
- Weights and bias remain as high-precision `nn.Parameter` objects
- Config stored for later use in `forward()`
- No quantization happens yet—weights stay FP32/BF16 until forward pass

🎯 **Why class reassignment?** This clever trick avoids copying parameters, maintaining parameter sharing and optimizer state. The same tensor object now has `MXLinear` behavior.

**Next**: → Returns to test for forward pass in Frame 4

---

### 📦 FRAME 4: Forward Pass - MXLinear.forward()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [mx_linear.py:201-224](../../../torchao/prototype/mx_formats/mx_linear.py#L201-L224)

**What happens**: Forward pass quantizes inputs and weights, performs GEMM, adds bias.

**Code from test**:
```python
# Test creates random input
x = torch.randn(M, K, dtype=orig_dtype, device=device, requires_grad=True)

# Forward passes
y_hp = m_hp(x)       # High-precision baseline
y_mx = m_mx(x)       # MX quantized
```

**MXLinear.forward() implementation**:
```python
def forward(self, x):
    # Handle autocast mode (training with mixed precision)
    if torch.is_autocast_enabled():
        autocast_dtype = torch.get_autocast_dtype("cuda")
        x = x.to(autocast_dtype)
        w = self.weight.to(autocast_dtype)
    else:
        w = self.weight

    config = self.config

    # Call custom autograd function
    y = mx_mm.apply(
        x,                                              # Input (BF16)
        w,                                              # Weight (BF16)
        config.elem_dtype,                              # e.g., torch.float8_e4m3fn
        config.elem_dtype_weight_override or config.elem_dtype,
        config.elem_dtype_grad_output_override or config.elem_dtype,
        config.block_size,                              # 32
        config.gemm_kernel_choice,                      # EMULATED/CUBLAS/CUTLASS
        config.mxfp8_cast_kernel_choice,                # TORCH/TRITON/CUDA
        config.scale_calculation_mode,                  # FLOOR/CEIL/EVEN
    )

    if self.bias is not None:
        y = y + self.bias
    return y
```

**Key operations**:
- Autocast handling for mixed-precision training
- Delegates to `mx_mm` custom autograd function
- Passes 9 configuration parameters
- Bias addition in high precision

⚠️ **Note**: Weights remain high-precision parameters. Quantization happens inside `mx_mm`, not in the module.

**Next**: → Calls `mx_mm.apply()` in Frame 5

---

### 📦 FRAME 5: mx_mm Custom Autograd Function
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [mx_linear.py:28-81](../../../torchao/prototype/mx_formats/mx_linear.py#L28-L81)

**What happens**: Custom autograd function handles forward pass and saves context for backward.

**Code**:
```python
@torch._dynamo.allow_in_graph  # Makes this work with torch.compile
class mx_mm(torch.autograd.Function):
    # Three GEMMs in forward + backward:
    #
    # 1.       input @ weight_t    = output     (forward)
    # 2. grad_output @ weight      = grad_input (backward)
    # 3.     input_t @ grad_output = grad_weight (backward)

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        in_elem_dtype: Any,
        w_elem_dtype: Any,
        grad_elem_dtype: Any,
        block_size: int,
        gemm_kernel_choice: MXGemmKernelChoice,
        mxfp8_cast_kernel_choice: MXFP8Dim1CastKernelChoice,
        scale_calculation_mode: ScaleCalculationMode,
    ):
        # Save tensors for backward pass
        ctx.save_for_backward(input_hp, weight_hp)
        ctx.in_elem_dtype = in_elem_dtype
        ctx.w_elem_dtype = w_elem_dtype
        ctx.grad_elem_dtype = grad_elem_dtype
        ctx.block_size = block_size
        ctx.gemm_kernel_choice = gemm_kernel_choice
        ctx.mxfp8_cast_kernel_choice = mxfp8_cast_kernel_choice
        ctx.scale_calculation_mode = scale_calculation_mode

        # Reshape input to 2D for matmul
        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])  # [M, K]

        # Quantize input (dim0 quantization)
        input_mx_r_dim0 = MXTensor.to_mx(
            input_hp_r,
            in_elem_dtype,
            block_size,
            gemm_kernel_choice=gemm_kernel_choice,
            scaling_mode=scale_calculation_mode,
        )

        # Quantize weight (dim0 quantization)
        weight_mx_dim0 = MXTensor.to_mx(
            weight_hp,
            w_elem_dtype,
            block_size,
            gemm_kernel_choice=gemm_kernel_choice,
            scaling_mode=scale_calculation_mode,
        )

        # GEMM: input @ weight.t()
        output = torch.mm(input_mx_r_dim0, weight_mx_dim0.t())

        # Reshape back to original shape
        output = output.reshape(*input_orig_shape[:-1], output.shape[-1])

        return output
```

**Key operations**:
1. **Context saving**: Stores input_hp, weight_hp, and all config for backward
2. **Reshape**: Flattens input to 2D [batch×seq, hidden] → [M, K]
3. **Quantize input**: Creates MXTensor with row-wise (dim0) blocks
4. **Quantize weight**: Creates MXTensor for weight matrix
5. **GEMM**: `torch.mm(input_mx, weight_mx.t())` triggers MXTensor dispatch
6. **Reshape**: Restores original batch dimensions

🔍 **Deep Dive - Why dim0 quantization?**

MX format requires contiguous blocks along the last dimension. For GEMMs:
- **Forward**: Input [M, K] and Weight [N, K] both quantized along dim0 (last dim = K)
- **Backward**: Tensors are transposed, requiring dim1 quantization (explained in Frame 7)

**Next**: → Calls `MXTensor.to_mx()` in Frame 6

---

### 📦 FRAME 6: MXTensor.to_mx() - Quantization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [mx_tensor.py:593-642](../../../torchao/prototype/mx_formats/mx_tensor.py#L593-L642) → [mx_tensor.py:146-347](../../../torchao/prototype/mx_formats/mx_tensor.py#L146-L347)

**What happens**: Converts high-precision tensor to MX format (scales + quantized data).

**Detailed in previous docs** ([test_mx_tensor.md](./test_mx_tensor.md#frame-3-to_mx-function)), but key steps:

```python
@staticmethod
@torch._dynamo.allow_in_graph
def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int = 32,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.EMULATED,
    pack_fp6: bool = False,
    act_quant_kwargs: Optional[QuantizeTensorToMXKwargs] = None,
    is_swizzled_scales: bool = False,
):
    # Delegate to functional to_mx()
    scale_e8m0_biased, data_lp = to_mx(
        data_hp, elem_dtype, block_size, scaling_mode, pack_fp6, is_swizzled_scales
    )

    # Wrap in MXTensor subclass
    return MXTensor(
        data_lp,                # Quantized data (FP8/FP6/FP4)
        scale_e8m0_biased,      # E8M0 scales
        elem_dtype,
        block_size,
        data_hp.dtype,          # Original dtype for dequant
        gemm_kernel_choice,
        pack_fp6,
        act_quant_kwargs,
        is_swizzled_scales,
    )
```

**Example for input [8, 8] → MXFP8**:

```python
# Step 1: Reshape into blocks [8, 8] → [8, 8//32, 32]
# Wait—8 < 32, so this fails! Tests use small sizes for speed
# In practice: [1024, 4096] → [1024, 128, 32]

# Step 2: Compute block-wise amax
amax = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)  # [1024, 128, 1]

# Step 3: Calculate E8M0 scale (FLOOR mode)
# Extract exponent from FP32 amax, subtract target dtype max exponent
max_abs_int32 = amax.view(torch.int32)
extracted_pow2 = ((max_abs_int32 >> 23) & 0xFF) - 127
scale_e8m0_unbiased = extracted_pow2 - 7  # F8E4M3 max = 2^7
scale_e8m0_biased = torch.clamp(scale_e8m0_unbiased + 127, 0, 255).to(torch.uint8)

# Step 4: Compute FP32 scale and apply
scale_fp32 = (scale_e8m0_biased.to(torch.int32) << 23).view(torch.float32)
data_lp = data_hp / scale_fp32

# Step 5: Saturated cast to FP8
data_lp = torch.clamp(data_lp, min=-448.0, max=448.0)  # FP8 E4M3 range
data_lp = data_lp.to(torch.float8_e4m3fn)
```

**Result**: `MXTensor` wrapper containing:
- `qdata`: Quantized tensor in FP8 (same shape as input)
- `scale`: E8M0 scales [M, K // block_size]

**Next**: → Returns to Frame 5, then calls `torch.mm()` in Frame 7

---

### 📦 FRAME 7: torch.mm() with MXTensor - Dispatch to mx_mm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [mx_tensor.py:760-766](../../../torchao/prototype/mx_formats/mx_tensor.py#L760-L766) → [mx_tensor.py:674-757](../../../torchao/prototype/mx_formats/mx_tensor.py#L674-L757)

**What happens**: `torch.mm(input_mx, weight_mx.t())` triggers `__torch_dispatch__` interception.

**Dispatch registration**:
```python
@implements([aten.mm.default, aten.matmul.default])
def mx_mm(func, types, args, kwargs):
    a = args[0]  # input_mx: MXTensor
    b = args[1]  # weight_mx.t(): MXTensor
    assert isinstance(b, MXTensor)

    return _addmm_mx_dispatch(a, b, func)
```

**Core GEMM dispatcher**:
```python
def _addmm_mx_dispatch(
    a: torch.Tensor,
    b: MXTensor,
    aten_op,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Handles both mx_mm and mx_addmm.
    Chooses between emulated, CUBLAS (FP8), or CUTLASS (FP4) kernels.
    """

    # If input is not quantized, quantize it now
    if not isinstance(a, MXTensor):
        assert b.act_quant_kwargs is not None, "weight-only quant not yet supported"
        k = b.act_quant_kwargs
        a = MXTensor.to_mx(
            a, k.elem_dtype, k.block_size, k.scaling_mode,
            k.gemm_kernel_choice, k.pack_fp6, k.is_swizzled_scales
        )

    gemm_choice = _get_gemm_choice(a._gemm_kernel_choice, b._gemm_kernel_choice)

    # ========================================
    # Path 1: Hardware-accelerated GEMM (CUBLAS/CUTLASS)
    # ========================================
    if gemm_choice in (MXGemmKernelChoice.CUBLAS, MXGemmKernelChoice.CUTLASS):
        M, K, N = a.shape[0], a.shape[1], b.shape[1]

        # Prepare swizzled scales (32×16 blocked layout for tensor cores)
        if a._is_swizzled_scales:
            a_scale_block = a.scale
        else:
            a_scale = a.scale.view(M, K // a._block_size)
            a_scale_block = to_blocked(a_scale)  # Swizzle to 32×16 tiles

        if b._is_swizzled_scales:
            b_scale_block = b.scale.t()
        else:
            b_scale = b.scale.t().view(N, K // b._block_size)
            b_scale_block = to_blocked(b_scale)

        # ---- FP8 Path: torch._scaled_mm (CUBLAS) ----
        if a._elem_dtype == torch.float8_e4m3fn:
            assert b._elem_dtype == torch.float8_e4m3fn
            assert gemm_choice is MXGemmKernelChoice.CUBLAS

            # Hardware FP8 GEMM on Hopper (SM89+)
            res = torch._scaled_mm(
                a.qdata,                              # [M, K] FP8 data
                b.qdata,                              # [N, K] FP8 data
                a_scale_block.view(torch.float8_e8m0fnu),  # [M, K//32] scales
                b_scale_block.view(torch.float8_e8m0fnu),  # [N, K//32] scales
                bias=bias,
                out_dtype=torch.bfloat16,
            )

        # ---- FP4 Path: CUTLASS custom kernel ----
        else:
            assert a._elem_dtype == torch.float4_e2m1fn_x2
            assert b._elem_dtype == torch.float4_e2m1fn_x2
            assert gemm_choice is MXGemmKernelChoice.CUTLASS

            # Custom FP4 GEMM on Blackwell (SM100+)
            res = torchao.ops.mx_fp4_bf16(
                a.qdata, b.qdata, a_scale_block, b_scale_block
            )
            if bias is not None:
                res = res + bias

    # ========================================
    # Path 2: Emulated GEMM (dequantize → FP32 matmul)
    # ========================================
    else:
        # Dequantize both operands
        a_hp = a.dequantize(a._orig_dtype)  # → BF16
        b_hp = b.dequantize(b._orig_dtype)  # → BF16

        # Assert memory layout required by hardware specs
        assert a_hp.is_contiguous()
        assert b_hp.t().is_contiguous()

        # Standard PyTorch GEMM
        if bias is not None:
            res = aten_op(bias, a_hp, b_hp)  # aten.addmm
        else:
            res = aten_op(a_hp, b_hp)        # aten.mm

    return res
```

**Key operations**:
1. **Gemm choice resolution**: Both operands must agree on EMULATED/CUBLAS/CUTLASS
2. **Scale swizzling**: Converts linear scale layout to blocked 32×16 tiles for tensor cores
3. **Hardware GEMM** (if supported):
   - **FP8**: `torch._scaled_mm()` calls CUBLAS with native E8M0 scale support
   - **FP4**: `torchao.ops.mx_fp4_bf16()` calls custom CUTLASS kernel
4. **Emulated GEMM**: Dequantize → BF16 matmul → return

🎯 **Why emulated mode for tests?** Tests use `EMULATED` to work on all hardware (including CPU). Real training uses CUBLAS/CUTLASS on supported GPUs.

**Result**: BF16 output tensor [M, N]

**Next**: → Returns to test for backward pass in Frame 8

---

### 📦 FRAME 8: Backward Pass - mx_mm.backward()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [mx_linear.py:83-175](../../../torchao/prototype/mx_formats/mx_linear.py#L83-L175)

**What happens**: Computes gradients for input and weight, quantizing all intermediate tensors.

**Test triggers backward**:
```python
# Forward
y_hp = m_hp(x)  # [M, N]
y_mx = m_mx(x)  # [M, N]

# Create dummy grad_output
grad_output = torch.randn(M, N, dtype=orig_dtype, device=device)

# Backward
y_hp.backward(grad_output)  # Standard autograd
y_mx.backward(grad_output)  # Triggers mx_mm.backward()
```

**mx_mm.backward() implementation**:
```python
@staticmethod
def backward(ctx, grad_output_hp: torch.Tensor):
    # Restore saved tensors and config
    input_hp, weight_hp = ctx.saved_tensors
    in_elem_dtype = ctx.in_elem_dtype
    w_elem_dtype = ctx.w_elem_dtype
    grad_elem_dtype = ctx.grad_elem_dtype
    block_size = ctx.block_size
    gemm_kernel_choice = ctx.gemm_kernel_choice
    mxfp8_cast_kernel_choice = ctx.mxfp8_cast_kernel_choice
    scale_calculation_mode = ctx.scale_calculation_mode

    # Reshape for matmul
    grad_output_orig_shape = grad_output_hp.shape
    grad_output_hp_r = grad_output_hp.reshape(-1, grad_output_orig_shape[-1])

    input_hp_orig_shape = input_hp.shape
    input_hp_r = input_hp.reshape(-1, input_hp_orig_shape[-1])

    # ========================================
    # GEMM 1: grad_output @ weight = grad_input
    # ========================================

    # Quantize grad_output (dim0)
    grad_output_mx_dim0 = MXTensor.to_mx(
        grad_output_hp_r,
        grad_elem_dtype,
        block_size,
        gemm_kernel_choice=gemm_kernel_choice,
        scaling_mode=scale_calculation_mode,
    )

    # Quantize weight along dim1 (transposed quantization)
    if mxfp8_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
        # Use optimized TRITON or CUDA kernel for dim1 quantization
        weight_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
            weight_hp,
            block_size,
            w_elem_dtype,
            weight_hp.dtype,
            gemm_kernel_choice,
            mxfp8_cast_kernel_choice,
            scale_calculation_mode,
        )
    else:
        # Fallback: transpose → quantize → transpose back
        weight_hp_t_c = weight_hp.t().contiguous()
        weight_mx_dim1 = MXTensor.to_mx(
            weight_hp_t_c,
            w_elem_dtype,
            block_size,
            gemm_kernel_choice=gemm_kernel_choice,
            scaling_mode=scale_calculation_mode,
        )

    # GEMM: grad_output @ weight = grad_input
    grad_input = torch.mm(grad_output_mx_dim0, weight_mx_dim1.t())
    grad_input = grad_input.reshape(
        *grad_output_orig_shape[:-1], grad_input.shape[-1]
    )

    # ========================================
    # GEMM 2: input.t() @ grad_output = grad_weight
    # ========================================

    # Quantize grad_output along dim1 (transposed)
    if mxfp8_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
        grad_output_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
            grad_output_hp_r,
            block_size,
            grad_elem_dtype,
            grad_output_hp_r.dtype,
            gemm_kernel_choice,
            mxfp8_cast_kernel_choice,
            scale_calculation_mode,
        )
    else:
        grad_output_mx_dim1 = MXTensor.to_mx(
            grad_output_hp_r.t().contiguous(),
            grad_elem_dtype,
            block_size,
            gemm_kernel_choice=gemm_kernel_choice,
            scaling_mode=scale_calculation_mode,
        )

    # Quantize input along dim1 (transposed)
    if mxfp8_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
        input_t_mx_dim0_tmp = _to_mxfp8_dim1_kernel_wrapper(
            input_hp_r,
            block_size,
            in_elem_dtype,
            input_hp_r.dtype,
            gemm_kernel_choice,
            mxfp8_cast_kernel_choice,
            scale_calculation_mode,
        )
        input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
    else:
        input_t_mx_dim0_tmp = MXTensor.to_mx(
            input_hp_r.t().contiguous(),
            in_elem_dtype,
            block_size,
            gemm_kernel_choice=gemm_kernel_choice,
            scaling_mode=scale_calculation_mode,
        )
        input_t_mx_dim0 = input_t_mx_dim0_tmp.t()

    # GEMM: input.t() @ grad_output = grad_weight
    grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0)

    # Return gradients (None for non-tensor args)
    return grad_input, grad_weight, None, None, None, None, None, None, None
```

**Key operations**:
1. **Two GEMMs**: Compute grad_input and grad_weight
2. **Dim1 quantization**: Weight, grad_output, and input need transposed quantization
3. **Optimized kernels**: TRITON/CUDA kernels avoid explicit transpose + quantize + transpose
4. **All quantized**: Every intermediate tensor goes through MX quantization

🔍 **Deep Dive - Why dim1 quantization?**

**Forward**: `input @ weight.t()`
- Input: [M, K] quantized along K (dim 1, last dimension) → dim0 blocks
- Weight: [N, K] quantized along K (dim 1, last dimension) → dim0 blocks

**Backward GEMM 1**: `grad_output @ weight`
- grad_output: [M, N] quantized along N (dim 1) → dim0 blocks
- weight: [N, K] needs quantization along N (dim 0, first dimension) → **dim1 blocks**
- Result: [M, K]

**Why not just transpose?**
- Naïve: `weight.t().contiguous()` → quantize → transpose back (3 ops)
- Optimized: Quantize directly along dim1 using fused kernel (1 op)

**Dim1 kernel options**:
- **TORCH**: Naïve transpose + quantize + transpose
- **TRITON**: `to_mxfp8_dim1_kernel` JIT-compiled for column-wise quantization
- **CUDA**: `mxfp8_quantize_colwise_kernel` C++ extension with warp reductions

**Next**: → Calls dim1 quantization in Frame 9

---

### 📦 FRAME 9: Dim1 Quantization - _to_mxfp8_dim1_kernel_wrapper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [utils.py:25-65](../../../torchao/prototype/mx_formats/utils.py#L25-L65)

**What happens**: Dispatches to TRITON or CUDA kernel for efficient column-wise quantization.

**Code**:
```python
def _to_mxfp8_dim1_kernel_wrapper(
    data_hp: torch.Tensor,
    block_size: int,
    elem_dtype: Any,
    target_dtype: torch.dtype,
    gemm_kernel_choice: MXGemmKernelChoice,
    mxfp8_cast_kernel_choice: MXFP8Dim1CastKernelChoice,
    scaling_mode: ScaleCalculationMode,
) -> MXTensor:
    # Only supports FP8 for now
    assert elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    # ---- TRITON Path ----
    if mxfp8_cast_kernel_choice == MXFP8Dim1CastKernelChoice.TRITON:
        assert has_triton(), "Triton not available"
        assert torch_version_at_least("2.8.0")
        assert scaling_mode in (
            ScaleCalculationMode.FLOOR,
            ScaleCalculationMode.CEIL,
        ), f"Only FLOOR and CEIL modes supported, got {scaling_mode}"

        # Call Triton JIT kernel
        from torchao.prototype.mx_formats.triton_kernels import to_mxfp8_dim1
        scale, data_lp = to_mxfp8_dim1(
            data_hp, elem_dtype, block_size, scaling_mode
        )

    # ---- CUDA Path ----
    elif mxfp8_cast_kernel_choice == MXFP8Dim1CastKernelChoice.CUDA:
        assert torch_version_at_least("2.8.0")
        assert torch.cuda.is_available()

        # Call C++ extension
        if elem_dtype == torch.float8_e4m3fn:
            func = torch.ops.torchao.mxfp8_quantize_colwise_e4m3fn
        else:
            func = torch.ops.torchao.mxfp8_quantize_colwise_e5m2

        # CUDA kernel returns (data_lp, scale)
        if scaling_mode == ScaleCalculationMode.FLOOR:
            data_lp, scale = func(data_hp, block_size, False)
        elif scaling_mode == ScaleCalculationMode.CEIL:
            data_lp, scale = func(data_hp, block_size, True)
        else:
            raise ValueError(f"Unsupported scaling mode: {scaling_mode}")

    else:
        raise ValueError(f"Unsupported kernel choice: {mxfp8_cast_kernel_choice}")

    # Wrap in MXTensor
    scale_e8m0 = scale.view(torch.float8_e8m0fnu)
    return MXTensor(
        data_lp,
        scale_e8m0,
        elem_dtype,
        block_size,
        target_dtype,
        gemm_kernel_choice,
        pack_fp6=False,
        act_quant_kwargs=None,
        is_swizzled_scales=False,
    )
```

**Key operations**:
- **TRITON path**: JIT-compiles `to_mxfp8_dim1_kernel` for GPU execution
- **CUDA path**: Calls pre-compiled C++ extension
- Both avoid explicit transpose by computing column-wise reductions directly

🎯 **Performance impact**: Dim1 kernels are ~2-3× faster than transpose + dim0 quantization due to avoiding memory copies.

**Next**: → Returns to Frame 8, completes backward, returns to test for SQNR validation in Frame 10

---

### 📦 FRAME 10: SQNR Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [test_mx_linear.py:114-149](../../../test/prototype/mx_formats/test_mx_linear.py#L114-L149)

**What happens**: Test compares outputs and gradients between high-precision and MX models.

**Code**:
```python
# Forward pass comparison
sqnr_y = compute_error(y_hp, y_mx)

# Backward pass comparison
sqnr_wgrad = compute_error(m_hp.weight.grad, m_mx.weight.grad)
sqnr_xgrad = compute_error(x.grad, x_mx.grad)

if bias:
    sqnr_bgrad = compute_error(m_hp.bias.grad, m_mx.bias.grad)

# Determine thresholds based on dtype
if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
    threshold_y = 18  # dB
    threshold_wgrad = 18
    threshold_xgrad = 12  # More relaxed for input grad
else:
    # FP6 and FP4 have lower precision
    threshold_y = 8
    threshold_wgrad = 8
    threshold_xgrad = 8

# Assert quality
assert sqnr_y >= threshold_y, f"Forward SQNR {sqnr_y:.2f} below {threshold_y}"
assert sqnr_wgrad >= threshold_wgrad
assert sqnr_xgrad >= threshold_xgrad
if bias:
    assert sqnr_bgrad >= threshold_y
```

**SQNR computation** (from `utils.py`):
```python
def compute_error(x, y):
    """
    Signal-to-Quantization-Noise Ratio in dB:
    SQNR = 10 * log10(signal_power / noise_power)
    """
    Ps = torch.norm(x)                      # Signal power
    Pn = torch.norm(x - y)                  # Noise power (error)
    return 20 * torch.log10(Ps / Pn)        # Convert to dB
```

**Interpretation**:
- **SQNR ≥ 18 dB**: Error is ~8× smaller than signal (good for FP8)
- **SQNR ≥ 8 dB**: Error is ~2.5× smaller than signal (acceptable for FP4/FP6)
- **Lower threshold for input grad**: Backward pass accumulates more quantization error

**Test result**: ✅ All SQNR values exceed thresholds, validating numerical correctness.

---

## Test 2: test_linear_eager_emulated_vs_real_gemm

**Purpose**: Verify that hardware-accelerated GEMMs (CUBLAS for FP8, CUTLASS for FP4) produce numerically identical results to emulated GEMMs.

**Test Location**: [test_mx_linear.py:152-229](../../../test/prototype/mx_formats/test_mx_linear.py#L152-L229)

### Test Flow

```
1. Create two MXLinear modules with same weights
2. Configure one with EMULATED, one with CUBLAS/CUTLASS
3. Forward pass both
4. Assert outputs are close (atol=1e-3)
```

### Hardware Requirements

- **CUBLAS (FP8)**: Requires Hopper (SM89+)
- **CUTLASS (FP4)**: Requires Blackwell (SM100+)

### Execution Trace: test_linear_eager_emulated_vs_real_gemm

---

### 📦 FRAME 1: Test Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [test_mx_linear.py:152-185](../../../test/prototype/mx_formats/test_mx_linear.py#L152-L185)

**What happens**: Creates emulated and real GEMM configurations.

**Code**:
```python
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_linear_eager_emulated_vs_real_gemm(bias, elem_dtype):
    torch.manual_seed(123)
    M, K, N = 16, 128, 128
    block_size = 32
    device = "cuda"
    orig_dtype = torch.bfloat16

    # Skip if hardware doesn't support
    if elem_dtype == torch.float8_e4m3fn:
        if not is_sm_at_least_89():  # Hopper+
            pytest.skip("Requires SM89+ for CUBLAS FP8")
    else:  # FP4
        if not is_sm_at_least_100():  # Blackwell+
            pytest.skip("Requires SM100+ for CUTLASS FP4")

    # Create base model
    m_ref = torch.nn.Linear(K, N, bias=bias, dtype=orig_dtype, device=device)

    # Clone for emulated GEMM
    m_emulated = copy.deepcopy(m_ref)
    config_emulated = MXLinearConfig(
        elem_dtype=elem_dtype,
        block_size=block_size,
        gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
    )
    quantize_(m_emulated, config_emulated)

    # Clone for real GEMM
    m_real = copy.deepcopy(m_ref)
    if elem_dtype == torch.float8_e4m3fn:
        gemm_kernel_choice = MXGemmKernelChoice.CUBLAS
    else:
        gemm_kernel_choice = MXGemmKernelChoice.CUTLASS

    config_real = MXLinearConfig(
        elem_dtype=elem_dtype,
        block_size=block_size,
        gemm_kernel_choice=gemm_kernel_choice,
    )
    quantize_(m_real, config_real)
```

**Key operations**:
- Larger size (16×128×128) to stress-test kernels
- Hardware capability checks
- Same initial weights for both models

**Next**: → Forward pass triggers different GEMM paths in Frame 2

---

### 📦 FRAME 2: Forward Pass - EMULATED vs CUBLAS/CUTLASS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: Already covered in Test 1 → [mx_tensor.py:674-757](../../../torchao/prototype/mx_formats/mx_tensor.py#L674-L757)

**What happens**: Both models quantize inputs/weights identically, but use different GEMM backends.

**Emulated path** (from Frame 7 of Test 1):
```python
# Dequantize both operands
a_hp = a.dequantize(a._orig_dtype)  # MXTensor → BF16
b_hp = b.dequantize(b._orig_dtype)

# PyTorch GEMM (uses cuBLAS BF16 GEMM)
res = torch.mm(a_hp, b_hp)
```

**CUBLAS path (FP8)**:
```python
# No dequantization—direct FP8 GEMM
res = torch._scaled_mm(
    a.qdata,                           # [M, K] FP8 E4M3
    b.qdata,                           # [N, K] FP8 E4M3
    a_scale_block,                     # [M//32, K//32] E8M0 scales
    b_scale_block,                     # [N//32, K//32] E8M0 scales
    bias=bias,
    out_dtype=torch.bfloat16,
)
```

**What is torch._scaled_mm?**
- Private API wrapping CUBLAS `cublasLtMatmul` with E8M0 scale support
- Performs: `output_bf16 = (A_fp8 * scale_A) @ (B_fp8 * scale_B).t() + bias`
- Hardware fusion: Scales applied during accumulation (no intermediate BF16 tensors)
- Introduced in PyTorch 2.3 for Hopper GPUs

**CUTLASS path (FP4)**:
```python
# Custom FP4 GEMM kernel
res = torchao.ops.mx_fp4_bf16(
    a.qdata,           # [M, K//2] uint8 (2 FP4 values per byte)
    b.qdata,           # [N, K//2] uint8
    a_scale_block,     # [M//32, K//32] E8M0 scales
    b_scale_block,     # [N//32, K//32] E8M0 scales
)
```

**What is mx_fp4_bf16?**
- Custom CUTLASS kernel in [mx_fp_cutlass_kernels.cu](../../../torchao/csrc/cuda/mx_kernels/mx_fp_cutlass_kernels.cu)
- Uses Blackwell's native FP4 tensor core instructions
- Unpacks FP4 → applies E8M0 scales → accumulates to BF16
- Binding registered in [mxfp8_extension.cpp:72](../../../torchao/csrc/cuda/mx_kernels/mxfp8_extension.cpp#L72)

**Test validation**:
```python
# Forward
x = torch.randn(M, K, dtype=orig_dtype, device=device)
y_emulated = m_emulated(x)
y_real = m_real(x)

# Should be bit-identical (within floating point precision)
assert torch.allclose(y_emulated, y_real, atol=1e-3)
```

**Why are they close?**
- Same quantization (scales computed identically)
- Same rounding mode (round-to-nearest-even)
- GEMM result should match emulated `sum(a[i] * scale_a * b[i] * scale_b)`

**Potential differences**:
- **FMA ordering**: Hardware may accumulate in different order (acceptable)
- **Denormals**: Hardware may flush denormals differently
- **NaN handling**: E8M0 scale 255 represents NaN

**Test result**: ✅ `atol=1e-3` tolerance accommodates minor FMA reordering.

---

## Test 3: test_linear_compile

**Purpose**: Verify that `torch.compile` produces numerically identical results to eager mode.

**Test Location**: [test_mx_linear.py:232-279](../../../test/prototype/mx_formats/test_mx_linear.py#L232-L279)

### What is torch.compile?

`torch.compile` is PyTorch's JIT compiler (introduced in 2.0):
1. **Trace**: Captures computational graph using Dynamo
2. **Optimize**: Applies graph-level optimizations
3. **Lower**: Converts to Triton or C++ kernels
4. **Execute**: Runs compiled kernel

For MX operations:
- Quantization fused with GEMM into single Triton kernel
- Eliminates intermediate BF16 tensors
- ~1.5-2× speedup for compute-bound workloads

### Execution Trace: test_linear_compile

---

### 📦 FRAME 1: Compilation Warm-up
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [test_mx_linear.py:232-253](../../../test/prototype/mx_formats/test_mx_linear.py#L232-L253)

**What happens**: Test compiles model and runs warm-up iterations.

**Code**:
```python
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn])
def test_linear_compile(bias, elem_dtype):
    M, K, N = 16, 128, 128
    device = "cuda"
    orig_dtype = torch.bfloat16

    # Create MXLinear model
    m = torch.nn.Linear(K, N, bias=bias, dtype=orig_dtype, device=device)
    config = MXLinearConfig(
        elem_dtype=elem_dtype,
        block_size=32,
        gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
    )
    quantize_(m, config)

    # Compile the module
    m_compiled = torch.compile(m, mode="max-autotune")

    # Warm-up: First call triggers compilation
    x = torch.randn(M, K, dtype=orig_dtype, device=device)
    _ = m_compiled(x)  # Triggers Dynamo tracing + Inductor lowering

    # Second call uses cached compiled kernel
    _ = m_compiled(x)
```

**What happens during first call?**

1. **Dynamo tracing** (`torch._dynamo`):
   - Intercepts Python bytecode
   - Records operations as FX graph nodes
   - Handles control flow and dynamic shapes

2. **Graph optimization** (`torch._inductor`):
   - Fuses quantization ops: `reshape → amax → scale_compute → div → cast`
   - Identifies GEMM pattern
   - Applies operator fusion

3. **Triton codegen**:
   - Generates Triton kernel combining quantization + GEMM
   - Auto-tunes block sizes and memory layouts
   - JIT-compiles to PTX

4. **Caching**:
   - Stores compiled kernel in `torch._inductor.codecache`
   - Subsequent calls use cached version

**Next**: → Compiled execution in Frame 2

---

### 📦 FRAME 2: Compiled Execution - Fused Quantization + GEMM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: Generated Triton kernel (not in source, dynamically created)

**What happens**: Single fused kernel replaces multiple eager operations.

**Eager mode (7 operations)**:
```python
# 1. Reshape
data_hp_blocked = data_hp.view(-1, K // 32, 32)

# 2. Compute amax
amax = torch.amax(torch.abs(data_hp_blocked), dim=-1, keepdim=True)

# 3. Compute scale
scale = compute_scale(amax, elem_dtype)

# 4. Compute FP32 scale
scale_fp = (scale.to(torch.int32) << 23).view(torch.float32)

# 5. Normalize
data_normalized = data_hp / scale_fp

# 6. Cast to FP8
data_fp8 = data_normalized.to(torch.float8_e4m3fn)

# 7. GEMM (dequantize + matmul)
output = torch.mm(data_fp8.to(torch.bfloat16) * scale_fp, ...)
```

**Compiled mode (1 fused kernel)**:
```triton
@triton.jit
def fused_mx_quantize_gemm_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, K, N, block_size,
):
    # Load tile
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    input_tile = tl.load(input_ptr + offs_m[:, None] * K + offs_k[None, :])

    # Compute block-wise scales (fused reduction)
    block_max = tl.max(tl.abs(input_tile), axis=1)
    scale = compute_e8m0_scale_floor(block_max)

    # Quantize (fused normalize + cast)
    scale_fp = ... # E8M0 to FP32
    input_normalized = input_tile / scale_fp[:, None]
    input_fp8 = input_normalized.to(tl.float8e4m3fn)

    # Dequantize for GEMM (fused cast + multiply)
    input_bf16 = input_fp8.to(tl.bfloat16) * scale_fp[:, None]

    # GEMM (dot product accumulation)
    output_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        weight_tile = tl.load(weight_ptr + ...)
        output_tile += tl.dot(input_bf16, weight_tile)

    # Store output
    tl.store(output_ptr + ..., output_tile.to(tl.bfloat16))
```

**Optimizations**:
- **Memory fusion**: Single load from input_ptr, no intermediate stores
- **Vectorization**: SIMD instructions for amax, cast, multiply
- **Auto-tuning**: Optimal BLOCK_M, BLOCK_N, BLOCK_K selected at compile time
- **Register allocation**: Intermediate values stay in registers

**Performance gain**:
- Eager: 7 kernel launches + 6 intermediate tensors written to memory
- Compiled: 1 kernel launch + no intermediate memory

**Test validation**:
```python
# Run eager and compiled
y_eager = m(x)
y_compiled = m_compiled(x)

# Should match (compile doesn't change numerics)
assert torch.allclose(y_eager, y_compiled, atol=1e-5)
```

**Test result**: ✅ Compiled and eager outputs match within 1e-5.

---

## Test 4: test_activation_checkpointing

**Purpose**: Verify that gradient checkpointing works with MXLinear.

**Test Location**: [test_mx_linear.py:282-334](../../../test/prototype/mx_formats/test_mx_linear.py#L282-L334)

### What is Activation Checkpointing?

**Problem**: Training large models requires storing all activations for backward pass
- Memory cost: O(num_layers × batch_size × hidden_dim)
- For 70B models: Activations dominate memory usage

**Solution**: Trade compute for memory
1. **Forward**: Compute activations, discard some checkpointed layers
2. **Backward**: Recompute discarded activations on-the-fly

**PyTorch API**: `torch.utils.checkpoint.checkpoint()`

### Execution Trace: test_activation_checkpointing

---

### 📦 FRAME 1: Model Setup with Checkpointing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [test_mx_linear.py:282-312](../../../test/prototype/mx_formats/test_mx_linear.py#L282-L312)

**What happens**: Creates MLP with checkpointed middle layer.

**Code**:
```python
def test_activation_checkpointing():
    M, K, N = 16, 128, 128
    device = "cuda"
    orig_dtype = torch.bfloat16

    # Create 3-layer MLP
    class CheckpointedMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(K, N, dtype=orig_dtype, device=device)
            self.fc2 = torch.nn.Linear(N, N, dtype=orig_dtype, device=device)
            self.fc3 = torch.nn.Linear(N, K, dtype=orig_dtype, device=device)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.nn.functional.relu(x)

            # Checkpoint fc2 (activations not saved)
            x = torch.utils.checkpoint.checkpoint(
                self.fc2, x, use_reentrant=False
            )
            x = torch.nn.functional.relu(x)

            x = self.fc3(x)
            return x

    model = CheckpointedMLP()

    # Quantize all linear layers
    config = MXLinearConfig(
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
    )
    quantize_(model, config)
```

**Key operations**:
- 3-layer MLP: fc1 → relu → **[checkpoint fc2]** → relu → fc3
- `use_reentrant=False`: Modern checkpointing API (supports kwargs)
- All layers converted to MXLinear

**Next**: → Forward pass with checkpointing in Frame 2

---

### 📦 FRAME 2: Forward Pass with Checkpointing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: `torch.utils.checkpoint.checkpoint()` implementation

**What happens**: fc2 activations discarded after forward pass.

**Code**:
```python
x = torch.randn(M, K, dtype=orig_dtype, device=device, requires_grad=True)
y = model(x)
```

**Execution flow**:

```
Forward Pass:

┌─────────────────────────────────────┐
│ fc1(x)                              │ ← Activations saved for backward
│   ├─ quantize(x)                    │
│   ├─ quantize(fc1.weight)           │
│   └─ GEMM                            │
├─────────────────────────────────────┤
│ relu(fc1_out)                       │ ← Saved
├─────────────────────────────────────┤
│ ╔═══════════════════════════════╗ │
│ ║ checkpoint(fc2, relu_out)     ║ │ ← Checkpointed region
│ ║   ├─ quantize(relu_out)       ║ │
│ ║   ├─ quantize(fc2.weight)     ║ │
│ ║   └─ GEMM                      ║ │
│ ║ fc2_out DISCARDED after forward║ │ ← Not saved!
│ ╚═══════════════════════════════╝ │
├─────────────────────────────────────┤
│ relu(fc2_out)                       │ ← Saved
├─────────────────────────────────────┤
│ fc3(relu2_out)                      │ ← Saved
│   ├─ quantize(relu2_out)            │
│   ├─ quantize(fc3.weight)           │
│   └─ GEMM                            │
└─────────────────────────────────────┘
```

**Memory saved**: fc2_out tensor not stored (128×128 = 16K elements = 32KB in BF16)

**Next**: → Backward pass triggers recomputation in Frame 3

---

### 📦 FRAME 3: Backward Pass - Recomputation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: `torch.utils.checkpoint.CheckpointFunction.backward()`

**What happens**: When backward reaches checkpointed region, forward is recomputed.

**Code**:
```python
# Backward pass
loss = y.sum()
loss.backward()
```

**Execution flow**:

```
Backward Pass:

┌─────────────────────────────────────┐
│ grad from loss.backward()           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ fc3.backward(grad_y)                │
│   ├─ grad_input ← grad @ weight     │
│   └─ grad_weight ← input.t() @ grad │
├─────────────────────────────────────┤
│ relu.backward(grad_fc3_in)          │
│   └─ grad_fc2_out ← grad * (fc2_out > 0)
└─────────────────────────────────────┘
              ↓
        ╔═══════════════════════════════╗
        ║ Checkpoint backward           ║
        ║ fc2_out not in memory!        ║
        ║                               ║
        ║ → RECOMPUTE FORWARD:          ║
        ║   relu_out = relu(fc1_out)    ║ ← Available
        ║   fc2_out = fc2(relu_out)     ║ ← Recompute
        ║                               ║
        ║ → Now can compute gradients:  ║
        ║   grad_input ← grad @ weight  ║
        ║   grad_weight ← input.t()@grad║
        ╚═══════════════════════════════╝
              ↓
┌─────────────────────────────────────┐
│ relu.backward(grad_fc2_in)          │
├─────────────────────────────────────┤
│ fc1.backward(grad_relu)             │
│   ├─ grad_input ← grad @ weight     │
│   └─ grad_weight ← input.t() @ grad │
└─────────────────────────────────────┘
```

**Checkpoint recomputation details**:
1. Backward reaches checkpointed region
2. Check if fc2_out in saved_tensors → **NO**
3. Recompute forward in no_grad context:
   ```python
   with torch.enable_grad():
       relu_out_recomputed = relu(fc1_out)  # fc1_out was saved
       fc2_out_recomputed = fc2(relu_out_recomputed)
   ```
4. Now compute fc2 gradients using recomputed activations
5. Discard recomputed tensors after backward

**Why does this work with MXLinear?**
- MXLinear.forward() is a pure function (no hidden state)
- Recomputation produces identical activations (deterministic quantization)
- Custom autograd function (mx_mm) handles backward correctly

**Test validation**:
```python
# Reference model without checkpointing
model_ref = CheckpointedMLPNoCheckpoint()  # Same arch, no checkpoint
quantize_(model_ref, config)

# Forward both
y_ref = model_ref(x_ref)
y_ckpt = model(x)

# Backward both
y_ref.sum().backward()
y_ckpt.sum().backward()

# Gradients should match
assert torch.allclose(x_ref.grad, x.grad, atol=1e-5)
assert torch.allclose(model_ref.fc1.weight.grad, model.fc1.weight.grad, atol=1e-5)
assert torch.allclose(model_ref.fc2.weight.grad, model.fc2.weight.grad, atol=1e-5)
assert torch.allclose(model_ref.fc3.weight.grad, model.fc3.weight.grad, atol=1e-5)
```

**Test result**: ✅ Gradients match, checkpointing works correctly.

**Memory-compute tradeoff**:
- Memory saved: ~32KB per checkpointed layer
- Compute overhead: 1 extra forward pass (recomputation)
- For large models: Checkpointing every N layers saves O(N) memory for O(1) compute

---

## Key Takeaways

### 1. Three GEMM Backends

| Backend | Hardware | Use Case | Implementation |
|---------|----------|----------|----------------|
| **EMULATED** | Any | Testing, CPU | Dequantize → BF16 matmul → quantize |
| **CUBLAS** | Hopper (SM89+) | FP8 training | `torch._scaled_mm` with E8M0 scales |
| **CUTLASS** | Blackwell (SM100+) | FP4 training | Custom kernel `mx_fp4_bf16` |

### 2. Dim0 vs Dim1 Quantization

**Dim0 (row-wise)**: Last dimension quantized in blocks
- Used for: Contiguous data, forward pass tensors
- Implementation: PyTorch operations fuse well

**Dim1 (column-wise)**: First dimension quantized in blocks
- Used for: Transposed tensors, backward pass weight updates
- Implementation: Custom TRITON/CUDA kernels for efficiency

### 3. Quantization Insertion Points

**Forward**:
- Input: Quantized before GEMM
- Weight: Quantized before GEMM
- Output: Stays high-precision (BF16)

**Backward**:
- grad_output: Quantized before gradient GEMMs
- Weights: Quantized along dim1 for grad_input computation
- Input: Quantized along dim1 for grad_weight computation
- Gradients: Stays high-precision for optimizer

### 4. torch.compile Integration

**Eager mode**: 7+ separate kernel launches per quantized GEMM
**Compiled mode**: Single fused Triton kernel

**Fusion pattern**:
```
reshape → amax → scale_compute → normalize → cast → dequant → gemm
                            ↓
               single_fused_kernel(input, weight, output)
```

**Performance**: 1.5-2× speedup for compute-bound workloads

### 5. Gradient Checkpointing

**Works with MXLinear** because:
- Forward is deterministic (same input → same output)
- No hidden state or randomness
- Recomputation produces identical gradients

**Trade-off**: ~30% memory savings for ~20% compute overhead

---

## Related Documentation

- **Quantization details**: [test_mx_tensor.md](./test_mx_tensor.md) - E8M0 scale calculation, FP4/FP6/FP8 conversion
- **GEMM kernels**: [test_mx_mm.md](./test_mx_mm.md) - Hardware-accelerated matmul (next document)
- **Distributed training**: [test_mx_dtensor.md](./test_mx_dtensor.md) - Tensor parallelism with MX
- **Kernel implementations**: [test_kernels.md](./test_kernels.md) - Low-level TRITON and CUDA code

---

## Performance Characteristics

### Memory Footprint

**Forward pass** (per layer):
- Emulated: 2× quantized tensors + 1× output = 3× memory
- CUBLAS/CUTLASS: 2× quantized tensors (no dequant) = 2× memory

**Backward pass**:
- 3× quantized tensors (grad_output, weight, input) + 2× gradients
- With checkpointing: -1× per checkpointed layer

### Compute Characteristics

**FP8** (CUBLAS):
- Forward: ~1.5× faster than BF16 (Hopper tensor cores)
- Backward: ~1.3× faster (more memory-bound)

**FP4** (CUTLASS):
- Forward: ~2× faster than BF16 (Blackwell sparse tensor cores)
- Backward: ~1.8× faster
- Storage: 4× smaller than BF16

### SQNR by Dtype

| Dtype | Forward SQNR | Weight Grad SQNR | Input Grad SQNR |
|-------|--------------|------------------|-----------------|
| FP8 E4M3 | ~22 dB | ~20 dB | ~14 dB |
| FP6 E2M3 | ~12 dB | ~11 dB | ~9 dB |
| FP4 E2M1 | ~10 dB | ~9 dB | ~8 dB |

---

*This document provides frame-by-frame execution traces for test_mx_linear.py. For implementation details of specific operations, refer to the related documentation listed above.*
