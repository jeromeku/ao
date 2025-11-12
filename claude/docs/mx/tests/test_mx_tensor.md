# test_mx_tensor.py - Execution Trace Walkthrough

## Overview

[test_mx_tensor.py](../../../test/prototype/mx_formats/test_mx_tensor.py) tests the core **MXTensor** class, which implements the OCP Microscaling (MX) specification. This includes:

- **Quantization** (high precision → MX format)
- **Dequantization** (MX format → high precision)
- **Tensor operations** (transpose, view, clone, indexing)
- **Numerical accuracy** (SQNR validation)
- **Scale calculation modes** (FLOOR, CEIL, RCEIL)
- **Multiple precisions** (FP8 E4M3, FP8 E5M2, FP6 E2M3, FP6 E3M2, FP4 E2M1)
- **Compilation** (torch.compile support)
- **Scale layouts** (swizzled vs non-swizzled)

## Test Summary

### Basic Functionality Tests

| Test | Lines | Purpose |
|------|-------|---------|
| `test_hello_world` | 86-89 | Smoke test: 8×8 tensor, block_size=4 |
| `test_realistic_numerics` | 95-98 | 128×128 tensor, block_size=32, all scale modes |
| `test_all_zeros` | 103-106 | Edge case: all-zero input |
| `test_some_zeros` | 111-116 | Mixed zeros and non-zeros |
| `test_ranks` | 397-405 | 1D, 2D, 3D, 4D tensor support |
| `test_block_sizes` | 411-420 | Block sizes: 1, 4, 32 |

### Numerical Edge Cases

| Test | Lines | Purpose |
|------|-------|---------|
| `test_to_mx_rceil` | 120-325 | RCEIL mode with NaN, denormals, normals |
| `test_exponent_nan_in` | 330-341 | NaN input → NaN scale |
| `test_exponent_nan_out` | 347-392 | NaN scale → NaN output |
| `test_cast_to_float8_e4m3fn_saturation_behavior` | 581-625 | Eager vs Triton cast behavior |

### Tensor Operations

| Test | Lines | Purpose |
|------|-------|---------|
| `test_transpose` | 425-443 | `tensor.t()` correctness |
| `test_view` | 448-452 | `tensor.view()` reshaping |
| `test_clone` | 456-466 | `tensor.clone()` deep copy |
| `test_index_select` | 558-573 | Indexing 3D tensors (expert weights) |

### Data Packing

| Test | Lines | Purpose |
|------|-------|---------|
| `test_fp6_packing` | 472-481 | FP6 packed vs unpacked formats |

### Compilation

| Test | Lines | Purpose |
|------|-------|---------|
| `test_to_mx_from_mx_compile_numerics` | 488-534 | torch.compile numerical equivalence |
| `test_to_mx_inductor_single_kernel` | 542-553 | Verify single kernel fusion |

### Advanced Features

| Test | Lines | Purpose |
|------|-------|---------|
| `test_to_blocked_from_blocked_roundtrip` | 645-660 | Scale swizzling roundtrip |
| `test_scale_shape_matches_qdata` | 673-714 | Scale shape validation |
| `test_swizzle` | 728-779 | Swizzled scale layout |

---

## Detailed Execution Traces

---

## test_hello_world Execution Trace

**Test code** ([test_mx_tensor.py:86-89](../../../test/prototype/mx_formats/test_mx_tensor.py#L86)):
```python
def test_hello_world(elem_dtype):
    data = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    block_size = 4
    _test_mx(data, elem_dtype, block_size)
```

This test validates basic MX quantization with a small 8×8 tensor and block_size=4.

---

### 📦 FRAME 1: Test Entry Point

📍 **Source**: [test_mx_tensor.py:55-82](../../../test/prototype/mx_formats/test_mx_tensor.py#L55)

**What happens**: The `_test_mx` helper function orchestrates quantization, dequantization, and validation.

```python
def _test_mx(data_hp, elem_dtype, block_size, scale_calculation_mode=ScaleCalculationMode.FLOOR):
    # 1. Quantize: high precision → MX format
    data_mx = MXTensor.to_mx(data_hp, elem_dtype, block_size, scale_calculation_mode)

    # 2. Dequantize: MX format → high precision
    data_mx_dq = data_mx.dequantize(data_hp.dtype)

    # 3. Validate SQNR (signal-to-quantization-noise ratio)
    assert_sqnr_gt_threshold(data_hp, data_mx_dq, threshold)

    # 4. Validate shapes
    assert data_mx.qdata.shape == (*prev_dims, K)  # or K//2 for FP4
    assert data_mx.scale.shape == (*prev_dims, K // block_size)
```

**Key operations**:
1. Call `MXTensor.to_mx()` for quantization
2. Call `data_mx.dequantize()` for reconstruction
3. Compute SQNR to verify accuracy (≥18 dB for FP8, ≥13 dB for others)
4. Validate tensor shapes

**Next**: → Calls `MXTensor.to_mx()` in Frame 2

---

### 📦 FRAME 2: MXTensor.to_mx() - Public API

📍 **Source**: [mx_tensor.py:594-642](../../../torchao/prototype/mx_formats/mx_tensor.py#L594)

**What happens**: Static method that creates an MXTensor from a high-precision tensor.

```python
@staticmethod
def to_mx(
    x: torch.Tensor,
    elem_dtype: torch.dtype,
    block_size: int,
    scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
    pack_fp6: bool = False,
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.EMULATED,
    is_swizzled_scales: bool = False,
    act_quant_kwargs: Optional[QuantizeTensorToMXKwargs] = None,
) -> "MXTensor":
    # Validate inputs
    assert x.dtype in (torch.float32, torch.bfloat16, torch.float16)

    # Call core quantization function
    qdata, scale = to_mx(
        x,
        elem_dtype,
        block_size,
        scaling_mode=scale_calculation_mode,
        pack_fp6=pack_fp6,
        is_swizzled_scales=is_swizzled_scales,
    )

    # Wrap in MXTensor
    return MXTensor(
        qdata, scale, elem_dtype, block_size, x.dtype,
        gemm_kernel_choice, pack_fp6, act_quant_kwargs, is_swizzled_scales
    )
```

**Key operations**:
1. Input validation (dtype, device)
2. Delegate to functional `to_mx()` for actual quantization
3. Construct and return `MXTensor` wrapper

**Next**: → Calls functional `to_mx()` in Frame 3

---

### 📦 FRAME 3: to_mx() - Core Quantization Logic

📍 **Source**: [mx_tensor.py:146-347](../../../torchao/prototype/mx_formats/mx_tensor.py#L146)

**What happens**: The core quantization algorithm implementing the MX specification.

```python
def to_mx(
    x_hp: torch.Tensor,
    elem_dtype: torch.dtype,
    block_size: int,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
    pack_fp6: bool = False,
    is_swizzled_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert high precision tensor to MX format.

    Returns:
        (qdata, scale) - Quantized data and E8M0 scales
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 1: Reshape input into blocks
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    orig_shape = x_hp.shape
    prev_dims, K = orig_shape[:-1], orig_shape[-1]

    # Reshape to [..., num_blocks, block_size]
    x_hp_blocked = x_hp.view(*prev_dims, K // block_size, block_size)
    # Shape: (8, 2, 4) for our 8×8 input with block_size=4
```

**Detailed step-by-step execution** for our test case (8×8 tensor, block_size=4):

#### Step 1: Reshape Input
```python
# Input: (8, 8) tensor
x_hp_blocked = x_hp.view(8, 2, 4)  # (M, num_blocks, block_size)
# Result: 8 rows × 2 blocks × 4 elements each
```

#### Step 2: Compute Block-wise Maximum Absolute Values
```python
# For each block, find max |value|
amax = torch.amax(torch.abs(x_hp_blocked), dim=-1, keepdim=True)
# Shape: (8, 2, 1) - one amax per block
```

**Example** for one row with values `[0.5, -1.2, 0.3, 0.8, -0.6, 2.1, -0.4, 0.9]`:
```
Block 0: [0.5, -1.2, 0.3, 0.8]  → amax = 1.2
Block 1: [-0.6, 2.1, -0.4, 0.9] → amax = 2.1
```

#### Step 3: Calculate E8M0 Scales

🎯 **Key Point**: This is where FLOOR vs RCEIL modes differ!

**FLOOR mode** (used in our test):
```python
# Lines 219-274
if scaling_mode == ScaleCalculationMode.FLOOR:
    # Compute floor(log2(amax / elem_max))
    # This gives us the shared exponent for the block

    # 1. Normalize by element dtype max value
    #    For FP8 E4M3: elem_max = 448.0
    amax_clipped = torch.clamp(amax, min=2**-127, max=2**127)

    # 2. Extract exponent from float representation
    amax_bits = amax_clipped.view(torch.float32).view(torch.int32)
    exponent = (amax_bits >> 23) & 0xFF  # Extract biased exponent

    # 3. Adjust for element dtype range
    elem_max = torch.finfo(elem_dtype).max
    elem_max_exp = math.floor(math.log2(elem_max))

    # 4. Compute shared exponent (E8M0)
    shared_exp = exponent - 127 - elem_max_exp  # Unbias and adjust

    # 5. Clamp to E8M0 range [0, 254]
    shared_exp_clamped = torch.clamp(shared_exp, 0, 254)

    # 6. Convert to E8M0 format
    scale_e8m0 = shared_exp_clamped.to(torch.uint8).view(torch.float8_e8m0fnu)
    # Shape: (8, 2, 1) - one scale per block
```

**Example calculation**:
```
Block with amax = 2.1:
  1. amax_clipped = 2.1
  2. float32 bits = 0x40066666
     exponent bits = 0x80 = 128 (biased)
  3. elem_max = 448.0 (FP8 E4M3)
     elem_max_exp = floor(log2(448)) = 8
  4. shared_exp = 128 - 127 - 8 = -7
     But negative! So this means scale UP
  5. After adjustments: scale_e8m0 = 120 (in E8M0 encoding)
  6. Actual scale value = 2^(120-127) = 2^(-7) ≈ 0.0078125
```

#### Step 4: Apply Scaling and Quantize

```python
# Lines 293-334

# 1. Broadcast scale to all elements in block
scale_broadcasted = scale_e8m0.view(*prev_dims, K // block_size, 1)

# 2. Convert E8M0 to high precision for division
scale_hp = scale_broadcasted.to(x_hp.dtype)  # E8M0 → BF16

# 3. Normalize input by scale
x_normalized = x_hp_blocked / scale_hp
# Each element divided by its block's scale

# 4. Quantize to element dtype
if elem_dtype == torch.float8_e4m3fn:
    x_quantized = x_normalized.to(torch.float8_e4m3fn)
elif elem_dtype == torch.float4_e2m1fn_x2:
    # FP4 requires special handling
    x_quantized = f32_to_f4_unpacked(x_normalized)
    x_quantized = pack_uint4(x_quantized)  # 2 values per byte
elif elem_dtype in (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2):
    # FP6 conversion
    x_quantized = f32_to_f6_e2m3_unpacked(x_normalized)
    if pack_fp6:
        x_quantized = pack_uint6(x_quantized)  # 4 values per 3 bytes
else:
    raise AssertionError("Unsupported dtype")
```

**Example for FP8 E4M3**:
```
Original values in block: [0.5, -1.2, 0.3, 0.8]
Scale: 2^(-7) ≈ 0.0078125
Normalized: [64, -153.6, 38.4, 102.4]
Quantized (FP8 E4M3): [64, -152, 38, 102] (after rounding)
```

#### Step 5: Reshape Back
```python
# Flatten blocks back to original shape
x_quantized = x_quantized.view(*prev_dims, K)  # (8, 8)
scale_e8m0 = scale_e8m0.view(*prev_dims, K // block_size)  # (8, 2)
```

#### Step 6: Optional Scale Swizzling
```python
# Lines 339-346
if is_swizzled_scales:
    # Convert scale layout for CUTLASS/CUBLAS
    scale_e8m0 = to_blocked(scale_e8m0, use_triton_kernel=True)
    # 128×4 scale blocks → 32×16 interleaved tiles
```

**Key operations summary**:
1. Reshape into blocks: `(M, K)` → `(M, K//B, B)`
2. Compute block-wise amax: `torch.amax(abs(x), dim=-1)`
3. Calculate E8M0 scales based on scaling mode (FLOOR/RCEIL)
4. Normalize by scale: `x / scale`
5. Cast to target dtype: FP8, FP6, or FP4
6. Pack if needed (FP4: 2 per byte, FP6: 4 per 3 bytes)
7. Optionally swizzle scales for accelerator layouts

**Next**: Returns to Frame 2 → MXTensor constructed

---

### 📦 FRAME 4: FP8 Quantization (if elem_dtype is FP8)

📍 **Source**: PyTorch internals

**What happens**: The `.to(torch.float8_e4m3fn)` cast triggers PyTorch's native FP8 conversion.

**In eager mode**:
```cpp
// aten/src/ATen/native/cuda/Float8Conversion.cu
__device__ float8_e4m3fn cast_to_f8_e4m3fn(float value) {
    // Unsaturated cast: values outside range → NaN
    if (abs(value) > FP8_E4M3_MAX) {
        return NaN;
    }
    // Round to nearest even
    return quantize_fp8_e4m3(value);
}
```

**In compiled mode** (torch.compile with Triton backend):
```python
# torch/_inductor/codegen/triton.py generates:
@triton.jit
def triton_kernel(...):
    x_f32 = tl.load(...)
    # Saturated cast in Triton!
    x_f8 = x_f32.to(tl.float8e4m3fn)  # Clamps instead of NaN
    tl.store(..., x_f8)
```

⚠️ **Note**: This behavioral difference is tested in `test_cast_to_float8_e4m3fn_saturation_behavior`!

**Next**: Returns quantized data to Frame 3

---

### 📦 FRAME 5: FP4 Quantization (if elem_dtype is FP4)

📍 **Source**: [kernels.py:68-101](../../../torchao/prototype/mx_formats/kernels.py#L68)

**What happens**: Custom FP4 quantization since PyTorch lacks native FP4 support.

```python
def f32_to_f4_unpacked(x: torch.Tensor) -> torch.Tensor:
    """
    Convert float32 to FP4 E2M1 (unpacked as uint8).

    FP4 E2M1 format:
    - 1 sign bit
    - 2 exponent bits (bias = 1)
    - 1 mantissa bit
    - Values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} and negatives
    """
    return _f32_to_floatx_unpacked(
        x,
        ebits=2,  # Exponent bits
        mbits=1,  # Mantissa bits
        exp_bias=1,
        max_norm=6.0,
        device=x.device,
    )
```

**Core conversion logic** ([kernels.py:530-633](../../../torchao/prototype/mx_formats/kernels.py#L530)):

```python
def _f32_to_floatx_unpacked(x, ebits, mbits, exp_bias, max_norm, device):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 1: Extract sign, exponent, mantissa from FP32
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    x_bits = x.view(torch.float32).view(torch.int32)
    sign = (x_bits >> 31) & 1
    exp_fp32 = (x_bits >> 23) & 0xFF  # FP32 exponent (biased by 127)
    mantissa_fp32 = x_bits & 0x7FFFFF  # 23-bit mantissa

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 2: Handle special cases
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    is_zero = (x_bits & 0x7FFFFFFF) == 0  # +0 or -0
    is_nan = (exp_fp32 == 0xFF) & (mantissa_fp32 != 0)
    is_inf = (exp_fp32 == 0xFF) & (mantissa_fp32 == 0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 3: Compute target exponent for FP4
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    exp_fp4_biased = exp_fp32 - 127 + exp_bias  # Re-bias for FP4

    # Clamp to FP4 range: [0, 2^ebits - 1]
    exp_fp4_clamped = torch.clamp(exp_fp4_biased, 0, (1 << ebits) - 1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 4: Round mantissa to target precision
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FP32 has 23 mantissa bits, FP4 has 1
    # Need to round 23 → 1 bits

    mantissa_shift = 23 - mbits  # 22 bits for FP4
    mantissa_fp4 = mantissa_fp32 >> mantissa_shift

    # Tie-to-even rounding
    # If discarded bits are exactly 0.5 (GRS = 100), round to even
    remainder = mantissa_fp32 & ((1 << mantissa_shift) - 1)
    halfway = 1 << (mantissa_shift - 1)

    # Round up if: remainder > halfway, or (remainder == halfway and result is odd)
    round_up = (remainder > halfway) | (
        (remainder == halfway) & ((mantissa_fp4 & 1) == 1)
    )
    mantissa_fp4 = torch.where(round_up, mantissa_fp4 + 1, mantissa_fp4)

    # Handle mantissa overflow (carry into exponent)
    mantissa_overflow = mantissa_fp4 >> mbits
    exp_fp4_final = torch.where(mantissa_overflow, exp_fp4_clamped + 1, exp_fp4_clamped)
    mantissa_fp4_final = torch.where(mantissa_overflow, 0, mantissa_fp4)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5: Assemble FP4 bits
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fp4_bits = (sign << (ebits + mbits)) | (exp_fp4_final << mbits) | mantissa_fp4_final

    # Handle special cases
    fp4_bits = torch.where(is_zero, 0, fp4_bits)
    fp4_bits = torch.where(is_nan, (1 << ebits) - 1, fp4_bits)  # NaN encoding
    fp4_bits = torch.where(is_inf, (1 << (ebits + mbits)) - 1, fp4_bits)

    return fp4_bits.to(torch.uint8)
```

**Example trace** for value `1.2`:

```
Input: 1.2 (FP32)
  Bits: 0x3F99999A
  Sign: 0
  Exp:  0x7F = 127 (unbiased: 0)
  Mant: 0x19999A

Step 1: Re-bias exponent for FP4 (bias=1)
  exp_fp4 = 127 - 127 + 1 = 1

Step 2: Round mantissa 23 bits → 1 bit
  mantissa_fp32 = 0x19999A = 0b000110011001100110011010
  Shift right 22 bits: 0b00 = 0
  Remainder = 0x19999A (discarded bits)
  Halfway = 2^21 = 0x200000
  remainder (0x19999A) < halfway → round down
  mantissa_fp4 = 0

Step 3: Assemble FP4
  fp4_bits = (0 << 3) | (1 << 1) | 0 = 0b0010 = 2

Step 4: Interpret as FP4
  Sign: 0 (positive)
  Exp:  01 (biased) → unbias: 1-1 = 0
  Mant: 0 → actual: 1.0 (implicit leading bit)
  Value: (+1) × 2^0 × 1.0 = 1.0

Result: 1.2 → 1.0 (quantization error: 0.2)
```

**Next**: Returns uint8 tensor to Frame 3

---

### 📦 FRAME 6: FP4 Packing

📍 **Source**: [kernels.py:723-764](../../../torchao/prototype/mx_formats/kernels.py#L723)

**What happens**: Pack two 4-bit values into one uint8 byte.

```python
def pack_uint4(x: torch.Tensor) -> torch.Tensor:
    """
    Pack FP4 values (stored as uint8) into compact format.

    Input:  [val0, val1, val2, val3, ...] where each val ∈ [0, 15]
    Output: [val1:val0, val3:val2, ...] where : means bit concatenation

    Layout: uint8[7:4] = val1, uint8[3:0] = val0
    """
    assert x.dtype == torch.uint8

    # Reshape: (N,) → (N//2, 2)
    x = x.contiguous().view(-1, 2)

    # Pack: [val0, val1] → (val1 << 4) | val0
    packed = (x[:, 1] << 4) | x[:, 0]

    return packed
```

**Example**:
```
Input:  [0x0, 0x2, 0x5, 0x7]  # 4 FP4 values
Reshape: [[0x0, 0x2], [0x5, 0x7]]
Pack:    [(0x2 << 4) | 0x0, (0x7 << 4) | 0x5]
       = [0x20, 0x75]
Output: [0x20, 0x75]  # 2 bytes instead of 4
```

**Next**: Returns packed tensor to Frame 3

---

### 📦 FRAME 7: Dequantization - data_mx.dequantize()

📍 **Source**: [mx_tensor.py:565-591](../../../torchao/prototype/mx_formats/mx_tensor.py#L565)

**What happens**: Convert MX format back to high precision.

```python
def dequantize(self, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """
    Dequantize MXTensor to high precision.

    Returns:
        torch.Tensor in specified dtype
    """
    # Unswizzle scales if needed
    if self._is_swizzled_scales:
        scale_unswizzled = from_blocked(
            self.scale,
            math.prod(self.shape[:-1]),
            self.shape[-1] // self._block_size
        )
    else:
        scale_unswizzled = self.scale

    # Call functional dequantization
    return to_dtype(
        self.qdata,
        scale_unswizzled,
        self._elem_dtype,
        self._block_size,
        dtype,
        self._pack_fp6,
    )
```

**Next**: → Calls functional `to_dtype()` in Frame 8

---

### 📦 FRAME 8: to_dtype() - Dequantization Logic

📍 **Source**: [mx_tensor.py:60-143](../../../torchao/prototype/mx_formats/mx_tensor.py#L60)

**What happens**: Core dequantization implementing the inverse of quantization.

```python
def to_dtype(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    elem_dtype: torch.dtype,
    block_size: int,
    orig_dtype: torch.dtype,
    pack_fp6: bool = False,
) -> torch.Tensor:
    """
    Convert MX format tensors back to high precision.

    Args:
        qdata: Quantized data (FP8/FP6/FP4)
        scale: E8M0 scales
        elem_dtype: Element data type
        block_size: Block size
        orig_dtype: Target dtype
        pack_fp6: Whether FP6 data is packed
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 1: Unpack data if needed
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if elem_dtype == torch.float4_e2m1fn_x2:
        # FP4: Unpack 2 values per byte
        qdata_unpacked = unpack_uint4(qdata)  # uint8 → 2× uint8
    elif elem_dtype in (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2) and pack_fp6:
        # FP6: Unpack 4 values per 3 bytes
        qdata_unpacked = unpack_uint6(qdata)  # 3× uint8 → 4× uint8
    else:
        qdata_unpacked = qdata

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 2: Convert quantized data to high precision
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # Native FP8: Direct cast
        data_hp = qdata_unpacked.to(orig_dtype)
    elif elem_dtype == torch.float4_e2m1fn_x2:
        # FP4: Custom conversion
        data_hp = f4_unpacked_to_f32(qdata_unpacked).to(orig_dtype)
    elif elem_dtype == DTYPE_FP6_E2M3:
        data_hp = f6_e2m3_unpacked_to_f32(qdata_unpacked).to(orig_dtype)
    elif elem_dtype == DTYPE_FP6_E3M2:
        data_hp = f6_e3m2_unpacked_to_f32(qdata_unpacked).to(orig_dtype)
    else:
        raise AssertionError("Unsupported elem_dtype")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 3: Reshape to blocks
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    orig_shape = data_hp.shape
    prev_dims, K = orig_shape[:-1], orig_shape[-1]

    data_hp_blocked = data_hp.view(*prev_dims, K // block_size, block_size)
    # Example: (8, 8) → (8, 2, 4)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 4: Broadcast and apply scales
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    scale_broadcasted = scale.view(*prev_dims, K // block_size, 1)
    # Example: (8, 2) → (8, 2, 1)

    # Convert E8M0 scale to high precision
    scale_hp = scale_broadcasted.to(orig_dtype)

    # Multiply by scale to recover original magnitude
    data_hp_scaled = data_hp_blocked * scale_hp

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5: Reshape back to original shape
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    data_reconstructed = data_hp_scaled.view(*orig_shape)
    # Example: (8, 2, 4) → (8, 8)

    return data_reconstructed
```

**Example reconstruction** for one block:

```
Quantized values: [64, -152, 38, 102] (FP8 E4M3)
Scale: 2^(-7) ≈ 0.0078125 (E8M0)

Step 1: Cast to high precision
  [64.0, -152.0, 38.0, 102.0] (BF16)

Step 2: Multiply by scale
  [64.0 × 0.0078125, -152.0 × 0.0078125, 38.0 × 0.0078125, 102.0 × 0.0078125]
= [0.5, -1.1875, 0.296875, 0.796875]

Original values (for comparison):
  [0.5, -1.2, 0.3, 0.8]

Reconstruction error:
  [0.0, 0.0125, -0.003125, -0.003125]
```

**Next**: Returns reconstructed tensor to Frame 7 → Frame 1

---

### 📦 FRAME 9: FP4 Dequantization

📍 **Source**: [kernels.py:103-133](../../../torchao/prototype/mx_formats/kernels.py#L103)

**What happens**: Convert FP4 bits back to float32.

```python
def f4_unpacked_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Convert FP4 E2M1 (unpacked as uint8) to float32."""
    return _floatx_unpacked_to_f32(
        x,
        ebits=2,
        mbits=1,
        exp_bias=1,
    )
```

**Core conversion** ([kernels.py:635-720](../../../torchao/prototype/mx_formats/kernels.py#L635)):

```python
def _floatx_unpacked_to_f32(x, ebits, mbits, exp_bias):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 1: Extract sign, exponent, mantissa from FP4 bits
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    x_int = x.to(torch.int32)

    sign = (x_int >> (ebits + mbits)) & 1  # Top bit
    exp_floatx = (x_int >> mbits) & ((1 << ebits) - 1)  # Next 2 bits
    mantissa_floatx = x_int & ((1 << mbits) - 1)  # Bottom bit

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 2: Handle zero
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    is_zero = (exp_floatx == 0) & (mantissa_floatx == 0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 3: Handle denormals (exp == 0, mantissa != 0)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    is_denorm = (exp_floatx == 0) & (mantissa_floatx != 0)

    # For denormals: value = 2^(1-bias) × 0.mantissa
    denorm_exp_fp32 = 1 - exp_bias + 127  # FP32 biased exponent

    # Shift mantissa to FP32 position (bit 22 for FP4's 1-bit mantissa)
    denorm_mantissa_fp32 = mantissa_floatx << (23 - mbits)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 4: Handle normals (exp != 0)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # For normals: value = 2^(exp-bias) × 1.mantissa
    normal_exp_fp32 = exp_floatx - exp_bias + 127  # Re-bias for FP32
    normal_mantissa_fp32 = mantissa_floatx << (23 - mbits)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 5: Assemble FP32 bits
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    exp_fp32 = torch.where(is_denorm, denorm_exp_fp32, normal_exp_fp32)
    mantissa_fp32 = torch.where(is_denorm, denorm_mantissa_fp32, normal_mantissa_fp32)

    fp32_bits = (sign << 31) | (exp_fp32 << 23) | mantissa_fp32
    fp32_bits = torch.where(is_zero, 0, fp32_bits)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Step 6: Reinterpret as float32
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    return fp32_bits.view(torch.float32)
```

**Example**: Dequantize FP4 value `0b0010` (decimal 2):

```
Input: 0b0010 = 2 (uint8)

Step 1: Extract components
  sign = (2 >> 3) & 1 = 0 (positive)
  exp  = (2 >> 1) & 0b11 = 1
  mant = 2 & 1 = 0

Step 2: Check if zero
  is_zero = (exp==0 && mant==0) = False

Step 3: Check if denormal
  is_denorm = (exp==0 && mant!=0) = False

Step 4: Normal number conversion
  exp_fp32 = 1 - 1 + 127 = 127
  mantissa_fp32 = 0 << 22 = 0

Step 5: Assemble FP32
  fp32_bits = (0 << 31) | (127 << 23) | 0
            = 0x3F800000

Step 6: Reinterpret
  0x3F800000 as float32 = 1.0

Result: FP4 value 2 → 1.0
```

**Next**: Returns float32 tensor to Frame 8

---

## torch.compile Integration

### test_to_mx_from_mx_compile_numerics Execution Trace

**Test code** ([test_mx_tensor.py:488-534](../../../test/prototype/mx_formats/test_mx_tensor.py#L488)):
```python
def test_to_mx_from_mx_compile_numerics(elem_dtype, hp_dtype, all_zeros):
    x = torch.randn(4, 8, dtype=hp_dtype, device="cuda")
    block_size = 4

    # Compile the quantization function
    to_mx_c = torch.compile(MXTensor.to_mx, fullgraph=True)

    # Compare eager vs compiled
    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_mx_c = to_mx_c(x, elem_dtype, block_size)

    # Validate numerics match exactly
    torch.testing.assert_close(x_mx.scale, x_mx_c.scale, atol=0, rtol=0)
    torch.testing.assert_close(x_mx.qdata, x_mx_c.qdata, atol=0, rtol=0)
```

---

### 📦 FRAME 10: torch.compile Entry

📍 **Source**: PyTorch compiler

**What happens**: `torch.compile()` traces the function and generates optimized code.

```python
# When to_mx_c(x, ...) is called:

# 1. Dynamo captures the Python bytecode
torch._dynamo.eval_frame.compile_function(MXTensor.to_mx)

# 2. Traces execution to build FX graph
fx_graph = torch._dynamo.symbolic_convert(MXTensor.to_mx, args, kwargs)

# 3. Inductor generates Triton kernel code
triton_code = torch._inductor.compile_fx(fx_graph)

# 4. JIT compiles Triton kernel
compiled_kernel = triton.compile(triton_code)
```

---

### 📦 FRAME 11: Inductor Code Generation

📍 **Source**: torch/_inductor/

**What happens**: Inductor analyzes the FX graph and generates fused Triton kernel.

**FX Graph** (simplified):
```python
graph():
    %x : torch.Tensor = placeholder[target=x]
    %elem_dtype : torch.dtype = placeholder[target=elem_dtype]
    %block_size : int = placeholder[target=block_size]

    # Reshape to blocks
    %view_1 : Tensor = call_function[target=torch.ops.aten.view](
        args=(%x, [4, 2, 4])
    )

    # Compute amax
    %abs_1 : Tensor = call_function[target=torch.ops.aten.abs](%view_1)
    %amax_1 : Tensor = call_function[target=torch.ops.aten.amax](
        args=(%abs_1,),
        kwargs={dim: -1, keepdim: True}
    )

    # Calculate scale (bit manipulation ops)
    %scale_e8m0 : Tensor = call_function[target=...calculate_scale...](%amax_1)

    # Normalize and cast
    %div_1 : Tensor = call_function[target=torch.ops.aten.div](%view_1, %scale_e8m0)
    %to_1 : Tensor = call_function[target=torch.ops.aten._to_copy](
        args=(%div_1,),
        kwargs={dtype: torch.float8_e4m3fn}
    )

    return (%to_1, %scale_e8m0)
```

**Inductor fusion** (if possible):

```python
# Inductor tries to fuse operations into a single Triton kernel
@triton.jit
def fused_to_mx_kernel(
    x_ptr, scale_ptr, out_ptr,
    M, K, block_size,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Load input block
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :])

    # Compute max per block (reduction)
    x_abs = tl.abs(x)
    x_max = tl.max(x_abs, axis=1)  # Reduce over K dimension

    # Calculate E8M0 scale
    scale = compute_e8m0_scale(x_max)
    tl.store(scale_ptr + pid_m, scale)

    # Normalize and cast to FP8
    x_normalized = x / scale[:, None]
    x_fp8 = x_normalized.to(tl.float8e4m3fn)  # Saturated cast in Triton

    # Store result
    tl.store(out_ptr + offs_m[:, None] * K + offs_k[None, :], x_fp8)
```

🎯 **Key optimization**: Inductor fuses reshape, abs, amax, scale computation, division, and FP8 cast into a single kernel!

---

### 📦 FRAME 12: test_to_mx_inductor_single_kernel

**Test code** ([test_mx_tensor.py:542-553](../../../test/prototype/mx_formats/test_mx_tensor.py#L542)):
```python
def test_to_mx_inductor_single_kernel():
    x = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
    block_size = 32
    to_mx_c = torch.compile(MXTensor.to_mx, fullgraph=True)

    # Get generated code
    out, code = run_and_get_code(to_mx_c, x, torch.float8_e4m3fn, block_size)

    # Verify only one kernel was generated (full fusion)
    FileCheck().check("def call(").check_count(".run(", 1, exactly=True).run(code[0])
```

**What it validates**: Confirms that Inductor successfully fuses all operations into a single Triton kernel, achieving optimal performance.

---

## Scale Swizzling Traces

### test_swizzle Execution Trace

**Test code** ([test_mx_tensor.py:728-779](../../../test/prototype/mx_formats/test_mx_tensor.py#L728)):
```python
def test_swizzle(elem_dtype, transpose, shape):
    block_size = 32
    x_hp = torch.randn(*shape, device="cuda")

    # Regular layout
    x = MXTensor.to_mx(x_hp, elem_dtype, block_size, is_swizzled_scales=False)

    # Swizzled layout
    xs = MXTensor.to_mx(x_hp, elem_dtype, block_size, is_swizzled_scales=True)

    # Validate qdata is identical (only scales differ)
    torch.testing.assert_close(x.qdata, xs.qdata, atol=0, rtol=0)

    # Validate scales are equivalent after unswizzling
    xs_scale_unblocked = from_blocked(xs.scale, M, K // block_size)
    torch.testing.assert_close(x.scale, xs_scale_unblocked, atol=0, rtol=0)
```

---

### 📦 FRAME 13: to_blocked() - Scale Swizzling

📍 **Source**: [utils.py:32-83](../../../torchao/prototype/mx_formats/utils.py#L32)

**What happens**: Converts linear scale layout to blocked (swizzled) layout for CUTLASS.

**Why swizzle?** CUTLASS tensor cores expect scales in a specific interleaved layout for efficient memory access.

```python
def to_blocked(
    scales: torch.Tensor,
    use_triton_kernel: bool = False
) -> torch.Tensor:
    """
    Convert linear scale layout to 32×16 blocked layout.

    Input:  (M, K) scales
    Output: (M', K') blocked scales where M'×K' covers same data

    Blocking pattern:
    - 128 rows → 32-row block
    - 4 cols  → 16-col block
    """
    M, K = scales.shape

    # Constants for CUTLASS layout
    ROWS_PER_BLOCK = 128
    COLS_PER_BLOCK = 4
    BLOCK_HEIGHT = 32
    BLOCK_WIDTH = 16

    # Pad to block boundaries
    M_padded = ceil_div(M, ROWS_PER_BLOCK) * ROWS_PER_BLOCK
    K_padded = ceil_div(K, COLS_PER_BLOCK) * COLS_PER_BLOCK

    if M_padded != M or K_padded != K:
        scales_padded = torch.zeros(M_padded, K_padded, dtype=scales.dtype, device=scales.device)
        scales_padded[:M, :K] = scales
    else:
        scales_padded = scales

    if use_triton_kernel:
        return triton_mx_block_rearrange(scales_padded)
    else:
        return to_blocked_pytorch(scales_padded)
```

**PyTorch implementation** ([utils.py:86-161](../../../torchao/prototype/mx_formats/utils.py#L86)):

```python
def to_blocked_pytorch(scales: torch.Tensor) -> torch.Tensor:
    M, K = scales.shape

    # Step 1: Reshape into 128×4 tiles
    num_tiles_m = M // 128
    num_tiles_k = K // 4

    scales_tiled = scales.reshape(num_tiles_m, 128, num_tiles_k, 4)
    # Shape: (num_tiles_m, 128, num_tiles_k, 4)

    # Step 2: Subdivide each 128×4 tile into 4 subtiles of 32×1
    scales_subtiled = scales_tiled.reshape(
        num_tiles_m, 4, 32, num_tiles_k, 4, 1
    )
    # Shape: (num_tiles_m, 4, 32, num_tiles_k, 4, 1)

    # Step 3: Interleave subtiles
    # Reorder dimensions to create interleaved layout
    scales_interleaved = scales_subtiled.permute(0, 3, 1, 4, 2, 5)
    # Shape: (num_tiles_m, num_tiles_k, 4, 4, 32, 1)

    # Step 4: Reshape to final 32×16 block layout
    scales_blocked = scales_interleaved.reshape(
        num_tiles_m, num_tiles_k, 32, 16
    ).reshape(num_tiles_m * 32, num_tiles_k * 16)
    # Shape: (M/4, K*4)

    return scales_blocked
```

**Visual example** for 128×4 input:

```
Input (128×4):
┌─────────────────────────┐
│  0  1  2  3            │  ← Row 0
│  4  5  6  7            │  ← Row 1
│  ...                    │
│  508 509 510 511        │  ← Row 127
└─────────────────────────┘

After to_blocked (32×16):
┌─────────────────────────────────────────────────────┐
│  0   4   8  12  | 128 132 136 140 | 256 260 ... │  ← Rows 0-31
│  1   5   9  13  | 129 133 137 141 | 257 261 ... │    interleaved
│  2   6  10  14  | 130 134 138 142 | 258 262 ... │    from input
│  3   7  11  15  | 131 135 139 143 | 259 263 ... │    rows 0-127
│  ...                                             │
│  124 ... (16 columns total)                      │  ← Row 31
└─────────────────────────────────────────────────────┘
```

🎯 **Key insight**: This layout enables coalesced memory access in CUTLASS kernels by grouping scales that will be accessed together.

---

### 📦 FRAME 14: Triton Scale Swizzling Kernel

📍 **Source**: [kernels.py:1434-1489](../../../torchao/prototype/mx_formats/kernels.py#L1434)

**What happens**: GPU-accelerated scale swizzling using Triton.

```python
@triton.jit
def triton_mx_block_rearrange_kernel(
    input_ptr,
    output_ptr,
    M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Rearrange scale matrix from linear to blocked layout.

    Each program handles one 32×16 output block.
    """
    # Get program ID
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Output block coordinates
    out_row_base = pid_m * BLOCK_M
    out_col_base = pid_k * BLOCK_K

    # Each 32×16 output block comes from:
    # - 128 input rows (ROWS_PER_TILE = 128)
    # - 1 input column (COLS_PER_TILE = 4)

    # Compute input tile coordinates
    tile_m = pid_m // 4  # Which 128-row tile?
    subtile_m = pid_m % 4  # Which 32-row subtile within tile?
    tile_k = pid_k // 4

    # Load from input in interleaved pattern
    for i in range(32):
        for j in range(16):
            # Compute source location
            src_row = tile_m * 128 + subtile_m * 32 + i
            src_col = tile_k * 4 + (j % 4)

            # Load and store
            val = tl.load(input_ptr + src_row * K + src_col)
            out_offset = (out_row_base + i) * (K * 4) + out_col_base + j
            tl.store(output_ptr + out_offset, val)
```

**Performance**: ~10× faster than PyTorch implementation for large tensors.

---

## Summary of Execution Flows

### Quantization Path (High Precision → MX)

```
User: x_mx = MXTensor.to_mx(x_hp, torch.float8_e4m3fn, block_size=32)
  ↓
📦 Frame 1: _test_mx() - Test wrapper
  ↓
📦 Frame 2: MXTensor.to_mx() - Public API
  ↓
📦 Frame 3: to_mx() - Core quantization logic
  ├─ Reshape to blocks: (M, K) → (M, K//B, B)
  ├─ Compute amax per block: torch.amax(abs(x), dim=-1)
  ├─ Calculate E8M0 scales (FLOOR/RCEIL mode)
  ├─ Normalize: x / scale
  ├─ Cast to target dtype
  │  ├─ FP8 → 📦 Frame 4: PyTorch native cast
  │  ├─ FP4 → 📦 Frame 5: Custom f32_to_f4_unpacked()
  │  └─ FP6 → Custom f32_to_f6_unpacked()
  ├─ Pack if needed
  │  └─ 📦 Frame 6: pack_uint4() or pack_uint6()
  └─ Optionally swizzle scales
     └─ 📦 Frame 13-14: to_blocked()
  ↓
Returns: MXTensor(qdata, scale, metadata)
```

### Dequantization Path (MX → High Precision)

```
User: x_hp = x_mx.dequantize(torch.bfloat16)
  ↓
📦 Frame 7: MXTensor.dequantize()
  ├─ Unswizzle scales if needed: from_blocked()
  └─ Delegate to to_dtype()
     ↓
📦 Frame 8: to_dtype() - Core dequantization
  ├─ Unpack data if needed
  ├─ Cast to high precision
  │  ├─ FP8 → PyTorch native cast
  │  ├─ FP4 → 📦 Frame 9: f4_unpacked_to_f32()
  │  └─ FP6 → Custom f6_unpacked_to_f32()
  ├─ Reshape to blocks
  ├─ Broadcast scales
  ├─ Multiply: x * scale
  └─ Reshape back
  ↓
Returns: torch.Tensor (high precision)
```

### Compiled Path (torch.compile)

```
User: to_mx_c = torch.compile(MXTensor.to_mx); x_mx = to_mx_c(x, ...)
  ↓
📦 Frame 10: torch.compile - Dynamo tracing
  ├─ Capture bytecode
  ├─ Build FX graph
  └─ Pass to Inductor
     ↓
📦 Frame 11: Inductor code generation
  ├─ Analyze FX graph
  ├─ Fuse operations
  ├─ Generate Triton kernel
  └─ JIT compile
     ↓
⚡ Optimized Triton kernel execution
  └─ Single fused kernel (validated by Frame 12)
```

---

## Key Takeaways

1. **MX quantization is block-based**: Each block of 32 (or configurable) elements shares one E8M0 exponent scale.

2. **Two scale calculation modes**:
   - **FLOOR** (OCP spec): `floor(log2(amax / elem_max))`
   - **RCEIL** (NVIDIA): Ceiling-based, handles denormals differently

3. **Multiple precisions supported**: FP8 (native), FP6 (custom), FP4 (custom)

4. **Packing for efficiency**: FP4 packs 2 per byte, FP6 packs 4 per 3 bytes

5. **Scale layouts**: Linear (default) vs swizzled (for CUTLASS/CUBLAS)

6. **torch.compile friendly**: Fuses into single kernel for maximum performance

7. **Saturated vs unsaturated casts**: Eager FP8 cast produces NaN on overflow, compiled Triton saturates

8. **Numerical accuracy**: FP8 achieves ≥18 dB SQNR, FP4/FP6 achieve ≥13 dB SQNR

This implementation provides a complete, hardware-accelerated path from Python API to GPU execution!
