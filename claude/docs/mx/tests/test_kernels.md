# test_kernels.py - Execution Trace Walkthrough

## Overview

[test_kernels.py](../../../test/prototype/mx_formats/test_kernels.py) validates the **low-level kernel implementations** for MX format operations. This includes:

- **FP format specifications** (FP4, FP6, FP8)
- **Conversion kernels** (FP32 ↔ FPX)
- **Packing/unpacking** kernels
- **Triton quantization kernels** (MXFP8, NVFP4)
- **CUDA kernels** (MXFP8 via C++ extension)
- **Scale swizzling** (blocked layout transforms)

These tests validate **numerical correctness** and **implementation equivalence** between different backends.

## Test Categories

### 1. FP Format Specification Tests

| Test | Lines | Purpose |
|------|-------|---------|
| `test_fp32` | 72-76 | FP32 bit encoding (skipped) |
| `test_bf16` | 82-86 | BF16 bit encoding (skipped) |
| `test_fp16` | 89-93 | FP16 bit encoding |
| `test_float8_e4m3fn` | 96-100 | FP8 E4M3 encoding |
| `test_float8_e5m2` | 103-107 | FP8 E5M2 encoding |
| `test_float4_e2m1_table` | 133-148 | FP4 E2M1 reference table |
| `test_float6_e3m2_table` | 151-166 | FP6 E3M2 reference table |
| `test_float6_e2m3_table` | 169-184 | FP6 E2M3 reference table |

### 2. FP4 Conversion Tests

| Test | Lines | Purpose |
|------|-------|---------|
| `test_fp4_0_0` through `test_fp4_6_0` | 242-318 | All FP4 encodings + tie-to-even rounding |
| `test_fp4_pack_unpack` | 321-348 | Pack 2 FP4 per byte, unpack, verify |
| `test_fp4_values` | 352-393 | All 64 FP6 values (E2M3 and E3M2) |

### 3. FP6 Conversion Tests

| Test | Lines | Purpose |
|------|-------|---------|
| `test_fp6_values` | 352-393 | All 64 FP6 values (E2M3 and E3M2) |
| `test_fp6_e3m2_rounding` | 418-423 | Rounding modes for FP6 E3M2 |
| `test_fp6_e2m3_pack_unpack` | 428-438 | Pack 4 FP6 per 3 bytes (E2M3) |
| `test_fp6_e3m2_pack_unpack` | 443-453 | Pack 4 FP6 per 3 bytes (E3M2) |

### 4. Triton MXFP8 Kernels

| Test | Lines | Purpose |
|------|-------|---------|
| `test_triton_mxfp8_dim1_randn` | 480-485 | Column-wise quantization (Triton) |
| `test_triton_mxfp8_dim0_randn` | 495-500 | Row-wise quantization (Triton) |
| `test_triton_mxfp8_dim0_zeros` | 508-514 | Edge case: all-zero input |
| `test_triton_mxfp8_dequant_dim0` | 525-537 | Dequantization kernel |

### 5. CUDA MXFP8 Kernels

| Test | Lines | Purpose |
|------|-------|---------|
| `test_cuda_mx_dim1_numerics` | 571-606 | C++ extension column-wise quantization |
| `test_cuda_mx_dim0_not_supported` | 613-630 | Verify dim0 raises error |
| `test_cuda_mx_dim1_invalid_block_size` | 637-654 | Invalid block size error |

### 6. Scale Swizzling

| Test | Lines | Purpose |
|------|-------|---------|
| `test_rearrange` | 554-558 | Eager vs Triton scale swizzling |

---

## Detailed Execution Traces

---

## FP4 Conversion Traces

### test_fp4_1_0 Execution Trace

**Test code** ([test_kernels.py:261-269](../../../test/prototype/mx_formats/test_kernels.py#L261)):
```python
def test_fp4_1_0():
    cases = [
        (1.25, 1.0, "010"),  # tie to even
        (1.1, 1.0, "010"),
        (1.0, 1.0, "010"),
        (0.9, 1.0, "010"),
        (0.75, 1.0, "010"),  # tie to even
    ]
    _test_fp4_cases(cases)
```

This tests **quantization to FP4 value 1.0**, including tie-to-even rounding cases.

---

### 📦 FRAME 1: _test_fp4_case() - Test Harness

📍 **Source**: [test_kernels.py:199-209](../../../test/prototype/mx_formats/test_kernels.py#L199)

**What happens**: Tests FP32 → FP4 → FP32 roundtrip for specific value.

```python
def _test_fp4_case(f32_val, f32_val_ref, f4_enc_ref):
    """
    Test FP4 conversion for a single value.

    Args:
        f32_val: Input FP32 value
        f32_val_ref: Expected FP4-quantized value (as FP32)
        f4_enc_ref: Expected FP4 bit encoding (as string)
    """

    # Step 1: FP32 → FP4
    f4_unpacked = f32_to_f4_unpacked(torch.tensor(f32_val))

    # Step 2: Verify bit encoding
    s_enc, e_enc, m_enc = get_sem_bits(f4_unpacked, bitwidth=4)
    assert s_enc + e_enc + m_enc == f4_enc_ref

    # Step 3: FP4 → FP32
    f32_dequantized = f4_unpacked_to_f32(f4_unpacked)
    assert f32_val_ref == f32_dequantized.item()
```

**Example trace** for input `1.25`:

---

### 📦 FRAME 2: f32_to_f4_unpacked() - Quantization

📍 **Source**: [kernels.py:68-73](../../../torchao/prototype/mx_formats/kernels.py#L68)

```python
def f32_to_f4_unpacked(x: torch.Tensor) -> torch.Tensor:
    """Convert FP32 to FP4 E2M1 (unpacked as uint8)."""
    return _f32_to_floatx_unpacked(
        x,
        ebits=2,        # Exponent bits
        mbits=1,        # Mantissa bits
        exp_bias=1,     # Bias
        max_norm=6.0,   # FP4 max value
        device=x.device,
    )
```

---

### 📦 FRAME 3: _f32_to_floatx_unpacked() - Core Conversion Logic

📍 **Source**: [kernels.py:530-633](../../../torchao/prototype/mx_formats/kernels.py#L530)

**What happens**: Implements FP32 → FPX conversion with tie-to-even rounding.

**Detailed trace for input `1.25`**:

```python
# Input: 1.25 (FP32)
x = torch.tensor(1.25)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Extract FP32 components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

x_bits = x.view(torch.int32)
# 1.25 in FP32: 0x3FA00000
#   Sign: 0 (bit 31)
#   Exp:  0x7F = 127 (bits 30-23) → unbiased: 0
#   Mant: 0x200000 (bits 22-0) → 0.25 fractional

sign = (x_bits >> 31) & 1  # = 0
exp_fp32 = (x_bits >> 23) & 0xFF  # = 127
mantissa_fp32 = x_bits & 0x7FFFFF  # = 0x200000

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Re-bias exponent for FP4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FP32 value: 1.25 = 2^0 × 1.25
# Unbiased FP32 exp: 127 - 127 = 0

# FP4 E2M1 has:
#   - 2 exponent bits → range [0, 3]
#   - bias = 1
#   - Representable exponents: -1 to 2

# Convert: FP32 unbiased exp → FP4 biased exp
exp_fp4 = (127 - 127) + 1  # = 1
# This is within FP4 range [0, 3] ✓

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Round mantissa (23 bits → 1 bit)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FP32 mantissa: 0x200000 = 0b001000000000000000000000
#                              ^
#                              23 bits

# FP4 needs only 1 mantissa bit
# Shift right by (23 - 1) = 22 bits
mantissa_fp4 = mantissa_fp32 >> 22
# = 0x200000 >> 22 = 0b0010... >> 22 = 0b00 = 0

# But wait! Need to check rounding
# Discarded bits: mantissa_fp32 & ((1 << 22) - 1)
#               = 0x200000 & 0x3FFFFF
#               = 0x200000

# This is exactly halfway (0x200000 = 2^21 = half of 2^22)
# Tie-to-even rule: Round to nearest even

# Current mantissa_fp4 = 0 (even)
# Halfway case → round down (keep 0) ✓

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Assemble FP4 bits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fp4_bits = (sign << 3) | (exp_fp4 << 1) | mantissa_fp4
# = (0 << 3) | (1 << 1) | 0
# = 0b0000 | 0b0010 | 0b0000
# = 0b0010
# = 2 (decimal)

# Binary breakdown:
#   Bit 3: sign = 0 (positive)
#   Bits 2-1: exp = 01 (biased exponent 1)
#   Bit 0: mant = 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Interpret FP4 value
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FP4 encoding: 0b0010
#   Sign: + (0)
#   Exp: 1 (biased) → unbiased: 1 - 1 = 0
#   Mant: 0 → actual mantissa: 1.0 (implicit leading 1)
#
# Value = (+1) × 2^0 × 1.0 = 1.0

return torch.tensor([2], dtype=torch.uint8)  # FP4 encoding
```

**Result**: `1.25 → FP4(0b0010) → 1.0` ✓

🎯 **Key insight**: Tie-to-even rounding is critical for FP4 due to limited precision. The halfway case `1.25` rounds down to `1.0` because the mantissa bit is already even (0).

---

### 📦 FRAME 4: get_sem_bits() - Bit Extraction

📍 **Source**: [fp_format_spec.py:169-180](../../../torchao/prototype/mx_formats/fp_format_spec.py#L169)

**What happens**: Extract sign, exponent, mantissa bits as strings.

```python
def get_sem_bits(x: torch.Tensor, bitwidth: int) -> tuple[str, str, str]:
    """
    Extract S-E-M bits from FP encoding.

    Args:
        x: uint8 tensor with FP bits
        bitwidth: 4 for FP4, 6 for FP6, 8 for FP8

    Returns:
        (sign_str, exp_str, mant_str) as binary strings
    """

    x_int = x.to(torch.int32).item()

    if bitwidth == 4:
        # FP4 E2M1: 1 sign + 2 exp + 1 mant
        sign = (x_int >> 3) & 1
        exp = (x_int >> 1) & 0b11
        mant = x_int & 0b1

        return (bin(sign)[2:], bin(exp)[2:].zfill(2), bin(mant)[2:])

    elif bitwidth == 6:
        # FP6 E2M3: 1 sign + 2 exp + 3 mant
        # FP6 E3M2: 1 sign + 3 exp + 2 mant
        # (handled based on dtype context)
        ...

    else:
        # FP8, etc.
        ...
```

**Example** for FP4 value `0b0010`:
```
Input: 0b0010 (decimal 2)

Extract bits:
  sign = (2 >> 3) & 1 = 0 → "0"
  exp  = (2 >> 1) & 0b11 = 1 → "01"
  mant = 2 & 1 = 0 → "0"

Return: ("0", "01", "0") → concatenate: "0010" ✓
```

---

### 📦 FRAME 5: f4_unpacked_to_f32() - Dequantization

📍 **Source**: [kernels.py:103-108](../../../torchao/prototype/mx_formats/kernels.py#L103)

**What happens**: Convert FP4 bits back to FP32.

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

---

### 📦 FRAME 6: _floatx_unpacked_to_f32() - Core Dequantization

📍 **Source**: [kernels.py:635-720](../../../torchao/prototype/mx_formats/kernels.py#L635)

**Detailed trace for FP4 value `0b0010`**:

```python
# Input: FP4 bits = 0b0010 (uint8 tensor)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Extract components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

x_int = 2  # 0b0010

sign = (x_int >> (ebits + mbits)) & 1
# = (2 >> 3) & 1 = 0

exp_fp4 = (x_int >> mbits) & ((1 << ebits) - 1)
# = (2 >> 1) & 0b11 = 1

mantissa_fp4 = x_int & ((1 << mbits) - 1)
# = 2 & 0b1 = 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Check special cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

is_zero = (exp_fp4 == 0) and (mantissa_fp4 == 0)  # False
is_denorm = (exp_fp4 == 0) and (mantissa_fp4 != 0)  # False

# This is a normal number

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Convert to FP32 encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# For normal numbers:
# FP4 value = 2^(exp_fp4 - bias) × (1 + mantissa_fraction)
#           = 2^(1 - 1) × (1 + 0/2)
#           = 2^0 × 1.0
#           = 1.0

# Convert exponent: FP4 bias → FP32 bias
exp_fp32 = exp_fp4 - exp_bias + 127
# = 1 - 1 + 127 = 127

# Convert mantissa: 1 bit → 23 bits (shift left 22)
mantissa_fp32 = mantissa_fp4 << (23 - mbits)
# = 0 << 22 = 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Assemble FP32 bits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fp32_bits = (sign << 31) | (exp_fp32 << 23) | mantissa_fp32
# = (0 << 31) | (127 << 23) | 0
# = 0x00000000 | 0x3F800000 | 0x00000000
# = 0x3F800000

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Reinterpret as float32
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fp32_value = fp32_bits.view(torch.float32)
# 0x3F800000 as float32 = 1.0

return torch.tensor([1.0], dtype=torch.float32)
```

**Result**: FP4(0b0010) → 1.0 ✓

---

## FP4 Packing/Unpacking Trace

### test_fp4_pack_unpack Execution Trace

**Test code** ([test_kernels.py:321-348](../../../test/prototype/mx_formats/test_kernels.py#L321)):
```python
def test_fp4_pack_unpack():
    orig_vals = torch.Tensor([[0.0, 0.5, 4.0, -0.0], [-0.0, 1.0, -6.0, 3.0]])

    # Quantize to FP4
    orig_vals_f4_unpacked = f32_to_f4_unpacked(orig_vals)
    # Result: [[0x0, 0x1, 0x6, 0x8], [0x8, 0x2, 0xF, 0x5]]

    # Pack 2 FP4 per byte
    orig_vals_f4_packed = pack_uint4(orig_vals_f4_unpacked)

    # Verify packing format
    expected_f4_packed = torch.tensor([
        [0b00010000, 0b10000110],  # Row 0: [1:0, 6:4] (note: negative 0 = 0x8)
        [0b00101000, 0b01011111],  # Row 1: [2:8, 5:F]
    ], dtype=torch.uint8)

    assert torch.all(orig_vals_f4_packed == expected_f4_packed)

    # Unpack
    orig_vals_f4_packed_unpacked = unpack_uint4(orig_vals_f4_packed)

    # Dequantize
    orig_vals_dq = f4_unpacked_to_f32(orig_vals_f4_packed_unpacked)

    # Verify roundtrip
    assert torch.all(orig_vals_dq == orig_vals)
```

---

### 📦 FRAME 7: pack_uint4() - Pack 2 FP4 per Byte

📍 **Source**: [kernels.py:723-764](../../../torchao/prototype/mx_formats/kernels.py#L723)

**What happens**: Pack two 4-bit values into one uint8 byte.

```python
def pack_uint4(x: torch.Tensor) -> torch.Tensor:
    """
    Pack FP4 values (stored as uint8) into compact format.

    Layout per byte:
      Bits [7:4] = second value
      Bits [3:0] = first value

    Example: pack([0x3, 0xA]) → 0xA3
    """

    # Reshape to pairs
    x_reshaped = x.contiguous().view(-1, 2)
    # Shape: (N, 2) where N = total_elements // 2

    # Pack: byte = (val1 << 4) | val0
    packed = (x_reshaped[:, 1] << 4) | x_reshaped[:, 0]

    return packed
```

**Example trace** for first row `[0x0, 0x1, 0x6, 0x8]`:

```
Input: [0x0, 0x1, 0x6, 0x8]  # 4 FP4 values

Reshape to pairs: [[0x0, 0x1], [0x6, 0x8]]

Pack each pair:
  Pair 0: (0x1 << 4) | 0x0 = 0x10
  Pair 1: (0x8 << 4) | 0x6 = 0x86

Result: [0x10, 0x86]

Binary visualization:
  0x10 = 0b00010000
         ^^^^====
         val1 val0
         0x1  0x0

  0x86 = 0b10000110
         ^^^^====
         val1 val0
         0x8  0x6
```

---

### 📦 FRAME 8: unpack_uint4() - Unpack 2 FP4 per Byte

📍 **Source**: [kernels.py:766-804](../../../torchao/prototype/mx_formats/kernels.py#L766)

**What happens**: Unpack two 4-bit values from one uint8 byte.

```python
def unpack_uint4(x: torch.Tensor) -> torch.Tensor:
    """
    Unpack FP4 values from compact format.

    Input:  uint8 bytes where each byte contains 2 FP4 values
    Output: uint8 tensor with unpacked values (doubled size)
    """

    # Extract lower 4 bits (first value)
    val0 = x & 0x0F

    # Extract upper 4 bits (second value)
    val1 = (x >> 4) & 0x0F

    # Interleave back to original order
    unpacked = torch.stack([val0, val1], dim=-1).view(*x.shape[:-1], -1)

    return unpacked
```

**Example trace** for packed `[0x10, 0x86]`:

```
Input: [0x10, 0x86]

Unpack each byte:
  Byte 0x10:
    val0 = 0x10 & 0x0F = 0x0
    val1 = (0x10 >> 4) & 0x0F = 0x1

  Byte 0x86:
    val0 = 0x86 & 0x0F = 0x6
    val1 = (0x86 >> 4) & 0x0F = 0x8

Stack and interleave:
  [[0x0, 0x1], [0x6, 0x8]]

Flatten:
  [0x0, 0x1, 0x6, 0x8]

Result matches original unpacked values! ✓
```

---

## Triton MXFP8 Quantization Trace

### test_triton_mxfp8_dim1_randn Execution Trace

**Test code** ([test_kernels.py:480-485](../../../test/prototype/mx_formats/test_kernels.py#L480)):
```python
@pytest.mark.skipif(not is_sm_at_least_89(), reason="requires sm_89+")
@pytest.mark.parametrize("M", (256, 2048))
@pytest.mark.parametrize("K", (256, 2048))
def test_triton_mxfp8_dim1_randn(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    # Reference: PyTorch implementation
    x_mx_ref, x_s_ref = triton_to_mxfp8_dim1_reference(x, block_size=32)

    # Triton kernel
    x_mx_t, x_s_t = triton_to_mxfp8_dim1(x, inner_block_size=32)

    # Verify exact match
    torch.testing.assert_close(x_mx_t, x_mx_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_s_t, x_s_ref, rtol=0, atol=0)
```

This validates that the Triton kernel produces **bit-identical** results to the reference.

---

### 📦 FRAME 9: triton_to_mxfp8_dim1_reference() - PyTorch Reference

📍 **Source**: [test_kernels.py:61-66](../../../test/prototype/mx_formats/test_kernels.py#L61)

**What happens**: Reference implementation using existing `to_mx` function.

```python
def triton_to_mxfp8_dim1_reference(x_hp: torch.Tensor, block_size) -> tuple:
    """
    Reference version of column-wise (dim1) MXFP8 quantization.
    """
    # Transpose for column-wise processing
    x_hp_t = x_hp.t().contiguous()

    # Quantize using standard to_mx
    scale_d1, data_d1 = to_mx(
        x_hp_t,
        torch.float8_e4m3fn,
        block_size,
        scaling_mode=ScaleCalculationMode.FLOOR
    )

    # Transpose back
    return data_d1.t(), scale_d1
```

**For input (256, 2048)**:
```
x: (256, 2048) BF16
  ↓ transpose
x_t: (2048, 256) BF16
  ↓ to_mx (block_size=32)
data: (2048, 256) FP8
scale: (2048, 8) E8M0  # 256 / 32 = 8 blocks per row
  ↓ transpose
data_t: (256, 2048) FP8
scale_t: (8, 2048) E8M0  # Wait, this doesn't match!
```

⚠️ **Note**: The actual reference keeps scales in `(2048, 8)` form (column-wise blocks).

---

### 📦 FRAME 10: triton_to_mxfp8_dim1() - Triton Kernel

📍 **Source**: [kernels.py:1179-1247](../../../torchao/prototype/mx_formats/kernels.py#L1179)

**What happens**: GPU-accelerated column-wise MXF P8 quantization.

```python
def triton_to_mxfp8_dim1(
    x_hp: torch.Tensor,
    inner_block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Column-wise MXFP8 quantization using Triton.

    Processes columns (dim=1) in blocks of inner_block_size.
    """

    M, K = x_hp.shape

    # Allocate outputs
    x_mx = torch.empty_like(x_hp, dtype=torch.float8_e4m3fn)
    scale_e8m0 = torch.empty(
        (M, K // inner_block_size),
        dtype=torch.float8_e8m0fnu,
        device=x_hp.device
    )

    # Launch kernel
    grid = (M, triton.cdiv(K, inner_block_size))

    to_mxfp8_dim1_kernel[grid](
        x_hp, x_mx, scale_e8m0,
        M, K,
        inner_block_size,
    )

    return x_mx, scale_e8m0
```

---

### 📦 FRAME 11: to_mxfp8_dim1_kernel - Triton JIT Kernel

📍 **Source**: [kernels.py:903-1045](../../../torchao/prototype/mx_formats/kernels.py#L903)

**What happens**: Column-wise quantization kernel (each program handles one column block).

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
    ],
    key=["M"],
)
@triton.jit
def to_mxfp8_dim1_kernel(
    x_ptr,             # Input (M, K) BF16
    x_mx_ptr,          # Output (M, K) FP8
    scale_ptr,         # Output (M, K//block_size) E8M0
    M, K,
    BLOCK_SIZE: tl.constexpr,  # inner_block_size
):
    """
    Column-wise MXFP8 quantization.

    Each program processes one block of BLOCK_SIZE elements along a column.
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Get program IDs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pid_m = tl.program_id(0)  # Row index
    pid_k = tl.program_id(1)  # Block index along K

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Load input block
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    offs_k = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_k = offs_k < K

    x_block = tl.load(
        x_ptr + pid_m * K + offs_k,
        mask=mask_k,
        other=0.0
    )
    # Shape: (BLOCK_SIZE,) - one block from row pid_m

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Compute E8M0 scale (FLOOR mode)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Find max absolute value in block
    x_abs = tl.abs(x_block)
    x_amax = tl.max(x_abs, axis=0)  # Scalar

    # Compute E8M0 scale
    FP8_E4M3_MAX = 448.0
    scale = _triton_calculate_scale_floor(x_amax, FP8_E4M3_MAX)
    # Returns E8M0 value

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Normalize and cast to FP8
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Broadcast scale to block
    scale_fp = scale.to(x_block.dtype)

    # Normalize
    x_normalized = x_block / scale_fp

    # Cast to FP8 (saturated in Triton)
    x_fp8 = x_normalized.to(tl.float8e4m3fn)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Store results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Store quantized data
    tl.store(
        x_mx_ptr + pid_m * K + offs_k,
        x_fp8,
        mask=mask_k
    )

    # Store scale (one per block)
    tl.store(
        scale_ptr + pid_m * (K // BLOCK_SIZE) + pid_k,
        scale
    )
```

**Execution pattern** for (256, 2048) with block_size=32:

```
Grid: (256 rows, 2048/32=64 blocks) = 16,384 programs

Each program:
  - Processes 32 elements from one row
  - Computes 1 E8M0 scale
  - Quantizes 32 elements to FP8
  - Stores 32 FP8 values + 1 E8M0 scale

Parallelism: 16,384 independent programs on GPU

Memory pattern:
  - Coalesced loads: 32 consecutive BF16 values
  - Coalesced stores: 32 consecutive FP8 values
  - Scale stores: Sparse (one per block)
```

⚡ **Performance**: ~10-50× faster than CPU for large tensors due to massive parallelism.

---

### 📦 FRAME 12: _triton_calculate_scale_floor() - E8M0 Scale Calculation

📍 **Source**: [kernels.py:832-876](../../../torchao/prototype/mx_formats/kernels.py#L832)

**What happens**: Compute E8M0 scale in Triton (FLOOR mode).

```python
@triton.jit
def _triton_calculate_scale_floor(
    amax: tl.float32,
    elem_max: tl.constexpr,  # 448.0 for FP8 E4M3
) -> tl.float8e8m0fnu:
    """
    Calculate E8M0 scale using FLOOR mode.

    E8M0 format: 8-bit exponent only, bias=127
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Handle edge cases
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Zero amax → scale = 0 (all quantized values = 0)
    if amax == 0.0:
        return tl.zeros((1,), dtype=tl.float8e8m0fnu)[0]

    # NaN → scale = NaN
    if tl.math.isnan(amax):
        return tl.full((1,), float('nan'), dtype=tl.float8e8m0fnu)[0]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Compute shared exponent
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Goal: Find scale such that amax / scale ≈ elem_max
    # Scale format: 2^exp (E8M0)

    # Compute: exp = floor(log2(amax / elem_max))
    ratio = amax / elem_max
    exp_unbias = tl.math.floor(tl.math.log2(ratio))

    # Bias for E8M0 (same as FP32 bias)
    exp_biased = exp_unbias + 127

    # Clamp to E8M0 range [0, 254]
    # (255 is reserved for NaN/Inf)
    exp_clamped = tl.clamp(exp_biased, 0, 254)

    # Convert to E8M0
    scale_e8m0 = exp_clamped.to(tl.uint8).to(tl.float8e8m0fnu)

    return scale_e8m0
```

**Example calculation** for amax=120.5:

```
amax = 120.5
elem_max = 448.0

ratio = 120.5 / 448.0 = 0.269

exp_unbias = floor(log2(0.269))
           = floor(-1.89)
           = -2

exp_biased = -2 + 127 = 125

exp_clamped = clamp(125, 0, 254) = 125

E8M0 value = 125 (uint8)

Interpreted as scale:
  2^(125 - 127) = 2^(-2) = 0.25

Verification:
  amax / scale = 120.5 / 0.25 = 482
  elem_max = 448

Hmm, 482 > 448! This will saturate in FP8.
This is expected behavior for FLOOR mode.
```

---

## CUDA MXFP8 Quantization Trace

### test_cuda_mx_dim1_numerics Execution Trace

**Test code** ([test_kernels.py:571-606](../../../test/prototype/mx_formats/test_kernels.py#L571)):
```python
@pytest.mark.skipif(not is_sm_at_least_100(), reason="requires sm100+")
@pytest.mark.parametrize("M", (32, 64, 2048))
@pytest.mark.parametrize("K", (32, 64, 2048))
@pytest.mark.parametrize("input_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("scaling_mode", (ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL))
def test_cuda_mx_dim1_numerics(M, K, input_dtype, scaling_mode):
    from torchao.prototype import mxfp8_cuda

    x = torch.arange(0, M * K, dtype=input_dtype, device="cuda").reshape(M, K)
    block_size = 32

    # Reference: PyTorch
    y_d1_ref, s_d1_ref = to_mx_dim1_reference(x, block_size, scaling_mode)

    # CUDA extension
    _, y_d1, _, s_d1 = mxfp8_cuda.quantize(
        x,
        rowwise=False,
        colwise=True,
        scaling_mode="floor" if scaling_mode == ScaleCalculationMode.FLOOR else "rceil",
        scale_dim_x=1,
        scale_dim_y=block_size,
    )

    # Verify exact match
    torch.testing.assert_close(s_d1, s_d1_ref, rtol=0, atol=0)
    torch.testing.assert_close(y_d1, y_d1_ref, rtol=0, atol=0)
```

This validates the **C++/CUDA implementation** against the PyTorch reference.

---

### 📦 FRAME 13: mxfp8_cuda.quantize() - Python Binding

📍 **Source**: [mxfp8_extension.cpp:49-124](../../../torchao/csrc/cuda/mx_kernels/mxfp8_extension.cpp#L49)

**What happens**: Python wrapper that validates inputs and calls CUDA.

```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp8_quantize(
    torch::Tensor input,
    bool rowwise,
    bool colwise,
    std::string scaling_mode,
    int scale_dim_x,
    int scale_dim_y
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(rowwise || colwise, "Must specify at least one dimension");

    int M = input.size(0);
    int K = input.size(1);

    // Create output tensors
    torch::Tensor output_rowwise, scales_rowwise;
    torch::Tensor output_colwise, scales_colwise;

    if (colwise) {
        // Column-wise quantization
        output_colwise = torch::empty_like(input, torch::kFloat8_e4m3fn);
        scales_colwise = torch::empty(
            {M, K / scale_dim_y},
            torch::TensorOptions()
                .dtype(torch::kFloat8_e8m0fnu)
                .device(input.device())
        );

        // Call CUDA kernel
        mxfp8_quantize_cuda(
            input,
            output_colwise,
            scales_colwise,
            scaling_mode,
            /*dim=*/1,  // Column-wise
            scale_dim_y
        );
    }

    if (rowwise) {
        // Row-wise quantization (not yet implemented)
        TORCH_CHECK(false, "Row-wise quantization not supported");
    }

    return std::make_tuple(
        output_rowwise, output_colwise,
        scales_rowwise, scales_colwise
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &mxfp8_quantize, "MXFP8 quantization");
}
```

---

### 📦 FRAME 14: mxfp8_quantize_cuda() - CUDA Bridge

📍 **Source**: [mxfp8_cuda.cu:47-110](../../../torchao/csrc/cuda/mx_kernels/mxfp8_cuda.cu#L47)

**What happens**: Type dispatch and stream management.

```cuda
void mxfp8_quantize_cuda(
    const torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& scales,
    const std::string& scaling_mode,
    int dim,
    int scale_block_size
) {
    // Dispatch based on input dtype
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input.scalar_type(),
        "mxfp8_quantize_cuda",
        [&] {
            // Dispatch based on scaling mode
            if (scaling_mode == "floor") {
                mxfp8_quantize_kernel_launcher<scalar_t, ScalingMode::FLOOR>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<at::Float8_e4m3fn>(),
                    scales.data_ptr<at::Float8_e8m0fnu>(),
                    input.size(0),  // M
                    input.size(1),  // K
                    dim,
                    scale_block_size,
                    at::cuda::getCurrentCUDAStream()
                );
            } else if (scaling_mode == "rceil") {
                mxfp8_quantize_kernel_launcher<scalar_t, ScalingMode::RCEIL>(
                    // ... same args ...
                );
            } else {
                TORCH_CHECK(false, "Unknown scaling mode: ", scaling_mode);
            }
        }
    );

    // Synchronize (optional, for debugging)
    // cudaDeviceSynchronize();
}
```

---

### 📦 FRAME 15: mxfp8_quantize_kernel - CUDA Kernel

📍 **Source**: [mxfp8_quantize.cuh:78-256](../../../torchao/csrc/cuda/mx_kernels/mxfp8_quantize.cuh#L78)

**What happens**: Column-wise MXFP8 quantization kernel (SM100+ optimized).

```cuda
template <typename InputT, ScalingMode MODE>
__global__ void mxfp8_quantize_colwise_kernel(
    const InputT* __restrict__ input,   // (M, K)
    float8_e4m3fn* __restrict__ output,  // (M, K)
    float8_e8m0fnu* __restrict__ scales, // (M, K/block_size)
    int M,
    int K,
    int block_size
) {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Thread indexing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    int row = blockIdx.x;
    int block_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= M) return;

    // Each block processes one scale block
    int block_start = block_idx * block_size;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Shared memory for reduction
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    __shared__ float block_max;  // Shared across block

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Phase 1: Compute amax (reduction)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    float thread_max = 0.0f;

    // Each thread processes multiple elements
    for (int i = tid; i < block_size; i += blockDim.x) {
        int col = block_start + i;
        if (col < K) {
            float val = static_cast<float>(input[row * K + col]);
            thread_max = fmaxf(thread_max, fabsf(val));
        }
    }

    // Block-wide reduction using warp shuffles
    __shared__ float shared_maxes[32];  // One per warp

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
    }

    // First lane of each warp writes to shared memory
    if (lane_id == 0) {
        shared_maxes[warp_id] = thread_max;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        thread_max = (lane_id < (blockDim.x / 32)) ? shared_maxes[lane_id] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
        }

        if (lane_id == 0) {
            block_max = thread_max;
        }
    }
    __syncthreads();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Phase 2: Compute E8M0 scale
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    float8_e8m0fnu scale;

    if (tid == 0) {
        if (MODE == ScalingMode::FLOOR) {
            scale = compute_e8m0_scale_floor(block_max);
        } else {  // RCEIL
            scale = compute_e8m0_scale_rceil(block_max);
        }

        // Store scale
        scales[row * (K / block_size) + block_idx] = scale;
    }
    __syncthreads();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Phase 3: Normalize and quantize
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    float scale_fp = static_cast<float>(scale);

    for (int i = tid; i < block_size; i += blockDim.x) {
        int col = block_start + i;
        if (col < K) {
            float val = static_cast<float>(input[row * K + col]);

            // Normalize
            float normalized = val / scale_fp;

            // Cast to FP8 E4M3 (hardware saturates)
            float8_e4m3fn quantized = static_cast<float8_e4m3fn>(normalized);

            // Store
            output[row * K + col] = quantized;
        }
    }
}
```

🎯 **Key optimizations**:
1. **Warp shuffle reductions**: Fast max computation without global memory
2. **Shared memory**: Minimize global memory accesses
3. **Coalesced access**: All threads in warp access consecutive elements
4. **Thread-level parallelism**: Each thread processes multiple elements

---

## Execution Flow Summary

### FP4 Conversion Flow (FP32 ↔ FP4)

```
Quantization: FP32 → FP4
  ↓
📦 Frame 2: f32_to_f4_unpacked()
  ↓
📦 Frame 3: _f32_to_floatx_unpacked()
  ├─ Extract FP32 S-E-M
  ├─ Re-bias exponent: 127 → 1
  ├─ Round mantissa: 23 bits → 1 bit (tie-to-even)
  ├─ Handle overflow (carry into exponent)
  └─ Assemble FP4 bits
  ↓
Returns: uint8 with FP4 encoding

Dequantization: FP4 → FP32
  ↓
📦 Frame 5: f4_unpacked_to_f32()
  ↓
📦 Frame 6: _floatx_unpacked_to_f32()
  ├─ Extract FP4 S-E-M
  ├─ Re-bias exponent: 1 → 127
  ├─ Extend mantissa: 1 bit → 23 bits (shift left 22)
  ├─ Handle denormals
  └─ Assemble FP32 bits
  ↓
Returns: float32 value
```

### FP4 Packing Flow

```
Pack: 2 FP4 → 1 byte
  ↓
📦 Frame 7: pack_uint4()
  ├─ Reshape to pairs: (N,) → (N//2, 2)
  ├─ Pack: byte = (val1 << 4) | val0
  └─ Return: (N//2,) uint8
  ↓
Unpacked size halved

Unpack: 1 byte → 2 FP4
  ↓
📦 Frame 8: unpack_uint4()
  ├─ Extract: val0 = byte & 0x0F
  ├─ Extract: val1 = (byte >> 4) & 0x0F
  ├─ Interleave: [val0, val1]
  └─ Return: (N,) uint8
  ↓
Unpacked size doubled
```

### Triton MXFP8 Quantization Flow

```
User: x_mx, scale = triton_to_mxfp8_dim1(x, block_size=32)
  ↓
📦 Frame 10: triton_to_mxfp8_dim1() - Python launcher
  ├─ Allocate outputs (FP8 data + E8M0 scales)
  └─ Launch Triton kernel
     ↓
📦 Frame 11: to_mxfp8_dim1_kernel - GPU kernel
  ├─ Load input block (32 elements)
  ├─ Compute amax: tl.max(tl.abs(x))
  ├─ Calculate E8M0 scale (Frame 12)
  ├─ Normalize: x / scale
  ├─ Cast to FP8: x.to(tl.float8e4m3fn)
  └─ Store results
     ↓
Returns: (M, K) FP8 + (M, K//32) E8M0
```

### CUDA MXFP8 Quantization Flow

```
User: mxfp8_cuda.quantize(x, colwise=True, ...)
  ↓
📦 Frame 13: mxfp8_quantize() - Python binding (C++)
  ├─ Validate inputs
  ├─ Allocate outputs
  └─ Call CUDA kernel
     ↓
📦 Frame 14: mxfp8_quantize_cuda() - Type dispatch
  ├─ Dispatch on dtype (FP32/BF16/FP16)
  ├─ Dispatch on mode (FLOOR/RCEIL)
  └─ Launch CUDA kernel
     ↓
📦 Frame 15: mxfp8_quantize_colwise_kernel - CUDA kernel
  ├─ Thread indexing (block per scale block)
  ├─ Phase 1: Warp-shuffle reduction for amax
  ├─ Phase 2: Compute E8M0 scale (shared memory)
  ├─ Phase 3: Normalize and quantize (coalesced stores)
  └─ Return
     ↓
Returns: (M, K) FP8 + (M, K//32) E8M0
```

---

## Key Takeaways

1. **FP4/FP6 are custom formats**: No native PyTorch/CUDA support, implemented in software

2. **Tie-to-even rounding**: Critical for numerical stability in low-precision formats

3. **Packing efficiency**: FP4 packs 2 per byte (50% storage), FP6 packs 4 per 3 bytes (37.5% storage)

4. **Triton advantages**: Auto-tuning, JIT compilation, Python-like syntax

5. **CUDA optimizations**: Warp shuffles, shared memory, coalesced access patterns

6. **Numerical equivalence**: Tests validate bit-identical outputs across implementations

7. **SM compute capability**: FP8 requires SM89+, NVFP4 requires SM100+ for PTX intrinsics

8. **Multi-backend strategy**: PyTorch (portable), Triton (fast, user-friendly), CUDA (maximum performance)

The test suite ensures **correctness** and **equivalence** across all backends, enabling confident deployment!
