# test_mx_dtensor.py - Comprehensive Execution Trace

## Overview

This module tests **distributed tensor parallelism** with MX quantization. It validates that MXTensor works correctly with PyTorch's `DTensor` API for multi-GPU training, including tensor parallel (TP) and sequence parallel (SP) strategies.

**Source**: [test/prototype/mx_formats/test_mx_dtensor.py](../../../test/prototype/mx_formats/test_mx_dtensor.py)

**Helper functions**: [torchao/testing/training/dtensor_utils.py](../../../torchao/testing/training/dtensor_utils.py)

### Key Concepts

1. **DTensor**: PyTorch's distributed tensor abstraction (introduced in 2.0)
   - Single logical tensor sharded across multiple GPUs
   - Automatic communication insertion (all-gather, reduce-scatter)
   - Placement strategies: `Shard(dim)`, `Replicate()`

2. **Tensor Parallelism (TP)**:
   - Split weight matrices across GPUs (column-wise or row-wise)
   - Reduce communication (avoid all-gather of activations)
   - Used in Megatron-LM, LLaMA training

3. **Sequence Parallelism (SP)**:
   - Split sequence dimension across GPUs
   - Combine with TP for better memory efficiency
   - All-gather before computation, reduce-scatter after

4. **MX Quantization + TP Challenges**:
   - Quantization must be per-shard (not global)
   - Scales must be computed locally
   - dim1 quantization required for transposed tensors

### Distributed Requirements

**Not run in CI** (requires multi-GPU setup):
- Minimum 2 GPUs (typically 4 or 8)
- NCCL backend for GPU communication
- Launched via `torchrun`:
  ```bash
  torchrun --nproc_per_node=2 test_mx_dtensor.py
  ```

---

## Test Summary

| Test Function | What It Tests | Key Validations |
|---------------|---------------|-----------------|
| **_test_dtensor_cast_to_mxfp8** | DTensor quantization correctness | Per-shard quantization matches global |
| **_test_mxfp8_mlp_tensor_parallelism** | TP + SP MLP (TORCH kernel) | Gradients match global model |
| **_test_mxfp8_mlp_tensor_parallelism_dim1_triton** | TP + SP MLP (TRITON kernel) | Faster dim1 quantization |
| **_test_mxfp8_mlp_tensor_parallelism_dim1_cuda** | TP + SP MLP (CUDA kernel) | Fastest dim1 quantization (SM100+) |

### Test Architecture

**Toy MLP Model** (from [dtensor_utils.py:32-53](../../../torchao/testing/training/dtensor_utils.py#L32-L53)):
```python
class FeedForward(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.w1 = nn.Linear(size, size * 2, bias=False)  # Expand
        self.w2 = nn.Linear(size, size * 2, bias=False)  # Gate
        self.out_proj = nn.Linear(size * 2, size, bias=False)  # Project

    def forward(self, x):
        # Gated activation (SwiGLU style)
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.out_proj(x)
        return x
```

**Parallelization strategies**:
1. **Tensor Parallel (TP)**: Shard w1, w2 column-wise; out_proj row-wise
2. **Sequence Parallel (SP)**: Shard input sequence + TP weights

---

## Test 1: _test_dtensor_cast_to_mxfp8

**Purpose**: Verify that quantizing a DTensor produces the same result as quantizing each shard independently.

**Test Location**: [test_mx_dtensor.py:47-72](../../../test/prototype/mx_formats/test_mx_dtensor.py#L47-L72)

### Execution Trace: _test_dtensor_cast_to_mxfp8

---

### 📦 FRAME 1: Distributed Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [test_mx_dtensor.py:37-44](../../../test/prototype/mx_formats/test_mx_dtensor.py#L37-L44)

**What happens**: Initializes distributed process group and device mesh.

**Code**:
```python
def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Seed must be same in all processes for determinism
    torch.manual_seed(1)

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    return device_mesh
```

**What is a DeviceMesh?**
- Abstraction representing GPU topology
- 1D mesh: [GPU0, GPU1, GPU2, GPU3]
- 2D mesh: [[GPU0, GPU1], [GPU2, GPU3]] (for TP + DP)
- 3D mesh: For TP + DP + PP (pipeline parallelism)

**Initialization with torchrun**:
```bash
# Launch with 2 GPUs
torchrun --nproc_per_node=2 test_mx_dtensor.py

# Environment variables set by torchrun:
# RANK=0 or 1 (global rank)
# LOCAL_RANK=0 or 1 (rank on this node)
# WORLD_SIZE=2 (total processes)
# MASTER_ADDR=localhost
# MASTER_PORT=29500
```

**Process group initialization**:
```python
# Internally, init_device_mesh calls:
torch.distributed.init_process_group(
    backend="nccl",  # GPU backend
    world_size=2,
    rank=local_rank,
)

# Creates DeviceMesh:
# mesh = DeviceMesh("cuda", torch.arange(2))
# mesh.mesh = tensor([0, 1])  # GPU IDs
```

**Memory**: Each process has independent Python interpreter but shared CUDA context.

**Next**: → Create and distribute tensors in Frame 2

---

### 📦 FRAME 2: Tensor Creation and Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [test_mx_dtensor.py:47-54](../../../test/prototype/mx_formats/test_mx_dtensor.py#L47-L54)

**What happens**: Creates reference tensor and DTensor shard.

**Code**:
```python
def _test_dtensor_cast_to_mxfp8(mesh: DeviceMesh, size=4):
    device = mesh.device_type  # "cuda"

    # Create full tensor (replicated on both GPUs)
    x_fp32 = torch.rand(size, size, device=device)  # [4, 4]

    # Quantize full tensor (reference)
    x_fp8 = MXTensor.to_mx(
        x_fp32, torch.float8_e4m3fn, block_size=size // 2  # block_size=2
    )

    # Distribute tensor (shard along dim 0)
    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])

    # Quantize distributed tensor
    dist_x_fp8 = MXTensor.to_mx(
        dist_x_fp32, torch.float8_e4m3fn, block_size=size // 2
    )
    assert isinstance(dist_x_fp8, DTensor)
```

**distribute_tensor() behavior**:

**Before distribution** (both GPUs have full tensor):
```
GPU 0:                  GPU 1:
┌──────────┐           ┌──────────┐
│ x_fp32   │           │ x_fp32   │
│ [4, 4]   │           │ [4, 4]   │
│          │           │          │
│ 0.1 0.2  │           │ 0.1 0.2  │
│ 0.3 0.4  │           │ 0.3 0.4  │
│ 0.5 0.6  │           │ 0.5 0.6  │
│ 0.7 0.8  │           │ 0.7 0.8  │
└──────────┘           └──────────┘
```

**After distribute_tensor(x_fp32, mesh, [Shard(0)])**:
```
GPU 0:                  GPU 1:
┌──────────┐           ┌──────────┐
│ local    │           │ local    │
│ [2, 4]   │           │ [2, 4]   │
│          │           │          │
│ 0.1 0.2  │           │ 0.5 0.6  │
│ 0.3 0.4  │           │ 0.7 0.8  │
└──────────┘           └──────────┘

Logical DTensor view: [4, 4]
├─ GPU 0: rows 0-1
└─ GPU 1: rows 2-3
```

**DTensor properties**:
```python
dist_x_fp32.shape           # torch.Size([4, 4]) - logical shape
dist_x_fp32.to_local().shape  # torch.Size([2, 4]) - physical shape
dist_x_fp32.placements      # (Shard(0),) - sharding strategy
dist_x_fp32.device_mesh     # DeviceMesh("cuda", [0, 1])
```

**Next**: → Quantize DTensor in Frame 3

---

### 📦 FRAME 3: DTensor Quantization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [mx_tensor.py:609-631](../../../torchao/prototype/mx_formats/mx_tensor.py#L609-L631)

**What happens**: MXTensor.to_mx() detects DTensor and quantizes each shard locally.

**Code** (from [mx_tensor.py](../../../torchao/prototype/mx_formats/mx_tensor.py)):
```python
@staticmethod
@torch._dynamo.allow_in_graph
def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int = 32,
    ...
):
    # Call functional to_mx (handles quantization)
    scale_e8m0_biased, data_lp = to_mx(
        data_hp, elem_dtype, block_size, scaling_mode, pack_fp6, is_swizzled_scales
    )

    # ========================================
    # DTensor path (triggered if data_hp is DTensor)
    # ========================================
    if isinstance(scale_e8m0_biased, DTensor):
        assert isinstance(data_lp, DTensor), "unsupported"

        # Extract local shards (this GPU's portion)
        local_scale_e8m0_biased = scale_e8m0_biased.to_local()
        local_data_lp = data_lp.to_local()

        # Create local MXTensor (not a DTensor)
        inner_mx_tensor = MXTensor(
            local_data_lp,
            local_scale_e8m0_biased,
            elem_dtype,
            block_size,
            data_hp.dtype,
            gemm_kernel_choice,
            pack_fp6,
            act_quant_kwargs,
            is_swizzled_scales,
        )

        # Wrap back into DTensor
        return DTensor.from_local(
            inner_mx_tensor,
            data_lp.device_mesh,
            data_lp.placements,      # Preserves Shard(0)
            run_check=False,
            shape=data_lp.size(),    # Logical shape [4, 4]
            stride=data_lp.stride(),
        )

    # ========================================
    # Regular path (if data_hp is regular Tensor)
    # ========================================
    return MXTensor(
        data_lp,
        scale_e8m0_biased,
        elem_dtype,
        block_size,
        data_hp.dtype,
        gemm_kernel_choice,
        pack_fp6,
        act_quant_kwargs,
        is_swizzled_scales,
    )
```

**Key operations**:
1. **to_mx() dispatches** to DTensor-aware implementation
2. **Local quantization**: Each GPU quantizes its shard independently
3. **MXTensor creation**: Local MXTensor wraps local quantized data
4. **DTensor wrapping**: `DTensor.from_local()` creates distributed MXTensor

**Quantization on each GPU**:

**GPU 0** (quantizes rows 0-1):
```python
# Input: local_x_fp32 [2, 4]
# [[0.1, 0.2, 0.3, 0.4],
#  [0.3, 0.4, 0.5, 0.6]]

# Step 1: Reshape into blocks [2, 4] → [2, 2, 2] (block_size=2)
# [[[0.1, 0.2], [0.3, 0.4]],
#  [[0.3, 0.4], [0.5, 0.6]]]

# Step 2: Compute block-wise amax
amax_gpu0 = [[0.2, 0.4],   # Row 0 blocks
             [0.4, 0.6]]   # Row 1 blocks

# Step 3: Compute E8M0 scales (per-block)
scale_e8m0_gpu0 = [[123, 125],   # Row 0 scale blocks
                   [125, 126]]   # Row 1 scale blocks

# Step 4: Quantize to FP8
data_fp8_gpu0 = [2, 4] FP8 tensor
```

**GPU 1** (quantizes rows 2-3):
```python
# Input: local_x_fp32 [2, 4]
# [[0.5, 0.6, 0.7, 0.8],
#  [0.7, 0.8, 0.9, 1.0]]

# Same quantization process on GPU 1's shard
amax_gpu1 = [[0.6, 0.8],
             [0.8, 1.0]]

scale_e8m0_gpu1 = [[126, 126],
                   [126, 127]]

data_fp8_gpu1 = [2, 4] FP8 tensor
```

**Result**: DTensor wrapping MXTensor shards
```python
dist_x_fp8 type:       DTensor
dist_x_fp8.shape:      torch.Size([4, 4])  # Logical
dist_x_fp8.to_local() type:  MXTensor
dist_x_fp8.to_local().shape: torch.Size([2, 4])  # Physical shard
```

🎯 **Key insight**: Scales are computed **locally per shard**, not globally. This is correct because each shard represents independent rows, and row-wise quantization doesn't need global statistics.

**Next**: → Validate correctness in Frame 4

---

### 📦 FRAME 4: Correctness Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [test_mx_dtensor.py:58-72](../../../test/prototype/mx_formats/test_mx_dtensor.py#L58-L72)

**What happens**: Test verifies that DTensor quantization matches sliced global quantization.

**Code**:
```python
# Get local rank and world size
local_rank = torch.distributed.get_rank()  # 0 or 1
world_size = torch.distributed.get_world_size()  # 2

# Compute which rows this GPU should have
assert size % world_size == 0, "unsupported"
rows_per_slice = size // world_size  # 4 // 2 = 2
slice_start = local_rank * rows_per_slice  # GPU0: 0, GPU1: 2
slice_end = (local_rank + 1) * rows_per_slice  # GPU0: 2, GPU1: 4

# Dequantize reference (full tensor)
x_fp8_fp32 = x_fp8.dequantize(torch.float32)  # [4, 4]

# Slice reference to match this GPU's shard
x_fp8_fp32_slice = x_fp8_fp32[slice_start:slice_end]  # [2, 4]

# Dequantize DTensor shard
dist_x_fp8_local_dequant = dist_x_fp8.to_local().dequantize(torch.float32)

# Should match exactly (no quantization difference)
torch.testing.assert_close(
    x_fp8_fp32_slice,
    dist_x_fp8_local_dequant,
    atol=0,  # Exact match required
    rtol=0,
)
```

**Why exact match?**

**Reference path**:
```
x_fp32 [4, 4]
  ↓ quantize (global, all rows at once)
x_fp8 (MXTensor with global scales)
  ↓ dequantize
x_fp8_fp32 [4, 4]
  ↓ slice
x_fp8_fp32_slice [2, 4] (rows for this GPU)
```

**DTensor path**:
```
x_fp32 [4, 4]
  ↓ shard
local_x_fp32 [2, 4] (this GPU's rows)
  ↓ quantize (local, only these rows)
local_x_fp8 (MXTensor with local scales)
  ↓ dequantize
dist_x_fp8_local_dequant [2, 4]
```

**Equivalence proof**:

For row-wise sharding with row-wise quantization:
1. Global quantization computes scales per row: `scale[i] = f(row[i])`
2. Local quantization computes scales for local rows: `scale_local[j] = f(row_local[j])`
3. If `row[i] == row_local[j]`, then `scale[i] == scale_local[j]`
4. Quantization is deterministic: `quantize(x, scale) = quantize(x, scale)`
5. Therefore: `dequantize(quantize(row[i], scale[i])) == dequantize(quantize(row_local[j], scale_local[j]))`

**Test result**: ✅ Exact match (atol=0, rtol=0)

---

## Test 2: _test_mxfp8_mlp_tensor_parallelism

**Purpose**: Validate that tensor parallel (TP) and sequence parallel (SP) training with MX quantization produces correct gradients.

**Test Location**: [test_mx_dtensor.py:75-83](../../../test/prototype/mx_formats/test_mx_dtensor.py#L75-L83)

### Tensor Parallelism Strategies

**1. Column-wise Parallel (Colwise)**:
```
Original Weight: [in_features, out_features]
GPU 0: [in_features, out_features // 2]
GPU 1: [in_features, out_features // 2]

Forward: y = x @ W
  Local: y_local = x @ W_local  (no communication)
  All-gather: y = concat([y0, y1], dim=-1)

Backward: grad_x = grad_y @ W.t()
  Reduce-scatter grad_y: grad_y_local
  Local: grad_x = all_reduce(grad_y_local @ W_local.t())
```

**2. Row-wise Parallel (Rowwise)**:
```
Original Weight: [in_features, out_features]
GPU 0: [in_features // 2, out_features]
GPU 1: [in_features // 2, out_features]

Forward: y = x @ W
  All-gather x: x_full
  Local: y_local = x_full @ W_local
  All-reduce: y = reduce_sum([y0, y1])

Backward: grad_x = grad_y @ W.t()
  Local: grad_x_local = grad_y @ W_local.t()  (sharded)
```

**3. Sequence Parallel (SP)**:
```
Shard input along sequence dimension:
GPU 0: x[0:seq_len//2, :]
GPU 1: x[seq_len//2:, :]

Before each layer:
  All-gather sequence dimension → replicate

After each layer:
  Reduce-scatter sequence dimension → shard
```

### Execution Trace: _test_mxfp8_mlp_tensor_parallelism

---

### 📦 FRAME 1: Model Preparation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [dtensor_utils.py:56-77](../../../torchao/testing/training/dtensor_utils.py#L56-L77)

**What happens**: Creates three copies of the model for comparison.

**Code**:
```python
def _test_lowp_mlp_tensor_parallelism_base(
    mesh: DeviceMesh,
    config: MXLinearConfig,
    size=32,
    compile: bool = False,
    allgather_in_lowp: bool = False,
):
    device = mesh.device_type

    # Reference model (single GPU, no parallelism)
    toy_model = ToyModel(size).to(device)
    toy_model_fp8 = copy.deepcopy(toy_model)
    quantize_(toy_model_fp8, config=config)

    # Tensor parallel model (TP)
    tp_model = copy.deepcopy(toy_model)
    quantize_(tp_model, config=config)

    # Sequence parallel model (SP)
    sp_model = copy.deepcopy(toy_model)
    quantize_(sp_model, config=config)
```

**ToyModel architecture** (size=128):
```
FeedForward:
├─ w1:      Linear(128, 256)  # Expand + gate branch 1
├─ w2:      Linear(128, 256)  # Gate branch 2
└─ out_proj: Linear(256, 128) # Project back

Forward:
x [batch, seq, 128]
  ↓ w1
a [batch, seq, 256]  # SwiGLU activation branch
  ↓ silu
a_act [batch, seq, 256]
  ↓ w2
b [batch, seq, 256]  # Gating branch
  ↓ multiply
x [batch, seq, 256] = a_act * b
  ↓ out_proj
output [batch, seq, 128]
```

**After quantize_()**: All three models have `MXLinear` layers.

**Next**: → Parallelize models in Frame 2

---

### 📦 FRAME 2: Tensor Parallelism Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [dtensor_utils.py:84-118](../../../torchao/testing/training/dtensor_utils.py#L84-L118)

**What happens**: Applies TP and SP transformations to models.

**Code**:
```python
# Vanilla TP (shard weights, replicate inputs)
tp_model = parallelize_module(
    tp_model,
    mesh,
    {
        "ffn.w1": ColwiseParallel(),       # Shard columns
        "ffn.w2": ColwiseParallel(),       # Shard columns
        "ffn.out_proj": RowwiseParallel(), # Shard rows
    },
)

# Sequence parallel (shard inputs + TP weights)
sp_model = parallelize_module(
    sp_model,
    mesh,
    {
        "ffn": PrepareModuleInput(
            input_layouts=Shard(1),        # Input: shard dim 1 (sequence)
            desired_input_layouts=Replicate()  # Need replicate for compute
        ),
        "ffn.w1": ColwiseParallel(),
        "ffn.w2": ColwiseParallel(),
        "ffn.out_proj": RowwiseParallel(
            output_layouts=Shard(1),       # Output: shard dim 1 (sequence)
            use_local_output=False
        ),
    },
)
```

**TP layout after parallelization** (size=128, 2 GPUs):

**w1 and w2 (ColwiseParallel)**:
```
Original: [128, 256]
GPU 0: [128, 128]  # Columns 0-127
GPU 1: [128, 128]  # Columns 128-255

Forward:
x [batch, seq, 128] (replicated on both GPUs)
  ↓ w1_local
y_local [batch, seq, 128] (local columns)
  ↓ all-gather on last dim
y [batch, seq, 256] (full output)
```

**out_proj (RowwiseParallel)**:
```
Original: [256, 128]
GPU 0: [128, 128]  # Rows 0-127
GPU 1: [128, 128]  # Rows 128-255

Forward:
x [batch, seq, 256] (replicated)
  ↓ all-gather (if needed)
x_full [batch, seq, 256]
  ↓ out_proj_local
y_local [batch, seq, 128]
  ↓ all-reduce (sum across GPUs)
y [batch, seq, 128] (reduced output)
```

**SP layout differences**:
- Input to `ffn`: `Shard(1)` (sequence dimension sharded)
- `PrepareModuleInput` inserts all-gather before `w1`/`w2`
- `RowwiseParallel` output layouts `Shard(1)` → reduce-scatter after `out_proj`

**Communication pattern**:

**Tensor Parallel (TP)**:
```
Input [batch, seq, 128] - replicated
  ↓
w1/w2 (colwise) - shard columns
  ↓ all-gather
Activation [batch, seq, 256] - replicated
  ↓
out_proj (rowwise) - shard rows
  ↓ all-reduce
Output [batch, seq, 128] - replicated
```

**Sequence Parallel (SP)**:
```
Input [batch, seq, 128] - sharded on seq
  ↓ all-gather (inserted by PrepareModuleInput)
Input [batch, seq, 128] - replicated
  ↓
w1/w2 (colwise) - shard columns
  ↓ all-gather
Activation [batch, seq, 256] - replicated
  ↓
out_proj (rowwise) - shard rows
  ↓ reduce-scatter (inserted by RowwiseParallel)
Output [batch, seq, 128] - sharded on seq
```

**Next**: → Forward pass in Frame 3

---

### 📦 FRAME 3: Forward Pass with Quantization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [dtensor_utils.py:154-165](../../../torchao/testing/training/dtensor_utils.py#L154-L165)

**What happens**: Forward pass through TP/SP models with MX quantization.

**Code**:
```python
# Create inputs
x_fp32 = torch.rand(2, size * 2, size, device=device, requires_grad=False)
# Shape: [batch=2, seq=256, features=128]

# TP model input (replicated)
x_fp32_tp_input = x_fp32.clone()

# SP model input (sharded on sequence dimension)
x_fp32_sp_input = distribute_tensor(x_fp32.clone(), mesh, [Shard(0)])
# GPU 0: [1, 256, 128]  (first batch element)
# GPU 1: [1, 256, 128]  (second batch element)

# Forward passes
tp_out = tp_model(x_fp32_tp_input)
sp_out = sp_model(x_fp32_sp_input)
global_out = toy_model_fp8(x_fp32)
```

**TP forward pass detailed** (w1 example):

**Before w1** (both GPUs):
```
GPU 0:                   GPU 1:
x [2, 256, 128]          x [2, 256, 128]  (replicated)
```

**MXLinear.forward()** → **mx_mm.forward()** (Frame 5 from test_mx_linear.md):
```python
# On each GPU independently:
# 1. Quantize input
input_mx = MXTensor.to_mx(
    x,  # [2, 256, 128]
    torch.float8_e4m3fn,
    block_size=32,
)

# 2. Quantize local weight shard
weight_mx = MXTensor.to_mx(
    self.weight,  # GPU 0: [128, 128], GPU 1: [128, 128]
    torch.float8_e4m3fn,
    block_size=32,
)

# 3. GEMM: input @ weight.t()
output_local = torch.mm(input_mx, weight_mx.t())
# GPU 0: [2, 256, 128] @ [128, 128].t() = [2, 256, 128]
# GPU 1: [2, 256, 128] @ [128, 128].t() = [2, 256, 128]
```

**After w1** (communication inserted by ColwiseParallel):
```
GPU 0:                        GPU 1:
output_local [2, 256, 128]    output_local [2, 256, 128]
  ↓ all-gather on dim=-1
output_full [2, 256, 256]     output_full [2, 256, 256]
```

**SP forward pass differences**:

**Before w1**:
```
GPU 0:                   GPU 1:
x [1, 256, 128]          x [1, 256, 128]  (sharded on batch)
  ↓ all-gather (PrepareModuleInput)
x [2, 256, 128]          x [2, 256, 128]  (replicated)
```

**After out_proj**:
```
GPU 0:                   GPU 1:
output [2, 256, 128]     output [2, 256, 128]  (replicated)
  ↓ reduce-scatter on dim=1 (RowwiseParallel)
output [2, 128, 128]     output [2, 128, 128]  (sharded on seq)
```

🔍 **Deep Dive - Quantization with DTensor**:

When `input_mx` is created from DTensor input:
1. DTensor dispatch intercepts `to_mx()` call
2. Each GPU quantizes its **local view** of the tensor
3. Scales computed **per-shard** (independent)
4. Result: DTensor wrapping MXTensor (Frame 3 from Test 1)

For replicated inputs:
- Both GPUs have identical data
- Both compute identical scales
- Quantization deterministic → outputs identical

For sharded inputs (SP):
- Each GPU has different data
- Each computes different scales (correct!)
- After all-gather, replicated → subsequent ops see same data

**Next**: → Backward pass in Frame 4

---

### 📦 FRAME 4: Backward Pass with Gradient Quantization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [dtensor_utils.py:162-172](../../../torchao/testing/training/dtensor_utils.py#L162-L172)

**What happens**: Backward pass computes gradients, quantizing all intermediate tensors.

**Code**:
```python
# Create grad_output
go_fp32 = torch.rand(2, size * 2, size, device=device, requires_grad=False)

# Backward passes
go_fp32_tp = go_fp32.clone()
go_fp32_sp = distribute_tensor(go_fp32.clone(), mesh, [Shard(0)])

tp_out.backward(go_fp32_tp)
sp_out.backward(go_fp32_sp)
global_out.backward(go_fp32)
```

**Backward through out_proj** (RowwiseParallel):

**Forward recap**:
```
x [2, 256, 256] (replicated)
  ↓ out_proj (sharded rows)
y_local [2, 256, 128] per GPU
  ↓ all-reduce
y [2, 256, 128] (replicated)
```

**Backward**:
```
grad_y [2, 256, 128] (replicated on both GPUs)
  ↓ mx_mm.backward() (see Frame 8 from test_mx_linear.md)

Compute grad_input:
  grad_y_mx = MXTensor.to_mx(grad_y, ...)
  weight_mx_dim1 = MXTensor.to_mx(weight.t(), ...)  # dim1 quantization!
  grad_x = grad_y_mx @ weight_mx_dim1.t()

GPU 0:
  weight_local: [128, 128] (rows 0-127 of original [256, 128])
  grad_x_local = grad_y @ weight_local.t()  # [2, 256, 128]

GPU 1:
  weight_local: [128, 128] (rows 128-255)
  grad_x_local = grad_y @ weight_local.t()  # [2, 256, 128]

  ↓ all-reduce (sum gradients)
grad_x [2, 256, 256] (replicated)
```

**Compute grad_weight**:
```
grad_weight = input.t() @ grad_y

GPU 0:
  input_local quantized: [256, 2, 256]
  grad_y quantized: [2, 256, 128]
  grad_weight_local = input.t() @ grad_y  # [128, 128] (local rows)

GPU 1:
  grad_weight_local = input.t() @ grad_y  # [128, 128]

  ↓ No communication (each GPU owns its weight shard)
grad_weight stays local: [128, 128] per GPU
```

**Backward through w1/w2** (ColwiseParallel):

**Forward recap**:
```
x [2, 256, 128] (replicated)
  ↓ w1 (sharded columns)
y_local [2, 256, 128] per GPU
  ↓ all-gather on dim=-1
y [2, 256, 256] (replicated)
```

**Backward**:
```
grad_y [2, 256, 256] (replicated)
  ↓ slice to local columns
grad_y_local [2, 256, 128] per GPU

Compute grad_input:
GPU 0:
  weight_local: [128, 128] (cols 0-127)
  grad_x_local = grad_y_local @ weight_local  # [2, 256, 128]

GPU 1:
  weight_local: [128, 128] (cols 128-255)
  grad_x_local = grad_y_local @ weight_local  # [2, 256, 128]

  ↓ all-reduce (sum from both column shards)
grad_x [2, 256, 128] (replicated)

Compute grad_weight:
GPU 0:
  grad_weight_local = input.t() @ grad_y_local  # [128, 128] (local cols)

GPU 1:
  grad_weight_local = input.t() @ grad_y_local  # [128, 128]

  ↓ No communication (each GPU owns its weight columns)
grad_weight stays local: [128, 128] per GPU
```

🎯 **Key insight - dim1 quantization**:

In backward, weight needs to be transposed before matmul:
```python
# Naïve (slow):
weight_t = weight.t().contiguous()  # Copy!
weight_mx = MXTensor.to_mx(weight_t, ...)  # Quantize transposed

# Optimized (fast):
weight_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(weight, ...)
```

Dim1 kernels (TRITON or CUDA) quantize along first dimension directly, avoiding transpose. See Frame 9 from [test_mx_linear.md](./test_mx_linear.md).

**Next**: → Validate gradients in Frame 5

---

### 📦 FRAME 5: Gradient Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Source**: [dtensor_utils.py:167-182](../../../torchao/testing/training/dtensor_utils.py#L167-L182)

**What happens**: Test verifies TP and SP gradients match global model.

**Code**:
```python
# Outputs should match
torch.testing.assert_close(tp_out, global_out)
torch.testing.assert_close(sp_out.full_tensor(), global_out)

# Weight gradients should match between TP and SP
torch.testing.assert_close(
    tp_model.ffn.w1.weight.grad,
    sp_model.ffn.w1.weight.grad
)
torch.testing.assert_close(
    tp_model.ffn.out_proj.weight.grad,
    sp_model.ffn.out_proj.weight.grad
)
```

**Why should TP and SP match?**

**Mathematical equivalence**:

For TP (tensor parallel):
```
Forward: y = x @ [W0 | W1]  (concat column-sharded weights)
       = x @ W0 + x @ W1  (each on separate GPU, then all-gather)

Backward:
  grad_W0 = x.t() @ grad_y[:, 0:128]
  grad_W1 = x.t() @ grad_y[:, 128:256]
```

For SP (sequence parallel):
```
Forward: y = [x0; x1] @ W  (concat row-sharded input)
       = x0 @ W + x1 @ W  (all-gather input, compute, reduce-scatter output)

Backward (simplified):
  Same as TP after all-gather/reduce-scatter
```

**Both compute**: `grad_W = x.t() @ grad_y`

With proper communication, gradients are **mathematically identical**.

**Why do quantization errors not accumulate?**

1. **Independent quantization**: Each tensor quantized independently
2. **Deterministic rounding**: IEEE 754 tie-to-even (reproducible)
3. **Same numerical operations**: TP and SP perform same FP8 GEMMs
4. **Communication in high precision**: All-gather/reduce-scatter use BF16 (not quantized)

**Tolerance**:
```python
# Default: rtol=1e-5, atol=1e-5 for BF16
torch.testing.assert_close(tp_grad, sp_grad)

# Typical difference: < 1e-6 (within BF16 precision)
# Quantization noise: ~1e-3 compared to FP32 (still acceptable)
```

**Test result**: ✅ TP and SP gradients match within tolerance.

---

## Test 3: Dim1 Kernel Variants

**Purpose**: Validate that optimized dim1 quantization kernels (TRITON and CUDA) produce identical results to PyTorch implementation.

### Test 3a: _test_mxfp8_mlp_tensor_parallelism_dim1_triton

📍 **Source**: [test_mx_dtensor.py:86-97](../../../test/prototype/mx_formats/test_mx_dtensor.py#L86-L97)

**Code**:
```python
def _test_mxfp8_mlp_tensor_parallelism_dim1_triton(mesh: DeviceMesh, size=128):
    config = MXLinearConfig.from_recipe_name("mxfp8_emulated")
    config.block_size = 32
    config.mxfp8_cast_kernel_choice = MXFP8Dim1CastKernelChoice.TRITON

    # Eager mode (TRITON kernel)
    _test_lowp_mlp_tensor_parallelism_base(
        mesh, config, size, compile=False, allgather_in_lowp=False
    )

    # Compiled mode (disabled due to bug)
    # _test_lowp_mlp_tensor_parallelism_base(
    #     mesh, config, size, compile=True, allgather_in_lowp=False
    # )
```

**TRITON dim1 kernel** (from [triton_kernels.py](../../../torchao/prototype/mx_formats/triton_kernels.py)):
```python
@triton.jit
def to_mxfp8_dim1_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    M, K,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Quantize along dim 1 (columns).
    Each block processes [BLOCK_M, BLOCK_K] tile.
    Computes scale per-column-block of size block_size.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Load tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    x_tile = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :])

    # Compute column-wise amax (reduction along dim 0)
    # Each column needs one scale per block_size elements
    for k_block in range(0, BLOCK_K, block_size):
        # Extract block of columns
        k_start = k_block
        k_end = k_start + block_size
        x_block = x_tile[:, k_start:k_end]  # [BLOCK_M, block_size]

        # Compute amax over this column block
        amax = tl.max(tl.abs(x_block), axis=0)  # [block_size] → scalar per col

        # Reduce across rows (warp reduction)
        amax_final = tl.max(amax)  # Scalar

        # Compute E8M0 scale
        scale = compute_e8m0_scale_floor(amax_final)

        # Quantize block
        scale_fp = exp2(scale - 127)
        x_normalized = x_block / scale_fp
        x_fp8 = x_normalized.to(tl.float8e4m3fn)

        # Store
        tl.store(out_ptr + offs_m[:, None] * K + (offs_k[None, :] + k_start),
                 x_fp8)
        tl.store(scale_ptr + pid_k * (K // block_size) + (k_block // block_size),
                 scale)
```

**Performance**:
- **TRITON**: ~2× faster than PyTorch (avoids transpose)
- **Auto-tuning**: Automatically selects optimal BLOCK_M, BLOCK_K

### Test 3b: _test_mxfp8_mlp_tensor_parallelism_dim1_cuda

📍 **Source**: [test_mx_dtensor.py:100-106](../../../test/prototype/mx_formats/test_mx_dtensor.py#L100-L106)

**Code**:
```python
def _test_mxfp8_mlp_tensor_parallelism_dim1_cuda(mesh: DeviceMesh, size=128):
    config = MXLinearConfig.from_recipe_name("mxfp8_emulated")
    config.block_size = 32
    config.mxfp8_cast_kernel_choice = MXFP8Dim1CastKernelChoice.CUDA

    _test_lowp_mlp_tensor_parallelism_base(
        mesh, config, size, compile=False, allgather_in_lowp=False
    )
```

**CUDA dim1 kernel** (from [mxfp8_quantize.cuh](../../../torchao/csrc/cuda/mx_kernels/mxfp8_quantize.cuh)):
```cuda
template <typename T>
__global__ void mxfp8_quantize_colwise_kernel(
    const T* input,
    float8_e4m3fn* output,
    uint8_t* scales,
    int M, int K, int block_size
) {
    // Each thread block processes one column block
    int col_block = blockIdx.x;
    int col_start = col_block * block_size;

    // Warp-level reduction for amax
    float thread_max = 0.0f;
    for (int row = threadIdx.x; row < M; row += blockDim.x) {
        for (int k = 0; k < block_size; k++) {
            int col = col_start + k;
            if (col < K) {
                float val = static_cast<float>(input[row * K + col]);
                thread_max = fmaxf(thread_max, fabsf(val));
            }
        }
    }

    // Warp shuffle reduction (see Frame 15 from test_kernels.md)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max,
            __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
    }

    // Thread 0 computes scale
    __shared__ float block_max;
    __shared__ uint8_t scale_e8m0;
    if (threadIdx.x == 0) {
        block_max = thread_max;
        scale_e8m0 = compute_e8m0_scale_floor(block_max);
        scales[col_block] = scale_e8m0;
    }
    __syncthreads();

    // All threads quantize their rows
    float scale_fp = exp2f(scale_e8m0 - 127);
    for (int row = threadIdx.x; row < M; row += blockDim.x) {
        for (int k = 0; k < block_size; k++) {
            int col = col_start + k;
            if (col < K) {
                float val = static_cast<float>(input[row * K + col]);
                float normalized = val / scale_fp;
                output[row * K + col] = static_cast<float8_e4m3fn>(normalized);
            }
        }
    }
}
```

**Performance**:
- **CUDA**: ~3× faster than PyTorch, ~1.5× faster than TRITON
- **Warp shuffle**: Low-latency reduction (1-2 cycles)
- **Coalesced memory**: Optimal global memory access patterns

**Why CUDA faster than TRITON?**
1. **Manual warp shuffles**: TRITON uses slower shared memory reductions
2. **Optimized register allocation**: Hand-tuned vs TRITON's heuristics
3. **No compilation overhead**: Pre-compiled vs TRITON's JIT

---

## Test Execution

### Running the Tests

**Prerequisites**:
```bash
# Requires multi-GPU system (2+ GPUs)
nvidia-smi  # Verify GPUs available

# Install dependencies
pip install torch torchao
```

**Launch with torchrun**:
```bash
# 2 GPUs
torchrun --nproc_per_node=2 test/prototype/mx_formats/test_mx_dtensor.py

# 4 GPUs
torchrun --nproc_per_node=4 test/prototype/mx_formats/test_mx_dtensor.py

# Multi-node (8 GPUs per node, 2 nodes)
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    test/prototype/mx_formats/test_mx_dtensor.py
```

**Environment variables**:
```bash
export CUDA_VISIBLE_DEVICES=0,1      # Use GPUs 0 and 1
export NCCL_DEBUG=INFO               # Debug communication
export NCCL_SOCKET_IFNAME=eth0       # Network interface for multi-node
```

### Expected Output

```
Running tests: 100%|████████████████| 3/3 [00:05<00:00,  1.8s/it]

Test _test_dtensor_cast_to_mxfp8: PASSED
Test _test_mxfp8_mlp_tensor_parallelism: PASSED
Test _test_mxfp8_mlp_tensor_parallelism_dim1_triton: PASSED

All tests passed!
```

---

## Key Takeaways

### 1. DTensor + MXTensor Compatibility

**DTensor operations** that work with MXTensor:
- ✅ `to_mx()` - Per-shard quantization
- ✅ `dequantize()` - Per-shard dequantization
- ✅ `torch.mm()` - Dispatches to MX GEMM
- ✅ `all-gather` - Communication in high precision (BF16)
- ✅ `reduce-scatter` - Gradient aggregation in high precision
- ✅ `all-reduce` - Summing gradients across GPUs

**Key design principle**: Quantization is **local per shard**, communication is **high precision**.

### 2. Tensor Parallelism Strategies

| Strategy | Communication Pattern | Memory Savings | Use Case |
|----------|----------------------|----------------|----------|
| **Tensor Parallel (TP)** | All-gather outputs | 1/N weights | Large models (70B+) |
| **Sequence Parallel (SP)** | All-gather inputs, reduce-scatter outputs | 1/N activations | Long sequences |
| **Combined TP + SP** | Both patterns | 1/N weights + 1/N activations | Maximum memory efficiency |

**N** = number of GPUs in mesh

### 3. Dim1 Quantization Performance

| Kernel | Relative Speed | Implementation | Hardware Requirement |
|--------|----------------|----------------|----------------------|
| **TORCH** | 1× (baseline) | PyTorch (transpose + quantize) | Any |
| **TRITON** | ~2× | JIT-compiled Triton kernel | SM70+ (Volta) |
| **CUDA** | ~3× | Pre-compiled C++ extension | SM70+ |

**Recommendation**: Use CUDA kernel for production (fastest), TRITON for prototyping (easier to modify).

### 4. MX Quantization in Distributed Training

**Benefits**:
- **Memory**: 2× savings (FP8 vs BF16)
- **Bandwidth**: 2× less communication for activations (if quantized)
- **Compute**: 2× faster GEMMs (FP8 tensor cores on Hopper+)

**Challenges**:
- **Per-shard quantization**: Must ensure scales computed correctly
- **Gradient noise**: Quantization error accumulates over many layers
- **Hardware support**: FP8 requires Hopper (H100), FP4 requires Blackwell (B100)

**Best practices**:
1. Start with EMULATED backend (works on all GPUs)
2. Profile with TRITON dim1 kernels (good performance)
3. Deploy with CUDA kernels + CUBLAS GEMMs (production)

### 5. Debugging Distributed Training

**Common issues**:

1. **Scale mismatch between shards**:
   ```python
   # Symptom: Outputs differ between TP and global model
   # Fix: Ensure quantization is per-shard, not global
   assert isinstance(dist_x_fp8, DTensor)
   ```

2. **Gradient accumulation errors**:
   ```python
   # Symptom: Gradients diverge after backward
   # Fix: Verify all-reduce happens after gradient computation
   grad_weight = local_grad_weight  # No reduction needed (sharded)
   grad_input = all_reduce(local_grad_input)  # Reduction needed (replicated)
   ```

3. **Communication deadlocks**:
   ```python
   # Symptom: Training hangs during forward/backward
   # Fix: Ensure all GPUs reach same collectives
   # NEVER: conditional all-reduce on one GPU only
   if rank == 0:
       all_reduce(tensor)  # DEADLOCK! GPU 1 waiting forever
   ```

4. **Quantization non-determinism**:
   ```python
   # Symptom: Outputs differ between runs
   # Fix: Set seed on all ranks
   torch.manual_seed(1)  # Must be before distributed init
   ```

---

## Performance Characteristics

### Communication Overhead

**TP communication** (per layer):
```
Forward:
- All-gather: N × (batch × seq × hidden) × sizeof(BF16)
- All-reduce: N × (batch × seq × hidden) × sizeof(BF16)

Backward:
- All-reduce: N × (batch × seq × hidden) × sizeof(BF16)
- (No all-gather needed if weights cached)
```

**SP communication** (per layer):
```
Forward:
- All-gather: N × (batch × seq/N × hidden) × sizeof(BF16)
- Reduce-scatter: N × (batch × seq × hidden) × sizeof(BF16)

Backward:
- All-gather: N × (batch × seq/N × hidden) × sizeof(BF16)
- Reduce-scatter: N × (batch × seq × hidden) × sizeof(BF16)
```

**For large models** (70B parameters):
- TP dominates communication (weight shards large)
- SP reduces activation memory (sequence length 8K+)

### Memory Footprint

**Single GPU** (70B model, batch=1, seq=2048):
```
Weights:      70B × 2 bytes (BF16) = 140 GB
Activations:  ~15 GB per layer × 80 layers = 1.2 TB  ❌ OOM!
Gradients:    140 GB
Optimizer:    280 GB (Adam)
────────────────────────────────────────────
Total:        ~1.6 TB  ❌ Doesn't fit on single GPU!
```

**8× TP** (each GPU):
```
Weights:      140 GB / 8 = 17.5 GB
Activations:  1.2 TB / 1 = 1.2 TB  ❌ Still too large!
Gradients:    17.5 GB
Optimizer:    35 GB
────────────────────────────────────────────
Total:        ~1.27 TB  ❌ Still OOM!
```

**8× TP + SP + Gradient Checkpointing**:
```
Weights:      17.5 GB
Activations:  1.2 TB / 8 / 4 = 37.5 GB  ✅ (SP + checkpointing)
Gradients:    17.5 GB
Optimizer:    35 GB
────────────────────────────────────────────
Total:        ~107.5 GB  ✅ Fits on H100 80GB with headroom!
```

**With MX quantization (FP8)**:
```
Weights:      17.5 GB / 2 = 8.75 GB
Activations:  37.5 GB / 2 = 18.75 GB
Gradients:    17.5 GB / 2 = 8.75 GB
Optimizer:    35 GB (stays FP32)
────────────────────────────────────────────
Total:        ~71.25 GB  ✅ 33% memory savings!
```

---

## Related Documentation

- **Training integration**: [test_mx_linear.md](./test_mx_linear.md) - MXLinear autograd and backward pass
- **GEMM kernels**: [test_mx_mm.md](./test_mx_mm.md) - Hardware-accelerated matmul
- **Quantization details**: [test_mx_tensor.md](./test_mx_tensor.md) - E8M0 scale calculation, FP4/FP6/FP8 conversion
- **Kernel implementations**: [test_kernels.md](./test_kernels.md) - Low-level TRITON and CUDA code

---

*This document provides frame-by-frame execution traces for test_mx_dtensor.py. For complete MX format documentation, see the [README](./README.md).*
