import bitsandbytes as bnb
import torch
import triton
import triton.language as tl
from bitsandbytes.functional import (
    create_dynamic_map,
    dequantize_blockwise,
    dequantize_nf4,
)

from torchao.dtypes.nf4tensor import to_nf4

BLOCK_SIZE = 64
NESTED_BLOCK_SIZE = 256
SEED = 0

#https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/csrc/kernels.cu#L25
NF4_CODE = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


def create_input_weight(embed_dim, device, dtype):
    input_weight = torch.empty(embed_dim, embed_dim, device=device, dtype=dtype)
    input_weight.normal_(0, 1)
    return input_weight


def lookup_scaler_code(quantized_scalers, code):
    assert code.ndim == 1
    return code.index_select(0, quantized_scalers.view(-1).to(torch.long))

def interleave(t1, t2):
    assert t1.shape == t2.shape
    assert t1.ndim == 1
    return torch.vstack([t1, t2]).permute(1, 0).contiguous().view(t1.numel() * 2)

def test_bnb_dequant():
    torch.manual_seed(SEED)
    dim = 512
    device = "cuda"
    dtype = torch.bfloat16
    input_weight = create_input_weight(dim, device, dtype)

    block_code = create_dynamic_map().to(device)
    param = bnb.nn.Params4bit(
        input_weight, requires_grad=False, quant_type="nf4"
    ).cuda()
    qstate = param.quant_state
    nested_qstate = param.quant_state.state2

    quantized_data = param
    quantized_scalers = qstate.absmax
    nested_scale_factors = nested_qstate.absmax
    quantized_scaler_mean = qstate.offset

    # First test scaler dequantization
    ref = dequantize_blockwise(quantized_scalers, nested_qstate)
    ref += quantized_scaler_mean

    # Dequantize manually
    # 1. Lookup the scaler code
    decoded = lookup_scaler_code(quantized_scalers, block_code)
    # 2. Apply the scaler
    scalers = decoded.reshape(-1, NESTED_BLOCK_SIZE) * nested_scale_factors[:, None]
    # 3. Apply the offset
    scalers += quantized_scaler_mean
    scalers = scalers.reshape(-1, 1)

    if not torch.allclose(ref, scalers.view(-1)):
        print("ref: ", ref.view(-1)[:5])
        print("scaled: ", scalers.view(-1)[:5])
        raise ValueError("Scaler dequantization failed")

    first_elements = quantized_data >> 4
    second_elements = quantized_data & 0xF

    nf4_code = torch.tensor(NF4_CODE, device=device, dtype=torch.float32)
    decoded_first_elements = lookup_scaler_code(first_elements, nf4_code)
    decoded_second_elements = lookup_scaler_code(second_elements, nf4_code)
    interleaved = interleave(decoded_first_elements, decoded_second_elements)
    dq = interleaved.reshape(-1, BLOCK_SIZE) * scalers
    dq = dq.to(dtype).reshape(input_weight.shape)
    
    ref_dq = dequantize_nf4(quantized_data, qstate)
    print(ref_dq.shape, ref_dq.dtype, ref_dq.view(-1)[:5])
    print(dq.shape, dq.dtype, dq.view(-1)[:5])
    


@triton.jit
def lookup_code(q, code_ptr):
    # Need to cast to use as pointer index
    q = q.to(tl.int32)
    return tl.load(code_ptr + q)

@triton.jit
def lookup_kernel(q_ptr, code_ptr, out_ptr, BLOCK_X: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_blocks = tl.num_programs(axis=0)

    load_idx = pid * BLOCK_X + tl.arange(0, BLOCK_X)
    qs = tl.load(q_ptr + load_idx)
    out = lookup_code(qs, code_ptr)
    tl.store(out_ptr + load_idx, out)
 
# Launch a 1-D grid 
# let QBLOCKS = number fo quantized blocks = len(quantized_data) // QBLOCK_SIZE
# let QBLOCKS_PER_CTA = number of quantized blocks to process per CTA
# Simple case is QBLOCKS_PER_CTA = 1
# for QBLOCK_SIZE = 64 -> must output 64 dequantized elements per CTA
# NOTE: since we have 2 elements per each quantized data element, we need to load only 32 elements per CTA
# each block processes ELEMENTS_PER_BLOCK elements and hence must load ELEMENTS_PER_BLOCK
# Each quantized block corresponds to 1 quantized absmax and len(quantized_absmax) // QBLOCK_SIZE qabs_max_scalers

@triton.jit
def dequant_nf4_kernel(
    # Quantized data
    q_ptr, 
    # NF4 code
    nf4_code_ptr, 
    # Compressed statistics
    # Quantized absmax
    qabsmax_ptr,
    # Scale factors for quantized absmax
    qabsmax_scalers_ptr,
    # Offset for quantized absmax
    qabsmax_mean_ptr,
    # Double quant block code
    block_code_ptr,
    # Output
    dq_ptr,
    # Quantized data block size
    QBLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    # Nested block size
    NESTED_QBLOCK_SIZE: tl.constexpr = NESTED_BLOCK_SIZE,
    QBLOCKS_PER_CTA: tl.constexpr = 1):
    
    output_elements_per_qblock: tl.constexpr = QBLOCKS_PER_CTA * QBLOCK_SIZE
    input_elements_per_qblock: tl.constexpr = output_elements_per_qblock // 2

    # Load quantized data
    block_idx = tl.program_id(axis=0)
    offset = block_idx * QBLOCKS_PER_CTA * input_elements_per_qblock
    q_load_idx = offset + tl.arange(0, input_elements_per_qblock)
    q = tl.load(q_ptr + q_load_idx)
    q_first_elements = q >> 4
    q_second_elements = q & 0xF
    dq_first_elements = lookup_code(q_first_elements, nf4_code_ptr)
    dq_second_elements = lookup_code(q_second_elements, nf4_code_ptr)
    interleaved = tl.interleave(dq_first_elements, dq_second_elements)
    store_idx = block_idx * output_elements_per_qblock + tl.arange(0, output_elements_per_qblock)
    tl.store(dq_ptr + store_idx, interleaved)

def test_triton_lookup():
    torch.manual_seed(SEED)
    dim = 512
    device = "cuda"
    dtype = torch.bfloat16
    input_weight = create_input_weight(dim, device, dtype)

    block_code = create_dynamic_map().to(device)
    param = bnb.nn.Params4bit(
        input_weight, requires_grad=False, quant_type="nf4"
    ).cuda()
    qstate = param.quant_state
    nested_qstate = param.quant_state.state2

    quantized_data = param
    quantized_scalers = qstate.absmax
    nested_scale_factors = nested_qstate.absmax
    quantized_scaler_mean = qstate.offset

    decoded = lookup_scaler_code(quantized_scalers, block_code)
    print(decoded.shape, decoded.dtype, decoded.view(-1)[:5])

    out = torch.empty_like(decoded)
    total_elements = quantized_scalers.numel()
    BLOCK_X = total_elements
    grid = (triton.cdiv(total_elements, BLOCK_X),)

    lookup_kernel[grid](quantized_scalers, block_code, out, BLOCK_X=BLOCK_X)
    print(out.shape, out.dtype, out.view(-1)[:5])
    if not torch.allclose(out, decoded):
        diff = (out - decoded).abs().max()
        print(f"diff: {diff}")

def create_nf4_code(device):
    return torch.tensor(NF4_CODE, device=device, dtype=torch.float32)

def get_ref_interleaved(quantized_data, nf4_code):
    first_elements = quantized_data >> 4
    second_elements = quantized_data & 0xF
    decoded_first_elements = lookup_scaler_code(first_elements, nf4_code)
    decoded_second_elements = lookup_scaler_code(second_elements, nf4_code)
    interleaved = interleave(decoded_first_elements, decoded_second_elements)
    return interleaved

def test_triton_dequant():
    torch.manual_seed(SEED)
    dim = 512
    device = "cuda"
    dtype = torch.bfloat16
    input_weight = create_input_weight(dim, device, dtype)

    param = bnb.nn.Params4bit(
        input_weight, requires_grad=False, quant_type="nf4"
    ).cuda()
    qstate = param.quant_state
    nested_qstate = param.quant_state.state2

    quantized_data = param
    quantized_scalers = qstate.absmax
    nested_scale_factors = nested_qstate.absmax
    quantized_scaler_mean = qstate.offset

    block_code = create_dynamic_map().to(device)
    nf4_code = create_nf4_code(device)

    grid = lambda meta: (triton.cdiv(quantized_data.numel(), meta['QBLOCKS_PER_CTA'] * meta['QBLOCK_SIZE']),)
    dq = torch.empty_like(input_weight)
    ref_interleaved = get_ref_interleaved(quantized_data, nf4_code).reshape(input_weight.shape).to(input_weight.dtype)

    dequant_nf4_kernel[grid](
        q_ptr=quantized_data, 
        nf4_code_ptr=nf4_code, 
        qabsmax_ptr=quantized_scalers, 
        qabsmax_scalers_ptr=nested_scale_factors, 
        qabsmax_mean_ptr=quantized_scaler_mean, 
        block_code_ptr=block_code, 
        dq_ptr=dq,
        QBLOCK_SIZE=BLOCK_SIZE, 
        NESTED_QBLOCK_SIZE=NESTED_BLOCK_SIZE,
        QBLOCKS_PER_CTA=1)

    if not torch.allclose(dq, ref_interleaved):
        for i in range(dq.numel()):
            if dq.view(-1)[i] != ref_interleaved.view(-1)[i]:
                print(f"diff at index {i}: {dq.view(-1)[i]} vs {ref_interleaved.view(-1)[i]}")
test_triton_dequant()
