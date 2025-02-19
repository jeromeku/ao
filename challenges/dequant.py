import math

import bitsandbytes as bnb
import torch
import triton
import triton.language as tl

#import unsloth
from bitsandbytes.functional import (
    create_dynamic_map,
    dequantize_blockwise,
    dequantize_nf4,
)

from torchao.dtypes.nf4tensor import to_nf4

#from unsloth.kernels.utils import fast_dequantize


DEVICE = "cuda"

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


# Reference implementation
def lookup_scaler_code(quantized_scalers, code):
    assert code.ndim == 1
    return code.index_select(0, quantized_scalers.view(-1).to(torch.long))

def interleave(t1, t2):
    assert t1.shape == t2.shape
    assert t1.ndim == 1
    return torch.vstack([t1, t2]).permute(1, 0).contiguous().view(t1.numel() * 2)

# Triton kernels
@triton.jit
def lut_device_kernel(q, code_ptr):
    # Need to cast to use as pointer index
    q = q.to(tl.int32)
    return tl.load(code_ptr + q)

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
    # For debugging only
    interleaved_ptr,
    dqabsmax_ptr,
    dqabsmax_scalers_ptr,
    # Quantized data block size
    QBLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    # Nested block size
    NESTED_QBLOCK_SIZE: tl.constexpr = NESTED_BLOCK_SIZE,
    QBLOCKS_PER_CTA: tl.constexpr = 1,
    NUM_PACKED_ELEMENTS: tl.constexpr = 2,
    OUTPUT_DTYPE: tl.constexpr = tl.bfloat16,
    DEBUG: tl.constexpr = False):
    
    OUTPUT_ELEMENTS_PER_QBLOCK: tl.constexpr = QBLOCKS_PER_CTA * QBLOCK_SIZE
    # Divide by NUM_PACKED_ELEMENTS to account for packed elements
    # Default quant storage is uint 8 => 8 bits // 4 bits => 2 packed elements

    tl.static_assert(OUTPUT_ELEMENTS_PER_QBLOCK % NUM_PACKED_ELEMENTS == 0, "OUTPUT_ELEMENTS_PER_QBLOCK must be divisible by NUM_PACKED_ELEMENTS")
    INPUT_ELEMENTS_PER_QBLOCK: tl.constexpr = OUTPUT_ELEMENTS_PER_QBLOCK // NUM_PACKED_ELEMENTS

    # Load quantized data
    block_offset = tl.program_id(axis=0) 
    input_offset = block_offset * INPUT_ELEMENTS_PER_QBLOCK
    output_offset = block_offset * OUTPUT_ELEMENTS_PER_QBLOCK
    #offset = block_idx * QBLOCKS_PER_CTA * INPUT_ELEMENTS_PER_QBLOCK
    q_load_idx = input_offset + tl.arange(0, INPUT_ELEMENTS_PER_QBLOCK)
    q_load_idx = tl.max_contiguous(tl.multiple_of(q_load_idx, INPUT_ELEMENTS_PER_QBLOCK), INPUT_ELEMENTS_PER_QBLOCK)

    q = tl.load(q_ptr + q_load_idx)
    q_first_elements = q >> 4
    q_second_elements = q & 0xF
    dq_first_elements = lut_device_kernel(q_first_elements, nf4_code_ptr)
    dq_second_elements = lut_device_kernel(q_second_elements, nf4_code_ptr)
    interleaved = tl.interleave(dq_first_elements, dq_second_elements)

    # Load qabsmax
    QABSMAX_ELEMENTS_TO_LOAD: tl.constexpr = QBLOCKS_PER_CTA
    # Previously block_offset = block_idx = tl.program_id(axis=0)
    qabsmax_offset = block_offset * QABSMAX_ELEMENTS_TO_LOAD
    qabsmax_load_idx = qabsmax_offset + tl.arange(0, QABSMAX_ELEMENTS_TO_LOAD)
    qabsmax_load_idx = tl.max_contiguous(tl.multiple_of(qabsmax_load_idx, QABSMAX_ELEMENTS_TO_LOAD), QABSMAX_ELEMENTS_TO_LOAD)
    qabsmax = tl.load(qabsmax_ptr + qabsmax_load_idx)
    dqabsmax = lut_device_kernel(qabsmax, block_code_ptr)
   
    qabsmax_scalers_offset = qabsmax_offset // NESTED_QBLOCK_SIZE
    NUM_QABSMAX_SCALERS: tl.constexpr = (QBLOCKS_PER_CTA + NESTED_QBLOCK_SIZE - 1) // NESTED_QBLOCK_SIZE
    tl.static_assert(NUM_QABSMAX_SCALERS == 1, "NUM_QABSMAX_SCALERS must be 1")
    dqabsmax_scalers = tl.load(qabsmax_scalers_ptr + qabsmax_scalers_offset)
    qabsmax_mean = tl.load(qabsmax_mean_ptr)

    # Dequantize qabsmax
    dqabsmax = dqabsmax * dqabsmax_scalers + qabsmax_mean
    tl.store(dqabsmax_ptr + qabsmax_load_idx, dqabsmax)

    interleaved = tl.reshape(interleaved, (QBLOCKS_PER_CTA, QBLOCK_SIZE))
    dqabsmax = tl.reshape(dqabsmax, (QBLOCKS_PER_CTA, 1))
        
    dq = interleaved * dqabsmax
    dq = dq.to(OUTPUT_DTYPE)

    if DEBUG:
        tl.static_print("dqabsmax", dqabsmax)
        tl.static_print("dqabsmax_scalers", dqabsmax_scalers)
        tl.static_print("interleaved", interleaved)
        tl.static_print("dqabsmax", dqabsmax)
        tl.static_print("dq", dq)

    store_idx = output_offset + tl.arange(0, OUTPUT_ELEMENTS_PER_QBLOCK)
    tl.store(dq_ptr + store_idx, tl.reshape(dq, OUTPUT_ELEMENTS_PER_QBLOCK))
    tl.store(interleaved_ptr + store_idx, tl.reshape(interleaved, OUTPUT_ELEMENTS_PER_QBLOCK))

def create_nf4_code(device):
    return torch.tensor(NF4_CODE, device=device, dtype=torch.float32)

def get_ref_interleaved(quantized_data, nf4_code):
    first_elements = quantized_data >> 4
    second_elements = quantized_data & 0xF
    decoded_first_elements = lookup_scaler_code(first_elements, nf4_code)
    decoded_second_elements = lookup_scaler_code(second_elements, nf4_code)
    interleaved = interleave(decoded_first_elements, decoded_second_elements)
    return interleaved

def torch_to_triton_dtype(dtype):
    parts = str(dtype).split(".")
    assert len(parts) == 2, f"dtype {dtype} is not a valid triton dtype, {parts}"
    dtype_str = parts[-1]
    tt_dtype = getattr(tl, dtype_str)
    return tt_dtype

def create_qparam(input_weight, quant_type="nf4", quant_storage=torch.uint8, compress_statistics=True):
    param = bnb.nn.Params4bit(
        input_weight, requires_grad=False, quant_type=quant_type, quant_storage=quant_storage, compress_statistics=compress_statistics
    ).cuda()
    return param

def test_triton_dequant(shape, dtype, qblocks_per_cta, quant_type="nf4", quant_storage=torch.uint8, compress_statistics=True):
    torch.manual_seed(SEED)
    device = "cuda"

    input_weight = torch.randn(shape, device=device, dtype=dtype)
    param = create_qparam(input_weight, quant_type=quant_type, quant_storage=quant_storage, compress_statistics=compress_statistics)

    qstate = param.quant_state
    nested_qstate = param.quant_state.state2

    quantized_data = param.data
    quantized_scalers = qstate.absmax
    nested_scale_factors = nested_qstate.absmax
    quantized_scaler_mean = qstate.offset

    block_code = create_dynamic_map().to(device)
    nf4_code = create_nf4_code(device)

    dq = torch.empty_like(input_weight)
    ref_interleaved = get_ref_interleaved(quantized_data, nf4_code) #.reshape(input_weight.shape).to(input_weight.dtype)

#    ref_dqabsmax = lookup_scaler_code(quantized_scalers, block_code)
    decoded = lookup_scaler_code(quantized_scalers, block_code)
    # 2. Apply the scaler
    if decoded.numel() < NESTED_BLOCK_SIZE:
        scalers = decoded * nested_scale_factors
    else:
        scalers = decoded.reshape(-1, NESTED_BLOCK_SIZE) * nested_scale_factors[:, None]
    # 3. Apply the offset
    scalers += quantized_scaler_mean
    ref_dqabsmax = scalers.reshape(-1, 1)
    ref_dq = ref_interleaved.reshape(-1, BLOCK_SIZE) * ref_dqabsmax
    ref_dq = ref_dq.to(input_weight.dtype).reshape(input_weight.shape)

    dqabsmax = torch.empty_like(ref_dqabsmax)
    dqabsmax_scalers = torch.empty_like(nested_scale_factors)
    interleaved = torch.empty_like(ref_interleaved)
    
    print("qblocks_per_cta", qblocks_per_cta)
    grid = lambda meta: (triton.cdiv(input_weight.numel(), meta['QBLOCKS_PER_CTA'] * meta['QBLOCK_SIZE']),)
    print("grid", grid(meta={'QBLOCKS_PER_CTA': qblocks_per_cta, 'QBLOCK_SIZE': BLOCK_SIZE}))
    total_blocks = input_weight.numel() // BLOCK_SIZE
    print("total_blocks", total_blocks)
    num_packed_elements = quantized_data.element_size() * 8 // 4
    print("num_packed_elements", num_packed_elements)
    output_dtype = torch_to_triton_dtype(qstate.dtype)
    print("output_dtype", output_dtype)
    reconstructed_dq = dequantize_nf4(quantized_data, qstate)

    dequant_nf4_kernel[grid](
        q_ptr=quantized_data, 
        nf4_code_ptr=nf4_code, 
        qabsmax_ptr=quantized_scalers, 
        qabsmax_scalers_ptr=nested_scale_factors, 
        qabsmax_mean_ptr=quantized_scaler_mean, 
        block_code_ptr=block_code, 
        dq_ptr=dq,
        interleaved_ptr=interleaved,
        dqabsmax_ptr=dqabsmax,
        dqabsmax_scalers_ptr=dqabsmax_scalers,
        QBLOCK_SIZE=BLOCK_SIZE, 
        NESTED_QBLOCK_SIZE=NESTED_BLOCK_SIZE,
        QBLOCKS_PER_CTA=qblocks_per_cta,
        NUM_PACKED_ELEMENTS=num_packed_elements,
        OUTPUT_DTYPE=output_dtype)

    if not torch.allclose(dq, ref_dq):
        print("dqabsmax", torch.allclose(dqabsmax, ref_dqabsmax))
        print("reconstructed_dq", torch.allclose(reconstructed_dq, ref_dq))
        print("interleaved", torch.allclose(interleaved, ref_interleaved))
        raise ValueError(f"Dequantization failed for qblocks_per_cta {qblocks_per_cta}")
    else:
        # Insert unicode checkmark
        print(f"\u2713 Dequantization passed for qblocks_per_cta {qblocks_per_cta}\n")

def test_fast_dequant(shape, dtype, quant_type="nf4", quant_storage=torch.uint8, compress_statistics=True):
    from unsloth.kernels.utils import fast_dequantize
    input_weight = torch.randn(shape, device=DEVICE, dtype=dtype)
    qparam = create_qparam(input_weight, quant_type=quant_type, quant_storage=quant_storage, compress_statistics=compress_statistics)
    dq = fast_dequantize(input_weight)
    dq_ref = dequantize_nf4(qparam, quant_state=qparam.quant_state)
    print("dq", torch.allclose(dq, dq_ref))

# def benchmark_dequant(shape, dtype):
#     input_weight = torch.randn(shape, device=DEVICE, dtype=dtype)
#     fast_dequantize(input_weight, qstate)

def triton_dequant_nf4(qparam: bnb.nn.Params4bit, qblocks_per_cta=1):
    qstate = qparam.quant_state
    nested_qstate = qparam.quant_state.state2

    quantized_data = qparam.data
    quantized_scalers = qstate.absmax
    nested_scale_factors = nested_qstate.absmax
    quantized_scaler_mean = qstate.offset

    nf4_code = qstate.code
    block_code = qstate.state2.code
    breakpoint()

if __name__ == "__main__":
    torch.manual_seed(SEED)
    
    MAX_BLOCKS_PER_CTA = int(math.log2(NESTED_BLOCK_SIZE))
    shape = (512, 512) # (4096, 4096), (4096, 14336)
    dtype = torch.bfloat16
    qblocks_per_cta = 1
    input_weight = torch.randn(shape, device=DEVICE, dtype=dtype)
    qparam = create_qparam(input_weight, quant_type="nf4", quant_storage=torch.uint8, compress_statistics=True)
    triton_dequant_nf4(qparam=qparam, qblocks_per_cta=qblocks_per_cta)
    # test_fast_dequant(shape=shape, dtype=dtype)
    # q_blocks_per_cta = [2 ** p for p in range(0, MAX_BLOCKS_PER_CTA + 1)]
    # for dtype in [torch.bfloat16, torch.float16]:
    #     for qblocks_per_cta in q_blocks_per_cta:
    #         test_triton_dequant(shape=shape, dtype=dtype, qblocks_per_cta=qblocks_per_cta)

#TODO
# optimizations: max_contiguous, eviction policies, static on-device LUT using tuple see (https://github.com/triton-lang/triton/issues/5864)