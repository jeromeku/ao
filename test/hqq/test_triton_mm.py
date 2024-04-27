# Skip entire test if triton is not available, otherwise CI failure
import pytest
try:
    import triton
    import hqq
    if int(triton.__version__.split(".")[0]) < 3:
        pytest.skip("triton >= 3.0.0 is required to run this test", allow_module_level=True)
except ImportError:
    pytest.skip("triton and hqq required to run this test", allow_module_level=True)

import itertools
import torch

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
from torchao.prototype.hqq import triton_mixed_mm, pack_2xint4


#Test configs
SHAPES = [
    [16, 128, 128],
    # [16, 4096, 4096],
]

DTYPES = [torch.float16] #[torch.bfloat16, torch.float16]
GROUP_SIZES = [128]
AXES = [1] #Only axis = 1 supported
TRANSPOSED = [False, True]
TRITON_KERNEL_TYPE = ["compute_bound"] #["max_autotune", "compute_bound"]

TEST_CONFIGS = list(itertools.product(SHAPES, GROUP_SIZES, AXES, DTYPES, TRANSPOSED, TRITON_KERNEL_TYPE))

BASE_QUANT_CONFIG = {
    "optimize": True,
    "view_as_float": False,
    "nbits": 4,
    "bitpack": False,
    "axis": 1,
}


def check(expected, actual, msg="", max_diff=1e-3, verbose=False):
    passed = torch.allclose(expected, actual, atol=max_diff, rtol=max_diff)
    if verbose:
        max_err = (expected - actual).abs().max()
        if not passed:
            print(f"{msg}:\nFailed! Max error: {max_err}")
        else:
            print(f"{msg}:\nPassed! Max error: {max_err}")

    return passed

def _arg_to_id(arg):
    if isinstance(arg, list):
        return "x".join([str(x) for x in arg])
    return str(arg)

@pytest.mark.parametrize("shape, group_size, axis, dtype, transposed, kernel_type", TEST_CONFIGS, ids=_arg_to_id)
def test_mixed_mm(shape, group_size, axis, dtype, transposed, kernel_type, quant_dtype=torch.uint8):
    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }
    M, N, K = shape
    
    linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cuda")

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})
    hqq_linear = HQQLinear(linear, quant_config, compute_dtype=dtype, del_orig=False)
    W_q, meta = hqq_linear.W_q, hqq_linear.meta
    W_q = W_q.to(dtype=quant_dtype)
    W_q = (
        W_q.reshape(meta["shape"])
        if quant_config["weight_quant_params"]["bitpack"] == False
        else W_q
    )
    W_dq = hqq_linear.dequantize()

    scales, zeros = meta["scale"], meta["zero"]
    scales = scales.reshape(N, -1)
    zeros = zeros.reshape(N, -1)
    
    packed_w = pack_2xint4(W_q.T)

    if transposed:
        x = torch.randn(M, N, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq         

        #Pack uint8 W_q, then run fused dequant matmul        
        # packed_w = pack_2xint4(W_q)
        # tt_out = triton_mixed_mm(
        #     x, packed_w, scales, zeros, transposed=True, group_size=group_size, fp8_fast_accum=False, kernel_type=kernel_type
        # )
        tt_out = triton_mixed_mm(
            x, packed_w, scales.T, zeros.T, transposed=True, group_size=group_size, fp8_fast_accum=False, kernel_type=kernel_type
        )

    else:
        x = torch.randn(M, K, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq.T    

        # packed_w = pack_2xint4(W_q.T)
        tt_out = triton_mixed_mm(
            x, packed_w, scales.T, zeros.T, transposed=False, group_size=group_size, fp8_fast_accum=False, kernel_type=kernel_type
        )
    assert check(hqq_out, tt_out, max_diff=1e-2 if dtype == torch.bfloat16 else 1e-3)

def _test_mixed_mm(shape, group_size, BLOCK_M, BLOCK_N, BLOCK_K, axis=1, dtype=torch.float16, transposed=True, kernel_type="debug", quant_dtype=torch.uint8):
    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }
    M, N, K = shape


    linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cuda")

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})
    hqq_linear = HQQLinear(linear, quant_config, compute_dtype=dtype, del_orig=False)
    W_q, meta = hqq_linear.W_q, hqq_linear.meta
    # W_q = W_q.to(dtype=quant_dtype)
    # W_q = (
    #     W_q.reshape(meta["shape"])
    #     if quant_config["weight_quant_params"]["bitpack"] == False
    #     else W_q
    # )
    # W_dq = hqq_linear.dequantize()
    W_q = torch.arange(K*N, dtype=quant_dtype, device="cuda").reshape(N, K) % 2 ** 4
    # print(W_q)
    scales = torch.arange(N * K // group_size, dtype=dtype, device="cuda").reshape(N, -1)
    zeros = torch.zeros_like(scales)
    # scales, zeros = meta["scale"], meta["zero"]
    # scales = scales.reshape(N, -1)
    # zeros = zeros.reshape(N, -1)
    W_dq = (W_q - zeros) * scales
    print("W_q: ", W_q.T)
    print(f"zeros: {zeros.T}")
    print(f"scales: {scales.T}")
    
    packed_w = pack_2xint4(W_q.T)

    if transposed:
        x = torch.randn(M, N, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq         

        #Pack uint8 W_q, then run fused dequant matmul        
        # packed_w = pack_2xint4(W_q)
        # tt_out = triton_mixed_mm(
        #     x, packed_w, scales, zeros, transposed=True, group_size=group_size, fp8_fast_accum=False, kernel_type=kernel_type
        # )
        tt_out = triton_mixed_mm(
                x, packed_w, scales.T, zeros.T, 
                transposed=True, 
                group_size=group_size, 
                fp8_fast_accum=False, 
                kernel_type=kernel_type, 
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    else:
        x = torch.randn(M, K, dtype=dtype, device="cuda")
        hqq_out = x @ W_dq.T    

        # packed_w = pack_2xint4(W_q.T)
        tt_out = triton_mixed_mm(
            x, packed_w, scales.T, zeros.T, transposed=False, 
            group_size=group_size, 
            fp8_fast_accum=False, 
            kernel_type=kernel_type, 
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    msg = f"shape={shape}, group_size={group_size}, axis={axis}, dtype={dtype}, transposed={transposed}, kernel_type={kernel_type}, quant_dtype={quant_dtype}"

    check(hqq_out, tt_out, msg=msg, max_diff=1e-2 if dtype == torch.bfloat16 else 1e-3, verbose=True)

if __name__ == "__main__":
    
    # _test_mixed_mm(transposed=False)
    shape = [16, 16, 16]
    BLOCK_M, BLOCK_N, BLOCK_K = shape
    group_size = BLOCK_K
    # _test_mixed_mm(shape, group_size=group_size, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, transposed=False)
    _test_mixed_mm(shape, group_size=group_size, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, transposed=True)