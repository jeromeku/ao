import itertools

import torch
from termcolor import colored

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
from hqq.kernels.custom_quant.triton import triton_mixed_mm, pack_2xint4
from torchao.prototype.hqq import triton_mixed_mm, pack_2xint4
from torchao.prototype.hqq.hqq_tinygemm_linear import HQQLinearTorchWeightOnlyInt4


#TODO: refactor to pytest

#Test configs
SHAPES = [
    # [16, 128],
    [16, 128, 128],
    [16, 4096, 4096],
    # [1024, 4096],
    # [4096, 4096],
    # [4096, 11008],
]

DTYPES = [torch.bfloat16, torch.float16]
GROUP_SIZES = [64, 128]
AXES = [1] #Only axis = 1 supported
TRITON_KERNEL_TYPE = ["compute_bound"] #["max_autotune", "compute_bound"]
TEST_CONFIGS = list(itertools.product(SHAPES, GROUP_SIZES, AXES, DTYPES, TRITON_KERNEL_TYPE))

BASE_QUANT_CONFIG = {
    "optimize": True,
    "view_as_float": False,
    "nbits": 4,
    # "quant_dtype": torch.uint8,
    "bitpack": False,
    "axis": 1,
}


def check(expected, actual, cfg_str, max_diff=1e-3):
    passed = torch.allclose(expected, actual, atol=max_diff, rtol=max_diff)
    max_err = (expected - actual).abs().max()
    if not passed:
        print(colored(f"{cfg_str}: Failed! Max error: {max_err}", "red", attrs=["bold"]))
    else:
        print(colored(f"{cfg_str}: Passed! Max error: {max_err}", "green", attrs=["bold"]))

def test_mixed_mm(shape, group_size, axis, dtype, kernel_type, quant_dtype=torch.uint8):
    # print(f"Test: {shape}, {group_size}, {axis}, {dtype}")
    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }
    M, N, K = shape

    x = torch.randn(M, K, dtype=dtype, device="cuda")
    linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cuda")

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})
    hqq_linear = HQQLinear(linear, quant_config, compute_dtype=dtype, del_orig=False)
    W_q, meta = hqq_linear.W_q, hqq_linear.meta
    W_q = (
        W_q.reshape(meta["shape"])
        if quant_config["weight_quant_params"]["bitpack"] == False
        else W_q
    )
    scales, zeros = meta["scale"], meta["zero"]
    
    #Reference
    hqq_out = hqq_linear.forward(x)
    
    ##Triton
    W_q = W_q.to(dtype=quant_dtype)
    packed_w = pack_2xint4(W_q.T)
    scales = scales.reshape(N, -1)
    zeros = zeros.reshape(N, -1)
    tt_out = triton_mixed_mm(
        x, packed_w, scales.T, zeros.T, group_size=group_size, fp8_fast_accum=False, kernel_type=kernel_type
    )

    cfg_str = f"Test config {shape} {group_size} {dtype}"
    # err = (hqq_out - tt_out).abs().max()
    check(hqq_out, tt_out, cfg_str + " triton", max_diff=1e-2 if dtype == torch.bfloat16 else 1e-3)

    if dtype == torch.bfloat16:
        _ = quant_config["weight_quant_params"].pop("bitpack")
        hqq_int4mm = HQQLinearTorchWeightOnlyInt4(
            linear, quant_config, compute_dtype=dtype, del_orig=False
        )
        hqq_int4_out = hqq_int4mm.forward(x)
        err = (hqq_int4_out - hqq_out).abs().max()
        check(hqq_out, hqq_int4_out, cfg_str + " torch_tinygemm", max_diff=1e-2)

    print()


for test in TEST_CONFIGS:
    test_mixed_mm(*test)