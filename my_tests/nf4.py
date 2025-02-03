import logging

logging.basicConfig(level=logging.DEBUG)
import bitsandbytes as bnb
import torch
from torch._inductor.utils import do_bench_using_profiling

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4


def _build_input_weight(embed_dim: int, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(0)
    input_weight = torch.empty(embed_dim, embed_dim, device=device, dtype=dtype)
    input_weight.normal_(0, 1)
    return input_weight


def _build_bnb_linear(input_weight, device):
    param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(
        device
    )
    bnb_linear = bnb.nn.LinearNF4(
        input_weight.size(0), input_weight.size(1), bias=False
    )
    bnb_linear.weight = param
    bnb_linear.to(device)
    return bnb_linear


def check_bnb_linear(dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(0)
    device = "cuda"
    embed_dim = 512
    input_weight = _build_input_weight(embed_dim, device, dtype)
    nf4_weight = to_nf4(input_weight)
    bnb_linear = _build_bnb_linear(input_weight, device)
    bnb_reconstruction = bnb_linear(
        torch.eye(embed_dim, embed_dim, dtype=dtype, device=device)
    )
    bnb_diff = (bnb_reconstruction.T - input_weight).abs().max()
    nugs_diff = (nf4_weight.get_original_weight() - input_weight).abs().max()
    # Since we are subtle different we assume that we both reconstruct with
    # a similar precision
    assert bnb_diff < 1
    assert nugs_diff < 1
    assert (nugs_diff - bnb_diff).abs() < 2e-1

def bench_bnb_linear(input, weight):
    bnb_linear = _build_bnb_linear(weight, device="cuda")
    return do_bench_using_profiling(lambda: bnb_linear(input))

def bench_linear_nf4(input, weight, compile=False, **compile_kwargs):
    nf4_weight = to_nf4(weight)
    if compile:
        linear_fn = torch.compile(linear_nf4, **compile_kwargs)
    else:
        linear_fn = linear_nf4
    return do_bench_using_profiling(lambda: linear_fn(input, nf4_weight))

def benchmark_bnb_linear(input_dim: int, embed_dim: int, dtype: torch.dtype = torch.bfloat16):
    input_weight = _build_input_weight(embed_dim, device="cuda", dtype=dtype)
    x = torch.randn(input_dim, embed_dim, dtype=dtype, device="cuda")

    # bnb_linear_time = bench_bnb_linear(x, input_weight)
    # print(f"bnb linear time: {bnb_linear_time} ms")
    nf4_linear_time = bench_linear_nf4(x, input_weight)
    print(f"nf4 linear time: {nf4_linear_time} ms")
    nf4_linear_time_compiled = bench_linear_nf4(x, input_weight, compile=True)
    print(f"nf4 linear time (compiled): {nf4_linear_time_compiled} ms")

if __name__ == "__main__":
    input_dim, embed_dim = 2, 512
    benchmark_bnb_linear(input_dim, embed_dim)