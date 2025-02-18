import bitsandbytes as bnb
import bitsandbytes.functional as bnb_func
import torch
import torch.nn as nn
import torch.nn.functional as F

bnb_func.dequantize_nf4
from torchao.dtypes.nf4tensor import (
    _INNER_TENSOR_NAMES_FOR_SHARDING,
    NF4Tensor,
    linear_nf4,
    nf4_weight_only,
    to_nf4,
)


class TestMod(nn.Module):
    def __init__(self, tensor, block_size, scaler_block_size):
        super().__init__()
        self.param = torch.nn.Parameter(
            to_nf4(tensor, block_size, scaler_block_size)
        )

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

def test_nf4_bnb_linear(dtype: torch.dtype):
    """
    This test ensures that nf4_linear is "no worse" than BNB by ensuring the
    error compared to a bf16 linear is not more than BNB's implementation.
    """
    torch.manual_seed(0)
    dim = 512
    device = "cuda"
    input_weight = _build_input_weight(dim, device, dtype)
    nf4_weight = to_nf4(input_weight)
    bnb_linear = _build_bnb_linear(input_weight, device)
    breakpoint()

    inp = torch.randn(2, 512, dtype=dtype, device="cuda")

    out_nf4 = linear_nf4(inp, nf4_weight).sum()
    out_bnb = bnb_linear(inp).sum()
    out_ref = F.linear(inp, input_weight).sum()

    err_bnb = (out_bnb - out_ref).abs().max()
    err_native = (out_nf4 - out_ref).abs().max()

def test_reconstruction_qlora_vs_bnb(dtype: torch.dtype):
    # From https://github.com/drisspg/transformer_nuggets/blob/f05afad68ad9086d342268f46a7f344617a02314/test/test_qlora.py#L65C1-L81C47
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
    # Since we are subtle different we assume 

if __name__ == "__main__":

    test_nf4_bnb_linear(torch.bfloat16)
    # dtype = torch.bfloat16
    # with torch.device("meta"):
    #     input_tensor = torch.randn(128, 128, dtype=dtype)
    #     base_mod = TestMod(input_tensor, block_size=64, scaler_block_size=256)
    
    # print(base_mod.param, base_mod.param.dtype, base_mod.param.device)
    # breakpoint()

    # sd = {'param': torch.zeros(128, 128, dtype=dtype, device="cuda")}
    # base_mod.load_state_dict(sd)
    # print(base_mod.param, base_mod.param.dtype, base_mod.param.device)


