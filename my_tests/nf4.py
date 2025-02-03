import bitsandbytes as bnb
import torch

from torchao.dtypes.nf4tensor import to_nf4


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

def main(dtype: torch.dtype = torch.bfloat16):
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

if __name__ == "__main__":
    main()
