# %%
import bitsandbytes as bnb
import bitsandbytes.functional as bnb_func
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
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



# %%
torch.manual_seed(0)
dim = 512
device = "cuda"
dtype = torch.bfloat16
input_weight = _build_input_weight(dim, device, dtype)
param = bnb.nn.Params4bit(input_weight, requires_grad=False, quant_type="nf4").cuda(
    device)
breakpoint()
# %%
bnb_func.dequantize_nf4(param, quant_state=param.quant_state)
# %%
