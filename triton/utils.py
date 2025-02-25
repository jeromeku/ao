import bitsandbytes as bnb
import torch
from bitsandbytes.functional import create_dynamic_map, create_normal_map


def create_qparam(
    input_weight, quant_type="nf4", quant_storage=torch.uint8, compress_statistics=True
) -> bnb.nn.Params4bit:
    param = bnb.nn.Params4bit(
        input_weight,
        requires_grad=False,
        quant_type=quant_type,
        quant_storage=quant_storage,
        compress_statistics=compress_statistics,
    ).cuda()
    return param


# From https://github.com/bitsandbytes-foundation/bitsandbytes/blob/5d468883cac85192c903cdce71a626317b241d70/csrc/kernels.cu#L25C40-L25C331
NF4_DATA = [
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

def get_nf4_codebook(use_hardcoded=True, device="cpu", dtype=torch.float32):
    """
    Returns:
        torch.Tensor: A tensor of shape (16,) containing bitsnbytes nf4 codebook values.
    """
    if use_hardcoded:
        return torch.tensor(NF4_DATA, device=device, dtype=dtype)
    else:
        normal_map = create_normal_map()
        nf4_map = torch.cat([normal_map[:8], normal_map[-8:]]).to(device=device, dtype=dtype)
        return nf4_map

if __name__ == "__main__":
    torch.set_printoptions(precision=8)

    nf4_map_functional = get_nf4_codebook(use_hardcoded=False)
    nf4_map_hardcoded = get_nf4_codebook(use_hardcoded=True)
    assert torch.equal(nf4_map_functional, nf4_map_hardcoded), "Functional and hardcoded NF4 maps do not match"