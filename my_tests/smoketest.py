import torch

import torchao


def test_smoketest_linear(dtype: torch.dtype, use_nf4: bool=False):
    weight = torch.randn(32, 32, dtype=dtype, device="cuda")
    weight = torchao.dtypes.to_nf4(weight, 16, 2) if use_nf4 else weight
    inp = torch.randn(2, 32, 32, dtype=weight.dtype, device=weight.device)
    linear = torch.compile(torch.nn.functional.linear)(inp, weight)
  #  linear_nf4 = torch.compile(torch.nn.functional.linear)(inp, weight_nf4)
    # _ = torch.nn.functional.linear(inp, a)
    # _ = torch.nn.functional.linear(inp, a_nf4)


test_smoketest_linear(torch.bfloat16, use_nf4=True)