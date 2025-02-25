import torch
import triton
import triton.language as tl


@triton.jit
def _lut(x):
    y = tl.where(x > 0 & x <=1, 2, tl.where(x > 1 & x <= 2, 3, tl.where(x > 2 & x <= 3, 4, 5)))
    return y

@triton.jit
def lut_kernel(x_ptr, y_ptr, N: tl.constexpr):
    load_idx = tl.arange(0, N)
    x = tl.load(x_ptr + load_idx)
    y = tl.where(x < 1, -2.0, tl.where(x < 2, -1.0, tl.where(x < 3, 0.0, 1.0)))
    tl.store(y_ptr + load_idx, y)


x = torch.arange(0, 8, dtype=torch.uint8, device="cuda")
N = x.numel()
y = torch.zeros(N, dtype=torch.float32, device="cuda")

grid = (1,)

print(x)
lut_kernel[grid](x, y, N)

print(y)
