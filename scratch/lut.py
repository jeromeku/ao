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
    y = tl.where(x > 0, tl.where(x > 1, tl.where(x > 2, 4, 5), 3), 2)
    tl.store(y_ptr + load_idx, y)


x = torch.arange(0, 8, dtype=torch.uint8, device="cuda")
y = torch.empty_like(x, dtype=torch.uint8, device="cuda")

grid = (1,)
N = x.numel()
print(x)
lut_kernel[grid](x, y, N)

print(y)
