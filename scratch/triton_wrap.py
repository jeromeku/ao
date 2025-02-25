import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.library import triton_op, wrap_triton

import triton
from triton import language as tl


@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # NB: we need to wrap the triton kernel in a call to wrap_triton
    wrap_triton(add_kernel)[grid](x, y, output, n_elements, 16)
    return output

wrapped_add = triton_op("mylib::add", mutates_args={})(_add)
compiled_add = torch.compile(_add)
compiled_wrapped_add = torch.compile(wrapped_add)

x = torch.randn(3, device="cuda")
y = torch.randn(3, device="cuda")

z = compiled_add(x, y)
assert torch.allclose(z, x + y), "compiled_add failed"

z = compiled_wrapped_add(x, y)
assert torch.allclose(z, x + y), "compiled_wrapped_add failed"

print("add graph module")
gm: torch.fx.GraphModule = make_fx(_add)(x, y)
gm.print_readable()
print("wrapped add graph module")
gm_wrapped: torch.fx.GraphModule = make_fx(wrapped_add)(x, y)
gm_wrapped.print_readable()

from triton.testing import do_bench

add_t = do_bench(lambda: _add(x, y))
wrapped_add_t = do_bench(lambda: wrapped_add(x, y))
print(f"add_t: {add_t}")
print(f"wrapped_add_t: {wrapped_add_t}")
compiled_add_t = do_bench(lambda: compiled_add(x, y))
compiled_wrapped_add_t = do_bench(lambda: compiled_wrapped_add(x, y))
print(f"compiled_add_t: {compiled_add_t}")
print(f"compiled_wrapped_add_t: {compiled_wrapped_add_t}")
