import torch
import triton
from torch.fx.experimental.proxy_tensor import make_fx
from torch.library import triton_op, wrap_triton
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

def print_graph_module(f, *args):
    gm: torch.fx.GraphModule = make_fx(f)(*args)
    gm.print_readable()

#print_graph_module(_add, x, y)
#print_graph_module(wrapped_add, x, y)



def bench_add(compiled=False):
    from triton.testing import do_bench

    add_t = do_bench(lambda: _add(x, y))
    wrapped_add_t = do_bench(lambda: wrapped_add(x, y))

    print(f"add_t: {add_t}")
    print(f"wrapped_add_t: {wrapped_add_t}")

    if compiled:
        compiled_add_t = do_bench(lambda: compiled_add(x, y))
        compiled_wrapped_add_t = do_bench(lambda: compiled_wrapped_add(x, y))
        print(f"compiled_add_t: {compiled_add_t}")
        print(f"compiled_wrapped_add_t: {compiled_wrapped_add_t}")


def profile_add(fn, num_runs=100):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(num_runs):
            fn()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

# profile_add(lambda: _add(x, y))
# profile_add(lambda: wrapped_add(x, y))

# profile_add(lambda: compiled_add(x, y))
# profile_add(lambda: compiled_wrapped_add(x, y))

# bench_add(compiled=False)

from triton_kernels.utils import direct_register_custom_op

library = torch.library.Library("test_lib", "FRAGMENT")
direct_register_custom_op(library, "add", _add)
print(dir(torch.ops.test_lib))
profile_add(lambda: torch.ops.test_lib.add(x, y))
