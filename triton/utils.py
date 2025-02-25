import sys
import textwrap

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import create_dynamic_map, create_normal_map

import triton
import triton.language as tl


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

# def _generate_lookup_table(lookup_values):
#     """
#     Given a list of lookup values, generate a nested tl.where
#     statement for a lookup table. The generated statement will be of the form:

#       y = tl.where(x < 1, v1, tl.where(x < 2, v2, tl.where(x < 3, v3, ... vN)))

#     where each value corresponds to the threshold condition:
#       - If x < 1, then use lookup_values[0] (v1)
#       - Else if x < 2, then use lookup_values[1] (v2)
#       - ...
#       - Else use the final value lookup_values[-1] (vN)
    
#     Parameters:
#       lookup_values (list of str): A list of lookup value strings.

#     Returns:
#       str: A string representing the nested tl.where statement.
#     """
#     # Start with the default value (the last value in the list).
#     expr = lookup_values[-1]
#     # Build the nested expression from the end toward the beginning.
#     for i in range(len(lookup_values) - 1, 0, -1):
#         # For each i, we use threshold i and value lookup_values[i-1].
#         expr = f"tl.where(x < {i}, {lookup_values[i-1]}, {expr})"
    
#     return expr

def _generate_lookup_table(lookup_values):
    """
    Given a list of lookup values, generate a nested, indented tl.where statement.
    
    The generated string will have the form:
    
        y = tl.where(
                x < 1,
                v1,
                tl.where(
                    x < 2,
                    v2,
                    tl.where(
                        x < 3,
                        v3,
                        ...,
                    )
                )
            )
    
    Implements a lookup table by mapping an uint8 array of values to a float32 array of values:
    E.g., if lookup_values = [-1, -.5, 0, .5], then:
    0 -> -1
    1 -> -0.5
    2 -> 0
    3 -> 0.5
    
    Parameters:
      lookup_values (list): List of lookup values (as strings or numbers).
      
    Returns:
      str: A multi-line, indented string of nested tl.where statements.
    
    Assumption is that lookup_values is a sorted list of floats and that the values to be mapped are uint8.
    """
    def rec(i, N, indent_level):
        indent = " " * (4 * indent_level)
        # Base case: when i == N, return the fallback value (last element).
        if i == N:
            return indent + str(lookup_values[-1])
        else:
            # Build the tl.where call:
            #   condition: x < i
            #   true branch: lookup_values[i-1]
            #   false branch: nested tl.where (for i+1)
            s = indent + "tl.where(\n"
            s += indent + "    x < " + str(i) + ",\n"
            s += indent + "    " + str(lookup_values[i-1]) + ",\n"
            s += rec(i + 1, N, indent_level + 1) + "\n"
            s += indent + ")"
            return s

    return rec(1, len(lookup_values), 0)
from dataclasses import dataclass


@dataclass
class LUTKernel:
    name: str
    source: str

def _generate_triton_lut_kernel(name, expr):
    """
    Given a name and an expression, generate a triton kernel that implements a lookup table.
    """

    TRITON_LUT_KERNEL_TEMPLATE = """
    # AUTOGENERATED LUT KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def {name}(x):
        return {expr}
    """
    # Remove leading whitespace from the template.
    _template = textwrap.dedent(TRITON_LUT_KERNEL_TEMPLATE)
    kernel = _template.format(name=name, expr=expr)
    
    return LUTKernel(name=name, source=kernel)

def generate_triton_nf4_lut_kernel(use_hardcoded=True, dtype=torch.float32, save_path=None, return_map=False):
    nf4_map = get_nf4_codebook(use_hardcoded=use_hardcoded, dtype=dtype)
    nf4_lut_expr = _generate_lookup_table(nf4_map.tolist())
    triton_kernel = _generate_triton_lut_kernel("nf4_lut_device_kernel", nf4_lut_expr)
    if save_path:
        if isinstance(save_path, str):
            with open(save_path, "w") as f:
                f.write(triton_kernel.source)
        else:
            save_path.write(triton_kernel.source)
    if return_map:
        return triton_kernel, nf4_map
    else:
        return triton_kernel

def _test_nf4_lut():
    nf4_map_functional = get_nf4_codebook(use_hardcoded=False)
    nf4_map_hardcoded = get_nf4_codebook(use_hardcoded=True)
    assert torch.equal(nf4_map_functional, nf4_map_hardcoded), "Functional and hardcoded NF4 maps do not match"

def _load_triton_kernel(kernel_path, kernel_name):
    import importlib.util
    from pathlib import Path
    
    kernel_path = Path(kernel_path)
    sys.path.insert(0, str(kernel_path.parent))
    spec = importlib.util.spec_from_file_location(kernel_path.stem, kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, kernel_name)
    return kernel

def _test_nf4_lut_kernel():
    import os
    from tempfile import TemporaryDirectory
    
    TEST_KERNEL_TEMPLATE = """
    @triton.jit
    def test_kernel(_x, _y, N: tl.constexpr):
        load_idx = tl.arange(0, N)
        x = tl.load(_x + load_idx)
        y = {device_kernel_name}(x)
        tl.store(_y + load_idx, y)
"""

    _template = textwrap.dedent(TEST_KERNEL_TEMPLATE)
    
    with TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, "_generated_nf4_lut.py")
        generated_kernel, nf4_map = generate_triton_nf4_lut_kernel(use_hardcoded=True, return_map=True)
        test_kernel = _template.format(device_kernel_name=generated_kernel.name)
        test_source = "\n".join([generated_kernel.source, test_kernel])

        with open(save_path, "w") as f:
            f.write(test_source)

        test_kernel = _load_triton_kernel(save_path, "test_kernel")
        
        
        N = 256
        nf4_min, nf4_max = 0, len(nf4_map) - 1
        assert nf4_min == 0 and nf4_max == 2 ** 4 - 1, "NF4 map is not the expected size"
        
        x = torch.randint(nf4_min, nf4_max, (N,), dtype=torch.uint8, device="cuda")
        y = torch.empty(N, dtype=torch.float32, device="cuda")
        
        test_kernel[(1,)](x, y, N)
        ref = nf4_map[x.cpu().tolist()]
        
        assert torch.equal(ref, y.cpu()), f"Triton kernel and reference lookup do not match, {sum(ref != y.cpu())} mismatches"
if __name__ == "__main__":
    torch.set_printoptions(precision=8)
#    _test_nf4_lut()
    _test_nf4_lut_kernel()
