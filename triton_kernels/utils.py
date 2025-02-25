import sys
import textwrap
from dataclasses import dataclass
from typing import Callable, List, Optional

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import create_dynamic_map, create_normal_map


def _get_package_version(package):
    from packaging import version

    _version = getattr(package, "__version__", None)
    if _version is None:
        raise ValueError(f"Package {package} does not have a __version__ attribute")

    ver = version.parse(_version)

    if isinstance(ver, version.Version):
        major = ver.major
        minor = ver.minor
        return major, minor
    elif isinstance(ver, version.LegacyVersion):
        # Handle legacy versions (non-PEP 440 compliant) if necessary.
        print("Legacy version format detected - further parsing may be required.")
    else:
        print("Invalid version format.")


# Copied from vLLM: https://github.com/vllm-project/vllm/blob/340e39e387d64160c019bcc553b194f070fa2748/vllm/utils.py#L1857
def direct_register_custom_op(
    library_name: str,
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.
    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.
    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    lib = torch.library.Library(library_name, "FRAGMENT")
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    lib.define(op_name + schema_str)
    lib.impl(op_name, op_func, "CUDA")
    if fake_impl is not None:
        lib._register_fake(op_name, fake_impl)

def create_nf4_param(
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
        nf4_map = torch.cat([normal_map[:8], normal_map[-8:]]).to(
            device=device, dtype=dtype
        )
        return nf4_map


def get_blockwise_codebook(device="cpu", dtype=torch.float32):
    """
    Returns:
        torch.Tensor: default dynamic blockwise quant map used for compressing absmax when quantizing with bitsandbytes nf4
        with `compress_statistics=True`.
    """
    codebook = create_dynamic_map()
    return codebook.to(device=device, dtype=dtype)


# --- Utility Functions for generating NF4 lookup table device kernel --- #


@dataclass

class LUTKernel:
    """
    A dataclass for storing a generated triton kernel.
    """

    name: str
    source: str


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
            s += indent + "    " + str(lookup_values[i - 1]) + ",\n"
            s += rec(i + 1, N, indent_level + 1) + "\n"
            s += indent + ")"
            return s

    return rec(1, len(lookup_values), 0)


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


def generate_triton_lut_kernel(codebook, kernel_name, save_path=None, return_map=False):
    lut_expr = _generate_lookup_table(codebook.tolist())
    triton_kernel = _generate_triton_lut_kernel(kernel_name, lut_expr)
    if save_path:
        if isinstance(save_path, str):
            with open(save_path, "w") as f:
                f.write(triton_kernel.source)
        else:
            save_path.write(triton_kernel.source)
    if return_map:
        return triton_kernel, codebook
    else:
        return triton_kernel


def generate_triton_nf4_lut_kernel(
    use_hardcoded=True, dtype=torch.float32, save_path=None, return_map=False
):
    """
    Generate a triton (device) kernel that implements a lookup table for nf4 quantization to work
    around triton's lack of support for static arrays.

    Meant to replicate the tree search per the original implementation: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu#L116.

    Args:
        use_hardcoded (bool): Whether to use the hardcoded nf4 codebook.
        dtype (torch.dtype): The dtype of the codebook.
        save_path (str): The path to save the generated kernel.
        return_map (bool): Whether to return the codebook.

    Returns:
        LUTKernel: The generated triton kernel.
    """
    nf4_map = get_nf4_codebook(use_hardcoded=use_hardcoded, dtype=dtype)
    return generate_triton_lut_kernel(
        nf4_map, "nf4_lut_device_kernel", save_path, return_map
    )


def _generate_triton_blockwise_lut_kernel(save_path=None, return_map=False):
    """
    Generate a triton (device) kernel that implements a lookup table for blockwise quantization used when
    compressing absmax with bitsandbytes nf4 with `compress_statistics=True`.

    NOT USED: The device kernel is implemented as a series of nested tl.where statements to map input values to codebook values
    since triton does not support static arrays.  However, since the nested codebook uses 256 values, results in syntax error due to
    too many nested parentheses.

    Also, in the original implementation, the codebooks is passed to the kernel as an array: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu#L636-L641
    wherease for nf4 dequant, a tree-like lookup is used: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu#L116
    """
    codebook = get_blockwise_codebook()
    return generate_triton_lut_kernel(
        codebook, "blockwise_lut_device_kernel", save_path, return_map
    )


### --- TESTS --- ###


def _test_nf4_lut():
    nf4_map_functional = get_nf4_codebook(use_hardcoded=False)
    nf4_map_hardcoded = get_nf4_codebook(use_hardcoded=True)
    assert torch.equal(nf4_map_functional, nf4_map_hardcoded), (
        "Functional and hardcoded NF4 maps do not match"
    )


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


def _test_lut_kernel(quant_type):
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

    if quant_type == "nf4":
        kernel_generator = generate_triton_nf4_lut_kernel
        codebook = get_nf4_codebook()
        assert len(codebook) == 16, "NF4 codebook is not the expected size"
    elif quant_type == "blockwise":
        kernel_generator = _generate_triton_blockwise_lut_kernel
        codebook = get_blockwise_codebook()
        assert len(codebook) == 256, "Blockwise codebook is not the expected size"
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")

    with TemporaryDirectory() as tempdir:
        save_path = os.path.join(tempdir, f"_generated_{quant_type}_lut.py")
        generated_kernel = kernel_generator()
        test_kernel = _template.format(device_kernel_name=generated_kernel.name)
        test_source = "\n".join([generated_kernel.source, test_kernel])

        with open(save_path, "w") as f:
            f.write(test_source)

        test_kernel = _load_triton_kernel(save_path, "test_kernel")

        N = 256
        _max = 2**4 - 1 if quant_type == "nf4" else 2**8 - 1

        x = torch.randint(0, _max, (N,), dtype=torch.uint8, device="cuda")
        y = torch.empty(N, dtype=torch.float32, device="cuda")

        test_kernel[(1,)](x, y, N)
        ref = codebook[x.cpu().tolist()]

        if not torch.equal(ref, y.cpu()):
            print(f"\u2718 {quant_type} LUT kernel test failed")
        else:
            print(f"\u2713 {quant_type} LUT kernel test passed")


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    # _test_nf4_lut()
    # _test_lut_kernel("nf4")

    print(_get_package_version(torch))
    # SyntaxError: too many nested parentheses
    # _test_lut_kernel("blockwise")

