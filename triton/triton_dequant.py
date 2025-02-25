"""
Triton nf4 dequant kernel

Dequantizes bnb.Params4bit to float16, bfloat16, or float32.

Tested for default bnb nf4 settings: 
- `block_size=64` and `nested_block_size=256` where `nested_block_size` is the quantization block size
for the quantized absmax.
- quant_storage: uint8

Benchmarked in the following env:
bitsandbytes                      0.45.2
triton                            3.2.0
unsloth                           2025.2.12
unsloth-zoo                       2025.2.5

On an H100, should get the following benchmark results:
```
python dequant.py --shape 4096 14336 --dtype float32 --benchmark

Benchmarking shape: (4096, 14336), dtype: torch.float32, qblocks_per_cta: autotuned
✓ triton / unsloth speedup: 1.55x
 Latency: triton 0.0978ms, unsloth 0.1513ms, bnb 0.1492ms
 Speedups:
  triton / unsloth: 1.55x
  triton / bnb: 1.53x
  unsloth / bnb: 0.99x
----------------------------------------
python dequant.py --shape 4096 14336 --dtype float16 --benchmark

Benchmarking shape: (4096, 14336), dtype: torch.float16, qblocks_per_cta: autotuned
✓ triton / unsloth speedup: 2.58x
 Latency: triton 0.0582ms, unsloth 0.1504ms, bnb 0.1516ms
 Speedups:
  triton / unsloth: 2.58x
  triton / bnb: 2.60x
  unsloth / bnb: 1.01x
----------------------------------------
python dequant.py --shape 4096 14336 --dtype bfloat16 --benchmark

Benchmarking shape: (4096, 14336), dtype: torch.bfloat16, qblocks_per_cta: autotuned
✓ triton / unsloth speedup: 2.62x
 Latency: triton 0.0580ms, unsloth 0.1519ms, bnb 0.1517ms
 Speedups:
  triton / unsloth: 2.62x
  triton / bnb: 2.62x
  unsloth / bnb: 1.00x
```

Correctness tests can be run by specifying `--test` to the script:
```
python dequant.py --test dtype float32 ...
```

Note that the tolerances depend on the dtype (float32 has lowest tolerance, bfloat16 the highest).

To see all available options:
```
python dequant.py --help
```
"""

import argparse
import math
import sys

import bitsandbytes as bnb
import torch
import triton
import triton.language as tl
from bitsandbytes.functional import dequantize_nf4
from triton.testing import do_bench
from unsloth.kernels.utils import fast_dequantize

DEVICE = "cuda"

BLOCK_SIZE = 64
NESTED_BLOCK_SIZE = 256

def get_dequant_configs(min_blocks_per_cta=1, max_blocks_per_cta=16):
    configs = []
    powers_of_two = [
        2**p
        for p in range(
            int(math.log2(min_blocks_per_cta)), int(math.log2(max_blocks_per_cta)) + 1
        )
    ]
    assert (
        max(powers_of_two) <= NESTED_BLOCK_SIZE
    ), f"max_blocks_per_cta must be less than NESTED_BLOCK_SIZE, {max(powers_of_two)} <= {NESTED_BLOCK_SIZE}"
    for qblocks_per_cta in powers_of_two:
        configs.append(triton.Config({"QBLOCKS_PER_CTA": qblocks_per_cta}))
    return configs


# Triton kernels
@triton.jit
def lut_device_kernel(q, code_ptr):
    # Need to cast to use as pointer index
    q = q.to(tl.int32)
    return tl.load(code_ptr + q)


@triton.jit
def _dequant_nf4_kernel(
    # Quantized data
    q_ptr,
    # NF4 code
    nf4_code_ptr,
    # Compressed statistics
    # Quantized absmax
    qabsmax_ptr,
    # Scale factors for quantized absmax
    qabsmax_scalers_ptr,
    # Offset for quantized absmax
    qabsmax_mean_ptr,
    # Double quant block code
    block_code_ptr,
    # Output
    dq_ptr,
    # Compile time constants
    N,
    # autotuned
    QBLOCKS_PER_CTA: tl.constexpr,
    # Quantized data block size
    QBLOCK_SIZE: tl.constexpr = BLOCK_SIZE,
    # Nested block size
    NESTED_QBLOCK_SIZE: tl.constexpr = NESTED_BLOCK_SIZE,
    # Number of packed elements
    NUM_PACKED_ELEMENTS: tl.constexpr = 2,
    # Output data type
    OUTPUT_DTYPE: tl.constexpr = tl.bfloat16,
):

    OUTPUT_ELEMENTS_PER_QBLOCK: tl.constexpr = QBLOCKS_PER_CTA * QBLOCK_SIZE
    # Divide by NUM_PACKED_ELEMENTS to account for packed elements
    # Default quant storage is uint 8 => 8 bits // 4 bits => 2 packed elements

    tl.static_assert(
        OUTPUT_ELEMENTS_PER_QBLOCK % NUM_PACKED_ELEMENTS == 0,
        "OUTPUT_ELEMENTS_PER_QBLOCK must be divisible by NUM_PACKED_ELEMENTS",
    )
    INPUT_ELEMENTS_PER_QBLOCK: tl.constexpr = (
        OUTPUT_ELEMENTS_PER_QBLOCK // NUM_PACKED_ELEMENTS
    )

    # Load quantized data
    block_offset = tl.program_id(axis=0)
    input_offset = block_offset * INPUT_ELEMENTS_PER_QBLOCK
    output_offset = block_offset * OUTPUT_ELEMENTS_PER_QBLOCK
    # offset = block_idx * QBLOCKS_PER_CTA * INPUT_ELEMENTS_PER_QBLOCK
    q_load_idx = input_offset + tl.arange(0, INPUT_ELEMENTS_PER_QBLOCK)
    q_load_idx = tl.max_contiguous(
        tl.multiple_of(q_load_idx, INPUT_ELEMENTS_PER_QBLOCK), INPUT_ELEMENTS_PER_QBLOCK
    )

    q = tl.load(q_ptr + q_load_idx)
    q_first_elements = q >> 4
    q_second_elements = q & 0xF
    dq_first_elements = lut_device_kernel(q_first_elements, nf4_code_ptr)
    dq_second_elements = lut_device_kernel(q_second_elements, nf4_code_ptr)
    interleaved = tl.interleave(dq_first_elements, dq_second_elements)

    # Load qabsmax
    QABSMAX_ELEMENTS_TO_LOAD: tl.constexpr = QBLOCKS_PER_CTA
    # Previously block_offset = block_idx = tl.program_id(axis=0)
    qabsmax_offset = block_offset * QABSMAX_ELEMENTS_TO_LOAD
    qabsmax_load_idx = qabsmax_offset + tl.arange(0, QABSMAX_ELEMENTS_TO_LOAD)
    qabsmax_load_idx = tl.max_contiguous(
        tl.multiple_of(qabsmax_load_idx, QABSMAX_ELEMENTS_TO_LOAD),
        QABSMAX_ELEMENTS_TO_LOAD,
    )
    qabsmax = tl.load(qabsmax_ptr + qabsmax_load_idx)
    dqabsmax = lut_device_kernel(qabsmax, block_code_ptr)

    qabsmax_scalers_offset = qabsmax_offset // NESTED_QBLOCK_SIZE
    NUM_QABSMAX_SCALERS: tl.constexpr = (
        QBLOCKS_PER_CTA + NESTED_QBLOCK_SIZE - 1
    ) // NESTED_QBLOCK_SIZE
    tl.static_assert(NUM_QABSMAX_SCALERS == 1, "NUM_QABSMAX_SCALERS must be 1")
    dqabsmax_scalers = tl.load(qabsmax_scalers_ptr + qabsmax_scalers_offset)
    qabsmax_mean = tl.load(qabsmax_mean_ptr)

    # Dequantize qabsmax
    dqabsmax = dqabsmax * dqabsmax_scalers + qabsmax_mean
    interleaved = tl.reshape(interleaved, (QBLOCKS_PER_CTA, QBLOCK_SIZE))
    dqabsmax = tl.reshape(dqabsmax, (QBLOCKS_PER_CTA, 1))

    dq = interleaved * dqabsmax
    dq = dq.to(OUTPUT_DTYPE)

    store_idx = output_offset + tl.arange(0, OUTPUT_ELEMENTS_PER_QBLOCK)
    tl.store(dq_ptr + store_idx, tl.reshape(dq, OUTPUT_ELEMENTS_PER_QBLOCK))


_triton_dequant_nf4_kernel_autotuned = triton.autotune(
    configs=get_dequant_configs(min_blocks_per_cta=1, max_blocks_per_cta=16), key=["N"]
)(_dequant_nf4_kernel)
_triton_dequant_nf4_kernel = _dequant_nf4_kernel


def torch_to_triton_dtype(dtype):
    parts = str(dtype).split(".")
    assert len(parts) == 2, f"dtype {dtype} is not a valid triton dtype, {parts}"
    dtype_str = parts[-1]
    tt_dtype = getattr(tl, dtype_str)
    return tt_dtype


def create_qparam(
    input_weight, quant_type="nf4", quant_storage=torch.uint8, compress_statistics=True
):
    param = bnb.nn.Params4bit(
        input_weight,
        requires_grad=False,
        quant_type=quant_type,
        quant_storage=quant_storage,
        compress_statistics=compress_statistics,
    ).cuda()
    return param


def triton_dequant_nf4(qparam: bnb.nn.Params4bit, QBLOCKS_PER_CTA=None, autotune=True):
    if not autotune:
        assert (
            QBLOCKS_PER_CTA is not None
        ), "QBLOCKS_PER_CTA must be provided if autotune is False"
    qstate = qparam.quant_state
    nested_qstate = qparam.quant_state.state2

    quantized_data = qparam.data
    quantized_scalers = qstate.absmax
    nested_scale_factors = nested_qstate.absmax
    quantized_scaler_mean = qstate.offset

    nf4_code = qstate.code
    block_code = qstate.state2.code

    NUM_PACKED_ELEMENTS = quantized_data.element_size() * 8 // 4
    OUTPUT_DTYPE = torch_to_triton_dtype(qstate.dtype)
    dq = torch.empty(qstate.shape, device=qparam.device, dtype=qstate.dtype)
    grid = lambda meta: (
        triton.cdiv(dq.numel(), meta["QBLOCKS_PER_CTA"] * meta["QBLOCK_SIZE"]),
    )

    kernel = (
        _triton_dequant_nf4_kernel_autotuned if autotune else _triton_dequant_nf4_kernel
    )
    kernel_args = {
        "q_ptr": quantized_data,
        "nf4_code_ptr": nf4_code,
        "qabsmax_ptr": quantized_scalers,
        "qabsmax_scalers_ptr": nested_scale_factors,
        "qabsmax_mean_ptr": quantized_scaler_mean,
        "block_code_ptr": block_code,
        "dq_ptr": dq,
        "N": dq.numel(),
        "QBLOCK_SIZE": BLOCK_SIZE,
        "NESTED_QBLOCK_SIZE": NESTED_BLOCK_SIZE,
        "NUM_PACKED_ELEMENTS": NUM_PACKED_ELEMENTS,
        "OUTPUT_DTYPE": OUTPUT_DTYPE,
    }
    if not autotune:
        kernel_args["QBLOCKS_PER_CTA"] = QBLOCKS_PER_CTA
    kernel[grid](**kernel_args)
    return dq


def unsloth_dequantize(qparam: bnb.nn.Params4bit):
    return fast_dequantize(qparam, qparam.quant_state)


def test_equivalence(
    shape,
    dtype,
    qblocks_per_cta=None,
    autotune=True,
    include_unsloth=False,
    atol=1e-3,
    rtol=1e-3,
):
    input_weight = torch.randn(shape, device=DEVICE, dtype=dtype)
    qparam = create_qparam(
        input_weight,
        quant_type="nf4",
        quant_storage=torch.uint8,
        compress_statistics=True,
    )
    ref_dq = dequantize_nf4(qparam, quant_state=qparam.quant_state)
    dq = triton_dequant_nf4(
        qparam=qparam, QBLOCKS_PER_CTA=qblocks_per_cta, autotune=autotune
    )

    passed = torch.allclose(dq, ref_dq, atol=atol, rtol=rtol)
    if passed:
        print(f"\u2713 triton kernel passed")
    else:
        diff = (dq - ref_dq).abs().max()
        print(f"\u2717 triton kernel failed, diff={diff}")

    if include_unsloth:
        unsloth_dq = unsloth_dequantize(qparam)
        assert torch.allclose(unsloth_dq, ref_dq), f"unsloth != bnb dequant"
        print(f"\u2713 unsloth bnb check passed")

    return passed


def benchmark_dequant(shape, dtype, qblocks_per_cta=None, autotune=True):
    input_weight = torch.randn(shape, device=DEVICE, dtype=dtype)
    qparam = create_qparam(
        input_weight,
        quant_type="nf4",
        quant_storage=torch.uint8,
        compress_statistics=True,
    )

    bnb_time = do_bench(lambda: dequantize_nf4(qparam, quant_state=qparam.quant_state))
    tt_time = do_bench(
        lambda: triton_dequant_nf4(
            qparam=qparam, QBLOCKS_PER_CTA=qblocks_per_cta, autotune=autotune
        )
    )
    unsloth_time = do_bench(lambda: unsloth_dequantize(qparam))

    tt_bnb_speedup = bnb_time / tt_time
    unsloth_bnb_speedup = bnb_time / unsloth_time
    tt_unsloth_speedup = unsloth_time / tt_time

    print(
        f"Benchmarking shape: {shape}, dtype: {dtype}, qblocks_per_cta: {'autotuned' if autotune else qblocks_per_cta}"
    )

    tt_unsloth_mark = "\u2713" if tt_unsloth_speedup > 1 else "\u2717"
    print(
        f"\033[1m{tt_unsloth_mark} triton / unsloth speedup: {tt_unsloth_speedup:.2f}x\033[0m",
        f" Latency: triton {tt_time:.4f}ms, unsloth {unsloth_time:.4f}ms, bnb {bnb_time:.4f}ms",
        f" Speedups:",
        f"  triton / unsloth: {tt_unsloth_speedup:.2f}x",
        f"  triton / bnb: {tt_bnb_speedup:.2f}x",
        f"  unsloth / bnb: {unsloth_bnb_speedup:.2f}x", 
        sep="\n"
    )


TOLERANCE = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Test and benchmark triton nf4 dequant kernel:
    --test - test for equivalence with the bnb `dequantize_nf4`.
    --benchmark - benchmark the kernel against the bnb `dequantize_nf4` and unsloth `fast_dequantize`.
    """, 
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--test", action="store_true", help="Test the kernel for equivalence with the bnb `dequantize_nf4` kernel.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the kernel against the bnb `dequantize_nf4` and unsloth `fast_dequantize` kernels.")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 14336])
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    atol, rtol = TOLERANCE[dtype]
    qblocks_per_cta = 8
    shape = tuple(args.shape)

    if args.test:
        print(f"Testing shape: {shape} dtype: {dtype} with atol={atol} rtol={rtol}")
        passed = test_equivalence(
            shape=shape, dtype=dtype, autotune=True, atol=atol, rtol=rtol
        )
        if not passed:
            raise AssertionError(
                f"triton kernel equivalence failed at shape {shape}, autotuned"
            )
        print("-" * 100)

    if args.benchmark:
        benchmark_dequant(shape=shape, dtype=dtype, autotune=True)