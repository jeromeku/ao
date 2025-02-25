from dataclasses import dataclass

import bitsandbytes as bnb
import torch

from triton_kernels.utils import create_nf4_param

num_chunks = 2
out_features, in_features = 8, 4
w = torch.arange(out_features * in_features).reshape(out_features, in_features).to(torch.int)

ref_chunks = torch.chunk(w, num_chunks, dim=1)

w2 = w.reshape(8, num_chunks, -1)
w3 = w2.permute(1, 0, 2)
for ref, test in zip(ref_chunks, w3):
    assert torch.equal(ref, test)

row_size_per_chunk = w.shape[1] // num_chunks
w4 = w.reshape(-1, row_size_per_chunk)
chunk1 = w4[::num_chunks]
chunk2 = w4[1::num_chunks]
assert torch.equal(chunk1, ref_chunks[0])
assert torch.equal(chunk2, ref_chunks[1])


@dataclass
class NestedNF4:
    quantized_data: torch.Tensor
    quantized_absmax: torch.Tensor
    absmax_offset: torch.Tensor
    absmax_scale_factors: torch.Tensor
    block_size: int
    nested_block_size: int
    original_shape: torch.Size

    def __repr__(self):
        return f"""NestedNF4(quantized_data={self.quantized_data.shape},
        quantized_absmax={self.quantized_absmax.shape},
        absmax_scale_factors={self.absmax_scale_factors.shape},
        block_size={self.block_size},
        nested_block_size={self.nested_block_size},
        original_shape={self.original_shape})"""

def unpack_nf4(w: bnb.nn.Params4bit):
    quant_state = w.quant_state
    quantized_data = w.data
    block_size = quant_state.blocksize
    quantized_absmax = quant_state.absmax
    absmax_offset = quant_state.offset
    nested_block_size = quant_state.state2.blocksize
    absmax_scale_factors = quant_state.state2.absmax
    original_shape = quant_state.shape
    return NestedNF4(quantized_data, quantized_absmax, absmax_offset, absmax_scale_factors, block_size, nested_block_size, original_shape)

w = torch.randn(512, 1024, dtype=torch.float16, device="cuda")
nf4_param = create_nf4_param(w)
nf4_meta = unpack_nf4(nf4_param)
num_blocks = w.numel() // nf4_meta.block_size
assert w.numel() == nf4_meta.quantized_data.numel() * 2, f"{w.numel()} != {nf4_meta.quantized_data.numel() * 2}"
assert (nf4_meta.quantized_data.numel() * 2) // nf4_meta.block_size == num_blocks, f"{(nf4_meta.quantized_data.numel() * 2) // nf4_meta.block_size} != {num_blocks}"
assert nf4_meta.quantized_absmax.numel() == num_blocks, f"{nf4_meta.quantized_absmax.numel()} != {num_blocks}"

blocks_per_row = w.shape[1] // nf4_meta.block_size
absmax_to_split = nf4_meta.quantized_absmax.reshape(-1, num_chunks, blocks_per_row // num_chunks).permute(1, 0, 2)
chunked_absmax = nf4_meta.quantized_absmax.reshape(-1, blocks_per_row).chunk(num_chunks, dim=1)
print(chunked_absmax[0].shape)
print(absmax_to_split[0].shape)
assert torch.equal(chunked_absmax[0], absmax_to_split[0])