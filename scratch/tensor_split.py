import torch

from triton_kernels.utils import create_nf4_param

num_chunks = 2
out_features, in_features = 8, 4
w = torch.arange(out_features * in_features).reshape(out_features, in_features).to(torch.int)

ref_chunks = torch.chunk(w, num_chunks, dim=1)
print(ref_chunks[0].shape)
for i, chunk in enumerate(ref_chunks):
    formatted_chunk =  "\n".join(str(row) for row in chunk.tolist()) 
    print(f"ref_chunk {i}:\n{formatted_chunk}")

w2 = w.reshape(8, num_chunks, -1)
w3 = w2.permute(1, 0, 2)
print(w3.shape)
for i, chunk in enumerate(w3):
    print(chunk.shape)
    print(chunk)

row_size_per_chunk = w.shape[1] // num_chunks
w4 = w.reshape(-1, row_size_per_chunk)
chunk1 = w4[::num_chunks]
chunk2 = w4[1::num_chunks]
print(torch.equal(chunk1, ref_chunks[0]))
print(torch.equal(chunk2, ref_chunks[1]))
