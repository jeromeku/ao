import math
import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torchao.dtypes.nf4tensor import NF4Tensor, linear_nf4, to_nf4

NUM_LINEARS = 1
SEED = 42
class LoRALinear(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        lora_rank: int = None,
        lora_alpha: float = 16,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        if lora_rank is None:
            lora_rank = hidden_dim // 2
        
        weight = torch.randn((hidden_dim, hidden_dim), dtype=dtype)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.register_parameter("weight", nn.Parameter(to_nf4(weight), requires_grad=False))
        self.lora_a = nn.Linear(in_features=hidden_dim, out_features=self.lora_rank, bias=False)
        self.lora_b = nn.Linear(in_features=self.lora_rank, out_features=hidden_dim, bias=False)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_b.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(SEED)
        out = linear_nf4(input=x, weight=self.weight)
        lora_out = self.lora_a(x)
        lora_out = (self.lora_alpha / self.lora_rank) * self.lora_b(lora_out)
        return out + lora_out

def _init_model(dim=128, device="cuda", dtype=torch.float32) -> nn.Module:
    with torch.device(device):

        modules = []
        for i in range(NUM_LINEARS):
            modules += [LoRALinear(hidden_dim=dim, dtype=dtype)]
        seq = nn.Sequential(*modules)

    return seq

def dist_print(*args, delay=.5):
    rank = dist.get_rank()
    time.sleep(delay * rank)
    print(f"[rank{rank}]: ", *args, flush=True)

def _print_params_and_grads(model, prefix=""):
    for name, param in model.named_parameters():
        if isinstance(param.data, NF4Tensor):
            assert not param.requires_grad
            param = param.get_original_weight()
            
        if param.grad is not None:
            dist_print(f"{prefix}::DEBUG::GRAD", name, type(param), param.sum().item(), param.grad.sum().item())
        else:
            dist_print(f"{prefix}::DEBUG::PARAM", name, type(param), param.sum().item(), "None")

def make_batch(bs, dim, dtype, device):
    batch = torch.randn((bs, dim), dtype=dtype, device=device)
    if dist.get_world_size() > 1:
        batch = batch.chunk(dist.get_world_size(), dim=0)[dist.get_rank()]
    return batch

def test_ddp(bs=2, dim=128, num_steps=3, device="cuda", dtype=torch.float32):
    model = _init_model(dim, device, dtype)
    model = DDP(model, device_ids=[device])
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    losses = []
    
    for i in range(num_steps):
        inp = make_batch(bs, dim, dtype, device)
        loss = model(inp).sum()
        losses.append(loss)
        #dist_print(f"LOSS::STEP_{i}", loss.item())
        loss.backward()
#        _print_params_and_grads(model, f"AFTER_BACKWARDS_{i}")
        optim.step()
        #_print_params_and_grads(model, f"PARAMS_AND_GRADS::AFTER_STEP_{i}")
        optim.zero_grad()
        #_print_params_and_grads(model, f"AFTER_ZERO_GRAD_{i}")
        #dist.barrier()

    dist.barrier()
    if dist.get_world_size() == 1:
        save_dir = "checkpoints/ref"
    else:
        save_dir = f"checkpoints/test"
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    save_path = f"{save_dir}/ddp-{dist.get_rank()}.pt"
    torch.save(model.state_dict(), save_path)
    dist_print("Saved model to", save_path)

def init_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    dist_print("Dist initialized with world size", dist.get_world_size())

def cleanup_dist():
    dist.barrier()
    if dist.get_rank() == 0:
        print("Cleaning up dist")
    dist.destroy_process_group()

@contextmanager
def distributed_context():
    init_dist()
    yield
    cleanup_dist()

if __name__ == "__main__":
    with distributed_context():
        torch.manual_seed(SEED)
        test_ddp()
