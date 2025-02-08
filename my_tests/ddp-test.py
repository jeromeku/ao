import copy
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

NUM_LINEARS = 1
def _init_model(dim=128, device="cuda", dtype=torch.float32) -> nn.Module:
    torch.manual_seed(42)
    modules = []
    for i in range(NUM_LINEARS):
        modules += [nn.Linear(dim, dim, device=device, bias=False, dtype=dtype)]
    seq = nn.Sequential(*modules)
    return seq

def dist_print(*args, delay=.5):
    rank = dist.get_rank()
    time.sleep(delay * rank)
    print(f"[rank{rank}]: ", *args, flush=True)

def _print_params_and_grads(model, prefix=""):
    for name, param in model.named_parameters():
        if param.grad is not None:
            dist_print(f"{prefix}::DEBUG::GRAD", name, param.sum().item(), param.grad.sum().item())
        else:
            dist_print(f"{prefix}::DEBUG::PARAM", name, param.sum().item(), "None")

def test_ddp(bs=2, dim=8, num_steps=3, device="cuda", dtype=torch.float32):
    rank = dist.get_rank()
    model = _init_model(dim, device, dtype)
    model = DDP(model, device_ids=[device])
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(rank + 1)
    losses = []
    
    for i in range(num_steps):
        inp = torch.randn((bs, dim), device=device, dtype=dtype)
        loss = model(inp).sum()
        losses.append(loss)
        dist_print(f"STEP_{i}", loss.item())
        _print_params_and_grads(model, f"BEFORE_BACKWARDS_{i}")
        loss.backward()
        _print_params_and_grads(model, f"AFTER_BACKWARDS_{i}")
        optim.step()
        _print_params_and_grads(model, f"AFTER_STEP_{i}")
        optim.zero_grad()
        #_print_params_and_grads(model, f"AFTER_ZERO_GRAD_{i}")
        dist.barrier()

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
        test_ddp()
