import os
import time

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4


class NF4(nn.Module):
    
    def __init__(
        self,
        device = None,
    ):
        super().__init__()

        self.lora_linear = nn.Linear(512, 512, bias=False, device=device)
        self.lora_linear.weight = nn.Parameter(to_nf4(self.lora_linear.weight), requires_grad=False)
        self.linear = nn.Linear(512, 512, bias=False, device=device)
    
    def forward(self, x):
        x = self.lora_linear(x)
        x = self.linear(x)
        return x

def dist_print(*msg, delay=.5):
    rank = torch.distributed.get_rank()
    delay = rank * delay
    time.sleep(delay)
    print(f"[rank{rank}]", *msg, sep=": ", flush=True)

if __name__ == "__main__":
    
    _local_rank = int(os.getenv("LOCAL_RANK", "0"))
    _device = f"cuda:{_local_rank}"

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(_local_rank),
    )

    model = NF4(_device)

    model = DistributedDataParallel(model)
    
    for name, param in model.named_parameters():
        dist_print(f"DEBUG::NF4_cat:: {name} {type(param)} {param.shape}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
