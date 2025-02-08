import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from torchao.dtypes.nf4tensor import to_nf4


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


class DummyDataset(Dataset):
    def __init__(self, d, size, dtype=torch.bfloat16):
        self.size = size
        self.data = [(torch.rand(d, dtype=dtype), torch.rand(1, dtype=dtype)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]



class NF4(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_classes: int = 1,
        device = None,
        use_lora: bool = False,
        dtype=torch.bfloat16,
    ):
        super().__init__()

        self.linear1 = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
        if use_lora:
            self.linear1.weight = nn.Parameter(to_nf4(self.linear1.weight))
        self.linear1.weight.requires_grad = False
        
        self.linear2 = nn.Linear(out_features, num_classes, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        load_snapshot: bool = False,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
       
        if load_snapshot and os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs(args):
    train_set = DummyDataset(d=args.in_features, size=args.dataset_size, dtype=args.dtype)  # load your dataset
    model = NF4(args.in_features, args.out_features, dtype=args.dtype, use_lora=args.use_lora)  # load your model
    optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=1e-4)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def dist_print(*msg, delay=.5):
    rank = torch.distributed.get_rank()
    delay = rank * delay
    time.sleep(delay)
    print(f"[rank{rank}]: ", *msg, flush=True)

def main(args):
    ddp_setup()
    rank = torch.distributed.get_rank()
    args.snapshot_path = f"rank{rank}.pt"
    dist_print(f"rank{rank} is ready")
    
    dataset, model, optimizer = load_train_objs(args)
    torch.distributed.barrier()
    batch = next(iter(dataset))
    data, label = batch
    dist_print(data.shape, label.shape)
    train_data = prepare_dataloader(dataset, args.batch_size)
    batch = next(iter(train_data))
    data, label = batch
    dist_print(data.shape, label.shape)
    trainer = Trainer(model, train_data, optimizer, args.save_every, args.snapshot_path)
    trainer.train(args.total_epochs)
    
    torch.distributed.barrier()
    for name, param in model.named_parameters():
        if param.grad is not None:
            dist_print(name, param.grad.min().item(), param.grad.max().item())
        else:
            dist_print(name, "No grad")
    
    torch.distributed.barrier()
    dist_print(f"rank{rank} is done")
    
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, default=5, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, default=None, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 16)')
    parser.add_argument('--in_features', default=128, type=int, help='Input features (default: 128)')
    parser.add_argument('--out_features', default=128, type=int, help='Output features (default: 128)')
    parser.add_argument('--dataset_size', default=2048, type=int, help='Dataset size (default: 2048)')
    parser.add_argument('--dtype', default=torch.bfloat16, type=torch.dtype, help='Data type (default: bfloat16)')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA (default: False)')

    args = parser.parse_args()

    if args.save_every is None:
        args.save_every = args.total_epochs

    main(args)
