import copy
import datetime
import math

import torch
from torch import nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy, OffloadPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight: torch.Tensor,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.register_parameter("weight", nn.Parameter(to_nf4(weight)))
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_b.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = linear_nf4(input=x, weight=self.weight)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out

def dist_breakpoint():
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            breakpoint()
        torch.distributed.barrier()
    else:
        breakpoint()

class TestQLoRA():
    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    def test_qlora_fsdp2(
        self,
        enable_activation_checkpointing: bool = False,
        offload_policy: "OffloadPolicy" = OffloadPolicy(),  # noqa: F821
    ):
        from torch.distributed._composable.fsdp import fully_shard
        from torch.testing._internal.distributed._tensor.common_dtensor import (
            ModelArgs,
            Transformer,
            TransformerBlock,
        )

        batch_size = 3
        lora_r = 8
        lora_alpha = 16
        vocab_size = 1024
        seq_len = 64
        model_args = ModelArgs(
            n_layers=3,
            n_heads=4,
            dim=1024,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dropout_p=0,
        )
        torch.manual_seed(42)
        with torch.device("cuda"):
            base_model = Transformer(model_args)
            dist_breakpoint()
            for layer in base_model.layers:
                # attention with lora adapters
                for attr in ["wq", "wk", "wv", "wo"]:
                    orig_linear = getattr(layer.attention, attr)
                    setattr(
                        layer.attention,
                        attr,
                        LoRALinear(
                            orig_linear.weight.shape[1],
                            orig_linear.weight.shape[0],
                            orig_linear.weight,
                            lora_r,
                            lora_alpha,
                        ),
                    )
                for attr in ["w1", "w2"]:
                    orig_linear = getattr(layer.feed_forward, attr)
                    setattr(
                        layer.feed_forward,
                        attr,
                        LoRALinear(
                            orig_linear.weight.shape[1],
                            orig_linear.weight.shape[0],
                            orig_linear.weight,
                            lora_r,
                            lora_alpha,
                        ),
                    )
        for name, param in base_model.named_parameters():
            param.requires_grad_(
                name.endswith("lora_a.weight") or name.endswith("lora_b.weight")
            )
        if enable_activation_checkpointing:
            apply_activation_checkpointing(
                base_model, auto_wrap_policy=ModuleWrapPolicy({TransformerBlock})
            )
        base_optim = torch.optim.AdamW(base_model.parameters(), lr=1e-2)

        fsdp_kwargs = {"offload_policy": offload_policy}
        fsdp_model = copy.deepcopy(base_model)
        for m in fsdp_model.modules():
            if enable_activation_checkpointing:
                if isinstance(m, CheckpointWrapper):
                    fully_shard(m, **fsdp_kwargs)
            else:
                if isinstance(m, TransformerBlock):
                    fully_shard(m, **fsdp_kwargs)
        fully_shard(fsdp_model, **fsdp_kwargs)
        fsdp_optim = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(5):
            inp = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
            fsdp_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = fsdp_model(inp).sum()
            fsdp_loss.backward()
            fsdp_optim.step()

            base_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            base_loss = base_model(inp).sum()
            base_loss.backward()
            for param in base_model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.AVG
                    )
            base_optim.step()
            self.assertEqual(fsdp_loss, base_loss)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    test = TestQLoRA()
    test.setUp()
    test.test_qlora_fsdp2()
