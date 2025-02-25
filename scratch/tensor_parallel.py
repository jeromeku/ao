    import os
    import time
    from contextlib import contextmanager
    from typing import Sequence

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed import DeviceMesh
    from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
    from torch.utils._python_dispatch import return_and_correct_aliasing

    from torchao.utils import fill_defaults


    class M(torch.nn.Module):
        def __init__(self, in_features, out_features, **kwargs) -> None:
            super().__init__(**kwargs)
            self.linear = torch.nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    def shard(
        full_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
    ) -> DTensor:
        """
        Add a shard function to simplify both colwise_shard and rowwise_shard.  The
        shard function accepts a full tensor, and returns a DTensor based on
        indicated placements.  Goal is to move the shard function as a static method
        of DTensor, e.g.
            dtensor = DTensor.shard(full_tensor, device_mesh, placement)
        """
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

        shape, offset = compute_local_shape_and_global_offset(
            full_tensor.shape, device_mesh, placements
        )
        slices = [
            slice(cur_offset, cur_offset + cur_shape)
            for cur_shape, cur_offset in zip(shape, offset)
        ]
        local_tensor = full_tensor[slices]
        return DTensor.from_local(local_tensor, device_mesh, placements)


    def colwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
        """
        Shard linear layer of the model in column-wise fashion
        """
        # Column-wise is wrt to A^T, so for A it is row-wise.
        orig_weight = m.linear.weight
        # Construct DTensor from local shard
        dtensor = shard(orig_weight, mesh, [Shard(0)])
        # Replace parameter in module
        m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
        return m


    def rowwise_shard(m: torch.nn.Module, mesh: DeviceMesh) -> torch.nn.Module:
        """
        Shard linear layer of the model in row-wise fashion
        """
        # Row-wise is wrt to A^T, so for A it is column-wise.
        orig_weight = m.linear.weight
        # Construct DTensor from local shard
        dtensor = shard(orig_weight, mesh, [Shard(1)])
        # Replace parameter in module
        m.linear.weight = torch.nn.Parameter(dtensor, requires_grad=False)
        return m


    def dist_print(*msg, delay=.5, rank0_only=False):
        rank = dist.get_rank()
        if dist.is_initialized():
            if rank0_only and rank != 0:
                return
            time.sleep(delay * rank)
            print(*msg, flush=True)
        else:
            print(*msg, flush=True)

    def run(rank=None, rendezvous=None, backend: str=None, world_size: int=None):
    #    dist.init_process_group(backend=backend, init_method=rendezvous, rank=rank, world_size=world_size)
        
        rank = rank or dist.get_rank()
        world_size = world_size or dist.get_world_size()
        torch.manual_seed(5)
        device = "cpu" if backend == "gloo" else "cuda"
        # Get rank and device
        # Original model
        proj_up = M(1024, 2048).to(device)
        proj_dn = M(2048, 1024).to(device)

        mesh = dist.init_device_mesh("cpu", mesh_shape=(world_size,), mesh_dim_names=("tp",))

        # Shard the models
        
        up_dist = colwise_shard(proj_up, mesh)
        dn_dist = rowwise_shard(proj_dn, mesh)


        # # We need to turn inputs into DTensor form as well -- just a format change
        # input_dtensor = DTensor.from_local(example_input, mesh, [Replicate()])

        # y_d = dn_dist(up_dist(input_dtensor))
        # print("Distributed result:", y_d)
        # print("Distributed works!")

        # up_compiled = torch.compile(up_dist)
        # y_up = up_compiled(input_dtensor)
        # dn_compiled = torch.compile(dn_dist)
        # y_dn = dn_compiled(y_up)
        # print("compiled result:", y_dn)
        # print("torch.compile works!")

    #    dist.destroy_process_group()


    @contextmanager
    def dist_local_context():
        # Get absolute path to avoid issues with relative paths
        rendezvous_file = os.path.abspath(filename)
        # Create the file *before* yielding the path
        open(rendezvous_file, "w").close()  # Or touch(filename)
        yield f"{FILE_SCHEMA}{rendezvous_file}"
        # No need to remove the file, torch.dist handles cleanup

    @contextmanager
    def dist_context(backend: str):
        dist.init_process_group(backend=backend)
        dist_print(f"Rank {dist.get_rank()} initialized")
        yield
        dist.barrier()
        dist_print(f"Rank {dist.get_rank()} finished")
        dist.destroy_process_group()

    if __name__ == "__main__":
        # FILE_SCHEMA = "file://"
        # filename = "rendezvous.txt"
        with dist_context("gloo"):
            run()        