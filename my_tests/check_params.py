import argparse
from pathlib import Path

import torch

from torchao.dtypes.nf4tensor import NF4Tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_checkpoint", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    args = parser.parse_args()

    ref_state_dict = torch.load(args.ref_checkpoint, weights_only=True, map_location="cpu")

    print(f"Ref checkpoint: {Path(args.ref_checkpoint).stem}")

    for path in Path(args.checkpoints_dir).glob("*.pt"):
        print(f"Checking checkpoint {path.stem}")
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        assert ref_state_dict.keys() == state_dict.keys()
        for name in ref_state_dict.keys():
            ref_param = ref_state_dict[name]
            test_param = state_dict[name]
            if not torch.allclose(ref_param, test_param):
                if isinstance(ref_param, NF4Tensor):
                    ref_param = ref_param.get_original_weight()
                    assert isinstance(test_param, NF4Tensor)
                    test_param = test_param.get_original_weight()
                diff = (ref_param - test_param).abs().max()
                print(f" \u2718 Param {name} differs by {diff}")
            else:
                print(f" \u2713 Param {name} is consistent")
        print(f"Passed!")

    state_dict = torch.load(args.path)
    print(state_dict)
