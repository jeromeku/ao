#!/bin/bash
set -euo pipefail

# git submodule update --init --recursive
# uv pip install torch --torch-backend=cu129

export DEBUG=1
# export TORCH_CUDA_ARCHITECTURES=sm100
USE_CUDA=1 uv pip install -v -e . --no-build-isolation 2>&1 | tee _build.log
