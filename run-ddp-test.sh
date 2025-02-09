#!/bin/bash

set -euo pipefail
WORLD_SIZE=${1:-2}
# Generate reference checkpoint
REF_CMD="torchrun --nproc_per_node 1 my_tests/ddp-test.py"
TEST_CMD="torchrun --nproc_per_node $WORLD_SIZE my_tests/ddp-test.py"
CHECK_CMD="python my_tests/check_params.py --ref_checkpoint checkpoints/ref/ddp-0.pt --checkpoints_dir checkpoints/test"
echo $REF_CMD
$REF_CMD
echo $TEST_CMD
$TEST_CMD
echo $CHECK_CMD
$CHECK_CMD
