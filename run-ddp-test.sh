#!/bin/bash

set -euo pipefail
WORLD_SIZE=${1:-2}


# Test params
GLOBAL_BS=8
DIM=128
NUM_LINEARS=1
NUM_STEPS=3

PARAMS="--global_bs $GLOBAL_BS --dim $DIM --num_linears $NUM_LINEARS --num_steps $NUM_STEPS"
SAVE_DIR="checkpoints"
REF_DIR="${SAVE_DIR}/ref"
TEST_DIR="${SAVE_DIR}/test"
REF_CMD="torchrun --nproc_per_node 1 my_tests/ddp-test.py $PARAMS --save_dir $REF_DIR"
TEST_CMD="torchrun --nproc_per_node $WORLD_SIZE my_tests/ddp-test.py $PARAMS --save_dir $TEST_DIR"
CHECK_CMD="python my_tests/check_params.py --ref_checkpoint_dir $REF_DIR --test_checkpoints_dir $TEST_DIR"
CLEANUP_CMD="rm -rf $SAVE_DIR"

# Generate reference checkpoint
echo $REF_CMD
$REF_CMD
echo "---"
sleep 2

# Generate test checkpoints
echo $TEST_CMD
$TEST_CMD
echo "---"
sleep 2

# Check params
echo $CHECK_CMD
$CHECK_CMD
echo "---"
sleep 2

# Cleanup
echo $CLEANUP_CMD
$CLEANUP_CMD
