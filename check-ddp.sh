#!/bin/bash
set -euo pipefail
CHECKPOINT_DIR=checkpoints
REF_CHECKPOINT=$CHECKPOINT_DIR/ref/ddp-1-0.pt
TEST_CHECKPOINT_DIR=$CHECKPOINT_DIR/test

# Check REF_CHECKPOINT exists
if [ ! -f $REF_CHECKPOINT ]; then
    echo "Reference checkpoint $REF_CHECKPOINT does not exist"
    exit 1
fi

#Check TEST_CHECKPOINT_DIR exists
if [ ! -d $TEST_CHECKPOINT_DIR ]; then
    echo "Test checkpoint directory $TEST_CHECKPOINT_DIR does not exist"
    exit 1
fi

CMD="python my_tests/check_params.py --checkpoints_dir $TEST_CHECKPOINT_DIR --ref_checkpoint $REF_CHECKPOINT"
echo $CMD
$CMD
