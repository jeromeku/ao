#! /bin/bash

set -euo pipefail

PROGRAM=$1
TRACE_DIR="./traces"
# Program name without the .py extension
PROGRAM_NAME=$(basename $PROGRAM .py)
SUBDIR=$TRACE_DIR/$PROGRAM_NAME
TRACE_OUTPUT_DIR=$SUBDIR/trace
PARSED_TRACE_OUTPUT_DIR=$SUBDIR/parsed_trace

mkdir -p $TRACE_DIR

CMD="TORCH_TRACE=${TRACE_OUTPUT_DIR} python ${PROGRAM} && \
tlparse ${TRACE_OUTPUT_DIR}/*.log -o ${PARSED_TRACE_OUTPUT_DIR}"

echo $CMD

eval $CMD




