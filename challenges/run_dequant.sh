#! /bin/bash

SHAPE="4096 14336"
DTYPES="float32 float16 bfloat16"

for dtype in $DTYPES; do
    CMD="python dequant.py --shape $SHAPE --dtype $dtype --test --benchmark"
    echo "Running $CMD"
    $CMD
    echo "----------------------------------------"
done

