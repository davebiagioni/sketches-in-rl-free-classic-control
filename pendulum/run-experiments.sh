#!/bin/bash
set -xe
clear
NUM_SEEDS=50
for K in 2 5 10 20 40 60 80 100
do
  python distributed_trials.py \
    --num-cpus 3 \
    --num-seeds $NUM_SEEDS \
    --max-steps 200 \
    --K $K
done

