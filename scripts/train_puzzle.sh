#!/bin/bash

MODEL_PATH=${1:-"models/policy_net.h5"}
PUZZLES_PATH=${2:-"data/puzzles.pgn"}
EPOCHS=${3:-50}
BATCH_SIZE=${4:-128}
OUTPUT_MODEL=${5:-"models/trained_policy_net.h5"}

python -m training.train_policy \
    --model "$MODEL_PATH" \
    --puzzles "$PUZZLES_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output "$OUTPUT_MODEL"
