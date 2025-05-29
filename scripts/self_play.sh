#!/bin/bash

MODEL_PATH=${1:-"models/trained_policy_net.h5"}
NUM_GAMES=${2:-100}
OUTPUT_DIR=${3:-"data/self_play"}

python -m self_play.data_generator \
    --model "$MODEL_PATH" \
    --games "$NUM_GAMES" \
    --output "$OUTPUT_DIR"
