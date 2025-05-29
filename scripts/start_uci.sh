#!/bin/bash

MODEL_PATH=${1:-"models/trained_policy_net.h5"}

python -m engine.uci --model "$MODEL_PATH"
