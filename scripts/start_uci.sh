#!/bin/bash

MODEL_PATH=${1:-"models/trained_policy_net.keras"}

python -m engine.uci --model "$MODEL_PATH"
