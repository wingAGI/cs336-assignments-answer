#!/bin/bash

# 检查是否传递了参数
if [ -z "$1" ]; then
    echo "Usage: $0 <training_method>"
    exit 1
fi

METHOD=$1
MODEL_SIZE=$2

# mmlu
python scripts/evaluate_mmlu_zero_shot.py $METHOD --model_size $MODEL_SIZE
# gsm8k
python scripts/evaluate_gsm8k_zero_shot.py $METHOD --model_size $MODEL_SIZE
# alpaca_eval
python scripts/evaluate_alpaca_eval_zero_shot.py --mode $METHOD --model_name 0.6B_$METHOD
python scripts/evaluate_alpaca_eval_zero_shot_scorer_ray.py 0.6B_$METHOD qwen3-1.7B $METHOD zero-shot
# SimpleSafetyTests
python scripts/evaluate_sst_zero_shot.py --mode $METHOD --model_name 0.6B_$METHOD
python scripts/evaluate_sst_zero_shot_scorer_ray.py $METHOD 0.6B_$METHOD