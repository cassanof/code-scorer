#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 LR" >&2
    exit 1
fi
python3 train.py \
  --seq_len 1000 \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --epochs 10 \
  --lr $1 \
  --weight_decay 0.01 \
  --save_dir "./results_$1" \
  --dataset "Roblox/code_score_gpt35" \
  --model "bigcode/starencoder" \
  --bf16 \
  --no_fp16
