#!/bin/bash
python3 train.py --seq_len 1000 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --epochs 10 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --save_dir "./results" \
  --dataset "Roblox/code_score_gpt35" \
  --model "bigcode/starencoder" \
  --bf16 \
  --no_fp16
