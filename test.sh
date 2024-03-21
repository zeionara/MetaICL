#!/bin/bash

# --use_demonstrations \

python test.py \
  --task 'huggingface:MicPie/unpredictable_5k' \
  --k 16 \
  --split test \
  --seed 17 \
  --test_batch_size 32 \
  --method direct \
  --checkpoint ../models/gpt2large-unpredictable5k.pt \
  --out_dir checkpoints/metaicl/unpredictable
