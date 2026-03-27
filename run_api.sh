#!/bin/bash
# model name: deepseek-v3.1-250821 ; claude-sonnet-4-5-20250929 ; gpt-4o; 

MODEL_NAME=""

INPUT_PATH=""
OUTPUT_PATH=""

python -u run_api.py \
  --source "${INPUT_PATH}" \
  --outfile "${OUTPUT_PATH}" \
  --model "${MODEL_NAME}" \
  --temperature 0.6 \
  --type llm_judge \
  --batch_size 16 \
  --n_shot 0 \
  --gene_type gene \
  --dataset_name nq \
  


