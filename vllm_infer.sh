#!/bin/bash

MODEL_PATH=""
INPUT_PATH=""
OUTPUT_PATH=""

python vllm_infer_distributed.py \
    --template qwen3 \
    --model_path "$MODEL_PATH" \
    --tp_size 1 \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --temperature 0.6 \
    --topp 0.95 \
    --topk 20 \
    --max_tokens 2048 \
    --repetition_penalty 1.0 \
    --sample_num 1 \
    --num_instances 1 \
    --batchsize 1 \
    --prompt_type short_qa \
    --logprobs 0 \
    --get_tokens 0 \
    --dataset_name nq \
    --think False
