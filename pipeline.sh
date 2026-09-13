#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B + H2O single-process inference example.
# Override any value from the shell, for example:
# MODEL_PATH=/models/Qwen3-8B CUDA_VISIBLE_DEVICES=1 bash pipeline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/qwen3-8b-h2o/inference}"
DATASET="${DATASET:-gsm8k}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
H2O_WINDOW_LENGTH="${H2O_WINDOW_LENGTH:-2048}"
H2O_NUM_HH_TOKENS="${H2O_NUM_HH_TOKENS:-1024}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}/${DATASET}"

python3 "${ROOT_DIR}/LightThinker/inference.py" \
  --model_tag qwen3-8b-h2o \
  --model_short_tag qwen3-8b-h2o \
  --ckpt 0 \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --compress_config "${ROOT_DIR}/configs/LightThinker/qwen/v1.json" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --repetition_penalty 1.1 \
  --output_tag "${OUTPUT_DIR}" \
  --model_type qwen \
  --bos_token '<|im_start|>' \
  --eos_token '<|im_end|>' \
  --rolling_rope false \
  --diagonal false \
  --bi_directional false \
  --see_current false \
  --exclude_continue false \
  --output_compress_instruction None \
  --prefill_compress false \
  --compress_prompt false \
  --update_attention_method local \
  --split_size 1 \
  --index 1 \
  --datasets "${DATASET}" \
  --use_EPL false \
  --use_h2o true \
  --h2o_window_length "${H2O_WINDOW_LENGTH}" \
  --h2o_num_hh_tokens "${H2O_NUM_HH_TOKENS}"
