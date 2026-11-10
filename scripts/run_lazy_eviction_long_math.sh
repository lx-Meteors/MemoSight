#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Evaluate the already-trained Vanilla Qwen3-8B checkpoint only.
export PROJECT_ROOT
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"
export TRAIN_VANILLA="false"
export RUN_LIGHTTHINKER="false"
export RUN_MEMOSIGHT="false"
export RUN_VANILLA="false"
export RUN_RKV="false"
export RUN_LAZY_EVICTION="true"

# Same long-math protocol as the completed rebuttal runs.
export TARGET_GPUS="0,1,2,3,4,5,6,7"
export PROCESS_PER_GPU="4"
export MAX_NEW_TOKENS="24576"
export REPETITION_PENALTY="1.1"
export EVAL_DATASETS="aime_2026,hmmt_feb_2025"

# Paper/repository AIME configuration adapted to the 24K generation protocol:
# alpha=1e-4 for Qwen, W=363 from the official AIME launcher comment, and
# B=4915 (30% of the paper's 16,384-token AIME generation limit).
export LAZY_BUDGET="${LAZY_BUDGET:-4915}"
export LAZY_WINDOW_SIZE="${LAZY_WINDOW_SIZE:-363}"
export LAZY_ALPHA="${LAZY_ALPHA:-0.0001}"

echo "[INFO] LazyEviction on the latest trained Vanilla checkpoint"
echo "[INFO] GPUs=${TARGET_GPUS}; shards=32; max_new_tokens=${MAX_NEW_TOKENS}"
echo "[INFO] B=${LAZY_BUDGET}; W=${LAZY_WINDOW_SIZE}; alpha=${LAZY_ALPHA}"
exec bash "${SCRIPT_DIR}/run_rebuttal_long_math.sh"
