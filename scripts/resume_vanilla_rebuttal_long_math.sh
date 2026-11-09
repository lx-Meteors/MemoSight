#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Resume the interrupted Vanilla training from this exact Trainer checkpoint.
VANILLA_RESUME_CHECKPOINT="${VANILLA_RESUME_CHECKPOINT:-${PROJECT_ROOT}/experiments/qwen3_8b_vanilla/train/checkpoint-786}"

[[ -d "${VANILLA_RESUME_CHECKPOINT}" ]] || {
    echo "[ERROR] Vanilla checkpoint not found: ${VANILLA_RESUME_CHECKPOINT}" >&2
    exit 1
}
[[ -f "${VANILLA_RESUME_CHECKPOINT}/trainer_state.json" ]] || {
    echo "[ERROR] trainer_state.json not found in: ${VANILLA_RESUME_CHECKPOINT}" >&2
    exit 1
}

# LightThinker and MemoSight were completed before the interrupted Vanilla
# stage. Continue only the remaining stages with the same rebuttal settings:
# Vanilla training -> Vanilla inference/eval -> R-KV inference/eval.
export PROJECT_ROOT
export OUTPUT_BASE_DIR="${PROJECT_ROOT}/experiments"
export VANILLA_RESUME_CHECKPOINT
export VANILLA_CKPT_PATH=""
export RUN_LIGHTTHINKER="false"
export RUN_MEMOSIGHT="false"
export TRAIN_VANILLA="true"
export RUN_RKV="true"
export TRAIN_GPUS="4,5,6,7"
export MAX_LENGTH="4096"
export EPOCHS="5"
export VANILLA_LR="2e-5"
export MICRO_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="16"
export WARMUP_RATIO="0.05"
export WARMUP_STEPS="0"
export SEED="42"
export TARGET_GPUS="4,5,6,7"
export PROCESS_PER_GPU="4"
export MAX_NEW_TOKENS="24576"
export REPETITION_PENALTY="1.1"
export EVAL_DATASETS="aime_2026,hmmt_feb_2025"
export RKV_BUDGET="1536"
export RKV_BUFFER_SIZE="128"
export RKV_WINDOW_SIZE="8"
export RKV_KERNEL_SIZE="7"
export RKV_MIX_LAMBDA="0.1"
export RKV_RETAIN_RATIO="0.2"
export RKV_RETAIN_DIRECTION="last"
export RKV_SIMILARITY_THRESHOLD="0.5"

echo "[INFO] Resuming Vanilla from: ${VANILLA_RESUME_CHECKPOINT}"
echo "[INFO] Remaining stages: Vanilla train -> Vanilla eval -> R-KV eval"
exec bash "${SCRIPT_DIR}/run_rebuttal_long_math.sh"
