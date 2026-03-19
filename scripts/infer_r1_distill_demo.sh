#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- 可按需覆盖的默认参数 ----------
MODEL_PATH="${MODEL_PATH:-/mnt/lxy/hf_models/DeepSeek-R1-Distill-Qwen-1.5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EXP_TAG="${EXP_TAG:-r1_distill_qwen_1p5b_infer}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/mnt/lxy/RRcot/experiments}"
COMP_CONFIG="${COMP_CONFIG:-configs/LightThinker/qwen/distillr1.json}"
TARGET_GPUS="${TARGET_GPUS:-0}"
PROCESS_PER_GPU="${PROCESS_PER_GPU:-1}"
DATASETS="${DATASETS:-mmlu,gsm8k,gpqa,bbh}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
USE_EPL="${USE_EPL:-false}"
SPEC_DECODE="${SPEC_DECODE:-false}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"

echo "[INFO] run infer with model_path=${MODEL_PATH}"
echo "[INFO] exp_tag=${EXP_TAG}, output_base_dir=${OUTPUT_BASE_DIR}"

bash "${ROOT_DIR}/scripts/pipeline.sh" \
  --stage infer \
  --root_dir "${ROOT_DIR}" \
  --exp_tag "${EXP_TAG}" \
  --output_base_dir "${OUTPUT_BASE_DIR}" \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --comp_config "${COMP_CONFIG}" \
  --model_type "${MODEL_TYPE}" \
  --target_gpus "${TARGET_GPUS}" \
  --process_per_gpu "${PROCESS_PER_GPU}" \
  --datasets "${DATASETS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --use_epl "${USE_EPL}" \
  --spec_decode "${SPEC_DECODE}" \
  "$@"
