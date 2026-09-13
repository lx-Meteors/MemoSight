#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- 可按需覆盖的默认参数 ----------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EXP_TAG="${EXP_TAG:-r1_distill_demo_infer}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${ROOT_DIR}/experiments}"
COMP_CONFIG="${COMP_CONFIG:-configs/LightThinker/qwen/distillr1.json}"
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
  --use_epl "${USE_EPL}" \
  --spec_decode "${SPEC_DECODE}" \
  "$@"
