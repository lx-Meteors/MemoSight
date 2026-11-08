#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# =============================================================================
# Server configuration. Every value can be overridden with an environment var.
# =============================================================================

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/yyk/yyk06/hpc/hf_sources/hf_Models/Qwen/Qwen3-8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL_PATH}}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/data/train/train.jsonl}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"

LIGHTTHINKER_EXP_TAG="${LIGHTTHINKER_EXP_TAG:-qwen3_8b_lightthinker}"
MEMOSIGHT_EXP_TAG="${MEMOSIGHT_EXP_TAG:-qwen3_8b_memosight}"
VANILLA_EXP_TAG="${VANILLA_EXP_TAG:-qwen3_8b_vanilla}"
RKV_EXP_TAG="${RKV_EXP_TAG:-qwen3_8b_vanilla_rkv}"

# Fixed rebuttal checkpoints. Override either path with an environment variable
# if a later checkpoint should be evaluated.
LIGHTTHINKER_CKPT_PATH="${LIGHTTHINKER_CKPT_PATH:-${PROJECT_ROOT}/experiments/qwen3_8b_lightthinker/train/checkpoint-1305}"
MEMOSIGHT_CKPT_PATH="${MEMOSIGHT_CKPT_PATH:-${PROJECT_ROOT}/experiments/qwen3_8b_memosight/train/checkpoint-1305}"
# Vanilla is trained by this workflow, so its latest checkpoint is resolved
# automatically unless an explicit path is supplied.
VANILLA_CKPT_PATH="${VANILLA_CKPT_PATH:-}"

# Vanilla training: identical common hyperparameters to the existing launcher.
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
EPOCHS="${EPOCHS:-5}"
VANILLA_LR="${VANILLA_LR:-2e-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
SEED="${SEED:-42}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_z3_offload_config.json}"
TRAIN_VANILLA="${TRAIN_VANILLA:-true}"

# Long-CoT inference: four GPUs with four workers each produce 16 shards.
# Reduce PROCESS_PER_GPU to 2 or 1 if 24K Vanilla KV cache pressure causes OOM.
TARGET_GPUS="${TARGET_GPUS:-4,5,6,7}"
PROCESS_PER_GPU="${PROCESS_PER_GPU:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24576}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
EVAL_DATASETS="${EVAL_DATASETS:-aime_2026,hmmt_feb_2025}"

# R-KV is training-free and reuses VANILLA_CKPT.  The default 1536-token
# budget is the paper's lossless fixed budget for its 8B AIME setting; Qwen3-8B
# is a new model adaptation, so override RKV_BUDGET for a budget sweep.
# For a controlled comparison, R-KV uses the same project decoding policy as
# Vanilla/LightThinker/MemoSight.  It does not switch only R-KV to the paper's
# temperature-0.6, top-p-0.95, 64-candidate sampling protocol.
RUN_RKV="${RUN_RKV:-true}"
RKV_BUDGET="${RKV_BUDGET:-1536}"
RKV_BUFFER_SIZE="${RKV_BUFFER_SIZE:-128}"
RKV_WINDOW_SIZE="${RKV_WINDOW_SIZE:-8}"
RKV_KERNEL_SIZE="${RKV_KERNEL_SIZE:-7}"
RKV_MIX_LAMBDA="${RKV_MIX_LAMBDA:-0.1}"
RKV_RETAIN_RATIO="${RKV_RETAIN_RATIO:-0.2}"
RKV_RETAIN_DIRECTION="${RKV_RETAIN_DIRECTION:-last}"
RKV_SIMILARITY_THRESHOLD="${RKV_SIMILARITY_THRESHOLD:-0.5}"

# Optional: point this to the server conda environment's bin directory. If it
# is empty, the already activated shell environment is used.
MEMOSIGHT_ENV_BIN="${MEMOSIGHT_ENV_BIN:-}"

PIPELINE_SCRIPT="${PROJECT_ROOT}/scripts/pipeline.sh"
LIGHTTHINKER_CONFIG="${PROJECT_ROOT}/configs/LightThinker/qwen/v1.json"
MEMOSIGHT_CONFIG="${PROJECT_ROOT}/configs/LightThinker/qwen/adaptive_mtp_v1.json"
VANILLA_CONFIG="${PROJECT_ROOT}/configs/LightThinker/qwen/v1.json"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/rebuttal_long_math_${RUN_TS}.log"
LOCK_FILE="${OUTPUT_BASE_DIR}/.rebuttal_long_math.lock"

if [[ -n "${MEMOSIGHT_ENV_BIN}" ]]; then
    export PATH="${MEMOSIGHT_ENV_BIN}:${PATH}"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "${path}" ]] || die "File not found: ${path}"
}

latest_checkpoint() {
    local explicit_path="$1"
    local exp_tag="$2"
    local display_name="$3"
    local train_dir="${OUTPUT_BASE_DIR}/${exp_tag}/train"
    local checkpoint=""

    if [[ -n "${explicit_path}" ]]; then
        [[ -d "${explicit_path}" ]] || die "${display_name} checkpoint does not exist: ${explicit_path}"
        echo "${explicit_path}"
        return 0
    fi

    if [[ -d "${train_dir}" ]]; then
        checkpoint="$(
            find "${train_dir}" -maxdepth 1 -type d -name 'checkpoint-*' \
                | sort -V \
                | tail -n 1
        )"
    fi
    [[ -n "${checkpoint}" ]] || die "No ${display_name} checkpoint found under ${train_dir}"
    echo "${checkpoint}"
}

run_inference() {
    local display_name="$1"
    local exp_tag="$2"
    local checkpoint="$3"
    local comp_config="$4"
    local use_epl="$5"
    local spec_decode="$6"
    local vanilla_inference="$7"
    local eval_method="$8"
    local rkv="$9"

    echo
    echo "================================================================"
    echo "[INFO] ${display_name} inference"
    echo "checkpoint=${checkpoint}"
    echo "datasets=${EVAL_DATASETS}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "use_epl=${use_epl}, spec_decode=${spec_decode}"
    echo "vanilla_inference=${vanilla_inference}"
    echo "rkv=${rkv}"
    echo "eval_method=${eval_method}"
    echo "================================================================"

    bash "${PIPELINE_SCRIPT}" \
        --stage infer \
        --root_dir "${PROJECT_ROOT}" \
        --exp_tag "${exp_tag}" \
        --output_base_dir "${OUTPUT_BASE_DIR}" \
        --model_type qwen \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --infer_model_path "${checkpoint}" \
        --comp_config "${comp_config}" \
        --use_epl "${use_epl}" \
        --spec_decode "${spec_decode}" \
        --vanilla_inference "${vanilla_inference}" \
        --rkv "${rkv}" \
        --rkv_budget "${RKV_BUDGET}" \
        --rkv_buffer_size "${RKV_BUFFER_SIZE}" \
        --rkv_window_size "${RKV_WINDOW_SIZE}" \
        --rkv_kernel_size "${RKV_KERNEL_SIZE}" \
        --rkv_mix_lambda "${RKV_MIX_LAMBDA}" \
        --rkv_retain_ratio "${RKV_RETAIN_RATIO}" \
        --rkv_retain_direction "${RKV_RETAIN_DIRECTION}" \
        --rkv_similarity_threshold "${RKV_SIMILARITY_THRESHOLD}" \
        --eval_method "${eval_method}" \
        --datasets "${EVAL_DATASETS}" \
        --target_gpus "${TARGET_GPUS}" \
        --process_per_gpu "${PROCESS_PER_GPU}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --repetition_penalty "${REPETITION_PENALTY}"
}

train_vanilla() {
    echo
    echo "================================================================"
    echo "[INFO] Vanilla training (mode=normal, no compression/EPL/MTP)"
    echo "output=${OUTPUT_BASE_DIR}/${VANILLA_EXP_TAG}/train"
    echo "================================================================"

    bash "${PIPELINE_SCRIPT}" \
        --stage train \
        --root_dir "${PROJECT_ROOT}" \
        --exp_tag "${VANILLA_EXP_TAG}" \
        --output_base_dir "${OUTPUT_BASE_DIR}" \
        --model_type qwen \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --train_model_path "${BASE_MODEL_PATH}" \
        --train_data_path "${TRAIN_DATA_PATH}" \
        --use_epl false \
        --mode normal \
        --conf_version v1 \
        --lr "${VANILLA_LR}" \
        --max_length "${MAX_LENGTH}" \
        --epochs "${EPOCHS}" \
        --save_strategy epoch \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --deepspeed_config "${DEEPSPEED_CONFIG}" \
        --train_gpus "${TRAIN_GPUS}" \
        --seed "${SEED}"
}

require_file "${PIPELINE_SCRIPT}"
require_file "${TRAIN_DATA_PATH}"
require_file "${PROJECT_ROOT}/data/eval/aime_2026.json"
require_file "${PROJECT_ROOT}/data/eval/aime_2026.jsonl"
require_file "${PROJECT_ROOT}/data/eval/hmmt_feb_2025.json"
require_file "${PROJECT_ROOT}/data/eval/hmmt_feb_2025.jsonl"
require_file "${LIGHTTHINKER_CONFIG}"
require_file "${MEMOSIGHT_CONFIG}"
require_file "${VANILLA_CONFIG}"

command -v python >/dev/null 2>&1 || die "python is not available; activate the MemoSight environment first"
command -v deepspeed >/dev/null 2>&1 || die "deepspeed is not available; activate the MemoSight environment first"
command -v flock >/dev/null 2>&1 || die "flock is required to prevent duplicate runs"

exec 9>"${LOCK_FILE}"
flock -n 9 || die "Another rebuttal long-math workflow is already running"

LIGHTTHINKER_CKPT="$(latest_checkpoint "${LIGHTTHINKER_CKPT_PATH}" "${LIGHTTHINKER_EXP_TAG}" "LightThinker")"
MEMOSIGHT_CKPT="$(latest_checkpoint "${MEMOSIGHT_CKPT_PATH}" "${MEMOSIGHT_EXP_TAG}" "MemoSight")"

echo "================================================================"
echo "Rebuttal long-math workflow"
echo "project_root=${PROJECT_ROOT}"
echo "base_model=${BASE_MODEL_PATH}"
echo "lightthinker_checkpoint=${LIGHTTHINKER_CKPT}"
echo "memosight_checkpoint=${MEMOSIGHT_CKPT}"
echo "train_vanilla=${TRAIN_VANILLA}"
echo "train_gpus=${TRAIN_GPUS}"
echo "target_gpus=${TARGET_GPUS}"
echo "process_per_gpu=${PROCESS_PER_GPU}"
echo "run_rkv=${RUN_RKV}"
echo "rkv_budget=${RKV_BUDGET}, rkv_buffer_size=${RKV_BUFFER_SIZE}"
echo "log=${RUN_LOG}"
echo "================================================================"

run_inference "LightThinker" "${LIGHTTHINKER_EXP_TAG}" "${LIGHTTHINKER_CKPT}" "${LIGHTTHINKER_CONFIG}" false false false anchor-thought false
run_inference "MemoSight" "${MEMOSIGHT_EXP_TAG}" "${MEMOSIGHT_CKPT}" "${MEMOSIGHT_CONFIG}" true true false anchor-thought false

if [[ "${TRAIN_VANILLA,,}" == "true" ]]; then
    train_vanilla
fi

VANILLA_CKPT="$(latest_checkpoint "${VANILLA_CKPT_PATH}" "${VANILLA_EXP_TAG}" "Vanilla")"
run_inference "Vanilla" "${VANILLA_EXP_TAG}" "${VANILLA_CKPT}" "${VANILLA_CONFIG}" false false true normal false

if [[ "${RUN_RKV,,}" == "true" ]]; then
    run_inference "R-KV (Vanilla checkpoint)" "${RKV_EXP_TAG}" "${VANILLA_CKPT}" "${VANILLA_CONFIG}" false false true normal true
fi

echo
echo "================================================================"
if [[ "${RUN_RKV,,}" == "true" ]]; then
    echo "[DONE] LightThinker, MemoSight, Vanilla, and R-KV finished on:"
else
    echo "[DONE] LightThinker, MemoSight, and Vanilla finished on:"
fi
echo "       aime_2026, hmmt_feb_2025"
echo "LightThinker results: ${OUTPUT_BASE_DIR}/${LIGHTTHINKER_EXP_TAG}"
echo "MemoSight results:    ${OUTPUT_BASE_DIR}/${MEMOSIGHT_EXP_TAG}"
echo "Vanilla results:      ${OUTPUT_BASE_DIR}/${VANILLA_EXP_TAG}"
if [[ "${RUN_RKV,,}" == "true" ]]; then
    echo "R-KV results:         ${OUTPUT_BASE_DIR}/${RKV_EXP_TAG}"
fi
echo "Log:                  ${RUN_LOG}"
echo "================================================================"
