#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =============================================================================
# 用户配置区：通常只需要修改这里
# 也可以在命令前用同名环境变量临时覆盖，例如：
# BASE_MODEL_PATH=/models/Qwen3-8B bash scripts/train_lightthinker_memosight.sh
# =============================================================================

# 项目、基础模型、Tokenizer、训练数据和训练产物目录
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL_PATH}}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/data/train/train.jsonl}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"

# 两个实验的目录名；checkpoint 会保存在：
#   ${OUTPUT_BASE_DIR}/${LIGHTTHINKER_EXP_TAG}/train/checkpoint-*
#   ${OUTPUT_BASE_DIR}/${MEMOSIGHT_EXP_TAG}/train/checkpoint-*
LIGHTTHINKER_EXP_TAG="${LIGHTTHINKER_EXP_TAG:-qwen3_8b_lightthinker}"
MEMOSIGHT_EXP_TAG="${MEMOSIGHT_EXP_TAG:-qwen3_8b_memosight}"

# 训练资源与公共超参数
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
EPOCHS="${EPOCHS:-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
SEED="${SEED:-42}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_z3_offload_config.json}"

# LightThinker：上下文压缩，不使用 EPL/MTP
LIGHTTHINKER_LR="${LIGHTTHINKER_LR:-2e-5}"
LIGHTTHINKER_MODE="${LIGHTTHINKER_MODE:-aug-wo-pc}"
LIGHTTHINKER_CONF_VERSION="${LIGHTTHINKER_CONF_VERSION:-v1}"

# MemoSight：自适应上下文压缩 + EPL + MTP
MEMOSIGHT_LR="${MEMOSIGHT_LR:-2e-5}"
MEMOSIGHT_MODE="${MEMOSIGHT_MODE:-aug-wo-pc-apa-mtp}"
MEMOSIGHT_CONF_VERSION="${MEMOSIGHT_CONF_VERSION:-adaptive_mtp_v1}"

# =============================================================================
# 执行逻辑：先 LightThinker，成功结束后再 MemoSight
# =============================================================================

PIPELINE_SCRIPT="${PROJECT_ROOT}/scripts/pipeline.sh"

[[ -f "${PIPELINE_SCRIPT}" ]] || {
    echo "[ERROR] 找不到训练入口：${PIPELINE_SCRIPT}" >&2
    exit 1
}

[[ -f "${TRAIN_DATA_PATH}" ]] || {
    echo "[ERROR] 找不到训练数据：${TRAIN_DATA_PATH}" >&2
    exit 1
}

command -v deepspeed >/dev/null 2>&1 || {
    echo "[ERROR] 当前环境找不到 deepspeed，请先激活 memosight 环境。" >&2
    exit 1
}

mkdir -p "${OUTPUT_BASE_DIR}"
cd "${PROJECT_ROOT}"

print_configuration() {
    echo "============================================================"
    echo "串行训练配置"
    echo "PROJECT_ROOT=${PROJECT_ROOT}"
    echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
    echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
    echo "TRAIN_DATA_PATH=${TRAIN_DATA_PATH}"
    echo "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
    echo "TRAIN_GPUS=${TRAIN_GPUS}"
    echo "MAX_LENGTH=${MAX_LENGTH}"
    echo "EPOCHS=${EPOCHS}"
    echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
    echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
    echo "============================================================"
}

show_resume_state() {
    local exp_tag="$1"
    local train_dir="${OUTPUT_BASE_DIR}/${exp_tag}/train"
    local latest_checkpoint=""

    if [[ -d "${train_dir}" ]]; then
        latest_checkpoint="$(
            find "${train_dir}" -maxdepth 1 -type d -name 'checkpoint-*' \
                | sort -V \
                | tail -n 1
        )"
    fi

    if [[ -n "${latest_checkpoint}" ]]; then
        echo "[INFO] 检测到已有 checkpoint，将自动续训：${latest_checkpoint}"
    else
        echo "[INFO] 未检测到已有 checkpoint，将从基础模型开始训练。"
    fi
}

run_training() {
    local display_name="$1"
    local exp_tag="$2"
    local use_epl="$3"
    local mode="$4"
    local conf_version="$5"
    local learning_rate="$6"

    echo
    echo "============================================================"
    echo "开始训练 ${display_name}"
    echo "exp_tag=${exp_tag}"
    echo "mode=${mode}"
    echo "use_epl=${use_epl}"
    echo "conf_version=${conf_version}"
    echo "learning_rate=${learning_rate}"
    echo "============================================================"

    show_resume_state "${exp_tag}"

    bash "${PIPELINE_SCRIPT}" \
        --stage train \
        --root_dir "${PROJECT_ROOT}" \
        --exp_tag "${exp_tag}" \
        --output_base_dir "${OUTPUT_BASE_DIR}" \
        --model_type qwen \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --train_model_path "${BASE_MODEL_PATH}" \
        --train_data_path "${TRAIN_DATA_PATH}" \
        --use_epl "${use_epl}" \
        --mode "${mode}" \
        --conf_version "${conf_version}" \
        --lr "${learning_rate}" \
        --max_length "${MAX_LENGTH}" \
        --epochs "${EPOCHS}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --deepspeed_config "${DEEPSPEED_CONFIG}" \
        --train_gpus "${TRAIN_GPUS}" \
        --seed "${SEED}"

    echo "[INFO] ${display_name} 训练完成。"
}

print_configuration

run_training \
    "LightThinker" \
    "${LIGHTTHINKER_EXP_TAG}" \
    "false" \
    "${LIGHTTHINKER_MODE}" \
    "${LIGHTTHINKER_CONF_VERSION}" \
    "${LIGHTTHINKER_LR}"

run_training \
    "MemoSight" \
    "${MEMOSIGHT_EXP_TAG}" \
    "true" \
    "${MEMOSIGHT_MODE}" \
    "${MEMOSIGHT_CONF_VERSION}" \
    "${MEMOSIGHT_LR}"

echo
echo "============================================================"
echo "LightThinker 和 MemoSight 已全部训练完成。"
echo "LightThinker: ${OUTPUT_BASE_DIR}/${LIGHTTHINKER_EXP_TAG}/train"
echo "MemoSight:    ${OUTPUT_BASE_DIR}/${MEMOSIGHT_EXP_TAG}/train"
echo "============================================================"
