#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# =============================================================================
# 用户配置区：可以直接修改，也可以用同名环境变量临时覆盖
# =============================================================================

# Qwen3 基础模型和 tokenizer。可以填写 Hugging Face 模型名或本地绝对路径。
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL_PATH}}"

# 训练数据及输出目录。
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/data/train/train.jsonl}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"
EXP_TAG="${EXP_TAG:-qwen3_8b_lightthinker}"

# GPU 和训练超参数。
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
SEED="${SEED:-42}"

# LightThinker 配置：上下文压缩，不启用 EPL/MTP。
TRAIN_MODE="${TRAIN_MODE:-aug-wo-pc}"
CONF_VERSION="${CONF_VERSION:-v1}"

# Python/Conda 环境。
MEMOSIGHT_ENV_BIN="${MEMOSIGHT_ENV_BIN:-/opt/conda/envs/memosight/bin}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_z3_offload_config.json}"

# =============================================================================
# 后台启动与断点恢复
# =============================================================================

PIPELINE_SCRIPT="${PROJECT_ROOT}/scripts/pipeline.sh"
TRAIN_DIR="${OUTPUT_BASE_DIR}/${EXP_TAG}/train"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qwen3_8b_lightthinker_train_resume_${RUN_TS}.log"
LOCK_FILE="${OUTPUT_BASE_DIR}/${EXP_TAG}/.lightthinker_train.lock"

[[ -f "${PIPELINE_SCRIPT}" ]] || {
    echo "错误：找不到训练入口：${PIPELINE_SCRIPT}" >&2
    exit 1
}

[[ -f "${TRAIN_DATA_PATH}" ]] || {
    echo "错误：找不到训练数据：${TRAIN_DATA_PATH}" >&2
    exit 1
}

[[ -x "${MEMOSIGHT_ENV_BIN}/python" ]] || {
    echo "错误：找不到 Python：${MEMOSIGHT_ENV_BIN}/python" >&2
    exit 1
}

[[ -x "${MEMOSIGHT_ENV_BIN}/deepspeed" ]] || {
    echo "错误：找不到 DeepSpeed：${MEMOSIGHT_ENV_BIN}/deepspeed" >&2
    exit 1
}

command -v flock >/dev/null 2>&1 || {
    echo "错误：系统没有 flock，无法安全阻止重复训练任务。" >&2
    exit 1
}

mkdir -p "${TRAIN_DIR}" "${LOG_DIR}" "$(dirname "${LOCK_FILE}")"

# FD 9 会被后台训练进程继承，因此启动终端退出后锁仍然有效。
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "错误：相同实验目录已有训练任务运行：${OUTPUT_BASE_DIR}/${EXP_TAG}" >&2
    echo "请先检查现有进程，不要让两个任务同时写入同一 checkpoint 目录。" >&2
    exit 1
fi

LATEST_CHECKPOINT="$(
    find "${TRAIN_DIR}" \
        -maxdepth 1 \
        -type d \
        -name 'checkpoint-*' \
        | sort -V \
        | tail -n 1
)"

echo "============================================================"
echo "LightThinker-Qwen3 训练配置"
echo "基础模型: ${BASE_MODEL_PATH}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "训练数据: ${TRAIN_DATA_PATH}"
echo "训练目录: ${TRAIN_DIR}"
echo "GPU: ${TRAIN_GPUS}"
echo "max_length: ${MAX_LENGTH}"
echo "epochs: ${EPOCHS}"
echo "learning_rate: ${LEARNING_RATE}"
echo "checkpoint: 每 ${SAVE_STEPS} steps 保存，只保留最新 1 个"
echo "日志: ${LOG_FILE}"
if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    echo "恢复训练: ${LATEST_CHECKPOINT}"
else
    echo "恢复训练: 未找到 checkpoint，将从基础模型开始"
fi
echo "============================================================"

nohup env \
    PATH="${MEMOSIGHT_ENV_BIN}:${PATH}" \
    bash "${PIPELINE_SCRIPT}" \
        --stage train \
        --root_dir "${PROJECT_ROOT}" \
        --exp_tag "${EXP_TAG}" \
        --output_base_dir "${OUTPUT_BASE_DIR}" \
        --model_type qwen \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --train_model_path "${BASE_MODEL_PATH}" \
        --train_data_path "${TRAIN_DATA_PATH}" \
        --use_epl false \
        --mode "${TRAIN_MODE}" \
        --conf_version "${CONF_VERSION}" \
        --lr "${LEARNING_RATE}" \
        --max_length "${MAX_LENGTH}" \
        --epochs "${EPOCHS}" \
        --save_strategy steps \
        --save_steps "${SAVE_STEPS}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --deepspeed_config "${DEEPSPEED_CONFIG}" \
        --train_gpus "${TRAIN_GPUS}" \
        --seed "${SEED}" \
        > "${LOG_FILE}" 2>&1 9>&9 &

RUN_PID=$!

echo "启动成功"
echo "PID=${RUN_PID}"
echo "日志=${LOG_FILE}"
echo "训练目录=${TRAIN_DIR}"
