#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# =============================================================================
# 用户配置区
# =============================================================================

# 从该训练目录中自动选择编号最大的 checkpoint-*。
CHECKPOINT_TRAIN_DIR="${CHECKPOINT_TRAIN_DIR:-${PROJECT_ROOT}/experiments/qwen3_8b_train_eval/train}"

# SepLLM 推理结果：${OUTPUT_BASE_DIR}/${EXP_TAG}/inference
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"
EXP_TAG="${EXP_TAG:-vanilla_sepllm}"

# 重新启动续跑时必须保持 GPU 数和每卡进程数不变。
TARGET_GPUS="${TARGET_GPUS:-0,1,2,3,4,5,6,7}"
PROCESS_PER_GPU="${PROCESS_PER_GPU:-4}"

DATASETS="${DATASETS:-bbh,gpqa,gsm8k,mmlu}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
RUN_EVALUATION="${RUN_EVALUATION:-true}"

# Python/Conda 环境。环境已经激活时也可以保持这个默认值。
MEMOSIGHT_ENV_BIN="${MEMOSIGHT_ENV_BIN:-/opt/conda/envs/memosight/bin}"

# =============================================================================
# 自动寻找 checkpoint 并在后台启动
# =============================================================================

[[ -d "${CHECKPOINT_TRAIN_DIR}" ]] || {
    echo "错误：训练目录不存在：${CHECKPOINT_TRAIN_DIR}" >&2
    exit 1
}

CKPT_PATH="$(
    find "${CHECKPOINT_TRAIN_DIR}" \
        -maxdepth 1 \
        -type d \
        -name 'checkpoint-*' \
        | sort -V \
        | tail -n 1
)"

if [[ -z "${CKPT_PATH}" ]]; then
    echo "错误：没有找到训练 checkpoint：${CHECKPOINT_TRAIN_DIR}" >&2
    exit 1
fi

CKPT_PATH="$(readlink -f "${CKPT_PATH}")"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/qwen3_8b_sepllm_resume_${RUN_TS}.log"
WORKER_SCRIPT="${PROJECT_ROOT}/scripts/run_sepllm_qwen3.sh"

[[ -x "${WORKER_SCRIPT}" ]] || {
    echo "错误：找不到可执行的 SepLLM 脚本：${WORKER_SCRIPT}" >&2
    exit 1
}

[[ -x "${MEMOSIGHT_ENV_BIN}/python" ]] || {
    echo "错误：找不到 Python：${MEMOSIGHT_ENV_BIN}/python" >&2
    exit 1
}

mkdir -p "${LOG_DIR}"

echo "Checkpoint: ${CKPT_PATH}"
echo "输出目录: ${OUTPUT_BASE_DIR}/${EXP_TAG}/inference"
echo "GPU: ${TARGET_GPUS}"
echo "每卡进程数: ${PROCESS_PER_GPU}"
echo "数据集: ${DATASETS}"
echo "日志文件: ${LOG_FILE}"

nohup env \
    PATH="${MEMOSIGHT_ENV_BIN}:${PATH}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    MODEL_PATH="${CKPT_PATH}" \
    TOKENIZER_PATH="${CKPT_PATH}" \
    OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
    EXP_TAG="${EXP_TAG}" \
    TARGET_GPUS="${TARGET_GPUS}" \
    PROCESS_PER_GPU="${PROCESS_PER_GPU}" \
    DATASETS="${DATASETS}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    RUN_EVALUATION="${RUN_EVALUATION}" \
    PYTHON_BIN="${MEMOSIGHT_ENV_BIN}/python" \
    bash "${WORKER_SCRIPT}" \
    > "${LOG_FILE}" 2>&1 &

RUN_PID=$!

echo "启动成功"
echo "PID=${RUN_PID}"
echo "日志=${LOG_FILE}"
