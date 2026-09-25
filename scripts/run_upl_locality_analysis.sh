#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# =============================================================================
# 用户配置区：可以直接修改，也可以在命令前用同名环境变量覆盖
# =============================================================================

# 两个训练目录。脚本会自动选择其中编号最大的 checkpoint-*。
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"
MEMOSIGHT_TRAIN_DIR="${MEMOSIGHT_TRAIN_DIR:-${OUTPUT_BASE_DIR}/qwen3_8b_memosight/train}"
LIGHTTHINKER_TRAIN_DIR="${LIGHTTHINKER_TRAIN_DIR:-${OUTPUT_BASE_DIR}/qwen3_8b_lightthinker/train}"

# 如果不想自动找 checkpoint，可直接指定这两个变量。
MEMOSIGHT_MODEL_PATH="${MEMOSIGHT_MODEL_PATH:-}"
LIGHTTHINKER_MODEL_PATH="${LIGHTTHINKER_MODEL_PATH:-}"

# 默认分别使用各自 checkpoint 中保存的 tokenizer；也可以手动覆盖。
MEMOSIGHT_TOKENIZER_PATH="${MEMOSIGHT_TOKENIZER_PATH:-}"
LIGHTTHINKER_TOKENIZER_PATH="${LIGHTTHINKER_TOKENIZER_PATH:-}"

# 必须和两个 checkpoint 训练时使用的配置一致。
MEMOSIGHT_CONFIG="${MEMOSIGHT_CONFIG:-${PROJECT_ROOT}/configs/LightThinker/qwen/adaptive_mtp_v1.json}"
LIGHTTHINKER_CONFIG="${LIGHTTHINKER_CONFIG:-${PROJECT_ROOT}/configs/LightThinker/qwen/v1.json}"

DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/train/train.jsonl}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/analysis_results/upl_locality}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/upl_locality}"

# 三个样本并行运行，一项对应一块 GPU。GPU 不够时减少 SAMPLE_INDICES。
SAMPLE_INDICES="${SAMPLE_INDICES:-1,2,4}"
ANALYSIS_GPUS="${ANALYSIS_GPUS:-0,1,2}"

# 分析参数。
MAX_LENGTH="${MAX_LENGTH:-4096}"
MIN_SEG_LEN="${MIN_SEG_LEN:-128}"
MAX_SEGMENTS="${MAX_SEGMENTS:-3}"
LAYERS="${LAYERS:-mid}"
BUCKET_EDGES="${BUCKET_EDGES:-64,128,256}"
PER_HEAD="${PER_HEAD:-true}"
ROW_NORMALIZE="${ROW_NORMALIZE:-true}"
DTYPE="${DTYPE:-bfloat16}"
STRICT_VOCAB="${STRICT_VOCAB:-true}"
LOGIT_CHUNK_SIZE="${LOGIT_CHUNK_SIZE:-128}"
PREFILL_COMPRESS="${PREFILL_COMPRESS:-false}"

# 固定使用该环境的 Python，避免 deepspeed/系统 Python 串环境。
MEMOSIGHT_ENV_BIN="${MEMOSIGHT_ENV_BIN:-/opt/conda/envs/memosight/bin}"
PYTHON_BIN="${PYTHON_BIN:-${MEMOSIGHT_ENV_BIN}/python}"
ANALYSIS_SCRIPT="${PROJECT_ROOT}/LightThinker/analysis_upl_attention.py"

find_latest_checkpoint() {
    local train_dir="$1"
    [[ -d "${train_dir}" ]] || return 1
    find "${train_dir}" \
        -maxdepth 1 \
        -type d \
        -name 'checkpoint-*' \
        | sort -V \
        | tail -n 1
}

if [[ -z "${MEMOSIGHT_MODEL_PATH}" ]]; then
    MEMOSIGHT_MODEL_PATH="$(find_latest_checkpoint "${MEMOSIGHT_TRAIN_DIR}" || true)"
fi
if [[ -z "${LIGHTTHINKER_MODEL_PATH}" ]]; then
    LIGHTTHINKER_MODEL_PATH="$(find_latest_checkpoint "${LIGHTTHINKER_TRAIN_DIR}" || true)"
fi

[[ -n "${MEMOSIGHT_MODEL_PATH}" && -d "${MEMOSIGHT_MODEL_PATH}" ]] || {
    echo "错误：没有找到 MemoSight checkpoint：${MEMOSIGHT_TRAIN_DIR}/checkpoint-*" >&2
    echo "可通过 MEMOSIGHT_MODEL_PATH=/绝对路径/checkpoint-N 手动指定。" >&2
    exit 1
}
[[ -n "${LIGHTTHINKER_MODEL_PATH}" && -d "${LIGHTTHINKER_MODEL_PATH}" ]] || {
    echo "错误：没有找到 LightThinker checkpoint：${LIGHTTHINKER_TRAIN_DIR}/checkpoint-*" >&2
    echo "可通过 LIGHTTHINKER_MODEL_PATH=/绝对路径/checkpoint-N 手动指定。" >&2
    exit 1
}

MEMOSIGHT_MODEL_PATH="$(cd "${MEMOSIGHT_MODEL_PATH}" && pwd)"
LIGHTTHINKER_MODEL_PATH="$(cd "${LIGHTTHINKER_MODEL_PATH}" && pwd)"
MEMOSIGHT_TOKENIZER_PATH="${MEMOSIGHT_TOKENIZER_PATH:-${MEMOSIGHT_MODEL_PATH}}"
LIGHTTHINKER_TOKENIZER_PATH="${LIGHTTHINKER_TOKENIZER_PATH:-${LIGHTTHINKER_MODEL_PATH}}"

for required_file in \
    "${ANALYSIS_SCRIPT}" \
    "${DATA_PATH}" \
    "${MEMOSIGHT_CONFIG}" \
    "${LIGHTTHINKER_CONFIG}"; do
    [[ -f "${required_file}" ]] || {
        echo "错误：找不到文件：${required_file}" >&2
        exit 1
    }
done

[[ -x "${PYTHON_BIN}" ]] || {
    echo "错误：找不到 Python：${PYTHON_BIN}" >&2
    exit 1
}
command -v flock >/dev/null 2>&1 || {
    echo "错误：系统没有 flock，无法防止同一样本重复运行。" >&2
    exit 1
}

IFS=',' read -r -a SAMPLE_ARRAY <<< "${SAMPLE_INDICES}"
IFS=',' read -r -a GPU_ARRAY <<< "${ANALYSIS_GPUS}"

if (( ${#SAMPLE_ARRAY[@]} == 0 )); then
    echo "错误：SAMPLE_INDICES 不能为空。" >&2
    exit 1
fi
if (( ${#GPU_ARRAY[@]} < ${#SAMPLE_ARRAY[@]} )); then
    echo "错误：GPU 数量少于样本数量。" >&2
    echo "当前 SAMPLE_INDICES=${SAMPLE_INDICES}，ANALYSIS_GPUS=${ANALYSIS_GPUS}" >&2
    echo "例如单卡运行：SAMPLE_INDICES=1 ANALYSIS_GPUS=0 bash $0" >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "UPL locality 分析"
echo "MemoSight checkpoint:   ${MEMOSIGHT_MODEL_PATH}"
echo "LightThinker checkpoint: ${LIGHTTHINKER_MODEL_PATH}"
echo "MemoSight config:        ${MEMOSIGHT_CONFIG}"
echo "LightThinker config:     ${LIGHTTHINKER_CONFIG}"
echo "数据:                    ${DATA_PATH}"
echo "样本:                    ${SAMPLE_INDICES}"
echo "GPU:                     ${ANALYSIS_GPUS}"
echo "层:                      ${LAYERS}"
echo "结果目录:                ${RESULT_DIR}"
echo "============================================================"

PIDS=()
for index in "${!SAMPLE_ARRAY[@]}"; do
    sample_index="${SAMPLE_ARRAY[$index]//[[:space:]]/}"
    gpu_id="${GPU_ARRAY[$index]//[[:space:]]/}"
    sample_dir="${RESULT_DIR}/sample${sample_index}"
    log_file="${LOG_DIR}/sample${sample_index}_${RUN_TS}.log"
    lock_file="${sample_dir}/.analysis.lock"
    pid_file="${sample_dir}/analysis.pid"

    [[ "${sample_index}" =~ ^[0-9]+$ ]] || {
        echo "错误：非法 sample index：${sample_index}" >&2
        exit 1
    }
    [[ -n "${gpu_id}" ]] || {
        echo "错误：sample ${sample_index} 没有对应 GPU。" >&2
        exit 1
    }

    mkdir -p "${sample_dir}"

    nohup env \
        PATH="${MEMOSIGHT_ENV_BIN}:${PATH}" \
        PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/LightThinker${PYTHONPATH:+:${PYTHONPATH}}" \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TOKENIZERS_PARALLELISM=false \
        bash -c '
            lock_file="$1"
            shift
            exec 9>"${lock_file}"
            if ! flock -n 9; then
                echo "错误：sample 目录已有分析任务运行：${lock_file}" >&2
                exit 73
            fi
            exec "$@"
        ' bash "${lock_file}" \
        "${PYTHON_BIN}" "${ANALYSIS_SCRIPT}" \
            --model_path "${MEMOSIGHT_MODEL_PATH}" \
            --baseline_model_path "${LIGHTTHINKER_MODEL_PATH}" \
            --tokenizer_path "${MEMOSIGHT_TOKENIZER_PATH}" \
            --baseline_tokenizer_path "${LIGHTTHINKER_TOKENIZER_PATH}" \
            --compress_config "${MEMOSIGHT_CONFIG}" \
            --baseline_compress_config "${LIGHTTHINKER_CONFIG}" \
            --data_path "${DATA_PATH}" \
            --sample_index "${sample_index}" \
            --max_length "${MAX_LENGTH}" \
            --min_seg_len "${MIN_SEG_LEN}" \
            --max_segments "${MAX_SEGMENTS}" \
            --layers "${LAYERS}" \
            --bucket_edges "${BUCKET_EDGES}" \
            --per_head "${PER_HEAD}" \
            --row_normalize "${ROW_NORMALIZE}" \
            --dtype "${DTYPE}" \
            --strict_vocab "${STRICT_VOCAB}" \
            --logit_chunk_size "${LOGIT_CHUNK_SIZE}" \
            --prefill_compress "${PREFILL_COMPRESS}" \
            --device cuda \
            --out_dir "${sample_dir}" \
        > "${log_file}" 2>&1 &

    run_pid=$!
    PIDS+=("${run_pid}")
    echo "${run_pid}" > "${pid_file}"
    echo "已启动 sample=${sample_index} GPU=${gpu_id} PID=${run_pid}"
    echo "  日志: ${log_file}"
    echo "  结果: ${sample_dir}"
done

echo "============================================================"
echo "所有任务已用 nohup 放到后台；SSH 断开不会停止。"
echo "PIDs: ${PIDS[*]}"
echo "查看日志: tail -f ${LOG_DIR}/sample1_${RUN_TS}.log"
echo "结束全部本次任务: kill ${PIDS[*]}"
echo "============================================================"
