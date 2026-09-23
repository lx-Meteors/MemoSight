#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =============================================================================
# 用户配置区：可直接修改，也可用同名环境变量覆盖
# =============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

# Qwen3 模型和 tokenizer。可以是 Hugging Face 模型名或本地绝对路径。
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

# 输出会写入：${OUTPUT_BASE_DIR}/${EXP_TAG}/inference/<dataset>/
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"
EXP_TAG="${EXP_TAG:-qwen3_8b_sepllm}"

# 多卡与分片。重新启动时必须保持这两项不变，否则旧分片无法正确续跑。
TARGET_GPUS="${TARGET_GPUS:-0,1,2,3,4,5,6,7}"
PROCESS_PER_GPU="${PROCESS_PER_GPU:-4}"

# 推理与评测设置
DATASETS="${DATASETS:-bbh,gpqa,gsm8k,mmlu}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
COMPRESS_CONFIG="${COMPRESS_CONFIG:-${PROJECT_ROOT}/configs/LightThinker/qwen/v1.json}"
RUN_EVALUATION="${RUN_EVALUATION:-true}"
EVAL_METHOD="${EVAL_METHOD:-normal}"
EVAL_CACHE_SIZE="${EVAL_CACHE_SIZE:-1024}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# =============================================================================
# 以下通常不需要修改
# =============================================================================

INFERENCE_PY="${PROJECT_ROOT}/LightThinker/inference.py"
EVALUATE_SH="${PROJECT_ROOT}/run_sh/evaluate.sh"
OUTPUT_ROOT="${OUTPUT_BASE_DIR}/${EXP_TAG}"
INFERENCE_DIR="${OUTPUT_ROOT}/inference"
SHARD_LOG_DIR="${INFERENCE_DIR}/inference_log"
LOCK_FILE="${OUTPUT_ROOT}/.sepllm_inference.lock"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

[[ -f "${INFERENCE_PY}" ]] || {
    echo "[ERROR] 找不到推理入口：${INFERENCE_PY}" >&2
    exit 1
}

[[ -f "${COMPRESS_CONFIG}" ]] || {
    echo "[ERROR] 找不到压缩配置：${COMPRESS_CONFIG}" >&2
    exit 1
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "[ERROR] 找不到 Python：${PYTHON_BIN}" >&2
    exit 1
}

mkdir -p "${OUTPUT_ROOT}" "${SHARD_LOG_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/LightThinker:${PYTHONPATH:-}"

# 防止 SSH 断开后旧任务仍在运行，而用户又误启动一套写入相同 JSONL。
if command -v flock >/dev/null 2>&1; then
    exec 9>"${LOCK_FILE}"
    if ! flock -n 9; then
        echo "[ERROR] 相同输出目录已有 SepLLM 任务运行：${OUTPUT_ROOT}" >&2
        echo "[ERROR] 请先检查进程，不要让两套任务同时写同一批 JSONL。" >&2
        exit 1
    fi
else
    echo "[WARNING] 系统没有 flock，无法自动阻止重复启动。"
fi

csv_to_array() {
    local input="$1"
    local output_name="$2"
    local old_ifs="${IFS}"
    local raw_items=()
    local cleaned_items=()
    local item

    IFS=',' read -r -a raw_items <<< "${input}"
    IFS="${old_ifs}"
    for item in "${raw_items[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -n "${item}" ]] && cleaned_items+=("${item}")
    done
    eval "${output_name}=(\"\${cleaned_items[@]}\")"
}

gpu_array=()
dataset_array=()
csv_to_array "${TARGET_GPUS}" gpu_array
csv_to_array "${DATASETS}" dataset_array

[[ "${#gpu_array[@]}" -gt 0 ]] || {
    echo "[ERROR] TARGET_GPUS 不能为空。" >&2
    exit 1
}

[[ "${#dataset_array[@]}" -gt 0 ]] || {
    echo "[ERROR] DATASETS 不能为空。" >&2
    exit 1
}

[[ "${PROCESS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] PROCESS_PER_GPU 必须是正整数。" >&2
    exit 1
}

split_size=$(( ${#gpu_array[@]} * PROCESS_PER_GPU ))
pids=()

stop_children() {
    local pid
    echo "[INFO] 收到停止信号，正在终止本次启动的推理进程..."
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait || true
    exit 130
}
trap stop_children INT TERM

echo "============================================================"
echo "SepLLM-Qwen3 推理配置"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TARGET_GPUS=${TARGET_GPUS}"
echo "PROCESS_PER_GPU=${PROCESS_PER_GPU}"
echo "SPLIT_SIZE=${split_size}"
echo "DATASETS=${DATASETS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "RUN_EVALUATION=${RUN_EVALUATION}"
echo "============================================================"

logical_id=0
for device in "${gpu_array[@]}"; do
    start_index_0based=$(( logical_id * PROCESS_PER_GPU ))
    end_index_0based=$(( start_index_0based + PROCESS_PER_GPU - 1 ))
    echo "[INFO] 在 GPU ${device} 启动分片 $((start_index_0based + 1))-$((end_index_0based + 1))/${split_size}"

    for ((idx=start_index_0based; idx<=end_index_0based; idx++)); do
        real_index=$(( idx + 1 ))
        shard_log="${SHARD_LOG_DIR}/${real_index}_${EXP_TAG}_${RUN_TS}.log"

        CUDA_VISIBLE_DEVICES="${device}" nohup "${PYTHON_BIN}" "${INFERENCE_PY}" \
            --model_tag "${EXP_TAG}" \
            --model_short_tag "${EXP_TAG}" \
            --ckpt 0 \
            --model_path "${MODEL_PATH}" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --compress_config "${COMPRESS_CONFIG}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --repetition_penalty "${REPETITION_PENALTY}" \
            --output_tag "${INFERENCE_DIR}" \
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
            --split_size "${split_size}" \
            --index "${real_index}" \
            --use_EPL false \
            --datasets "${dataset_array[@]}" \
            > "${shard_log}" 2>&1 &

        pids+=("$!")
        echo "[INFO] shard=${real_index}/${split_size} pid=$! log=${shard_log}"
        sleep 2
    done
    logical_id=$(( logical_id + 1 ))
done

echo "[INFO] 所有推理分片已启动，等待完成。"
failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "[ERROR] 推理子进程失败：PID=${pid}" >&2
        failed=1
    fi
done

[[ "${failed}" -eq 0 ]] || {
    echo "[ERROR] 至少一个分片失败。修复问题后使用相同配置重启即可续跑。" >&2
    exit 1
}

echo "[INFO] 所有 SepLLM 推理分片已完成。"

case "${RUN_EVALUATION}" in
true|True|TRUE|1|yes|Yes|YES)
    [[ -f "${EVALUATE_SH}" ]] || {
        echo "[ERROR] 找不到评估脚本：${EVALUATE_SH}" >&2
        exit 1
    }

    missing_reference=false
    for dataset in "${dataset_array[@]}"; do
        if [[ ! -f "${PROJECT_ROOT}/data/eval/${dataset}.jsonl" ]]; then
            missing_reference=true
            break
        fi
    done
    if [[ "${missing_reference}" == "true" ]]; then
        echo "[INFO] 正在生成评估 reference JSONL..."
        "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation/init.py"
    fi

    for dataset in "${dataset_array[@]}"; do
        echo "[INFO] 开始评估 ${dataset}..."
        bash "${EVALUATE_SH}" \
            "${EVAL_METHOD}" \
            "${TOKENIZER_PATH}" \
            "${dataset}" \
            "${INFERENCE_DIR}/${dataset}" \
            "${COMPRESS_CONFIG}" \
            qwen \
            '<|im_start|>' \
            '<|im_end|>' \
            "${EVAL_CACHE_SIZE}" \
            false
    done
    ;;
false|False|FALSE|0|no|No|NO)
    ;;
*)
    echo "[ERROR] RUN_EVALUATION 只能设置为 true 或 false：${RUN_EVALUATION}" >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "SepLLM-Qwen3 任务完成"
echo "推理结果：${INFERENCE_DIR}"
echo "分片日志：${SHARD_LOG_DIR}"
echo "============================================================"
