#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Shared server paths.  These defaults match the existing rebuttal launcher.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/yyk/yyk06/hpc/hf_sources/hf_Models/Qwen/Qwen3-8B}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/data/train/train.jsonl}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"
VANILLA_CKPT_PATH="${VANILLA_CKPT_PATH:-}"
LIGHTTHINKER_CKPT_PATH="${LIGHTTHINKER_CKPT_PATH:-${OUTPUT_BASE_DIR}/qwen3_8b_lightthinker/train/checkpoint-1305}"

# Run switches.  The three small wrapper scripts below select one task only.
RUN_TASK_1="${RUN_TASK_1:-true}"
RUN_TASK_2="${RUN_TASK_2:-true}"
RUN_TASK_3="${RUN_TASK_3:-true}"
TRAIN_TASK_1="${TRAIN_TASK_1:-true}"
TRAIN_TASK_2="${TRAIN_TASK_2:-true}"
TRAIN_TASK_3="${TRAIN_TASK_3:-true}"

FORESIGHT_EXP_TAG="${FORESIGHT_EXP_TAG:-qwen3_8b_foresight_mtp_only}"
BREADCRUMBS_EXP_TAG="${BREADCRUMBS_EXP_TAG:-qwen3_8b_breadcrumbs}"
MEDUSA_EXP_TAG="${MEDUSA_EXP_TAG:-qwen3_8b_lightthinker_medusa}"
FORESIGHT_CKPT_PATH="${FORESIGHT_CKPT_PATH:-}"
BREADCRUMBS_CKPT_PATH="${BREADCRUMBS_CKPT_PATH:-}"
MEDUSA_HEAD_PATH="${MEDUSA_HEAD_PATH:-}"

# The same train/eval budget is used for all three methods.
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
TARGET_GPUS="${TARGET_GPUS:-0,1,2,3,4,5,6,7}"
PROCESS_PER_GPU="${PROCESS_PER_GPU:-4}"
EVAL_DATASETS="${EVAL_DATASETS:-aime_2026,hmmt_feb_2025}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24576}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
EPOCHS="${EPOCHS:-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SEED="${SEED:-42}"
FORESIGHT_LR="${FORESIGHT_LR:-2e-5}"
BREADCRUMBS_LR="${BREADCRUMBS_LR:-2e-5}"
MEDUSA_LR="${MEDUSA_LR:-1e-3}"
BREADCRUMBS_COMPRESSION_RATIO="${BREADCRUMBS_COMPRESSION_RATIO:-16}"
MEDUSA_NUM_HEADS="${MEDUSA_NUM_HEADS:-4}"
MEDUSA_NUM_LAYERS="${MEDUSA_NUM_LAYERS:-1}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/ds_z3_offload_config.json}"
MEMOSIGHT_ENV_BIN="${MEMOSIGHT_ENV_BIN:-}"

PIPELINE="${PROJECT_ROOT}/scripts/pipeline.sh"
BASE_CONFIG="${PROJECT_ROOT}/configs/LightThinker/qwen/v1.json"
FORESIGHT_CONFIG="${PROJECT_ROOT}/configs/LightThinker/qwen/foresight_mtp_only_v1.json"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/three_new_methods_${RUN_TS}.log"

if [[ -n "${MEMOSIGHT_ENV_BIN}" ]]; then
    export PATH="${MEMOSIGHT_ENV_BIN}:${PATH}"
fi
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_BASE_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

die() { echo "[ERROR] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "File not found: $1"; }
require_dir() { [[ -d "$1" ]] || die "Directory not found: $1"; }
require_bool() {
    local value="${2,,}"
    [[ "${value}" == "true" || "${value}" == "false" ]] \
        || die "$1 must be true or false (got: $2)"
}

configure_torch_cuda_libraries() {
    local nvjitlink_lib_dir=""
    nvjitlink_lib_dir="$(python -c '
import pathlib, site
bases = list(site.getsitepackages())
user = site.getusersitepackages()
if isinstance(user, str): bases.append(user)
for base in bases:
    candidate = pathlib.Path(base) / "nvidia" / "nvjitlink" / "lib"
    if (candidate / "libnvJitLink.so.12").is_file():
        print(candidate)
        break
')"
    if [[ -n "${nvjitlink_lib_dir}" ]]; then
        export LD_LIBRARY_PATH="${nvjitlink_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

latest_checkpoint() {
    local explicit="$1"
    local train_dir="$2"
    local result=""
    if [[ -n "${explicit}" ]]; then
        result="${explicit}"
    elif [[ -d "${train_dir}" ]]; then
        result="$(find "${train_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    fi
    [[ -n "${result}" && -d "${result}" ]] || die "No checkpoint found under ${train_dir}"
    echo "${result}"
}

validate_model_checkpoint() {
    local path="$1"
    require_dir "${path}"
    require_file "${path}/config.json"
    compgen -G "${path}/model*.safetensors" >/dev/null \
        || compgen -G "${path}/pytorch_model*.bin" >/dev/null \
        || die "No model weights found in ${path}"
}

validate_medusa_checkpoint() {
    local path="$1"
    require_dir "${path}"
    require_file "${path}/medusa_config.json"
    [[ -f "${path}/model.safetensors" || -f "${path}/pytorch_model.bin" ]] \
        || die "No Medusa head weights found in ${path}"
}

run_inference() {
    local display_name="$1" exp_tag="$2" model_path="$3" tokenizer_path="$4"
    local config_path="$5" vanilla="$6" spec_decode="$7" eval_method="$8"
    shift 8
    echo "================================================================"
    echo "[INFER] ${display_name}"
    echo "model=${model_path}"
    echo "tokenizer=${tokenizer_path}"
    echo "datasets=${EVAL_DATASETS}, max_new_tokens=${MAX_NEW_TOKENS}"
    echo "gpus=${TARGET_GPUS}, process_per_gpu=${PROCESS_PER_GPU}"
    echo "================================================================"
    bash "${PIPELINE}" \
        --stage infer \
        --root_dir "${PROJECT_ROOT}" \
        --exp_tag "${exp_tag}" \
        --output_base_dir "${OUTPUT_BASE_DIR}" \
        --model_type qwen \
        --tokenizer_path "${tokenizer_path}" \
        --infer_model_path "${model_path}" \
        --comp_config "${config_path}" \
        --use_epl false \
        --spec_decode "${spec_decode}" \
        --vanilla_inference "${vanilla}" \
        --eval_method "${eval_method}" \
        --datasets "${EVAL_DATASETS}" \
        --target_gpus "${TARGET_GPUS}" \
        --process_per_gpu "${PROCESS_PER_GPU}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --repetition_penalty "${REPETITION_PENALTY}" \
        "$@"
}

train_foresight() {
    echo "[TRAIN] Task 1: Qwen3-8B + MemoSight foresight MTP only"
    bash "${PIPELINE}" \
        --stage train \
        --root_dir "${PROJECT_ROOT}" \
        --exp_tag "${FORESIGHT_EXP_TAG}" \
        --output_base_dir "${OUTPUT_BASE_DIR}" \
        --model_type qwen \
        --tokenizer_path "${BASE_MODEL_PATH}" \
        --train_model_path "${BASE_MODEL_PATH}" \
        --train_data_path "${TRAIN_DATA_PATH}" \
        --use_epl false \
        --mode mtp-only \
        --conf_version foresight_mtp_only_v1 \
        --lr "${FORESIGHT_LR}" \
        --max_length "${MAX_LENGTH}" \
        --epochs "${EPOCHS}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --deepspeed_config "${DEEPSPEED_CONFIG}" \
        --train_gpus "${TRAIN_GPUS}" \
        --seed "${SEED}"
}

train_breadcrumbs() {
    local teacher_path="$1"
    local output_dir="${OUTPUT_BASE_DIR}/${BREADCRUMBS_EXP_TAG}/train"
    mkdir -p "${output_dir}"
    echo "[TRAIN] Task 2: Breadcrumbs student; frozen Vanilla teacher=${teacher_path}"
    deepspeed --master_port=54222 --include "localhost:${TRAIN_GPUS}" \
        "${PROJECT_ROOT}/LightThinker/train_breadcrumbs.py" \
        --teacher_model_path "${teacher_path}" \
        --tokenizer_path "${teacher_path}" \
        --train_path "${TRAIN_DATA_PATH}" \
        --compress_config "${BASE_CONFIG}" \
        --output_dir "${output_dir}" \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --max_length "${MAX_LENGTH}" \
        --compression_ratio "${BREADCRUMBS_COMPRESSION_RATIO}" \
        --epochs "${EPOCHS}" \
        --lr "${BREADCRUMBS_LR}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --seed "${SEED}"
}

train_medusa() {
    local lightthinker_path="$1"
    local output_dir="${OUTPUT_BASE_DIR}/${MEDUSA_EXP_TAG}/train"
    mkdir -p "${output_dir}"
    echo "[TRAIN] Task 3: Medusa heads only; frozen LightThinker=${lightthinker_path}"
    deepspeed --master_port=54232 --include "localhost:${TRAIN_GPUS}" \
        "${PROJECT_ROOT}/LightThinker/train_medusa.py" \
        --base_model_path "${lightthinker_path}" \
        --tokenizer_path "${lightthinker_path}" \
        --train_path "${TRAIN_DATA_PATH}" \
        --compress_config "${BASE_CONFIG}" \
        --output_dir "${output_dir}" \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --max_length "${MAX_LENGTH}" \
        --num_heads "${MEDUSA_NUM_HEADS}" \
        --num_layers "${MEDUSA_NUM_LAYERS}" \
        --epochs "${EPOCHS}" \
        --lr "${MEDUSA_LR}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --seed "${SEED}"
}

for flag in RUN_TASK_1 RUN_TASK_2 RUN_TASK_3 TRAIN_TASK_1 TRAIN_TASK_2 TRAIN_TASK_3; do
    require_bool "${flag}" "${!flag}"
done
require_file "${PIPELINE}"
require_file "${BASE_CONFIG}"
require_file "${FORESIGHT_CONFIG}"
require_file "${TRAIN_DATA_PATH}"
require_file "${DEEPSPEED_CONFIG}"
require_file "${PROJECT_ROOT}/data/eval/aime_2026.jsonl"
require_file "${PROJECT_ROOT}/data/eval/hmmt_feb_2025.jsonl"
command -v python >/dev/null || die "python is unavailable"
command -v deepspeed >/dev/null || die "deepspeed is unavailable; activate the memosight environment"

configure_torch_cuda_libraries
python -c 'import torch; assert torch.cuda.is_available(); print(f"[PREFLIGHT] torch={torch.__version__}, cuda={torch.version.cuda}, visible_gpus={torch.cuda.device_count()}")' \
    || die "PyTorch/CUDA preflight failed"

echo "================================================================"
echo "Three-method rebuttal workflow"
echo "task1=${RUN_TASK_1}/${TRAIN_TASK_1}, task2=${RUN_TASK_2}/${TRAIN_TASK_2}, task3=${RUN_TASK_3}/${TRAIN_TASK_3}"
echo "train_gpus=${TRAIN_GPUS}; eval_gpus=${TARGET_GPUS}; workers_per_gpu=${PROCESS_PER_GPU}"
echo "datasets=${EVAL_DATASETS}; output_limit=${MAX_NEW_TOKENS}"
echo "log=${RUN_LOG}"
echo "================================================================"

if [[ "${RUN_TASK_1,,}" == "true" ]]; then
    [[ "${TRAIN_TASK_1,,}" == "false" ]] || train_foresight
    FORESIGHT_CKPT="$(latest_checkpoint "${FORESIGHT_CKPT_PATH}" "${OUTPUT_BASE_DIR}/${FORESIGHT_EXP_TAG}/train")"
    validate_model_checkpoint "${FORESIGHT_CKPT}"
    run_inference "Foresight-MTP-only" "${FORESIGHT_EXP_TAG}" \
        "${FORESIGHT_CKPT}" "${FORESIGHT_CKPT}" "${FORESIGHT_CONFIG}" false true normal
fi

if [[ "${RUN_TASK_2,,}" == "true" ]]; then
    VANILLA_CKPT="$(latest_checkpoint "${VANILLA_CKPT_PATH}" "${OUTPUT_BASE_DIR}/qwen3_8b_vanilla/train")"
    validate_model_checkpoint "${VANILLA_CKPT}"
    [[ "${TRAIN_TASK_2,,}" == "false" ]] || train_breadcrumbs "${VANILLA_CKPT}"
    BREADCRUMBS_CKPT="$(latest_checkpoint "${BREADCRUMBS_CKPT_PATH}" "${OUTPUT_BASE_DIR}/${BREADCRUMBS_EXP_TAG}/train")"
    validate_model_checkpoint "${BREADCRUMBS_CKPT}"
    run_inference "Breadcrumbs (Vanilla teacher distillation)" "${BREADCRUMBS_EXP_TAG}" \
        "${BREADCRUMBS_CKPT}" "${BREADCRUMBS_CKPT}" "${BASE_CONFIG}" true false normal \
        --breadcrumbs true \
        --breadcrumbs_compression_ratio "${BREADCRUMBS_COMPRESSION_RATIO}"
fi

if [[ "${RUN_TASK_3,,}" == "true" ]]; then
    validate_model_checkpoint "${LIGHTTHINKER_CKPT_PATH}"
    [[ "${TRAIN_TASK_3,,}" == "false" ]] || train_medusa "${LIGHTTHINKER_CKPT_PATH}"
    MEDUSA_CKPT="$(latest_checkpoint "${MEDUSA_HEAD_PATH}" "${OUTPUT_BASE_DIR}/${MEDUSA_EXP_TAG}/train")"
    validate_medusa_checkpoint "${MEDUSA_CKPT}"
    run_inference "LightThinker + Medusa-1" "${MEDUSA_EXP_TAG}" \
        "${LIGHTTHINKER_CKPT_PATH}" "${LIGHTTHINKER_CKPT_PATH}" "${BASE_CONFIG}" false false anchor-thought \
        --medusa true \
        --medusa_head_path "${MEDUSA_CKPT}"
fi

echo "================================================================"
echo "[DONE] Requested tasks completed."
echo "Task 1: ${OUTPUT_BASE_DIR}/${FORESIGHT_EXP_TAG}"
echo "Task 2: ${OUTPUT_BASE_DIR}/${BREADCRUMBS_EXP_TAG}"
echo "Task 3: ${OUTPUT_BASE_DIR}/${MEDUSA_EXP_TAG}"
echo "Log: ${RUN_LOG}"
echo "================================================================"
