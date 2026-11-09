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
VANILLA_RESUME_CHECKPOINT="${VANILLA_RESUME_CHECKPOINT:-}"

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
RUN_LIGHTTHINKER="${RUN_LIGHTTHINKER:-true}"
RUN_MEMOSIGHT="${RUN_MEMOSIGHT:-true}"

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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
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

require_dir() {
    local path="$1"
    [[ -d "${path}" ]] || die "Directory not found: ${path}"
}

require_bool() {
    local name="$1"
    local value="${2,,}"
    [[ "${value}" == "true" || "${value}" == "false" ]] \
        || die "${name} must be true or false (got: $2)"
}

configure_torch_cuda_libraries() {
    # PyTorch cu124 wheels install libcusparse and libnvJitLink under the
    # active Python environment.  Some clusters prepend an older system CUDA
    # directory to LD_LIBRARY_PATH, causing libcusparse.so.12 to resolve an
    # incompatible libnvJitLink.so.12.  Prefer the matching wheel library.
    local nvjitlink_lib_dir=""
    nvjitlink_lib_dir="$(python -c '
import pathlib
import site

bases = list(site.getsitepackages())
user_site = site.getusersitepackages()
if isinstance(user_site, str):
    bases.append(user_site)
for base in bases:
    candidate = pathlib.Path(base) / "nvidia" / "nvjitlink" / "lib"
    if (candidate / "libnvJitLink.so.12").is_file():
        print(candidate)
        break
')"

    if [[ -n "${nvjitlink_lib_dir}" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${nvjitlink_lib_dir}:"*) ;;
            *) export LD_LIBRARY_PATH="${nvjitlink_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
        esac
        echo "[PREFLIGHT] python=$(command -v python)"
        echo "[PREFLIGHT] prioritized nvJitLink=${nvjitlink_lib_dir}/libnvJitLink.so.12"
    else
        echo "[PREFLIGHT] No Python-wheel libnvJitLink.so.12 found; using the system loader configuration"
    fi
}

validate_checkpoint() {
    local path="$1"
    local display_name="$2"
    local weight_files=()
    [[ -d "${path}" ]] || die "${display_name} checkpoint does not exist: ${path}"
    [[ -f "${path}/config.json" ]] || die "${display_name} checkpoint has no config.json: ${path}"
    shopt -s nullglob
    weight_files=(
        "${path}"/model*.safetensors
        "${path}"/pytorch_model*.bin
        "${path}"/adapter_model*.safetensors
        "${path}"/adapter_model*.bin
    )
    shopt -u nullglob
    [[ "${#weight_files[@]}" -gt 0 ]] || die "${display_name} checkpoint has no model weights: ${path}"
}

validate_eval_dataset() {
    local name="$1"
    local json_path="${PROJECT_ROOT}/data/eval/${name}.json"
    local jsonl_path="${PROJECT_ROOT}/data/eval/${name}.jsonl"
    require_file "${json_path}"
    require_file "${jsonl_path}"
    python -c \
        'import json,sys; a=json.load(open(sys.argv[1], encoding="utf-8")); n=sum(len(v) for v in a.values()) if isinstance(a,dict) else len(a); rows=[json.loads(x) for x in open(sys.argv[2], encoding="utf-8") if x.strip()]; assert n==30 and len(rows)==30 and sorted(int(x["idx"]) for x in rows)==list(range(30)), (n,len(rows))' \
        "${json_path}" "${jsonl_path}" \
        || die "${name} must contain 30 examples in both JSON and JSONL"
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
        --resume_from_checkpoint "${VANILLA_RESUME_CHECKPOINT}" \
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
require_file "${LIGHTTHINKER_CONFIG}"
require_file "${MEMOSIGHT_CONFIG}"
require_file "${VANILLA_CONFIG}"
require_dir "${TOKENIZER_PATH}"
require_bool "TRAIN_VANILLA" "${TRAIN_VANILLA}"
require_bool "RUN_RKV" "${RUN_RKV}"
require_bool "RUN_LIGHTTHINKER" "${RUN_LIGHTTHINKER}"
require_bool "RUN_MEMOSIGHT" "${RUN_MEMOSIGHT}"

[[ "${MAX_NEW_TOKENS}" =~ ^[0-9]+$ ]] || die "MAX_NEW_TOKENS must be an integer"
[[ "${MAX_NEW_TOKENS}" -eq 24576 ]] || die "Rebuttal workflow requires MAX_NEW_TOKENS=24576"
[[ "${PROCESS_PER_GPU}" =~ ^[0-9]+$ ]] || die "PROCESS_PER_GPU must be an integer"
[[ "${PROCESS_PER_GPU}" -eq 4 ]] || die "Rebuttal workflow requires PROCESS_PER_GPU=4 (16 total workers)"
[[ "${TARGET_GPUS}" == "4,5,6,7" ]] || die "Rebuttal workflow requires TARGET_GPUS=4,5,6,7"
[[ "${EVAL_DATASETS}" == "aime_2026,hmmt_feb_2025" ]] \
    || die "Rebuttal workflow requires EVAL_DATASETS=aime_2026,hmmt_feb_2025"

command -v python >/dev/null 2>&1 || die "python is not available; activate the MemoSight environment first"
command -v flock >/dev/null 2>&1 || die "flock is required to prevent duplicate runs"
configure_torch_cuda_libraries
exec 9>"${LOCK_FILE}"
flock -n 9 || die "Another rebuttal long-math workflow is already running"

python -c 'import sys; assert float(sys.argv[1]) > 0' "${REPETITION_PENALTY}" \
    || die "REPETITION_PENALTY must be a positive number"

validate_eval_dataset "aime_2026"
validate_eval_dataset "hmmt_feb_2025"

if [[ "${RUN_RKV,,}" == "true" ]]; then
    [[ "${RKV_RETAIN_DIRECTION}" == "last" || "${RKV_RETAIN_DIRECTION}" == "first" ]] \
        || die "RKV_RETAIN_DIRECTION must be last or first"
    python -c \
        'import sys; b,buf,w,k=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]); lam,ratio,thr=map(float,sys.argv[5:8]); assert b>w>0 and buf>0 and k>0 and k%2==1 and 0<=lam<=1 and 0<ratio<=1 and -1<=thr<=1' \
        "${RKV_BUDGET}" "${RKV_BUFFER_SIZE}" "${RKV_WINDOW_SIZE}" "${RKV_KERNEL_SIZE}" \
        "${RKV_MIX_LAMBDA}" "${RKV_RETAIN_RATIO}" "${RKV_SIMILARITY_THRESHOLD}" \
        || die "Invalid R-KV configuration"
fi

python -c 'import torch; assert torch.cuda.is_available(); print(f"[PREFLIGHT] torch={torch.__version__}, cuda={torch.version.cuda}, visible_gpus={torch.cuda.device_count()}")' \
    || die "PyTorch/CUDA preflight failed"
python "${PROJECT_ROOT}/LightThinker/inference.py" --help >/dev/null \
    || die "Inference imports/CLI preflight failed"
python "${PROJECT_ROOT}/evaluation/eval_file.py" --help >/dev/null \
    || die "Evaluation imports/CLI preflight failed"

old_ifs="${IFS}"
IFS=',' read -r -a preflight_gpus <<< "${TARGET_GPUS}"
IFS="${old_ifs}"
for gpu in "${preflight_gpus[@]}"; do
    GPU_LABEL="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" python -c \
        'import os,torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print("[PREFLIGHT] physical_gpu={}, device={}".format(os.environ["GPU_LABEL"], torch.cuda.get_device_name(0)))' \
        || die "GPU ${gpu} is not available to PyTorch"
done

if [[ "${TRAIN_VANILLA,,}" == "true" ]]; then
    [[ "${TRAIN_GPUS}" == "4,5,6,7" ]] || die "Vanilla training requires TRAIN_GPUS=4,5,6,7"
    python -c \
        'import sys; ml,ep,mb,ga=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]); lr,wr=float(sys.argv[5]),float(sys.argv[6]); assert ml==4096 and ep>0 and mb==1 and ga==16 and lr>0 and 0<=wr<1' \
        "${MAX_LENGTH}" "${EPOCHS}" "${MICRO_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" \
        "${VANILLA_LR}" "${WARMUP_RATIO}" \
        || die "Invalid Vanilla training configuration (expected max_length=4096, micro_batch=1, grad_accumulation=16)"
    validate_checkpoint "${BASE_MODEL_PATH}" "Vanilla base model"
    require_file "${TRAIN_DATA_PATH}"
    if [[ -n "${VANILLA_RESUME_CHECKPOINT}" ]]; then
        validate_checkpoint "${VANILLA_RESUME_CHECKPOINT}" "Vanilla resume"
        require_file "${VANILLA_RESUME_CHECKPOINT}/trainer_state.json"
        expected_vanilla_train_dir="$(cd "${OUTPUT_BASE_DIR}/${VANILLA_EXP_TAG}/train" && pwd -P)"
        resume_parent_dir="$(cd "$(dirname "${VANILLA_RESUME_CHECKPOINT}")" && pwd -P)"
        [[ "${resume_parent_dir}" == "${expected_vanilla_train_dir}" ]] \
            || die "Vanilla resume checkpoint must be inside ${expected_vanilla_train_dir}"
    fi
    python -c \
        'import json,sys; f=open(sys.argv[1], encoding="utf-8"); n=sum(json.loads(line) is not None for line in f if line.strip()); assert n>0' \
        "${TRAIN_DATA_PATH}" \
        || die "Training data is empty or invalid JSONL: ${TRAIN_DATA_PATH}"
    if [[ "${DEEPSPEED_CONFIG}" == /* ]]; then
        require_file "${DEEPSPEED_CONFIG}"
    else
        require_file "${PROJECT_ROOT}/${DEEPSPEED_CONFIG}"
    fi
    command -v deepspeed >/dev/null 2>&1 || die "deepspeed is not available; activate the MemoSight environment first"
    python "${PROJECT_ROOT}/LightThinker/train.py" --help >/dev/null \
        || die "Training imports/CLI preflight failed"
fi

LIGHTTHINKER_CKPT=""
MEMOSIGHT_CKPT=""
if [[ "${RUN_LIGHTTHINKER,,}" == "true" ]]; then
    LIGHTTHINKER_CKPT="$(latest_checkpoint "${LIGHTTHINKER_CKPT_PATH}" "${LIGHTTHINKER_EXP_TAG}" "LightThinker")"
    validate_checkpoint "${LIGHTTHINKER_CKPT}" "LightThinker"
fi
if [[ "${RUN_MEMOSIGHT,,}" == "true" ]]; then
    MEMOSIGHT_CKPT="$(latest_checkpoint "${MEMOSIGHT_CKPT_PATH}" "${MEMOSIGHT_EXP_TAG}" "MemoSight")"
    validate_checkpoint "${MEMOSIGHT_CKPT}" "MemoSight"
fi
VANILLA_CKPT=""
if [[ "${TRAIN_VANILLA,,}" == "false" ]]; then
    VANILLA_CKPT="$(latest_checkpoint "${VANILLA_CKPT_PATH}" "${VANILLA_EXP_TAG}" "Vanilla")"
    validate_checkpoint "${VANILLA_CKPT}" "Vanilla"
fi

echo "================================================================"
echo "Rebuttal long-math workflow"
echo "project_root=${PROJECT_ROOT}"
echo "base_model=${BASE_MODEL_PATH}"
echo "lightthinker_checkpoint=${LIGHTTHINKER_CKPT:-SKIPPED}"
echo "memosight_checkpoint=${MEMOSIGHT_CKPT:-SKIPPED}"
echo "train_vanilla=${TRAIN_VANILLA}"
echo "vanilla_resume_checkpoint=${VANILLA_RESUME_CHECKPOINT:-auto/latest}"
echo "run_lightthinker=${RUN_LIGHTTHINKER}"
echo "run_memosight=${RUN_MEMOSIGHT}"
echo "train_gpus=${TRAIN_GPUS}"
echo "target_gpus=${TARGET_GPUS}"
echo "process_per_gpu=${PROCESS_PER_GPU}"
echo "run_rkv=${RUN_RKV}"
echo "rkv_budget=${RKV_BUDGET}, rkv_buffer_size=${RKV_BUFFER_SIZE}"
echo "log=${RUN_LOG}"
echo "================================================================"

if [[ "${RUN_LIGHTTHINKER,,}" == "true" ]]; then
    run_inference "LightThinker" "${LIGHTTHINKER_EXP_TAG}" "${LIGHTTHINKER_CKPT}" "${LIGHTTHINKER_CONFIG}" false false false anchor-thought false
fi
if [[ "${RUN_MEMOSIGHT,,}" == "true" ]]; then
    run_inference "MemoSight" "${MEMOSIGHT_EXP_TAG}" "${MEMOSIGHT_CKPT}" "${MEMOSIGHT_CONFIG}" true true false anchor-thought false
fi

if [[ "${TRAIN_VANILLA,,}" == "true" ]]; then
    train_vanilla
    VANILLA_CKPT="$(latest_checkpoint "${VANILLA_CKPT_PATH}" "${VANILLA_EXP_TAG}" "Vanilla")"
    validate_checkpoint "${VANILLA_CKPT}" "Vanilla"
fi

run_inference "Vanilla" "${VANILLA_EXP_TAG}" "${VANILLA_CKPT}" "${VANILLA_CONFIG}" false false true normal false

if [[ "${RUN_RKV,,}" == "true" ]]; then
    run_inference "R-KV (Vanilla checkpoint)" "${RKV_EXP_TAG}" "${VANILLA_CKPT}" "${VANILLA_CONFIG}" false false true normal true
fi

echo
echo "================================================================"
echo "[DONE] Requested rebuttal stages finished on:"
echo "       aime_2026, hmmt_feb_2025"
if [[ "${RUN_LIGHTTHINKER,,}" == "true" ]]; then
    echo "LightThinker results: ${OUTPUT_BASE_DIR}/${LIGHTTHINKER_EXP_TAG}"
fi
if [[ "${RUN_MEMOSIGHT,,}" == "true" ]]; then
    echo "MemoSight results:    ${OUTPUT_BASE_DIR}/${MEMOSIGHT_EXP_TAG}"
fi
echo "Vanilla results:      ${OUTPUT_BASE_DIR}/${VANILLA_EXP_TAG}"
if [[ "${RUN_RKV,,}" == "true" ]]; then
    echo "R-KV results:         ${OUTPUT_BASE_DIR}/${RKV_EXP_TAG}"
fi
echo "Log:                  ${RUN_LOG}"
echo "================================================================"
