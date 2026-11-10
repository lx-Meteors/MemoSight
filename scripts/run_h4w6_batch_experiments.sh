#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/experiments}"

VANILLA_CKPT_PATH="${VANILLA_CKPT_PATH:-}"
MEMOSIGHT_CKPT_PATH="${MEMOSIGHT_CKPT_PATH:-${OUTPUT_BASE_DIR}/qwen3_8b_memosight/train/checkpoint-1305}"
VANILLA_CONFIG="${VANILLA_CONFIG:-${PROJECT_ROOT}/configs/LightThinker/qwen/v1.json}"
MEMOSIGHT_CONFIG="${MEMOSIGHT_CONFIG:-${PROJECT_ROOT}/configs/LightThinker/qwen/adaptive_mtp_v1.json}"

BATCH_GPU="${BATCH_GPU:-0}"
BATCH_PYTHON="${BATCH_PYTHON:-python}"
BATCH_SAMPLE_COUNT="${BATCH_SAMPLE_COUNT:-256}"  # 0 means full GSM8K
BATCH_SIZES="${BATCH_SIZES:-4 16 64}"
BATCH_REPEATS="${BATCH_REPEATS:-3}"
BATCH_MAX_NEW_TOKENS="${BATCH_MAX_NEW_TOKENS:-10240}"
BATCH_MAX_PROMPT_LEN="${BATCH_MAX_PROMPT_LEN:-1100}"
BATCH_WARMUP_BATCHES="${BATCH_WARMUP_BATCHES:-1}"
BATCH_WARMUP_MAX_NEW_TOKENS="${BATCH_WARMUP_MAX_NEW_TOKENS:-128}"
BATCH_DTYPE="${BATCH_DTYPE:-bfloat16}"
BATCH_SEED="${BATCH_SEED:-42}"
BATCH_REPETITION_PENALTY="${BATCH_REPETITION_PENALTY:-1.1}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_RUN_ROOT="${BATCH_RUN_ROOT:-${OUTPUT_BASE_DIR}/h4w6_batch_${RUN_TS}}"

INFER_PY="${PROJECT_ROOT}/LightThinker/inference_batched.py"
COMPARE_PY="${PROJECT_ROOT}/scripts/compare_batch_outputs.py"
SUMMARY_PY="${PROJECT_ROOT}/scripts/summarize_batch_experiments.py"
MANIFEST="${BATCH_RUN_ROOT}/samples.json"
STATUS_FILE="${BATCH_RUN_ROOT}/status.tsv"

die() { echo "[ERROR] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "File not found: $1"; }
require_dir() { [[ -d "$1" ]] || die "Directory not found: $1"; }

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

configure_torch_cuda_libraries() {
    local nvjitlink_lib_dir=""
    nvjitlink_lib_dir="$("${BATCH_PYTHON}" -c '
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

require_file "${INFER_PY}"
require_file "${COMPARE_PY}"
require_file "${SUMMARY_PY}"
require_file "${VANILLA_CONFIG}"
require_file "${MEMOSIGHT_CONFIG}"
command -v "${BATCH_PYTHON}" >/dev/null 2>&1 || die "Python not found: ${BATCH_PYTHON}"

VANILLA_CKPT_PATH="$(latest_checkpoint "${VANILLA_CKPT_PATH}" "${OUTPUT_BASE_DIR}/qwen3_8b_vanilla/train")"
require_dir "${VANILLA_CKPT_PATH}"
require_dir "${MEMOSIGHT_CKPT_PATH}"
mkdir -p "${BATCH_RUN_ROOT}/samples"
cp "${VANILLA_CONFIG}" "${BATCH_RUN_ROOT}/vanilla_config.json"
cp "${MEMOSIGHT_CONFIG}" "${BATCH_RUN_ROOT}/memosight_config.json"
printf 'condition\tstatus\texit_code\n' > "${STATUS_FILE}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/LightThinker:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
configure_torch_cuda_libraries

CUDA_VISIBLE_DEVICES="${BATCH_GPU}" "${BATCH_PYTHON}" -c '
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "exactly one GPU must be visible"
print(f"[PREFLIGHT] torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")
'

export MANIFEST BATCH_SAMPLE_COUNT BATCH_SEED
"${BATCH_PYTHON}" - <<'PY'
import json
import os
import random
from pathlib import Path
from dataset_reader import GSM8KReader

reader = GSM8KReader()
count = int(os.environ["BATCH_SAMPLE_COUNT"])
if count < 0 or count > len(reader):
    raise SystemExit(f"invalid BATCH_SAMPLE_COUNT={count}; GSM8K has {len(reader)} samples")
if count == 0:
    sample_ids = list(range(len(reader)))
else:
    sample_ids = sorted(random.Random(int(os.environ["BATCH_SEED"])).sample(range(len(reader)), count))
path = Path(os.environ["MANIFEST"])
path.write_text(json.dumps({"gsm8k": sample_ids}, indent=2), encoding="utf-8")
print(f"[MANIFEST] {len(sample_ids)} unique GSM8K samples -> {path}")
PY

echo "================================================================"
echo "Reviewer H4W6 static-batch benchmark"
echo "run_root=${BATCH_RUN_ROOT}"
echo "vanilla=${VANILLA_CKPT_PATH}"
echo "memosight=${MEMOSIGHT_CKPT_PATH}"
echo "gpu=${BATCH_GPU}; batches=${BATCH_SIZES}; repeats=${BATCH_REPEATS}"
echo "samples=${BATCH_SAMPLE_COUNT}; max_new_tokens=${BATCH_MAX_NEW_TOKENS}; dtype=${BATCH_DTYPE}"
echo "================================================================"

unexpected_failure=0
oom_at_64=false
oom_at_16=false

run_condition() {
    local method="$1"
    local batch_size="$2"
    local model_path config_path use_epl spec_decode
    case "${method}" in
        vanilla_ar)
            model_path="${VANILLA_CKPT_PATH}"
            config_path="${VANILLA_CONFIG}"
            use_epl=false
            spec_decode=false
            ;;
        memosight_ar)
            model_path="${MEMOSIGHT_CKPT_PATH}"
            config_path="${MEMOSIGHT_CONFIG}"
            use_epl=true
            spec_decode=false
            ;;
        memosight_sd)
            model_path="${MEMOSIGHT_CKPT_PATH}"
            config_path="${MEMOSIGHT_CONFIG}"
            use_epl=true
            spec_decode=true
            ;;
        *) die "Unknown method: ${method}" ;;
    esac

    local stem="${method}.bs${batch_size}.gsm8k"
    local report="${BATCH_RUN_ROOT}/${stem}.json"
    local log="${BATCH_RUN_ROOT}/${stem}.log"
    local sample_dir="${BATCH_RUN_ROOT}/samples/${method}.bs${batch_size}"
    echo "[RUN] ${stem}"
    if CUDA_VISIBLE_DEVICES="${BATCH_GPU}" "${BATCH_PYTHON}" "${INFER_PY}" \
        --model_path "${model_path}" \
        --tokenizer_path "${model_path}" \
        --compress_config "${config_path}" \
        --model_type qwen \
        --bench_mode "${method}" \
        --spec_decode "${spec_decode}" \
        --use_EPL "${use_epl}" \
        --batch_size "${batch_size}" \
        --datasets gsm8k \
        --sample_manifest "${MANIFEST}" \
        --dtype "${BATCH_DTYPE}" \
        --seed "${BATCH_SEED}" \
        --repetition_penalty "${BATCH_REPETITION_PENALTY}" \
        --max_new_tokens "${BATCH_MAX_NEW_TOKENS}" \
        --max_prompt_len "${BATCH_MAX_PROMPT_LEN}" \
        --warmup_batches "${BATCH_WARMUP_BATCHES}" \
        --warmup_batch_size "${batch_size}" \
        --warmup_max_new_tokens "${BATCH_WARMUP_MAX_NEW_TOKENS}" \
        --repeats "${BATCH_REPEATS}" \
        --bench_report "${report}" \
        --output_tag "${sample_dir}" \
        > "${log}" 2>&1; then
        printf '%s\tOK\t0\n' "${stem}" >> "${STATUS_FILE}"
    else
        local exit_code=$?
        if grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "${log}"; then
            if [[ "${batch_size}" == "64" ]]; then oom_at_64=true; fi
            if [[ "${batch_size}" == "16" ]]; then oom_at_16=true; fi
            printf '%s\tOOM\t%s\n' "${stem}" "${exit_code}" >> "${STATUS_FILE}"
        else
            printf '%s\tFAILED\t%s\n' "${stem}" "${exit_code}" >> "${STATUS_FILE}"
            unexpected_failure=1
        fi
    fi
}

run_batch_group() {
    local batch_size="$1"
    for method in vanilla_ar memosight_ar memosight_sd; do
        run_condition "${method}" "${batch_size}"
    done
}

for batch_size in ${BATCH_SIZES}; do
    run_batch_group "${batch_size}"
done

has_oom_at() {
    local batch_size="$1"
    if [[ "${batch_size}" == "64" ]]; then [[ "${oom_at_64}" == "true" ]]; return; fi
    if [[ "${batch_size}" == "16" ]]; then [[ "${oom_at_16}" == "true" ]]; return; fi
    return 1
}

if has_oom_at 64; then
    echo "[FALLBACK] At least one batch-64 condition OOMed; running batch 32 for all methods."
    run_batch_group 32
fi
if has_oom_at 16; then
    echo "[FALLBACK] At least one batch-16 condition OOMed; running batch 8 for all methods."
    run_batch_group 8
fi

compare_pair() {
    local left="$1" right="$2" output="$3" label="$4"
    if [[ ! -f "${left}" || ! -f "${right}" ]]; then
        printf '%s\tSKIPPED_MISSING_REPORT\t0\n' "${label}" >> "${STATUS_FILE}"
        return
    fi
    if "${BATCH_PYTHON}" "${COMPARE_PY}" "${left}" "${right}" \
        --require-exact --output "${output}"; then
        printf '%s\tEXACT\t0\n' "${label}" >> "${STATUS_FILE}"
    else
        printf '%s\tDIFFERENT\t1\n' "${label}" >> "${STATUS_FILE}"
    fi
}

all_batches="${BATCH_SIZES}"
has_oom_at 64 && all_batches="${all_batches} 32"
has_oom_at 16 && all_batches="${all_batches} 8"
for batch_size in ${all_batches}; do
    compare_pair \
        "${BATCH_RUN_ROOT}/memosight_ar.bs${batch_size}.gsm8k.json" \
        "${BATCH_RUN_ROOT}/memosight_sd.bs${batch_size}.gsm8k.json" \
        "${BATCH_RUN_ROOT}/equivalence_memosight_ar_sd_bs${batch_size}.json" \
        "compare_memosight_ar_sd_bs${batch_size}"
done
for method in vanilla_ar memosight_ar memosight_sd; do
    for batch_size in ${all_batches}; do
        [[ "${batch_size}" == "4" ]] && continue
        compare_pair \
            "${BATCH_RUN_ROOT}/${method}.bs4.gsm8k.json" \
            "${BATCH_RUN_ROOT}/${method}.bs${batch_size}.gsm8k.json" \
            "${BATCH_RUN_ROOT}/equivalence_${method}_bs4_bs${batch_size}.json" \
            "compare_${method}_bs4_bs${batch_size}"
    done
done

"${BATCH_PYTHON}" "${SUMMARY_PY}" "${BATCH_RUN_ROOT}"
{
    echo "date=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_status_begin"
    git status --short
    echo "git_status_end"
    echo "inference_batched_sha256=$(sha256sum "${INFER_PY}" | awk '{print $1}')"
    echo "manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')"
    CUDA_VISIBLE_DEVICES="${BATCH_GPU}" "${BATCH_PYTHON}" -c \
        'import platform, torch, transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}")'
    CUDA_VISIBLE_DEVICES="${BATCH_GPU}" nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "${BATCH_RUN_ROOT}/metadata.txt"

echo "[DONE] H4W6 batch artifacts: ${BATCH_RUN_ROOT}"
echo "[DONE] Summary: ${BATCH_RUN_ROOT}/summary.csv"
exit "${unexpected_failure}"
