#!/usr/bin/env bash
# One-shot MTP self-speculative-decoding acceptance-rate analysis.
#
# Fill in the 5 fields in the CONFIG block below, then just run:
#     bash scripts/run_mtp_acceptance.sh
# It sweeps the draft length over the datasets, then aggregates into a
# CSV / plot / JSON under $RESULT_ROOT.
#
# Every field is also overridable from the command line, e.g.:
#     CKPT_PATH=/my/ckpt DRAFT_LENS="1 2" bash scripts/run_mtp_acceptance.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
# inference.py uses absolute imports (`from LightThinker.utils import *`), so the
# repo root must be on PYTHONPATH — activating a conda env does NOT do this.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ============================================================================
# CONFIG — edit these 5 lines, then run.  (all overridable via env vars)
# ============================================================================
CKPT_PATH="${CKPT_PATH:-/path/to/your/train_output/checkpoint-xxxx}"   # MTP-trained checkpoint dir
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/Qwen3-8B}"     # tokenizer dir
COMPRESS_CONFIG="${COMPRESS_CONFIG:-./configs/LightThinker/qwen/adaptive_mtp_v1.json}"  # config WITH an `mtp` block
DRAFT_LENS="${DRAFT_LENS:-1 2}"   # speculative register tokens/step to sweep; keep <= training max_offset (used as-is)
DATASETS="${DATASETS:-gsm8k}"     # space-separated: gsm8k mmlu bbh gpqa
# ============================================================================

# ---- knobs with sane defaults (usually leave as-is) ------------------------
MODEL_TYPE="${MODEL_TYPE:-qwen}"                       # qwen | llama (sets chat tokens below)
GPU="${GPU:-0}"                                        # CUDA device id
WITH_BASELINE="${WITH_BASELINE:-1}"                    # 1 = also run a non-speculative pass for a real speedup ref
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"   # match pipeline.sh infer (default 1.1)
UPDATE_ATTN="${UPDATE_ATTN:-local}"
INDEX="${INDEX:-1}"
SPLIT_SIZE="${SPLIT_SIZE:-1}"
RESULT_ROOT="${RESULT_ROOT:-mtp_accept_results}"

# inference flags — defaults match pipeline.sh infer for aug-wo-pc-apa-mtp + use_epl true
USE_EPL="${USE_EPL:-true}"
COMPRESS_PROMPT="${COMPRESS_PROMPT:-false}"
PREFILL_COMPRESS="${PREFILL_COMPRESS:-false}"
ROLLING_ROPE="${ROLLING_ROPE:-false}"
DIAGONAL="${DIAGONAL:-false}"
BI_DIRECTIONAL="${BI_DIRECTIONAL:-false}"
SEE_CURRENT="${SEE_CURRENT:-false}"
EXCLUDE_CONTINUE="${EXCLUDE_CONTINUE:-false}"
OUTPUT_COMPRESS_INSTRUCTION="${OUTPUT_COMPRESS_INSTRUCTION:-None}"

# chat tokens picked from MODEL_TYPE unless explicitly overridden
if [ "$MODEL_TYPE" = "llama" ]; then
    BOS_TOKEN="${BOS_TOKEN:-<|start_header_id|>}"
    EOS_TOKEN="${EOS_TOKEN:-<|eot_id|>}"
else
    BOS_TOKEN="${BOS_TOKEN:-<|im_start|>}"
    EOS_TOKEN="${EOS_TOKEN:-<|im_end|>}"
fi

ROOT_DIR="./LightThinker"

# ---- sanity checks ---------------------------------------------------------
[ -e "$CKPT_PATH" ]        || { echo "!! CKPT_PATH not found: $CKPT_PATH   (edit the CONFIG block)"; exit 1; }
[ -f "$COMPRESS_CONFIG" ]  || { echo "!! COMPRESS_CONFIG not found: $COMPRESS_CONFIG"; exit 1; }

# NOTE: DRAFT_LENS is used exactly as given. Keep each value <= the max_offset
# the checkpoint was trained with, or those extra positions were never trained
# and their acceptance is a fake signal.
echo ">>> DRAFT_LENS='${DRAFT_LENS}'  datasets='${DATASETS}'  gpu=${GPU}  ->  ${RESULT_ROOT}/"

mkdir -p "$RESULT_ROOT"
GROUP_ARGS=()

run_one () {          # $1 = draft_len ("baseline" -> spec_decode off)
    local dl="$1" spec out_tag extra_args
    if [ "$dl" = "baseline" ]; then
        spec="false"; out_tag="${RESULT_ROOT}/baseline"; extra_args=()
    else
        spec="true";  out_tag="${RESULT_ROOT}/dl${dl}"; extra_args=(--mtp_draft_len "$dl")
    fi
    echo ">>> running ${out_tag} (spec_decode=${spec})"
    CUDA_VISIBLE_DEVICES="$GPU" python "${ROOT_DIR}/inference.py" \
        --model_tag "mtp_accept" \
        --model_short_tag "mtp_dl_${dl}" \
        --ckpt "0" \
        --model_path "$CKPT_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --compress_config "$COMPRESS_CONFIG" \
        --model_type "$MODEL_TYPE" \
        --bos_token "$BOS_TOKEN" \
        --eos_token "$EOS_TOKEN" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --update_attention_method "$UPDATE_ATTN" \
        --use_EPL "$USE_EPL" \
        --compress_prompt "$COMPRESS_PROMPT" \
        --prefill_compress "$PREFILL_COMPRESS" \
        --rolling_rope "$ROLLING_ROPE" \
        --diagonal "$DIAGONAL" \
        --bi_directional "$BI_DIRECTIONAL" \
        --see_current "$SEE_CURRENT" \
        --exclude_continue "$EXCLUDE_CONTINUE" \
        --output_compress_instruction "$OUTPUT_COMPRESS_INSTRUCTION" \
        --output_tag "$out_tag" \
        --datasets $DATASETS \
        --split_size "$SPLIT_SIZE" \
        --index "$INDEX" \
        --spec_decode "$spec" \
        "${extra_args[@]}"
    # only speculative runs carry mtp_stats worth grouping.
    # group PER DATASET (not pooled) so each dataset gets its own acceptance
    # numbers — mirrors run_runtime_breakdown.sh. Group name = dl<N>_<dataset>.
    if [ "$dl" != "baseline" ]; then
        for ds in $DATASETS; do
            GROUP_ARGS+=(--group "dl${dl}_${ds}=${out_tag}/${ds}/**/*.jsonl")
        done
    fi
}

if [ "$WITH_BASELINE" = "1" ]; then
    run_one baseline
fi
for dl in $DRAFT_LENS; do
    run_one "$dl"
done

echo ">>> aggregating acceptance stats"
python scripts/analyze_mtp_acceptance.py \
    "${GROUP_ARGS[@]}" \
    --csv "${RESULT_ROOT}/summary.csv" \
    --plot "${RESULT_ROOT}/acceptance.png" \
    --json "${RESULT_ROOT}/summary.json"

echo ""
echo ">>> done. results:"
echo "      ${RESULT_ROOT}/summary.csv     (τ / α / tok-fwd / acc per draft length × dataset)"
echo "      ${RESULT_ROOT}/acceptance.png  (per-position acceptance curve + speedup)"
echo "      ${RESULT_ROOT}/summary.json    (full metrics incl. per-position α_k)"
