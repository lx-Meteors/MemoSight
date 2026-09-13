#!/usr/bin/env bash
# One-shot MTP decode-time runtime breakdown (prediction / verification /
# compression / other).
#
# Fill in the CONFIG block, then run:
#     bash scripts/run_runtime_breakdown.sh
# Runs inference with --profile_breakdown at a single draft length across the
# datasets, then aggregates a per-dataset breakdown table + stacked-bar plot.
#
# NOTE: profiling inserts torch.cuda.synchronize() around each phase, so this
# run's tokens/s is NOT a valid throughput number. Keep it separate from the
# end-to-end speed / acceptance-rate runs.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
# inference.py uses absolute imports (`from LightThinker.utils import *`), so the
# repo root must be on PYTHONPATH — activating a conda env does NOT do this.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ============================================================================
# CONFIG — edit these, then run.  (all overridable via env vars)
# ============================================================================
CKPT_PATH="${CKPT_PATH:-/path/to/your/train_output/checkpoint-xxxx}"   # MTP-trained checkpoint dir
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/Qwen3-8B}"     # tokenizer dir
COMPRESS_CONFIG="${COMPRESS_CONFIG:-./configs/LightThinker/qwen/adaptive_mtp_v1.json}"  # config WITH an `mtp` block
DRAFT_LEN="${DRAFT_LEN:-2}"       # single deployment draft length; keep <= training max_offset
DATASETS="${DATASETS:-gsm8k}"     # space-separated: gsm8k mmlu bbh gpqa
# ============================================================================

# ---- knobs with sane defaults ----------------------------------------------
MODEL_TYPE="${MODEL_TYPE:-qwen}"                       # qwen | llama
GPU="${GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10240}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"   # match pipeline.sh infer (default 1.1)
UPDATE_ATTN="${UPDATE_ATTN:-local}"
INDEX="${INDEX:-1}"
SPLIT_SIZE="${SPLIT_SIZE:-1}"
RESULT_ROOT="${RESULT_ROOT:-runtime_breakdown_results}"

# inference flags — defaults match pipeline.sh infer for aug-wo-pc-apa-mtp + use_epl true.
# these MUST be passed explicitly: inference.py argparse defaults are the opposite
# (use_EPL=False, prefill_compress=True, compress_prompt=True), so omitting them
# profiles a different decode path than the one trained/deployed.
USE_EPL="${USE_EPL:-true}"
COMPRESS_PROMPT="${COMPRESS_PROMPT:-false}"
PREFILL_COMPRESS="${PREFILL_COMPRESS:-false}"
ROLLING_ROPE="${ROLLING_ROPE:-false}"
DIAGONAL="${DIAGONAL:-false}"
BI_DIRECTIONAL="${BI_DIRECTIONAL:-false}"
SEE_CURRENT="${SEE_CURRENT:-false}"
EXCLUDE_CONTINUE="${EXCLUDE_CONTINUE:-false}"
OUTPUT_COMPRESS_INSTRUCTION="${OUTPUT_COMPRESS_INSTRUCTION:-None}"

if [ "$MODEL_TYPE" = "llama" ]; then
    BOS_TOKEN="${BOS_TOKEN:-<|start_header_id|>}"
    EOS_TOKEN="${EOS_TOKEN:-<|eot_id|>}"
else
    BOS_TOKEN="${BOS_TOKEN:-<|im_start|>}"
    EOS_TOKEN="${EOS_TOKEN:-<|im_end|>}"
fi

ROOT_DIR="./LightThinker"
[ -e "$CKPT_PATH" ]        || { echo "!! CKPT_PATH not found: $CKPT_PATH   (edit the CONFIG block)"; exit 1; }
[ -f "$COMPRESS_CONFIG" ]  || { echo "!! COMPRESS_CONFIG not found: $COMPRESS_CONFIG"; exit 1; }

OUT_TAG="${RESULT_ROOT}/dl${DRAFT_LEN}"
echo ">>> runtime breakdown: draft_len=${DRAFT_LEN}  datasets='${DATASETS}'  gpu=${GPU}  ->  ${OUT_TAG}/"
echo ">>> (profiling adds cuda.synchronize barriers — do NOT read tokens/s from this run)"

# one inference pass over all datasets, with profiling on
CUDA_VISIBLE_DEVICES="$GPU" python "${ROOT_DIR}/inference.py" \
    --model_tag "runtime_bd" \
    --model_short_tag "rbd_dl${DRAFT_LEN}" \
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
    --output_tag "$OUT_TAG" \
    --datasets $DATASETS \
    --split_size "$SPLIT_SIZE" \
    --index "$INDEX" \
    --spec_decode true \
    --mtp_draft_len "$DRAFT_LEN" \
    --profile_breakdown true

# aggregate, one analyzer group per dataset
echo ">>> aggregating runtime breakdown per dataset"
GROUP_ARGS=()
for ds in $DATASETS; do
    GROUP_ARGS+=(--group "${ds}=${OUT_TAG}/${ds}/**/*.jsonl")
done
python scripts/analyze_runtime_breakdown.py \
    "${GROUP_ARGS[@]}" \
    --csv "${RESULT_ROOT}/breakdown.csv" \
    --plot "${RESULT_ROOT}/breakdown.png" \
    --json "${RESULT_ROOT}/breakdown.json"

echo ""
echo ">>> done. results:"
echo "      ${RESULT_ROOT}/breakdown.csv    (per-phase seconds & % per dataset)"
echo "      ${RESULT_ROOT}/breakdown.png    (stacked-bar breakdown)"
echo "      ${RESULT_ROOT}/breakdown.json   (full metrics)"
