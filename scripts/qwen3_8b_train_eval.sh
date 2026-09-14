#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 必须指定训练 JSONL；模型默认使用 /personal/models/Qwen3-8B。
# 可将训练数据作为第一个参数传入，也可设置 QWEN3_TRAIN_DATA_PATH。
if [[ -n "${1:-}" ]] && [[ "${1}" != --* ]]; then
    export QWEN3_TRAIN_DATA_PATH="${1}"
    shift
fi

if [[ -z "${QWEN3_TRAIN_DATA_PATH:-}" ]]; then
    echo "[ERROR] 请设置训练数据路径，例如：" >&2
    echo "bash scripts/qwen3_8b_train_eval.sh /path/to/train.jsonl" >&2
    exit 1
fi

exec bash "${SCRIPT_DIR}/pipeline.sh" qwen3-8b-train-eval "$@"
