#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/pipeline.sh"

usage() {
    cat <<'EOF'
Qwen3-8B 一键训练、推理和评测

用法:
  bash scripts/qwen3_8b_train_eval.sh TRAIN_JSONL [pipeline 参数...]
  QWEN3_TRAIN_DATA_PATH=TRAIN_JSONL bash scripts/qwen3_8b_train_eval.sh [pipeline 参数...]

默认流程:
  1. 使用 GPU 0-7 训练 Qwen3-8B
  2. 自动选择最新 checkpoint，在 GPU 0-7 上以每卡 4 个进程并行推理（共 32 个）
  3. 评测 mmlu、gsm8k、gpqa、bbh

默认值:
  模型        /personal/models/Qwen3-8B
  输出目录    <项目目录>/experiments/qwen3_8b_train_eval
  训练轮数    5
  学习率      1e-5
  全局 batch  64（8 卡 × 每卡 2 × 梯度累积 4）

示例:
  bash scripts/qwen3_8b_train_eval.sh ./data/train/train.jsonl

  QWEN3_MODEL_PATH=/models/Qwen3-8B \
    bash scripts/qwen3_8b_train_eval.sh ./data/train/train.jsonl

  # 后置参数会覆盖默认值
  bash scripts/qwen3_8b_train_eval.sh ./data/train/train.jsonl \
    --train_gpus 0,1,2,3 \
    --target_gpus 0,1,2,3 \
    --epochs 1 \
    --datasets gsm8k

完整参数请运行:
  bash scripts/pipeline.sh --help
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
done

[[ -f "${PIPELINE_SCRIPT}" ]] || die "找不到流程脚本: ${PIPELINE_SCRIPT}"

# 必须指定训练 JSONL；模型默认使用 /personal/models/Qwen3-8B。
# 可将训练数据作为第一个参数传入，也可设置 QWEN3_TRAIN_DATA_PATH。
if [[ -n "${1:-}" ]] && [[ "${1}" != --* ]]; then
    train_data_path="${1}"
    if [[ "${train_data_path}" != /* ]]; then
        train_data_path="$(pwd)/${train_data_path}"
    fi
    export QWEN3_TRAIN_DATA_PATH="${train_data_path}"
    shift
fi

if [[ -n "${QWEN3_TRAIN_DATA_PATH:-}" ]] && [[ "${QWEN3_TRAIN_DATA_PATH}" != /* ]]; then
    export QWEN3_TRAIN_DATA_PATH="$(pwd)/${QWEN3_TRAIN_DATA_PATH}"
fi

# 也允许用 pipeline 原生的 --train_data_path 传参。
has_train_data_arg="false"
for arg in "$@"; do
    if [[ "${arg}" == "--train_data_path" ]]; then
        has_train_data_arg="true"
        break
    fi
done

if [[ -z "${QWEN3_TRAIN_DATA_PATH:-}" ]] && [[ "${has_train_data_arg}" != "true" ]]; then
    die "请提供训练 JSONL，例如: bash scripts/qwen3_8b_train_eval.sh /path/to/train.jsonl"
fi

if [[ -n "${QWEN3_TRAIN_DATA_PATH:-}" ]] && [[ ! -f "${QWEN3_TRAIN_DATA_PATH}" ]]; then
    die "训练数据文件不存在: ${QWEN3_TRAIN_DATA_PATH}"
fi

echo "[INFO] 项目目录: ${PROJECT_ROOT}"
echo "[INFO] 训练数据: ${QWEN3_TRAIN_DATA_PATH:---train_data_path 参数}"
echo "[INFO] 基础模型: ${QWEN3_MODEL_PATH:-/personal/models/Qwen3-8B}"
echo "[INFO] 启动训练 -> 推理 -> 评测"

exec bash "${PIPELINE_SCRIPT}" qwen3-8b-train-eval "$@"


# nohup bash scripts/pipeline.sh \
#   --stage infer \
#   --exp_tag qwen3_8b_train_eval \
#   --output_base_dir ./experiments \
#   --tokenizer_path /personal/models/Qwen3-8B \
#   --target_gpus 0,1,2,3,4,5,6,7 \
#   --process_per_gpu 4 \
#   --datasets mmlu,gsm8k,gpqa,bbh \
#   > qwen3_8b_infer_eval.log 2>&1 &