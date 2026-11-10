#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reviewer H4W6 batch scaling is the first stage. Set RUN_H4W6_BATCH=false
# only when intentionally skipping the batch matrix.
RUN_H4W6_BATCH="${RUN_H4W6_BATCH:-true}"
if [[ "${RUN_H4W6_BATCH,,}" == "true" ]]; then
    bash "${SCRIPT_DIR}/run_h4w6_batch_experiments.sh"
elif [[ "${RUN_H4W6_BATCH,,}" != "false" ]]; then
    echo "[ERROR] RUN_H4W6_BATCH must be true or false" >&2
    exit 1
fi

# Fair-comparison launcher:
#   Task 2: Breadcrumbs with five-token compression blocks (c=5).
#   Task 3: LightThinker + Medusa with two future-token heads (d=2).
# Separate experiment tags prevent old c=16/four-head checkpoints from being
# resumed or evaluated accidentally.
RUN_TASK_1=false \
RUN_TASK_2=true \
RUN_TASK_3=true \
BREADCRUMBS_COMPRESSION_RATIO=5 \
MEDUSA_NUM_HEADS=2 \
BREADCRUMBS_EXP_TAG=qwen3_8b_breadcrumbs_c5 \
MEDUSA_EXP_TAG=qwen3_8b_lightthinker_medusa_d2 \
    bash "${SCRIPT_DIR}/run_three_new_methods.sh" "$@"
