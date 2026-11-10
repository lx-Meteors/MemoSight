#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TASK_1=false RUN_TASK_2=false RUN_TASK_3=true \
    exec bash "${SCRIPT_DIR}/run_three_new_methods.sh" "$@"
