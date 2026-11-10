# Three new rebuttal methods

All launchers use the same defaults as the existing rebuttal workflow:

- Qwen3-8B base model and the existing training JSONL;
- 4K semantic-token training truncation, 5 epochs, seed 42;
- eight training/evaluation GPUs (`0,1,2,3,4,5,6,7`);
- four evaluation workers per GPU (32 shards);
- AIME 2026 and HMMT February 2025;
- a strict 24K (`24576`) output-token ceiling.

## Run all three tasks

```bash
bash scripts/run_three_new_methods.sh
```

Run it in the background:

```bash
mkdir -p logs
LOG=logs/three_new_methods_nohup_$(date +%Y%m%d_%H%M%S).log
nohup bash scripts/run_three_new_methods.sh > "${LOG}" 2>&1 < /dev/null &
echo "PID=$! LOG=${LOG}"
```

The launcher also writes an internal timestamped log under `logs/`.

## Run one task only

```bash
bash scripts/run_foresight_mtp_long_math.sh
bash scripts/run_breadcrumbs_long_math.sh
bash scripts/run_lightthinker_medusa_long_math.sh
```

Run the revised Task 2 and Task 3 together (Breadcrumbs `c=5`, Medusa
`d=2`/two future-token heads), followed by the Reviewer H4W6 static-batch
matrix:

```bash
bash scripts/run_breadcrumbs_c5_medusa_d2_long_math.sh
```

The appended batch task measures Vanilla AR, MemoSight AR, and MemoSight SD at
batch 4/16/64 with three repeats on a fixed seed-42 GSM8K performance subset.
It records exact output IDs, aggregate throughput, accuracy, completion times,
allocated/reserved memory, valid/physical KV occupancy, and SD acceptance.
Batch-64 OOM triggers a complete batch-32 comparison; batch-16 OOM similarly
triggers batch 8. Set `BATCH_SAMPLE_COUNT=0` for full GSM8K, or run only the
batch task with `bash scripts/run_h4w6_batch_experiments.sh`.

Task 1 trains the full Qwen3-8B model with MemoSight's foresight/register MTP
objective, without compression or EPL. Task 2 distills a Breadcrumbs student
from the frozen trained Vanilla checkpoint; this intentionally replaces the
paper's RL-teacher stage and uses a compression interval of five. Task 3
freezes the trained LightThinker checkpoint, trains only two Medusa-1 residual
heads to match MemoSight's `d=2`, and truncates the official 63-node sparse
tree to depth two for lossless greedy verification at inference.

Existing checkpoints are resumed automatically. To skip training and only
evaluate an existing artifact, set the corresponding switch and path, for
example:

```bash
TRAIN_TASK_2=false \
BREADCRUMBS_CKPT_PATH=/path/to/checkpoint-N \
bash scripts/run_breadcrumbs_long_math.sh
```

The server defaults require no edits when the repository and checkpoints use
the same layout as the existing rebuttal run. Common optional overrides are
`TRAIN_GPUS`, `TARGET_GPUS`, `PROCESS_PER_GPU`, `BASE_MODEL_PATH`,
`VANILLA_CKPT_PATH`, and `LIGHTTHINKER_CKPT_PATH`.

Method references:

- Breadcrumbs Reasoning: <https://arxiv.org/abs/2510.13797>
- Medusa: <https://arxiv.org/abs/2401.10774>
- Official Medusa implementation: <https://github.com/FasterDecoding/Medusa>
