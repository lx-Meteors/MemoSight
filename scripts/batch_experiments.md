# Reviewer H4W6 static-batch experiment

The formal launcher compares the same three generation paths on one visible
GPU:

- `vanilla_ar`: trained Vanilla checkpoint, compression and EPL disabled;
- `memosight_ar`: trained MemoSight checkpoint, compression and EPL enabled,
  speculative decoding disabled;
- `memosight_sd`: the same MemoSight target/config with only speculative
  decoding enabled.

Run only the batch matrix:

```bash
bash scripts/run_h4w6_batch_experiments.sh
```

It is also appended automatically after the revised Task 2/Task 3 workflow:

```bash
bash scripts/run_breadcrumbs_c5_medusa_d2_long_math.sh
```

## Defaults

- fixed random 256-example GSM8K subset (`seed=42`, sorted original IDs);
- batch sizes 4, 16, and 64;
- BF16, greedy decoding, repetition penalty 1.1;
- one independent 128-token warmup followed by three complete repeats;
- 10,240-token output ceiling, matching the original GSM8K protocol;
- one visible GPU (`BATCH_GPU=0`).

Use `BATCH_SAMPLE_COUNT=0` for full GSM8K. To deliberately use the rebuttal
long-output protocol instead, set `BATCH_MAX_NEW_TOKENS=24576` for the entire
matrix; never change it for only one method or batch size.

If any batch-64 condition is identified as CUDA OOM, the launcher runs batch
32 for all three methods. A batch-16 OOM similarly triggers batch 8. Other
failures remain classified as `FAILED`, not OOM.

## Artifacts

Each timestamped `experiments/h4w6_batch_*` directory contains:

- the immutable sample manifest and copied compression configs;
- one JSON report and log per method/batch condition;
- lossless per-request JSONL for every repeat;
- exact-output comparison reports (AR vs SD and batch 4 vs larger batches);
- `summary.csv` and `summary.json`, including same-batch speedups;
- `status.tsv` with OK/OOM/FAILED/equivalence status;
- checkpoint/config paths, git state, source/manifest SHA256, software, and GPU
  metadata.

The measured throughput is aggregate committed output tokens divided by the
GPU-synchronized generation wall time. It includes tokenization, prefill,
decode, real compression, KV compaction, SD verification, and output decode,
but excludes model loading, warmup, scoring, and file writes. This is an
offline static-batch result, not a continuous-batching service benchmark.

The Python entry also accepts `aime_2026` and `hmmt_feb_2025`, but each contains
only 30 unique questions. A run with `--batch_size 64` therefore has an actual
batch of 30 and must not be reported as a true batch-64 measurement.
