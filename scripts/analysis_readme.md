# MTP Inference Analysis

Two one-shot analyses of MemoSight's **self-speculative decoding**, each driven by a
convenience script in this directory:

| Script | Answers | Output |
|--------|---------|--------|
| [`run_mtp_acceptance.sh`](run_mtp_acceptance.sh) | *How many foresight tokens survive verification?* (acceptance rate) | `summary.csv` / `acceptance.png` / `summary.json` |
| [`run_runtime_breakdown.sh`](run_runtime_breakdown.sh) | *Where does each decode step's time go?* (prediction / verification / compression / other) | `breakdown.csv` / `breakdown.png` / `breakdown.json` |

**Background.** MemoSight decodes with self-speculative decoding: at each step the model
drafts `γ + 1` tokens in one forward (the mandatory next token plus `γ` speculative
*register* tokens), then a single *verify* forward confirms the longest matching prefix.
The **acceptance rate** — how many drafted tokens survive verification — is what turns the
extra register compute into real speedup. Verification is **greedy exact-match** (a draft
token is kept iff it equals the base model's argmax at that position), which makes the
speedup **lossless**: the output is identical to standard greedy decoding.

---

## Prerequisites

- **Environment:** run under a Python env that has PyTorch + Transformers (the `LightThinker`
  deps). At the time of writing that is the `lightthinker` conda env.
- **PYTHONPATH:** both scripts `cd` to the repo root and auto-export `PYTHONPATH=<repo root>`,
  because `inference.py` imports the package absolutely (`from LightThinker.utils import *`)
  and the repo ships no `setup.py`. **Activating a conda env alone does not put the repo on
  the import path.** If you invoke `LightThinker/inference.py` directly (bypassing these
  scripts), export `PYTHONPATH` yourself or you get
  `ModuleNotFoundError: No module named 'LightThinker'`.

```bash
# typical invocation
export PATH=/mnt/lxy/miniconda3/envs/lightthinker/bin:$PATH
CKPT_PATH=... TOKENIZER_PATH=... DATASETS=gsm8k \
  bash scripts/run_mtp_acceptance.sh
```

---

## 1. Acceptance-rate sweep — `run_mtp_acceptance.sh`

Chains *sweep draft length `γ` → run inference per dataset → aggregate per dataset* into one
command. Edit the 5 lines in the **CONFIG** block at the top of the script, then run it
(every field is also overridable via env vars):

```bash
# top of scripts/run_mtp_acceptance.sh
CKPT_PATH="/path/to/your/train_output/checkpoint-xxxx"   # MTP-trained checkpoint dir
TOKENIZER_PATH="/path/to/Qwen3-8B"          # tokenizer dir
COMPRESS_CONFIG="./configs/LightThinker/qwen/adaptive_mtp_v1.json"  # config WITH an `mtp` block
DRAFT_LENS="1 2"      # speculative register tokens/step; used as-is, keep <= training max_offset
DATASETS="gsm8k"      # space-separated: gsm8k mmlu bbh gpqa
```

```bash
bash scripts/run_mtp_acceptance.sh
# or, without editing the file:
CKPT_PATH=/my/ckpt DRAFT_LENS="1 2" DATASETS="gsm8k mmlu" GPU=0 \
  bash scripts/run_mtp_acceptance.sh
```

The script sweeps exactly the `DRAFT_LENS` you pass, picks the chat tokens from `MODEL_TYPE`
(`qwen`/`llama`), optionally runs a non-speculative baseline for a wall-clock reference, then
aggregates **per dataset** into a CSV / plot / JSON and prints the result paths.

| Field / env var | Default | Meaning |
|---------|---------|---------|
| `CKPT_PATH` | *(fill in)* | MTP-trained checkpoint dir |
| `TOKENIZER_PATH` | *(fill in)* | tokenizer dir |
| `COMPRESS_CONFIG` | `configs/.../adaptive_mtp_v1.json` | Must have an `mtp` block or the speculative path is skipped. |
| `DRAFT_LENS` | `1 2` | Speculative register tokens per step (`--mtp_draft_len`); used as-is — keep `<=` your training `max_offset`. |
| `DATASETS` | `gsm8k` | Space-separated: `gsm8k mmlu bbh gpqa` |
| `MODEL_TYPE` | `qwen` | `qwen` / `llama`; picks `BOS_TOKEN` / `EOS_TOKEN`. |
| `WITH_BASELINE` | `1` | Also run a non-speculative pass for a wall-clock speedup reference. |
| `MAX_NEW_TOKENS` | `10240` | Generation cap **and** buffer size — see the sizing note below. |
| `SPLIT_SIZE` / `INDEX` | `1` / `1` | Data sharding; `1`/`1` = **full test set**. See "Sub-sampling" below. |
| `GPU` | `0` | CUDA device id |
| `RESULT_ROOT` | `mtp_accept_results` | Output directory |

> **Constraint:** the draft length `γ` is set via `--mtp_draft_len`. Acceptance is only
> meaningful up to the `max_offset` the checkpoint was trained with (register offset was
> sampled in `[0, max_offset]`) — **the script uses your `DRAFT_LENS` as-is, so keep it within
> the training value yourself.** A single `dl2` run already contains the per-position
> acceptance for **pos1 / pos2 / pos3** (`draft_len = γ + 1 = 3`) — no need to run each
> position separately.

> **Does `max_offset` in `COMPRESS_CONFIG` need to match `DRAFT_LENS`? — No.** As long as
> `--mtp_draft_len` is passed (this script always does), it fully determines the inference
> draft length; the config's `max_offset` is overwritten and unused. The config's `mtp` block
> only needs to *exist* to trigger the speculative path. The real hard constraint is
> `DRAFT_LENS ≤ the max_offset the checkpoint was trained with` (baked into the weights).
> Still, keep the config's `max_offset` equal to the training value as a label for the
> checkpoint and for `inference_batched.py`'s fallback path.

### Reported metrics

| Metric | Definition | Reads as |
|--------|------------|----------|
| **mean accept length τ** | committed tokens / decode steps | tokens emitted per step (upper bound on speedup) |
| **overall acceptance α** | accepted speculative tokens / proposed speculative tokens | fraction of MTP drafts kept, `∈ [0, 1]` |
| **per-position αₖ** | acceptance of the k-th register position (unconditional and conditional-on-reached) | how prediction quality decays with distance |
| **tokens / forward** | committed tokens / forward passes | compute-bound speedup proxy vs. `1.0` for plain AR |
| **tokens / s** | output length / inference time | measured wall-clock throughput |
| **accept histogram** | frequency of committing 1, 2, … tokens per step | shape of the acceptance distribution |

Accuracy is reported alongside so speed gains are never read in isolation.

### Artifacts

- Each inference record (`<RESULT_ROOT>/dl<N>/<dataset>/<index>_<dataset>.jsonl`) gains an
  `mtp_stats` block plus `mtp_draft_len`, carrying raw counters for exact, loss-free
  re-aggregation across samples and files.
- `summary.csv` — one row per group `dl<N>_<dataset>`: `draft_len, accuracy, mean_accept_len,
  overall_accept_rate, tokens_per_forward, tokens_per_sec, …`
- `summary.json` — full derived metrics (including per-position αₖ and the accept histogram).
- `acceptance.png` — two panels: per-position acceptance curves, and mean accept length /
  tokens-per-forward vs. draft length.

### Analyzing manually

The script aggregates **per dataset** by default (group names `dl<N>_<dataset>`). To
re-aggregate any existing outputs produced with `--spec_decode true`:

```bash
# single run
python scripts/analyze_mtp_acceptance.py mtp_accept_results/dl2/gsm8k/**/*.jsonl

# compare several draft lengths × datasets, emit table + csv + plot
python scripts/analyze_mtp_acceptance.py \
  --group dl1_gsm8k=mtp_accept_results/dl1/gsm8k/**/*.jsonl \
  --group dl2_gsm8k=mtp_accept_results/dl2/gsm8k/**/*.jsonl \
  --csv mtp_accept_results/summary.csv \
  --plot mtp_accept_results/acceptance.png \
  --json mtp_accept_results/summary.json
```

---

## 2. Runtime breakdown — `run_runtime_breakdown.sh`

Splits each decode step's wall-clock time into **prediction / verification / compression /
other** by running inference at a single draft length with `--profile_breakdown true` (timing
is GPU-synchronized so the attribution is real).

```bash
CKPT_PATH=/path/to/your/ckpt \
TOKENIZER_PATH=/path/to/Qwen3-8B \
COMPRESS_CONFIG=./configs/LightThinker/qwen/adaptive_mtp_v1.json \
DRAFT_LEN=2 DATASETS="gsm8k" GPU=0 \
bash scripts/run_runtime_breakdown.sh
```

Phase → code: **prediction** = main forward + draft sampling; **verification** = verify
forward + sampling + KV trims; **compression** = the compression branch's cache/input-ids
reduction; **other** = `t_total` minus the three (mask construction, register bookkeeping,
Python overhead). Seconds are *pooled* across samples before dividing (long samples weigh
more — exactly what a "where does time go" figure wants). Results are grouped **per dataset**.

> **Note:** profiling wraps each phase in `torch.cuda.synchronize()`, so **this run's tokens/s
> is not a valid throughput number** — keep it separate from the end-to-end speed /
> acceptance-rate runs. The compression forward is *fused* into the main forward, so the
> compression bucket only counts its unique overhead (cache reduction); the compression
> forward cost lands inside prediction. This is an accounting convention — state it in the
> write-up.
>
> A short, barely-trained checkpoint (or any run whose output never emits `<|splitter|>`) will
> show **compression = 0%** simply because no compression step was triggered — not a bug.

You can also aggregate existing profiled outputs directly:

```bash
python scripts/analyze_runtime_breakdown.py \
  --group gsm8k=runtime_breakdown_results/dl2/gsm8k/**/*.jsonl \
  --group bbh=runtime_breakdown_results/dl2/bbh/**/*.jsonl \
  --csv breakdown.csv --plot breakdown.png --json breakdown.json
```

Outputs: `breakdown.csv` (per-dataset per-phase seconds & %), `breakdown.png` (stacked bar),
`breakdown.json` (full metrics).

**Run it per dataset?** The breakdown ratios are driven mostly by *mechanism* (forward-pass
count, verify/compression trigger frequency), not by content the way accuracy is, so one
representative dataset (e.g. GSM8k) is enough to support the "where time goes" claim. But this
profiling rides the *same* inference path as the acceptance-rate run — if you're already
sweeping benchmarks for acceptance, the breakdown comes at near-zero marginal cost.
Recommendation: use GSM8k as the main-text figure, and use the per-dataset grouping in the
appendix to show the split is stable across benchmarks (the compression-step fraction shifts a
bit with CoT length/structure, worth showing).

---

## Sub-sampling & gotchas (shared by both scripts)

### Sub-sampling with `SPLIT_SIZE` / `INDEX`

Default `SPLIT_SIZE=1 INDEX=1` runs the **full test set**. Sharding is applied
**independently per dataset** (inside each `eval_dataset` over `len(reader)`), *not* over a
mixed pool — so `SPLIT_SIZE=6` with four datasets takes the first ~1/6 of *each*:

| Dataset | Full | `SPLIT_SIZE=6 INDEX=1` |
|---------|------|------------------------|
| gsm8k | 1319 | 219 |
| mmlu | 1027 | 171 |
| bbh | 495 | 82 |
| gpqa | 198 | 33 |

`SPLIT_SIZE=N INDEX=1` ⇒ first `len/N` samples of each dataset. Acceptance is a per-token
statistic (dozens–hundreds of decode steps per sample), so 100–200 samples usually give a
stable curve without running the full set.

### `MAX_NEW_TOKENS` is also the buffer size — don't set it too small

The token/KV buffer is preallocated to `MAX_NEW_TOKENS + max_prompt_len` (1100). With prompt
compression on, the prefill *expands* — each prompt sentence appends ~300 register (`<|o_*|>`)
tokens, so a single GSM8k prompt prefills to ≈4k tokens. If `MAX_NEW_TOKENS + 1100` is smaller
than `prefill + generation`, you get `IndexError: index … is out of bounds` in
`set_input_ids`. Keep the default `10240` (buffer ≈11.3k), or size it to
`expected_prefill + expected_output` with headroom.

### Parallelizing across GPUs

One invocation runs its datasets **sequentially** (model loaded once). To go faster, launch
one process per dataset on a different GPU with a distinct `RESULT_ROOT`:

```bash
GPU=0 DATASETS=gsm8k RESULT_ROOT=acc_gsm8k ... bash scripts/run_mtp_acceptance.sh &
GPU=1 DATASETS=mmlu  RESULT_ROOT=acc_mmlu  ... bash scripts/run_mtp_acceptance.sh &
```

### Spec-decode + prompt-compression sanity check

If the speculative path corrupts structured output (e.g. `\boxed{D}` → `\boxedD}`) under
prompt compression, compare against the non-speculative path (`--spec_decode false`), which is
the clean reference: the acceptance path must reproduce it token-for-token under greedy
(rep-penalty 1.0). A divergence localizes to the EPL-verify / compression-step interaction.
