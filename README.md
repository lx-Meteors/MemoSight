# MemoSight

**English** | [中文](README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2604.14889-b31b1b.svg)](https://arxiv.org/abs/2604.14889)

Official PyTorch implementation of **[MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration](https://arxiv.org/abs/2604.14889)** ([arXiv:2604.14889](https://arxiv.org/abs/2604.14889)).

**MemoSight** (Memory-Foresight-Based Reasoning) unifies **context compression** and **multi-token prediction (MTP)** for chain-of-thought reasoning: it compresses historical tokens to reduce KV-cache growth and predicts future tokens in parallel to speed up decoding, while keeping reasoning accuracy close to vanilla supervised fine-tuning (SFT).

## Highlights

- Shared minimalist design with special tokens and token-specific positional layouts for both compression and parallel prediction.
- Up to **66%** lower KV-cache usage and **56%** faster inference vs. vanilla SFT, with under **3%** average accuracy drop on four reasoning benchmarks (see the [paper](https://arxiv.org/abs/2604.14889) for full results).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [MTP Inference Analysis](#mtp-inference-analysis)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Requirements

- Python 3.9
- CUDA-capable GPU(s) for training and inference
- Python dependencies in [`requirements.txt`](requirements.txt) (PyTorch 2.5.1, Transformers 4.46.3, DeepSpeed 0.15.3, etc.)

## Installation

```bash
git clone https://github.com/helldog-star/MemoSight.git
cd MemoSight

conda create -n memosight python=3.9 -y
conda activate memosight
pip install -r requirements.txt
```

## Data Preparation

Place training data under `data/` (e.g. `data/train/train.jsonl`). If you use the bundled archive:

```bash
cd data && unzip data.zip && cd ..
```

## Quick Start

We recommend the unified entry script [`scripts/pipeline.sh`](scripts/pipeline.sh) for training, inference, and evaluation instead of chaining commands manually.

The main Qwen path now targets Qwen3. When using `--model_type qwen`, load the model and tokenizer
from the same Qwen3 checkpoint (for example, `Qwen/Qwen3-8B`); Qwen2/Qwen2.5 tokenizers are rejected.

Show help:

```bash
bash scripts/pipeline.sh -h
```

### Pipeline stages

| `--stage` | Description |
|-----------|-------------|
| `train` | Training only |
| `infer` | Inference only (latest checkpoint by default) |
| `eval` | Evaluation only |
| `all` | Train → infer → eval |

### Common arguments

**General (required)**

| Argument | Description |
|----------|-------------|
| `--stage` | Pipeline stage |
| `--exp_tag` | Experiment name (also used as model tag) |
| `--output_base_dir` | Root directory for outputs |

**Training**

- `--use_epl`, `--lr`, `--mode`
- `--tokenizer_path`, `--model_path`, `--train_data_path`
- `--train_gpus`: comma-separated GPU ids, e.g. `0,1,2,3`

**Inference**

- `--target_gpus`, `--process_per_gpu`
- `--datasets`: comma-separated dataset names
- `--ckpt`: optional; uses the latest checkpoint if omitted

**Evaluation**

- `--eval_method` (default: `normal`)
- `--datasets`, `--comp_config`
- `--interaction`: `true` / `false`

### Examples

Replace `OUTPUT_DIR` and `TRAIN_DATA` with paths on your machine. The examples use `Qwen/Qwen3-8B` directly.

**Training only**

```bash
bash scripts/pipeline.sh \
  --stage train \
  --exp_tag vanilla_qwen \
  --output_base_dir OUTPUT_DIR \
  --use_epl false \
  --lr 1e-5 \
  --mode normal \
  --model_type qwen \
  --tokenizer_path Qwen/Qwen3-8B \
  --model_path Qwen/Qwen3-8B \
  --train_data_path TRAIN_DATA \
  --train_gpus 0,1,2,3
```

**Inference only**

```bash
bash scripts/pipeline.sh \
  --stage infer \
  --exp_tag vanilla_qwen \
  --output_base_dir OUTPUT_DIR \
  --use_epl false \
  --model_type qwen \
  --tokenizer_path Qwen/Qwen3-8B \
  --target_gpus 0,1,2,3 \
  --process_per_gpu 1 \
  --datasets mmlu,gsm8k,gpqa,bbh
```

**Full pipeline (train + infer + eval)**

```bash
bash scripts/pipeline.sh \
  --stage all \
  --exp_tag vanilla_qwen \
  --output_base_dir OUTPUT_DIR \
  --use_epl false \
  --lr 1e-5 \
  --mode normal \
  --model_type qwen \
  --tokenizer_path Qwen/Qwen3-8B \
  --model_path Qwen/Qwen3-8B \
  --train_data_path TRAIN_DATA \
  --train_gpus 0,1,2,3 \
  --target_gpus 0,1,2,3 \
  --process_per_gpu 1 \
  --datasets mmlu,gsm8k,gpqa,bbh
```

### Output layout

Artifacts are written under:

```text
<output_base_dir>/<exp_tag>/
```

| Path | Contents |
|------|----------|
| `train/` | Training logs and checkpoints |
| `inference/` | Inference outputs and worker logs |
| `eval/` | Evaluation logs and metrics |
| `run_*.txt` | Snapshot of run arguments |
| `pipeline_*.sh` | Snapshot of the invoked pipeline script |

Symlinks for the latest run: `run_latest.txt`, `pipeline_latest.sh`, and stage-specific `*_latest.log`.

### FAQ

1. **`--stage infer` cannot find a checkpoint**  
   Run training first, or pass an explicit path via `--ckpt`.

2. **Out-of-memory (OOM)**  
   Lower `--micro_batch_size` first, then `--max_length` and `--process_per_gpu`.

3. **Invalid arguments**  
   Run `bash scripts/pipeline.sh -h` to verify names and values.

## MTP Inference Analysis

Acceptance-rate and runtime-breakdown analyses of the self-speculative decoding (`run_mtp_acceptance.sh`, `run_runtime_breakdown.sh`) are documented in [`scripts/analysis_readme.md`](scripts/analysis_readme.md).

## Project Structure

```text
MemoSight/
├── LightThinker/           # Core model, training, and inference code
├── configs/LightThinker/   # Model and training configs (JSON)
├── scripts/                # pipeline.sh, MTP acceptance sweep & analysis, runners
├── evaluation/             # Evaluation scripts
├── data/                   # Training and benchmark data
└── requirements.txt
```

For a traditional MTP baseline, use [`scripts/pipeline_traditional_MTP.sh`](scripts/pipeline_traditional_MTP.sh).

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026memosight,
  title   = {MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration},
  author  = {Liu, Xinyu and Liu, Xin and Jin, Bo and Zhao, Runsong and Huang, Pengcheng and Ruan, Junhao and Li, Bei and Xiao, Chunyang and Wang, Chenglong and Xiao, Tong and Zhu, Jingbo},
  journal = {arXiv preprint arXiv:2604.14889},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.14889}
}
```

## Acknowledgments

This repository extends [LightThinker](https://github.com/ZJUNLP/LightThinker) and related open-source work. We thank the original authors and contributors.

## License

This project is licensed under the [MIT License](LICENSE).
