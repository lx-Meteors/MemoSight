
```
/*
 *                                                     __----~~~~~~~~~~~------___
 *                                    .  .   ~~//====......          __--~ ~~
 *                    -.            \_|//     |||\\  ~~~~~~::::... /~
 *                 ___-==_       _-~o~  \/    |||  \\            _/~~-
 *         __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~
 *     _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /
 *   .~       .~       |   \\ -_    /  /-   /   ||      \   /
 *  /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /
 *  |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
 *           '         ~-|      /|    |-~\~~       __--~~
 *                       |-~~-_/ |    |   ~\_   _-~            /\
 *                            /  \     \__   \/~                \__
 *                        _--~ _/ | .-~~____--~-/                  ~~==.
 *                       ((->/~   '.|||' -_|    ~~-/ ,              . _||
 *                                  -_     ~\      ~~---l__i__i__i--~~_/
 *                                  _-~-__   ~)  \--______________--~~
 *                                //.-~~~-~_--~- |-------~~~~~~~~
 *                                       //.-~~~--\
 *                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *                               神兽保佑            永无BUG
 */
```

## Table of Contents

- 🔧[Installation](#installation)
- 🏃[Quick Start](#quick-start)
- 🎁[Acknowledgement](#acknowledgement)
- 🚩[Citation](#citation)


## 🔧Installation

This branch targets `Qwen/Qwen3-8B` for DistillR1 inference and
training. The model and tokenizer must come from the same Qwen3 checkpoint;
Qwen2/Qwen2.5 checkpoints are rejected at startup.

```bash
git clone -b DistillR1-Qwen3 https://github.com/lx-Meteors/MemoSight.git
cd MemoSight
conda create -n lightthinker python=3.9 -y
conda activate lightthinker
pip install -r requirements.txt
cd data && unzip data.zip && cd ..
```

如需使用 `LightThinker/sglang_inference.py`，建议在独立的推理环境中执行
`pip install -r salang/requirements.txt`。


## 🏃Quick Start

本项目推荐通过统一脚本 `scripts/pipeline.sh` 运行训练、推理、评估，避免手动拼接多段命令。

脚本支持 4 个阶段：

- `--stage train`：只训练
- `--stage infer`：只推理（默认自动选择最新 checkpoint）
- `--stage eval`：只评估
- `--stage all`：训练 + 推理 + 评估全流程

可先查看帮助：

```bash
bash scripts/pipeline.sh -h
```

### 参数约定

必传通用参数：

- `--stage`
- `--exp_tag`：实验名（同时作为模型 tag）
- `--output_base_dir`：输出根目录

训练常用参数：

- `--use_epl`、`--lr`、`--mode`
- `--tokenizer_path`、`--model_path`、`--train_data_path`
- `--train_gpus`（逗号分隔，如 `0,1,2,3`）

推理常用参数：

- `--target_gpus`（逗号分隔）
- `--process_per_gpu`（每卡并发进程数）
- `--datasets`（逗号分隔）
- `--ckpt`（可选，不传时自动取最新）

评估常用参数：

- `--eval_method`（默认 `normal`）
- `--datasets`
- `--comp_config`
- `--interaction`（`true/false`）

### 示例 1：仅训练

```bash
bash scripts/pipeline.sh \
  --stage train \
  --exp_tag distillr1_qwen3_8b \
  --output_base_dir ./experiments \
  --use_epl false \
  --lr 1e-5 \
  --mode normal \
  --model_type qwen \
  --tokenizer_path Qwen/Qwen3-8B \
  --model_path Qwen/Qwen3-8B \
  --conf_version distillr1 \
  --train_data_path /path/to/train.jsonl \
  --train_gpus 0,1,2,3
```

### 示例 2：直接使用 Qwen3-8B 进行 DistillR1 推理

```bash
bash scripts/pipeline.sh \
  --stage infer \
  --exp_tag distillr1_qwen3_8b_infer \
  --output_base_dir ./experiments \
  --use_epl false \
  --model_type qwen \
  --tokenizer_path Qwen/Qwen3-8B \
  --model_path Qwen/Qwen3-8B \
  --comp_config configs/LightThinker/qwen/distillr1.json \
  --target_gpus 0 \
  --process_per_gpu 1 \
  --datasets gsm8k
```

### 示例 3：全流程（train + infer + eval）

```bash
bash scripts/pipeline.sh \
  --stage all \
  --exp_tag distillr1_qwen3_8b \
  --output_base_dir ./experiments \
  --use_epl false \
  --lr 1e-5 \
  --mode normal \
  --model_type qwen \
  --tokenizer_path Qwen/Qwen3-8B \
  --model_path Qwen/Qwen3-8B \
  --conf_version distillr1 \
  --train_data_path /path/to/train.jsonl \
  --train_gpus 0,1,2,3 \
  --target_gpus 0,1,2,3 \
  --process_per_gpu 1 \
  --datasets mmlu,gsm8k,gpqa,bbh
```

### 输出目录说明

运行后所有产物会放在：

`<output_base_dir>/<exp_tag>/`

常见内容包括：

- `train/`：训练日志和 checkpoint
- `inference/`：推理输出和子进程日志
- `eval/`：评估日志与结果
- `run_*.txt`：本次运行参数快照
- `pipeline_*.sh`：运行时脚本快照（便于复现）

并且会维护软链接：

- `run_latest.txt`
- `pipeline_latest.sh`
- 各阶段 `*_latest.log`

### 常见问题

1. `--stage infer` 报找不到 checkpoint  
请先执行训练，或手动指定 `--ckpt`。

2. 显存不足（OOM）  
优先降低 `--micro_batch_size`，其次减小 `--max_length`，并适当调低 `--process_per_gpu`。

3. 参数拼写错误导致脚本退出  
可先执行 `bash scripts/pipeline.sh -h`，确认参数名与取值。
