# MemoSight

[English](README.md) | **中文**

[arXiv](https://arxiv.org/abs/2604.14889)

论文 **[MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration](https://arxiv.org/abs/2604.14889)**（[arXiv:2604.14889](https://arxiv.org/abs/2604.14889)）的官方 PyTorch 实现。

**MemoSight**（Memory-Foresight-Based Reasoning）将**上下文压缩**与**多token预测**统一用于思维链推理：压缩历史 token 以抑制 KV cache 增长，并并行预测未来 token 以加速解码，同时使推理精度接近 vanilla SFT 基线。

## 亮点

- 压缩与并行预测共用基于特殊 token 与 token 专属位置编码的极简设计。
- 相较 vanilla SFT，KV cache 最多降低 **66%**、推理加速 **56%**，在四个推理基准上平均精度下降不足 **3%**（详见[论文](https://arxiv.org/abs/2604.14889)）。



## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [MTP 推理分析](#mtp-推理分析)
- [项目结构](#项目结构)
- [引用](#引用)
- [致谢](#致谢)
- [许可协议](#许可协议)



## 环境要求

- Python 3.9
- 训练与推理需支持 CUDA 的 GPU
- Python 依赖见 `[requirements.txt](requirements.txt)`（PyTorch 2.5.1、Transformers 4.46.3、DeepSpeed 0.15.3 等）



## 安装

```bash
git clone https://github.com/helldog-star/MemoSight.git
cd MemoSight

conda create -n memosight python=3.9 -y
conda activate memosight
pip install -r requirements.txt
```



## 数据准备

将训练数据放在 `data/` 下（例如 `data/train/train.jsonl`）。若使用仓库附带压缩包：

```bash
cd data && unzip data.zip && cd ..
```



## 快速开始

推荐使用统一入口脚本 `[scripts/pipeline.sh](scripts/pipeline.sh)` 完成训练、推理与评估，避免手动拼接多段命令。

Qwen 主流程现在对应 Qwen3。使用 `--model_type qwen` 时，模型和 tokenizer 都应来自同一个
Qwen3 checkpoint（例如 `Qwen/Qwen3-8B`），不能再搭配 Qwen2/Qwen2.5 tokenizer。

查看帮助：

```bash
bash scripts/pipeline.sh -h
```



### 流水线阶段


| `--stage` | 说明                     |
| --------- | ---------------------- |
| `train`   | 仅训练                    |
| `infer`   | 仅推理（默认使用最新 checkpoint） |
| `eval`    | 仅评估                    |
| `all`     | 训练 → 推理 → 评估           |




### 常用参数

**通用（必填）**


| 参数                  | 说明              |
| ------------------- | --------------- |
| `--stage`           | 执行阶段            |
| `--exp_tag`         | 实验名（同时作为模型 tag） |
| `--output_base_dir` | 输出根目录           |


**训练**

- `--use_epl`、`--lr`、`--mode`
- `--tokenizer_path`、`--model_path`、`--train_data_path`
- `--train_gpus`：逗号分隔 GPU 编号，如 `0,1,2,3`

**推理**

- `--target_gpus`、`--process_per_gpu`
- `--datasets`：逗号分隔数据集名
- `--ckpt`：可选；省略时使用最新 checkpoint

**评估**

- `--eval_method`（默认 `normal`）
- `--datasets`、`--comp_config`
- `--interaction`：`true` / `false`



### 示例

请将 `OUTPUT_DIR` 和 `TRAIN_DATA` 替换为本机路径；以下示例直接使用 `Qwen/Qwen3-8B`。

**仅训练**

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

**仅推理**

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

**全流程（train + infer + eval）**

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



### 输出目录

运行产物位于：

```text
<output_base_dir>/<exp_tag>/
```


| 路径              | 内容               |
| --------------- | ---------------- |
| `train/`        | 训练日志与 checkpoint |
| `inference/`    | 推理输出与子进程日志       |
| `eval/`         | 评估日志与结果          |
| `run_*.txt`     | 本次运行参数快照         |
| `pipeline_*.sh` | 运行时脚本快照（便于复现）    |


软链接便于追踪最近一次运行：`run_latest.txt`、`pipeline_latest.sh`、各阶段 `*_latest.log`。

### 常见问题

1. `**--stage infer` 找不到 checkpoint**
  先完成训练，或通过 `--ckpt` 显式指定路径。
2. **显存不足（OOM）**
  优先减小 `--micro_batch_size`，其次降低 `--max_length` 与 `--process_per_gpu`。
3. **参数错误**
  执行 `bash scripts/pipeline.sh -h` 核对参数名与取值。

## MTP 推理分析

自推测解码的接受率分析与运行时分解(`run_mtp_acceptance.sh`、`run_runtime_breakdown.sh`)的完整流程见 [`scripts/analysis_readme.md`](scripts/analysis_readme.md)。

## 项目结构

```text
MemoSight/
├── LightThinker/           # 模型、训练、推理核心代码
├── configs/LightThinker/   # 模型与训练配置（JSON）
├── scripts/                # pipeline.sh、MTP 接受率扫描与分析等运行脚本
├── evaluation/             # 评估脚本
├── data/                   # 训练/评测数据
└── requirements.txt
```

传统 MTP 对照实验可使用 `[scripts/pipeline_traditional_MTP.sh](scripts/pipeline_traditional_MTP.sh)`。

## 引用

若本工作对您的研究有帮助，请引用：

```bibtex
@article{liu2026memosight,
  title   = {MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration},
  author  = {Liu, Xinyu and Liu, Xin and Jin, Bo and Zhao, Runsong and Huang, Pengcheng and Ruan, Junhao and Li, Bei and Xiao, Chunyang and Wang, Chenglong and Xiao, Tong and Zhu, Jingbo},
  journal = {arXiv preprint arXiv:2604.14889},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.14889}
}
```



## 致谢

本仓库基于 [LightThinker](https://github.com/ZJUNLP/LightThinker) 等开源工作扩展实现，感谢所有相关作者与贡献者。

## 许可协议

本项目采用 [MIT 许可证](LICENSE)。
