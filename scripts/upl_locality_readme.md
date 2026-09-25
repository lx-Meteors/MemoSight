# UPL locality 分析运行说明

这套脚本比较训练完成的 MemoSight（UPL 位置）和 LightThinker（连续位置），输出注意力热图、逐 head 热图，以及 continuation NLL 的全局和分段长度统计。

## 直接启动

默认目录与 `scripts/train_lightthinker_memosight.sh` 一致，两个模型都训练完后运行：

```bash
cd /ossfs/workspace/MemoSight
bash scripts/run_upl_locality_analysis.sh
```

脚本会自动寻找：

- `experiments/qwen3_8b_memosight/train/checkpoint-*`
- `experiments/qwen3_8b_lightthinker/train/checkpoint-*`

并分别选择编号最大的 checkpoint。样本 1、2、4 默认并行放在 GPU 0、1、2，任务均由 `nohup` 启动，因此 SSH 断开不会停止。

## 常用覆盖方式

单卡只跑一个样本：

```bash
SAMPLE_INDICES=1 ANALYSIS_GPUS=0 \
  bash scripts/run_upl_locality_analysis.sh
```

手动指定 checkpoint 和 GPU：

```bash
MEMOSIGHT_MODEL_PATH=/path/to/memosight/checkpoint-500 \
LIGHTTHINKER_MODEL_PATH=/path/to/lightthinker/checkpoint-500 \
ANALYSIS_GPUS=4,5,6 \
  bash scripts/run_upl_locality_analysis.sh
```

缩短序列或只分析指定层（显存不足时有用）：

```bash
MAX_LENGTH=3072 LAYERS=12,16,20 \
  bash scripts/run_upl_locality_analysis.sh
```

可覆盖的主要变量都集中在启动脚本顶部，包括训练目录、模型路径、tokenizer、压缩配置、数据、样本、GPU、层范围和输出目录。

## 输出

每个样本写入 `analysis_results/upl_locality/sampleN/`：

- `sampleN_seg*_len*.png/.pdf`：MemoSight 与 LightThinker 的论文版配对平均-head 热图。
- `sampleN_seg*_perhead_MemoSight_UPL.png`：MemoSight 各 attention head。
- `sampleN_seg*_perhead_LightThinker.png`：LightThinker 各 attention head。
- `sampleN_continuation_nll_by_seglen.png`：按原始压缩段长度分桶的 NLL 差值。
- `sampleN_functional_metrics.json`：完整数值和运行元数据。
- `sampleN_summary.txt`：便于快速查看的摘要。

日志位于 `logs/upl_locality/`。例如：

```bash
tail -f logs/upl_locality/sample1_*.log
```

同一个样本运行时有文件锁，重复执行不会让两个进程同时写该样本目录。

## 比较口径

默认配置复现当前训练设置：MemoSight 使用 `adaptive_mtp_v1.json`，LightThinker 使用 `v1.json`。这是“两套已训练模型在各自原生配置下”的比较，因此 NLL 差异同时包含位置布局、压缩配置与训练结果差异。

如果论文需要只隔离 UPL 的严格消融，应让两个 checkpoint 使用完全相同的压缩配置和训练设置，仅令 `use_epl` 不同，然后把 `MEMOSIGHT_CONFIG` 和 `LIGHTTHINKER_CONFIG` 指向同一份配置。`STRICT_VOCAB=true` 建议保持开启；如果失败，应修正 checkpoint/tokenizer/config 的对应关系，不要用未训练的新 token 生成论文结果。
