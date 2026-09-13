#!/usr/bin/env python3
"""
对比 Vanilla（标准自回归、完整 KV）与 **脚本内模拟的 MemCoT**：
在不同 max_new_tokens 下的 wall-clock 推理时间与峰值序列长度（Peak tokens，proxy 显存）。

**不调用** LightThinker/inference.py。MemCoT 路径为极简模拟：
  每生成 `--memcot_chunk_tokens`（默认 500）个 token，就从 KV cache 末尾裁掉这 500 个位置，
  再在 cache 末尾**直接拼接** `--memcot_compressed_tokens`（默认 100）格占位 K/V（重复最后一格，
  不跑 forward），等价5× 压缩率（500→100）。**解码 forward 次数与 Vanilla 相同**（均为 max_new_tokens
  次单步 decode +同一次 prefill）。生成内容无意义，仅测耗时与峰值长度。

计时默认 `--runs 5` 取中位数；可用 `--warmup_runs` 在每个长度正式计时前先做不计时的完整跑。

用法（在仓库根目录 RRcot 下执行）:
  python scripts/benchmark_figure_c_scaling.py \\
    --model_path /path/to/checkpoint \\
    --tokenizer_path /path/to/tokenizer \\
    --model_type qwen \\
    --bos_token "<|im_start|>" \\
    --eos_token "<|redacted_im_end|>" \\
    --output_dir outputs/bench_figure_c

结果: output_dir 下保存 metrics.json 与 figure_c_scaling.pdf/png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LT_DIR = _PROJECT_ROOT / "LightThinker"
sys.path.insert(0, str(_LT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))


def str2bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y")


def build_synthetic_user_text(tokenizer: AutoTokenizer, target_tokens: int) -> str:
    """构造 user 段纯文本，使其 encode 后长度恰好为 target_tokens（不含 chat 模板）。"""
    if target_tokens <= 0:
        return ""
    unit = " synthbench "
    text = unit
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text += unit
        ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=False)


def build_chat_prompt_ids(
    bos: str,
    eos: str,
    system_prompt: str,
    question: str,
    tokenizer: AutoTokenizer,
) -> List[int]:
    """Qwen 风格 chat：system / user / assistant 起始（与原先 template 语义一致即可）。"""
    sys_text = system_prompt if (system_prompt or "").strip() else "You are a helpful assistant."
    text = f"{bos}system\n{sys_text}{eos}\n{bos}user\n{question}{eos}\n{bos}assistant\n"
    return tokenizer.encode(text, add_special_tokens=False)


def full_prompt_token_count(
    bos: str,
    eos: str,
    system_prompt: str,
    question: str,
    tokenizer: AutoTokenizer,
) -> int:
    return len(build_chat_prompt_ids(bos, eos, system_prompt, question, tokenizer))


def load_model_and_tokenizer(args: argparse.Namespace):
    """仅从 checkpoint 加载模型与 HF tokenizer（不经过 inference.py）。"""
    model_path = args.model_path or ""
    if not model_path:
        model_path = f"output/{args.model_tag}/checkpoint-{args.ckpt}"
    tok_path = args.tokenizer_path or model_path
    print(f"[bench] load model from `{model_path}` ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    dtype = torch.bfloat16
    if args.model_type.lower() == "qwen":
        from LightThinker.model_qwen import Qwen3ForCausalLM

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.model_type != "qwen3":
            raise ValueError(
                f"Expected a Qwen3 checkpoint, but `{model_path}` has model_type={config.model_type!r}."
            )
        missing_tokens = [token for token in ("<think>", "</think>") if token not in tokenizer.get_vocab()]
        if missing_tokens:
            raise ValueError(f"The tokenizer is not from Qwen3; missing native tokens: {missing_tokens}.")
        model = Qwen3ForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=dtype, device_map="auto"
        )
    elif args.model_type.lower() == "llama":
        from LightThinker.model_llama import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    else:
        raise ValueError(f"unknown --model_type {args.model_type}")
    return model, tokenizer


def _normalize_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, Cache):
        return past_key_values
    return DynamicCache.from_legacy_cache(past_key_values)


def _append_placeholder_kv(past_key_values: DynamicCache, num_slots: int) -> None:
    """在 DynamicCache 各层末尾拼接占位 K/V（沿序列维 repeat最后一格），不经过 model forward。"""
    if num_slots <= 0:
        return
    for layer_idx in range(len(past_key_values.key_cache)):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        if not isinstance(k, torch.Tensor) or k.numel() == 0:
            continue
        last_k = k[:, :, -1:, :]
        last_v = v[:, :, -1:, :]
        pad_k = last_k.repeat(1, 1, num_slots, 1)
        pad_v = last_v.repeat(1, 1, num_slots, 1)
        past_key_values.key_cache[layer_idx] = torch.cat([k, pad_k], dim=-2)
        past_key_values.value_cache[layer_idx] = torch.cat([v, pad_v], dim=-2)
    past_key_values._seen_tokens = past_key_values.get_seq_length(0)


def _dummy_token_id(tokenizer: AutoTokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


@dataclass
class InferenceTimings:
    """单次 greedy 跑完后的细粒度耗时（秒，CUDA 段前后均 synchronize）。"""

    prep_s: float = 0.0  # tokenizer 编码 + prompt张量准备
    prefill_s: float = 0.0  # 首次全序列 forward + 首步 logits / argmax
    decode_phase_s: float = 0.0  # 自回归循环内耗时，不含 memcot 维护块
    decode_forward_s: float = 0.0  # 循环内各步 model forward 累计（decode子项）
    memcot_maint_s: float = 0.0  #仅 MemCoT：KV crop + 占位拼接；Vanilla 恒为 0

    def decode_other_s(self) -> float:
        return max(0.0, self.decode_phase_s - self.decode_forward_s)

    def inner_sum_s(self) -> float:
        return self.prep_s + self.prefill_s + self.decode_phase_s + self.memcot_maint_s


def median_inference_timings(runs: List[InferenceTimings]) -> InferenceTimings:
    if not runs:
        return InferenceTimings()
    arr = np.array(
        [[r.prep_s, r.prefill_s, r.decode_phase_s, r.decode_forward_s, r.memcot_maint_s] for r in runs],
        dtype=np.float64,
    )
    m = np.median(arr, axis=0)
    return InferenceTimings(
        prep_s=float(m[0]),
        prefill_s=float(m[1]),
        decode_phase_s=float(m[2]),
        decode_forward_s=float(m[3]),
        memcot_maint_s=float(m[4]),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Figure (c)：Vanilla vs 脚本内 MemCoT 模拟（KV 裁切 + 占位压缩 token）")
    p.add_argument("--model_path", type=str, default=None, help="checkpoint 目录；不设则用 output/{model_tag}/checkpoint-{ckpt}")
    p.add_argument("--model_tag", type=str, default="dummy")
    p.add_argument("--ckpt", type=int, default=0)
    p.add_argument("--tokenizer_path", type=str, default=None, help="默认同 --model_path")
    p.add_argument("--model_type", type=str, choices=["qwen", "llama"], default="qwen")
    p.add_argument("--bos_token", type=str, required=True)
    p.add_argument("--eos_token", type=str, required=True)

    p.add_argument(
        "--lengths",
        type=str,
        default="1024,4096,32768",
        help="逗号分隔的目标生成长度（max_new_tokens）",
    )
    p.add_argument("--warmup_length", type=int, default=32)
    p.add_argument(
        "--warmup_runs",
        type=int,
        default=0,
        help="每个长度、每条路径在正式计时前先完整跑的次数（不计时）",
    )
    p.add_argument("--runs", type=int, default=5, help="每个长度正式计时的重复次数，取时间中位数")

    p.add_argument("--system_prompt", type=str, default="")
    p.add_argument("--question", type=str, default="", help="自定义 user 段；与 synthetic 互斥时以合成为准")
    p.add_argument(
        "--synthetic_prompt",
        type=str2bool,
        default=True,
        help="True：用合成占位 user 段（长度见 --prompt_user_tokens）",
    )
    p.add_argument("--prompt_user_tokens", type=int, default=125, help="合成时 user 段目标 token 数")

    p.add_argument(
        "--memcot_chunk_tokens",
        type=int,
        default=500,
        help="每积累多少个生成 token 触发一次 KV 裁切 + 压缩槽填充",
    )
    p.add_argument(
        "--memcot_compressed_tokens",
        type=int,
        default=100,
        help="每次裁切后在 KV末尾拼接多少格占位（直接拼 K/V，不 forward；如 500/5=100）",
    )

    p.add_argument("--repetition_penalty", type=float, default=1.0)

    p.add_argument("--ours_label", type=str, default="MemCoT-sim")
    p.add_argument("--vanilla_label", type=str, default="Vanilla")
    p.add_argument("--output_dir", type=str, default="outputs/benchmark_figure_c")
    p.add_argument("--skip_vanilla", action="store_true")
    p.add_argument("--skip_ours", action="store_true")
    return p.parse_args()


@torch.no_grad()
def vanilla_greedy_fixed_steps(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    bos: str,
    eos: str,
    system_prompt: str,
    question: str,
    max_new_tokens: int,
    repetition_penalty: float,
) -> Tuple[int, InferenceTimings]:
    """标准自回归，严格 max_new_tokens 步（忽略 EOS）；返回 peak KV 序列长度与分段耗时。"""
    timings = InferenceTimings()

    sync_cuda()
    t0 = time.perf_counter()
    input_ids = build_chat_prompt_ids(bos, eos, system_prompt, question, tokenizer)
    device = next(model.parameters()).device
    prompt_tensor = torch.as_tensor([input_ids], dtype=torch.long, device=device)
    generated: List[int] = []
    timings.prep_s = time.perf_counter() - t0

    sync_cuda()
    t0 = time.perf_counter()
    outputs = model(input_ids=prompt_tensor, use_cache=True, return_dict=True)
    sync_cuda()
    past_key_values = _normalize_past_key_values(outputs.past_key_values)
    logits = outputs.logits[0, -1, :]

    if repetition_penalty != 1.0:
        from transformers import RepetitionPenaltyLogitsProcessor

        processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        def apply_rep(logits_row: torch.Tensor, ctx_ids: torch.Tensor) -> torch.Tensor:
            return processor(ctx_ids, logits_row.unsqueeze(0)).squeeze(0)

    else:

        def apply_rep(logits_row: torch.Tensor, ctx_ids: torch.Tensor) -> torch.Tensor:
            return logits_row

    logits = apply_rep(logits, prompt_tensor)
    next_id = torch.argmax(logits).item()

    prompt_len = len(input_ids)
    peak = prompt_len
    timings.prefill_s = time.perf_counter() - t0

    decode_phase_acc = 0.0
    decode_forward_acc = 0.0
    for _ in range(max_new_tokens):
        sync_cuda()
        t_iter0 = time.perf_counter()
        generated.append(next_id)
        peak = max(peak, prompt_len + len(generated))
        step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        sync_cuda()
        tf0 = time.perf_counter()
        outputs = model(
            input_ids=step,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        sync_cuda()
        decode_forward_acc += time.perf_counter() - tf0
        past_key_values = _normalize_past_key_values(outputs.past_key_values)
        logits = outputs.logits[0, -1, :]
        ctx_for_rep = torch.cat(
            [prompt_tensor, torch.as_tensor([generated], dtype=torch.long, device=device)],
            dim=1,
        )
        logits = apply_rep(logits, ctx_for_rep)
        next_id = torch.argmax(logits).item()
        decode_phase_acc += time.perf_counter() - t_iter0

    timings.decode_phase_s = decode_phase_acc
    timings.decode_forward_s = decode_forward_acc
    return peak, timings


@torch.no_grad()
def memcot_sim_greedy_fixed_steps(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    bos: str,
    eos: str,
    system_prompt: str,
    question: str,
    max_new_tokens: int,
    repetition_penalty: float,
    chunk_tokens: int,
    compressed_tokens: int,
    dummy_id: int,
) -> Tuple[int, InferenceTimings]:
    """
    每生成 chunk_tokens 个 token：裁掉 KV 末尾 chunk_tokens，再在 cache 上拼接 compressed_tokens 格占位 K/V。
    单步 decode 的 model forward 次数与 Vanilla 相同（max_new_tokens 次）；压缩只做 tensor 裁剪/拼接。
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be >= 1")
    if compressed_tokens < 1:
        raise ValueError("compressed_tokens must be >= 1")

    timings = InferenceTimings()

    sync_cuda()
    t0 = time.perf_counter()
    input_ids = build_chat_prompt_ids(bos, eos, system_prompt, question, tokenizer)
    device = next(model.parameters()).device
    prompt_tensor = torch.as_tensor([input_ids], dtype=torch.long, device=device)
    prompt_len = len(input_ids)
    generated: List[int] = []
    timings.prep_s = time.perf_counter() - t0

    sync_cuda()
    t0 = time.perf_counter()
    outputs = model(input_ids=prompt_tensor, use_cache=True, return_dict=True)
    sync_cuda()
    past_key_values = _normalize_past_key_values(outputs.past_key_values)
    logits = outputs.logits[0, -1, :]

    if repetition_penalty != 1.0:
        from transformers import RepetitionPenaltyLogitsProcessor

        processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        def apply_rep(logits_row: torch.Tensor, ctx_ids: torch.Tensor) -> torch.Tensor:
            return processor(ctx_ids, logits_row.unsqueeze(0)).squeeze(0)

    else:

        def apply_rep(logits_row: torch.Tensor, ctx_ids: torch.Tensor) -> torch.Tensor:
            return logits_row

    logits = apply_rep(logits, prompt_tensor)
    next_id = torch.argmax(logits).item()

    peak = prompt_len
    tokens_since_compress = 0
    timings.prefill_s = time.perf_counter() - t0

    decode_phase_acc = 0.0
    decode_forward_acc = 0.0

    for _ in range(max_new_tokens):
        sync_cuda()
        t_iter0 = time.perf_counter()
        dt_memcot = 0.0
        generated.append(next_id)

        step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        sync_cuda()
        tf0 = time.perf_counter()
        outputs = model(
            input_ids=step,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        sync_cuda()
        decode_forward_acc += time.perf_counter() - tf0
        past_key_values = _normalize_past_key_values(outputs.past_key_values)
        peak = max(peak, past_key_values.get_seq_length())
        logits = outputs.logits[0, -1, :]
        ctx_for_rep = torch.cat(
            [prompt_tensor, torch.as_tensor([generated], dtype=torch.long, device=device)],
            dim=1,
        )
        logits = apply_rep(logits, ctx_for_rep)
        next_id = torch.argmax(logits).item()

        tokens_since_compress += 1
        if tokens_since_compress >= chunk_tokens:
            seq_len = past_key_values.get_seq_length()
            keep = seq_len - chunk_tokens
            if keep < 0:
                raise RuntimeError("internal: keep < 0")
            sync_cuda()
            tm0 = time.perf_counter()
            past_key_values.crop(keep)
            generated = generated[:-chunk_tokens]
            _append_placeholder_kv(past_key_values, compressed_tokens)
            generated.extend([dummy_id] * compressed_tokens)
            peak = max(peak, past_key_values.get_seq_length())
            tokens_since_compress = 0
            sync_cuda()
            dt_memcot = time.perf_counter() - tm0
            timings.memcot_maint_s += dt_memcot

        decode_phase_acc += time.perf_counter() - t_iter0 - dt_memcot

    timings.decode_phase_s = decode_phase_acc
    timings.decode_forward_s = decode_forward_acc
    return peak, timings


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def median(xs: List[float]) -> float:
    return float(np.median(np.array(xs, dtype=np.float64)))


def _format_timing_line(label: str, median_s: float, times: List[float], peak: int) -> str:
    if len(times) <= 1:
        return f"  {label}: time={median_s:.3f}s, peak_tokens={peak}"
    std = float(np.std(np.array(times, dtype=np.float64)))
    return (
        f"  {label}: time={median_s:.3f}s (min={min(times):.3f}, max={max(times):.3f}, std={std:.3f}), "
        f"peak_tokens={peak}"
    )


def _format_inference_breakdown(fine: InferenceTimings) -> str:
    """四段：prep / prefill / decode(非 memcot) / memcot；decode 内另报 forward 与 other子项。"""
    d_other = fine.decode_other_s()
    return (
        f"    └ breakdown (median): prep={fine.prep_s:.4f}s | prefill={fine.prefill_s:.4f}s | "
        f"decode={fine.decode_phase_s:.4f}s (forward={fine.decode_forward_s:.4f}s, other={d_other:.4f}s) | "
        f"memcot_maint={fine.memcot_maint_s:.4f}s | inner_sum={fine.inner_sum_s():.4f}s"
    )


@torch.no_grad()
def time_vanilla(
    model,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    max_new_tokens: int,
) -> Tuple[float, int, List[float], InferenceTimings, List[InferenceTimings]]:
    peak_last = 0
    for _ in range(args.warmup_runs):
        sync_cuda()
        peak_last, _ = vanilla_greedy_fixed_steps(
            model=model,
            tokenizer=tokenizer,
            bos=args.bos_token,
            eos=args.eos_token,
            system_prompt=args.system_prompt,
            question=args.question,
            max_new_tokens=max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        sync_cuda()
    times: List[float] = []
    fine_runs: List[InferenceTimings] = []
    for _ in range(args.runs):
        sync_cuda()
        t0 = time.perf_counter()
        peak_last, fine = vanilla_greedy_fixed_steps(
            model=model,
            tokenizer=tokenizer,
            bos=args.bos_token,
            eos=args.eos_token,
            system_prompt=args.system_prompt,
            question=args.question,
            max_new_tokens=max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        sync_cuda()
        times.append(time.perf_counter() - t0)
        fine_runs.append(fine)
    return median(times), peak_last, times, median_inference_timings(fine_runs), fine_runs


@torch.no_grad()
def time_memcot_sim(
    model,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    max_new_tokens: int,
    dummy_id: int,
) -> Tuple[float, int, List[float], InferenceTimings, List[InferenceTimings]]:
    peak_last = 0
    for _ in range(args.warmup_runs):
        sync_cuda()
        peak_last, _ = memcot_sim_greedy_fixed_steps(
            model=model,
            tokenizer=tokenizer,
            bos=args.bos_token,
            eos=args.eos_token,
            system_prompt=args.system_prompt,
            question=args.question,
            max_new_tokens=max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            chunk_tokens=args.memcot_chunk_tokens,
            compressed_tokens=args.memcot_compressed_tokens,
            dummy_id=dummy_id,
        )
        sync_cuda()
    times: List[float] = []
    fine_runs: List[InferenceTimings] = []
    for _ in range(args.runs):
        sync_cuda()
        t0 = time.perf_counter()
        peak_last, fine = memcot_sim_greedy_fixed_steps(
            model=model,
            tokenizer=tokenizer,
            bos=args.bos_token,
            eos=args.eos_token,
            system_prompt=args.system_prompt,
            question=args.question,
            max_new_tokens=max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            chunk_tokens=args.memcot_chunk_tokens,
            compressed_tokens=args.memcot_compressed_tokens,
            dummy_id=dummy_id,
        )
        sync_cuda()
        times.append(time.perf_counter() - t0)
        fine_runs.append(fine)
    return median(times), peak_last, times, median_inference_timings(fine_runs), fine_runs


def plot_figure_c(
    lengths: List[int],
    vanilla_time: List[float],
    ours_time: List[float],
    vanilla_peak: List[int],
    ours_peak: List[int],
    args: argparse.Namespace,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.array(lengths, dtype=float)
    xlog = np.log2(x)

    if not args.skip_vanilla:
        ax.plot(
            xlog,
            vanilla_time,
            color="#c0392b",
            linestyle="--",
            marker="s",
            linewidth=2,
            markersize=7,
            label=args.vanilla_label,
        )
    if not args.skip_ours:
        ax.plot(
            xlog,
            ours_time,
            color="#e67e22",
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=7,
            label=args.ours_label,
        )

    ax.set_xlabel("Generated Tokens")
    ax.set_ylabel("Time (s)")
    ax.set_title("(c) Inference time vs. generated tokens (and peak tokens in insets)")
    ax.xaxis.set_major_locator(ticker.FixedLocator(xlog))
    ax.set_xticks(xlog)
    ax.set_xticklabels([str(int(l)) for l in lengths])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    ymax = max(max(vanilla_time, default=0), max(ours_time, default=0), 1.0) * 1.15
    ax.set_ylim(0, ymax)

    labels = "abcdef"
    for i, L in enumerate(lengths):
        if i >= len(labels):
            break
        x0 = xlog[i]
        xa, xb = xlog.min(), xlog.max()
        rel = (x0 - xa) / (xb - xa + 1e-9) if xb > xa else 0.5
        inset_w, inset_h = 0.12, 0.18
        ax_in = ax.inset_axes(
            [rel - inset_w / 2, 0.55, inset_w, inset_h],
            transform=ax.transAxes,
            clip_on=False,
        )
        vt = vanilla_time[i] if not args.skip_vanilla else 0.0
        ot = ours_time[i] if not args.skip_ours else 0.0
        vp = vanilla_peak[i] if not args.skip_vanilla else 0
        op = ours_peak[i] if not args.skip_ours else 0

        bar_x = [0.0, 1.0, 2.3]
        bar_w = 0.35
        if not args.skip_vanilla and not args.skip_ours:
            ax_in.bar(
                [bar_x[0], bar_x[1]],
                [vt, ot],
                width=bar_w,
                color=["#c0392b", "#e67e22"],
            )
            if vt > 0:
                t_red = max(0.0, (vt - ot) / vt * 100)
                ax_in.annotate(
                    f"↓{t_red:.0f}%",
                    xy=(bar_x[1], ot),
                    xytext=(bar_x[1], ot + max(vt, ot) * 0.08),
                    ha="center",
                    fontsize=6,
                    color="#c0392b",
                )
            vmax = max(vt, ot, 1e-6)
            green_h = (op / max(vp, 1)) * vmax if vp > 0 else 0.0
            ax_in.bar(
                [bar_x[2]],
                [green_h],
                width=bar_w * 0.8,
                color="#27ae60",
            )
            if vp > 0 and op >= 0:
                p_green = max(0.0, (vp - op) / vp * 100)
                ax_in.annotate(
                    f"↑{p_green:.0f}%",
                    xy=(bar_x[2], green_h),
                    xytext=(bar_x[2], green_h + vmax * 0.08),
                    ha="center",
                    fontsize=6,
                    color="#27ae60",
                )
        elif not args.skip_vanilla:
            ax_in.bar([0], [vt], color="#c0392b", width=0.5)
        else:
            ax_in.bar([0], [ot], color="#e67e22", width=0.5)

        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title(f"({labels[i]}) {int(L)}", fontsize=7)
        for spine in ax_in.spines.values():
            spine.set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def timings_to_jsonable(t: InferenceTimings) -> dict:
    d = asdict(t)
    d["decode_other_s"] = t.decode_other_s()
    d["inner_sum_s"] = t.inner_sum_s()
    return d


def args_to_jsonable(ns: argparse.Namespace) -> dict:
    d = vars(ns).copy()
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def main():
    args = parse_args()
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    if not lengths:
        raise SystemExit("empty --lengths")

    if args.runs < 1:
        raise SystemExit("--runs 必须 >= 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup_runs 必须 >= 0")

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = Path(args.output_dir) / "metrics.json"

    model, tokenizer = load_model_and_tokenizer(args)
    model.eval()
    dummy_id = _dummy_token_id(tokenizer)

    if args.synthetic_prompt:
        args.question = build_synthetic_user_text(tokenizer, args.prompt_user_tokens)
    elif not (args.question or "").strip():
        raise SystemExit("请设置 --synthetic_prompt true，或提供非空 --question")

    user_tok = len(tokenizer.encode(args.question, add_special_tokens=False))
    full_tok = full_prompt_token_count(
        args.bos_token, args.eos_token, args.system_prompt, args.question, tokenizer
    )
    print(
        f"[bench] synthetic={args.synthetic_prompt} user_tokens={user_tok} "
        f"(target {args.prompt_user_tokens}) full_prompt_tokens={full_tok} "
        f"memcot_chunk={args.memcot_chunk_tokens} memcot_compressed={args.memcot_compressed_tokens} "
        f"dummy_token_id={dummy_id}"
    )
    if args.synthetic_prompt and user_tok != args.prompt_user_tokens:
        print(f"[WARN] user token count {user_tok} != target {args.prompt_user_tokens}")

    results = {
        "lengths": lengths,
        "vanilla_time_s": [],
        "ours_time_s": [],
        "vanilla_time_per_run_s": [],
        "ours_time_per_run_s": [],
        "vanilla_peak_tokens": [],
        "ours_peak_tokens": [],
        "vanilla_breakdown_median_s": [],
        "vanilla_breakdown_per_run_s": [],
        "ours_breakdown_median_s": [],
        "ours_breakdown_per_run_s": [],
        "prompt_stats": {
            "user_question_tokens": user_tok,
            "full_prompt_tokens": full_tok,
            "synthetic_prompt": args.synthetic_prompt,
        },
        "meta": args_to_jsonable(args),
    }

    if torch.cuda.is_available():
        wlen = min(args.warmup_length, lengths[0])
        if not args.skip_vanilla:
            _, _ = vanilla_greedy_fixed_steps(
                model,
                tokenizer,
                args.bos_token,
                args.eos_token,
                args.system_prompt,
                args.question,
                wlen,
                args.repetition_penalty,
            )
        if not args.skip_ours:
            warm_args = copy(args)
            warm_args.warmup_runs = 0
            warm_args.runs = 1
            _, _, _, _, _ = time_memcot_sim(model, tokenizer, warm_args, wlen, dummy_id)
        torch.cuda.empty_cache()

    for L in lengths:
        print(f"\n=== max_new_tokens = {L} ===")
        if not args.skip_vanilla:
            tv, pv, vt_times, fine_v_med, fine_v_runs = time_vanilla(model, tokenizer, args, L)
            print(_format_timing_line(args.vanilla_label, tv, vt_times, pv))
            print(_format_inference_breakdown(fine_v_med))
            results["vanilla_time_s"].append(tv)
            results["vanilla_time_per_run_s"].append(vt_times)
            results["vanilla_peak_tokens"].append(pv)
            results["vanilla_breakdown_median_s"].append(timings_to_jsonable(fine_v_med))
            results["vanilla_breakdown_per_run_s"].append([timings_to_jsonable(x) for x in fine_v_runs])
        else:
            results["vanilla_time_s"].append(None)
            results["vanilla_time_per_run_s"].append(None)
            results["vanilla_peak_tokens"].append(None)
            results["vanilla_breakdown_median_s"].append(None)
            results["vanilla_breakdown_per_run_s"].append(None)

        if not args.skip_ours:
            to, po, ot_times, fine_o_med, fine_o_runs = time_memcot_sim(model, tokenizer, args, L, dummy_id)
            print(_format_timing_line(args.ours_label, to, ot_times, po))
            print(_format_inference_breakdown(fine_o_med))
            results["ours_time_s"].append(to)
            results["ours_time_per_run_s"].append(ot_times)
            results["ours_peak_tokens"].append(po)
            results["ours_breakdown_median_s"].append(timings_to_jsonable(fine_o_med))
            results["ours_breakdown_per_run_s"].append([timings_to_jsonable(x) for x in fine_o_runs])
        else:
            results["ours_time_s"].append(None)
            results["ours_time_per_run_s"].append(None)
            results["ours_peak_tokens"].append(None)
            results["ours_breakdown_median_s"].append(None)
            results["ours_breakdown_per_run_s"].append(None)

        torch.cuda.empty_cache()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics -> {out_json}")

    vt = [x if x is not None else 0.0 for x in results["vanilla_time_s"]]
    ot = [x if x is not None else 0.0 for x in results["ours_time_s"]]
    vp = [x if x is not None else 0 for x in results["vanilla_peak_tokens"]]
    op = [x if x is not None else 0 for x in results["ours_peak_tokens"]]
    plot_figure_c(
        lengths,
        vt,
        ot,
        vp,
        op,
        args,
        Path(args.output_dir) / "figure_c_scaling",
    )
    print(f"Saved figure -> {Path(args.output_dir) / 'figure_c_scaling.pdf'}")


if __name__ == "__main__":
    main()
