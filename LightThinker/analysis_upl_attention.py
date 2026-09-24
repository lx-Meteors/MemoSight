#!/usr/bin/env python3
"""UPL locality analysis for Qwen3 MemoSight and LightThinker checkpoints.

The script produces paired memory-to-segment attention heatmaps, per-head
heatmaps for both checkpoints, and continuation-NLL summaries.  Models are
loaded sequentially and attention hooks retain only the requested sub-blocks,
so two Qwen3-8B checkpoints never reside on the GPU at the same time.
"""

import argparse
import gc
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(HERE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import Config
from dataset import MyDataCollator, MyDataset
from model_llama import LlamaForCausalLM
from model_qwen import Qwen3ForCausalLM
from tokenizer import Tokenizer


IGNORE_LABEL_ID = -100


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "t"}:
        return True
    if value in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UPL locality attention analysis")
    parser.add_argument("--model_path", required=True, help="MemoSight/UPL checkpoint")
    parser.add_argument(
        "--baseline_model_path",
        default=None,
        help="LightThinker checkpoint. Omit for a same-weights position-layout probe.",
    )
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--baseline_tokenizer_path", default=None)
    parser.add_argument("--compress_config", required=True, help="MemoSight compression config")
    parser.add_argument(
        "--baseline_compress_config",
        default=None,
        help="LightThinker compression config; defaults to --compress_config.",
    )
    parser.add_argument("--data_path", default="./data/train/train.jsonl")
    parser.add_argument("--model_type", default="qwen", choices=["qwen", "llama"])
    parser.add_argument("--bos_token", default="<|im_start|>")
    parser.add_argument("--eos_token", default="<|im_end|>")
    parser.add_argument("--output_compress_instruction", default="None")

    parser.add_argument("--diagonal", type=str2bool, default=False)
    parser.add_argument("--bi_directional", type=str2bool, default=False)
    parser.add_argument("--see_current", type=str2bool, default=False)
    parser.add_argument("--exclude_continue", type=str2bool, default=False)
    parser.add_argument("--prefill_compress", type=str2bool, default=False)
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--sample_index", type=int, default=1)
    parser.add_argument("--min_seg_len", type=int, default=128)
    parser.add_argument("--max_segments", type=int, default=3)
    parser.add_argument(
        "--layers",
        default="mid",
        help="mid | all | comma-separated layer ids, e.g. 12,15,18",
    )
    parser.add_argument("--per_head", type=str2bool, default=True)
    parser.add_argument("--row_normalize", type=str2bool, default=True)
    parser.add_argument("--bucket_edges", default="64,128,256")
    parser.add_argument("--logit_chunk_size", type=int, default=128)
    parser.add_argument(
        "--strict_vocab",
        type=str2bool,
        default=True,
        help="Fail if a checkpoint is missing tokens required by its analysis config.",
    )

    parser.add_argument("--out_dir", default="./analysis_results/upl_locality")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    return parser


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def read_sample_line(path: str, sample_index: int) -> str:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    with open(path, "r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == sample_index:
                if not line.strip():
                    raise ValueError(f"Sample {sample_index} is blank in {path}")
                json.loads(line)
                return line
    raise IndexError(f"sample_index {sample_index} is outside {path}")


def build_tokenizer_and_config(
    tokenizer_path: str,
    config_path: str,
    bos_token: str,
    eos_token: str,
    model_type: str,
) -> Tuple[Tokenizer, Config, List[str]]:
    comp_config = Config.from_file(config_path=config_path)
    tokenizer = Tokenizer(
        tokenizer_path=tokenizer_path,
        bos_token=bos_token,
        eos_token=eos_token,
        special_token_list=None,
        add_prefix_space=False,
    )
    if model_type == "qwen":
        tokenizer.validate_qwen3()

    vocabulary = tokenizer.get_vocab()
    added_tokens: List[str] = []
    for token in comp_config.special_token_name_list:
        if token not in vocabulary:
            added_tokens.append(token)
    if added_tokens:
        tokenizer.add_special_token(added_tokens)
    comp_config.convert2id(tokenizer)
    return tokenizer, comp_config, added_tokens


def build_condition_sample(
    args,
    sample_line: str,
    tokenizer: Tokenizer,
    comp_config: Config,
    use_epl: bool,
) -> Dict:
    instruction = "" if args.output_compress_instruction == "None" else args.output_compress_instruction
    padding_config = {
        "padding_side": "right",
        "label_padding_id": IGNORE_LABEL_ID,
        "input_padding_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "max_length": args.max_length,
        "position_ids_padding_id": 0,
    }

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", encoding="utf-8", delete=False
        ) as temp_stream:
            temp_stream.write(sample_line)
            temp_path = temp_stream.name

        dataset = MyDataset(
            file_path=temp_path,
            config=comp_config,
            tokenizer=tokenizer,
            padding_config=padding_config,
            train_on_input=False,
            change_rope=False,
            output_compress_instruction=instruction,
            use_EPL=use_epl,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    collator = MyDataCollator(
        dataset=dataset,
        attention_config={
            "diagonal": args.diagonal,
            "bi_directional": args.bi_directional,
            "see_current": args.see_current,
            "prefill_compress": args.prefill_compress,
        },
        exclude_continue=args.exclude_continue,
        sample_config={"mode": "aug-wo-pc", "hybrid": False},
    )
    instance = dataset[0]
    tokenized = instance[4]["tokenized"]
    batch = collator._aug_mode_wo_pc([instance])

    # The collator pads to max_length.  Trim padding before eager attention to
    # avoid materializing unnecessary max_length x max_length matrices.
    seq_len = len(tokenized["input_ids"])
    batch["input_ids"] = batch["input_ids"][:, :seq_len]
    batch["labels"] = batch["labels"][:, :seq_len]
    batch["position_ids"] = batch["position_ids"][:, :seq_len]
    batch["attention_mask"] = batch["attention_mask"][:, :, :seq_len, :seq_len]

    valid_comp = batch["column_comp_index"] < seq_len
    batch["column_comp_index"] = batch["column_comp_index"][valid_comp]
    batch["row_comp_index"] = batch["row_comp_index"][valid_comp]

    return {
        "batch": batch,
        "locate_index": [list(item) for item in tokenized["locate_index"]],
        "locate_indicator": list(tokenized["locate_indicator"]),
        "seq_len": seq_len,
    }


def pick_layers(spec: str, n_layers: int) -> List[int]:
    if spec == "all":
        layers = list(range(n_layers))
    elif spec == "mid":
        low = int(n_layers * 0.4)
        high = max(low + 1, int(n_layers * 0.7))
        layers = list(range(low, high))
    else:
        layers = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not layers:
        raise ValueError("No attention layers selected")
    invalid = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if invalid:
        raise ValueError(f"Layer ids outside [0, {n_layers}): {invalid}")
    return layers


def segment_is_valid(spec: Sequence[int], seq_len: int, min_seg_len: int) -> bool:
    start, end, instruction_len, n_comp, _ = spec
    query_end = end + instruction_len + n_comp
    return (
        end - start >= min_seg_len
        and n_comp >= 2
        and 0 <= start < end <= seq_len
        and query_end <= seq_len
    )


def pair_segments(upl_sample: Dict, baseline_sample: Dict, args) -> List[Dict]:
    upl_segments = upl_sample["locate_index"]
    baseline_segments = baseline_sample["locate_index"]
    paired: List[Dict] = []
    for index in range(min(len(upl_segments), len(baseline_segments))):
        upl_spec = upl_segments[index]
        baseline_spec = baseline_segments[index]
        if not segment_is_valid(upl_spec, upl_sample["seq_len"], args.min_seg_len):
            continue
        if not segment_is_valid(
            baseline_spec, baseline_sample["seq_len"], args.min_seg_len
        ):
            continue
        paired.append(
            {
                "index": index,
                "upl_spec": upl_spec,
                "baseline_spec": baseline_spec,
                "upl_len": upl_spec[1] - upl_spec[0],
                "baseline_len": baseline_spec[1] - baseline_spec[0],
            }
        )
    paired.sort(key=lambda item: -max(item["upl_len"], item["baseline_len"]))
    return paired[: args.max_segments]


class AttentionSubblockCapture:
    """Capture selected attention sub-blocks and discard full matrices per layer."""

    def __init__(self, model, layer_ids: Sequence[int], segment_specs: Dict[int, Sequence[int]]):
        self.layer_ids = set(layer_ids)
        self.segment_specs = segment_specs
        self.sums: Dict[int, torch.Tensor] = {}
        self.counts: Dict[int, int] = {key: 0 for key in segment_specs}
        self.handles = []
        for layer_id, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(self._make_hook(layer_id))
            )

    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError(f"Unexpected attention output at layer {layer_id}")
            attention = output[1]
            if attention is None:
                raise RuntimeError(
                    "Attention weights are unavailable. Ensure eager attention is enabled."
                )
            if layer_id in self.layer_ids:
                for segment_id, spec in self.segment_specs.items():
                    start, end, instruction_len, n_comp, _ = spec
                    query_start = end + instruction_len
                    query_end = query_start + n_comp
                    block = attention[
                        0, :, query_start:query_end, start:end
                    ].detach().float().cpu()
                    if segment_id not in self.sums:
                        self.sums[segment_id] = block
                    else:
                        self.sums[segment_id].add_(block)
                    self.counts[segment_id] += 1

            # Qwen3Model otherwise retains every full LxL matrix in its output.
            return (output[0], None, *output[2:])

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def averages(self) -> Dict[int, np.ndarray]:
        result = {}
        for segment_id in self.segment_specs:
            count = self.counts.get(segment_id, 0)
            if count == 0 or segment_id not in self.sums:
                raise RuntimeError(f"No attention block captured for segment {segment_id}")
            result[segment_id] = (self.sums[segment_id] / count).numpy()
        return result


def load_model(
    args,
    model_path: str,
    tokenizer: Tokenizer,
    added_tokens: Sequence[str],
    dtype: torch.dtype,
):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if args.model_type == "qwen" and model_config.model_type != "qwen3":
        raise ValueError(
            f"Expected Qwen3 checkpoint at {model_path}, got {model_config.model_type!r}"
        )
    model_class = Qwen3ForCausalLM if args.model_type == "qwen" else LlamaForCausalLM
    model = model_class.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    checkpoint_vocab = model.get_input_embeddings().num_embeddings
    required_vocab = len(tokenizer)
    if checkpoint_vocab < required_vocab:
        message = (
            f"Checkpoint vocab ({checkpoint_vocab}) is smaller than the tokenizer/config "
            f"requirement ({required_vocab}); {len(added_tokens)} tokens were added at analysis time."
        )
        if args.strict_vocab:
            raise ValueError(
                message
                + " Use the tokenizer/config that the checkpoint was trained with. "
                "Set --strict_vocab false only for a diagnostic, not a paper result."
            )
        print(f"[warning] {message} Resizing with untrained token rows.")
        model.resize_token_embeddings(required_vocab, mean_resizing=False)
    elif checkpoint_vocab > required_vocab:
        print(
            f"[warning] checkpoint vocab ({checkpoint_vocab}) exceeds tokenizer vocab "
            f"({required_vocab}); extra checkpoint rows remain available but are not input tokens."
        )

    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(args.device)
    return model


def continuation_segment_map(locate_index: Sequence[Sequence[int]], seq_len: int) -> np.ndarray:
    result = np.full(seq_len, -1, dtype=np.int64)
    events = []
    for start, end, instruction_len, n_comp, n_continue in locate_index:
        memory_end = end + instruction_len + n_comp + n_continue
        if memory_end <= seq_len:
            events.append((memory_end, end - start))
    events.sort()
    current_length = -1
    event_index = 0
    for position in range(seq_len):
        while event_index < len(events) and events[event_index][0] <= position:
            current_length = events[event_index][1]
            event_index += 1
        result[position] = current_length
    return result


@torch.inference_mode()
def continuation_nll(
    model,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    segment_lengths: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    labels = labels.detach().cpu().numpy()
    valid_positions = np.flatnonzero(
        (labels != IGNORE_LABEL_ID) & (segment_lengths >= 0) & (np.arange(len(labels)) > 0)
    )
    nll = np.full(len(labels), np.nan, dtype=np.float64)
    for offset in range(0, len(valid_positions), chunk_size):
        positions = valid_positions[offset : offset + chunk_size]
        predictor_positions = torch.as_tensor(
            positions - 1, dtype=torch.long, device=hidden_states.device
        )
        targets = torch.as_tensor(
            labels[positions], dtype=torch.long, device=hidden_states.device
        )
        logits = model.lm_head(hidden_states.index_select(0, predictor_positions)).float()
        losses = torch.logsumexp(logits, dim=-1) - logits.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        nll[positions] = losses.detach().cpu().numpy()
        del logits, losses
    return nll


def analyze_condition(
    args,
    name: str,
    model_path: str,
    tokenizer: Tokenizer,
    added_tokens: Sequence[str],
    sample: Dict,
    segment_specs: Dict[int, Sequence[int]],
) -> Dict:
    dtype = resolve_dtype(args.dtype, args.device)
    print(f"[info] loading {name}: {model_path}")
    model = load_model(args, model_path, tokenizer, added_tokens, dtype)
    layer_ids = pick_layers(args.layers, len(model.model.layers))
    print(f"[info] {name}: layers={layer_ids}")
    capture = AttentionSubblockCapture(model, layer_ids, segment_specs)

    batch = sample["batch"]
    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)
    position_ids = batch["position_ids"].to(args.device)
    row_comp_index = batch["row_comp_index"].to(args.device)
    column_comp_index = batch["column_comp_index"].to(args.device)

    try:
        with torch.inference_mode():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                row_comp_index=row_comp_index,
                column_comp_index=column_comp_index,
            )
            hidden_states = outputs.last_hidden_state[0]
            segment_lengths = continuation_segment_map(
                sample["locate_index"], sample["seq_len"]
            )
            nll = continuation_nll(
                model,
                hidden_states,
                batch["labels"][0],
                segment_lengths,
                args.logit_chunk_size,
            )
            blocks = capture.averages()
    finally:
        capture.close()

    result = {
        "blocks": blocks,
        "nll": nll,
        "segment_lengths": segment_lengths,
        "layers": layer_ids,
        "n_heads": model.config.num_attention_heads,
        "checkpoint_vocab": model.get_input_embeddings().num_embeddings,
        "tokenizer_vocab": len(tokenizer),
    }

    del outputs, hidden_states, model
    del input_ids, attention_mask, position_ids, row_comp_index, column_comp_index
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return result


def normalize_rows(block: np.ndarray) -> np.ndarray:
    denominator = block.sum(axis=-1, keepdims=True)
    denominator[denominator == 0] = 1.0
    return block / denominator


def mean_head_block(per_head_block: np.ndarray, row_normalize: bool) -> np.ndarray:
    block = per_head_block.mean(axis=0)
    return normalize_rows(block) if row_normalize else block


def band_localization_score(block: np.ndarray) -> float:
    n_comp, segment_length = block.shape
    centers = np.array(
        [int((index + 0.5) * segment_length / n_comp) for index in range(n_comp)]
    )
    maxima = block.argmax(axis=1).astype(float)
    return float(np.mean(np.abs(maxima - centers)) / segment_length)


def plot_pair(
    upl_block: np.ndarray,
    baseline_block: np.ndarray,
    item: Dict,
    upl_score: float,
    baseline_score: float,
    out_path: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = max(3.2, max(upl_block.shape[0], baseline_block.shape[0]) * 0.22 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(13, height))
    vmax = max(float(np.max(upl_block)), float(np.max(baseline_block)), 1e-12)
    conditions = [
        (axes[0], upl_block, "MemoSight + UPL", upl_score),
        (axes[1], baseline_block, "LightThinker + contiguous", baseline_score),
    ]
    for axis, block, title, score in conditions:
        image = axis.imshow(block, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        n_comp, segment_length = block.shape
        centers = [
            int((index + 0.5) * segment_length / n_comp)
            for index in range(n_comp)
        ]
        axis.plot(centers, range(n_comp), "r.", markersize=5, label="uniform center")
        axis.set_title(f"{title}\nnorm-offset={score:.3f}")
        axis.set_xlabel(f"original segment token (0..{segment_length - 1})")
        axis.set_ylabel("memory token")
        if n_comp > 20:
            ticks = np.linspace(0, n_comp - 1, 10, dtype=int)
            axis.set_yticks(ticks)
        axis.legend(loc="upper right", fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Memory-token → compressed-segment attention\n"
        f"segment #{item['index']} | raw lengths "
        f"{item['upl_len']} / {item['baseline_len']}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_per_head(per_head_block: np.ndarray, title: str, out_path: str, row_normalize: bool):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = normalize_rows(per_head_block) if row_normalize else per_head_block
    n_heads = blocks.shape[0]
    columns = 4
    rows = math.ceil(n_heads / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.5 * rows))
    axes = np.asarray(axes).reshape(-1)
    vmax = max(float(np.max(blocks)), 1e-12)
    for head in range(n_heads):
        axes[head].imshow(blocks[head], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        axes[head].set_title(f"head {head}", fontsize=8)
        axes[head].set_xticks([])
        axes[head].set_yticks([])
    for head in range(n_heads, len(axes)):
        axes[head].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def bucket_index(raw_length: int, edges: Sequence[int]) -> int:
    for index, edge in enumerate(edges):
        if raw_length < edge:
            return index
    return len(edges)


def bucket_labels(edges: Sequence[int]) -> List[str]:
    labels = []
    previous = 0
    for edge in edges:
        labels.append(f"{previous}-{edge}")
        previous = edge
    labels.append(f"{previous}+")
    return labels


def condition_mean(nll: np.ndarray, lengths: np.ndarray, bucket: Optional[int], edges) -> Tuple[float, int]:
    valid = ~np.isnan(nll) & (lengths >= 0)
    if bucket is not None:
        valid &= np.array(
            [bucket_index(int(length), edges) == bucket if length >= 0 else False for length in lengths]
        )
    count = int(valid.sum())
    return (float(np.mean(nll[valid])) if count else float("nan"), count)


def compute_metrics(upl_result: Dict, baseline_result: Dict, edges: Sequence[int]) -> Dict:
    upl_mean, upl_count = condition_mean(
        upl_result["nll"], upl_result["segment_lengths"], None, edges
    )
    baseline_mean, baseline_count = condition_mean(
        baseline_result["nll"], baseline_result["segment_lengths"], None, edges
    )
    metrics = {
        "global_nll_upl": upl_mean,
        "global_nll_contiguous": baseline_mean,
        "global_delta": baseline_mean - upl_mean,
        "n_positions_upl": upl_count,
        "n_positions_contiguous": baseline_count,
        "buckets": [],
    }
    labels = bucket_labels(edges)
    for bucket, label in enumerate(labels):
        upl_value, count_upl = condition_mean(
            upl_result["nll"], upl_result["segment_lengths"], bucket, edges
        )
        baseline_value, count_baseline = condition_mean(
            baseline_result["nll"], baseline_result["segment_lengths"], bucket, edges
        )
        delta = baseline_value - upl_value
        metrics["buckets"].append(
            {
                "label": label,
                "nll_upl": upl_value,
                "nll_contiguous": baseline_value,
                "delta": delta,
                "count_upl": count_upl,
                "count_contiguous": count_baseline,
            }
        )
    return metrics


def plot_bucket_delta(metrics: Dict, out_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = metrics["buckets"]
    labels = [item["label"] for item in buckets]
    deltas = [item["delta"] for item in buckets]
    values = [0.0 if np.isnan(value) else value for value in deltas]
    colors = ["#4c72b0" if value >= 0 else "#c44e52" for value in values]

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    positions = np.arange(len(labels))
    axis.bar(positions, values, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("compressed segment raw length (tokens)")
    axis.set_ylabel("ΔNLL = NLL(LightThinker) - NLL(MemoSight)")
    axis.set_title("Continuation-NLL benefit by segment length")
    for position, value, item in zip(positions, values, buckets):
        if not np.isnan(item["delta"]):
            axis.text(
                position,
                value,
                f"{value:+.3f}\nn={item['count_upl']}/{item['count_contiguous']}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> int:
    args = get_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sample_line = read_sample_line(args.data_path, args.sample_index)

    two_checkpoint = args.baseline_model_path is not None
    baseline_model_path = args.baseline_model_path or args.model_path
    baseline_config_path = args.baseline_compress_config or args.compress_config
    upl_tokenizer_path = args.tokenizer_path or args.model_path
    baseline_tokenizer_path = (
        args.baseline_tokenizer_path or args.baseline_model_path or upl_tokenizer_path
    )
    regime = "B (two checkpoints, each trained layout)" if two_checkpoint else "A (same-weights probe)"
    print(f"[info] comparison regime: {regime}")

    upl_tokenizer, upl_config, upl_added = build_tokenizer_and_config(
        upl_tokenizer_path,
        args.compress_config,
        args.bos_token,
        args.eos_token,
        args.model_type,
    )
    baseline_tokenizer, baseline_config, baseline_added = build_tokenizer_and_config(
        baseline_tokenizer_path,
        baseline_config_path,
        args.bos_token,
        args.eos_token,
        args.model_type,
    )

    upl_sample = build_condition_sample(
        args, sample_line, upl_tokenizer, upl_config, use_epl=True
    )
    baseline_sample = build_condition_sample(
        args, sample_line, baseline_tokenizer, baseline_config, use_epl=False
    )
    print(
        f"[info] sequence lengths: MemoSight={upl_sample['seq_len']} "
        f"LightThinker={baseline_sample['seq_len']}"
    )

    paired_segments = pair_segments(upl_sample, baseline_sample, args)
    if not paired_segments:
        stats = {
            "upl_segments": upl_sample["locate_index"],
            "baseline_segments": baseline_sample["locate_index"],
            "min_seg_len": args.min_seg_len,
        }
        stats_path = os.path.join(args.out_dir, f"sample{args.sample_index}_segment_stats.json")
        with open(stats_path, "w", encoding="utf-8") as stream:
            json.dump(stats, stream, indent=2)
        print(
            f"[warning] no paired segment meets min_seg_len={args.min_seg_len}; "
            f"see {stats_path}"
        )
        return 0

    upl_specs = {item["index"]: item["upl_spec"] for item in paired_segments}
    baseline_specs = {
        item["index"]: item["baseline_spec"] for item in paired_segments
    }
    upl_result = analyze_condition(
        args,
        "MemoSight/UPL",
        args.model_path,
        upl_tokenizer,
        upl_added,
        upl_sample,
        upl_specs,
    )
    baseline_result = analyze_condition(
        args,
        "LightThinker/contiguous",
        baseline_model_path,
        baseline_tokenizer,
        baseline_added,
        baseline_sample,
        baseline_specs,
    )

    edges = [int(value.strip()) for value in args.bucket_edges.split(",") if value.strip()]
    metrics = compute_metrics(upl_result, baseline_result, edges)
    print(
        f"[metric] continuation NLL: MemoSight={metrics['global_nll_upl']:.4f} "
        f"LightThinker={metrics['global_nll_contiguous']:.4f} "
        f"delta={metrics['global_delta']:+.4f}"
    )
    for bucket in metrics["buckets"]:
        if bucket["count_upl"] == 0 and bucket["count_contiguous"] == 0:
            print(f"[metric]   seglen {bucket['label']:>8s}: no positions")
        else:
            print(
                f"[metric]   seglen {bucket['label']:>8s}: "
                f"n={bucket['count_upl']}/{bucket['count_contiguous']} "
                f"MemoSight={bucket['nll_upl']:.4f} "
                f"LightThinker={bucket['nll_contiguous']:.4f} "
                f"delta={bucket['delta']:+.4f}"
            )
    plot_bucket_delta(
        metrics,
        os.path.join(
            args.out_dir, f"sample{args.sample_index}_continuation_nll_by_seglen.png"
        ),
    )

    segment_summaries = []
    for rank, item in enumerate(paired_segments):
        segment_id = item["index"]
        upl_per_head = upl_result["blocks"][segment_id]
        baseline_per_head = baseline_result["blocks"][segment_id]
        upl_block = mean_head_block(upl_per_head, args.row_normalize)
        baseline_block = mean_head_block(baseline_per_head, args.row_normalize)
        upl_score = band_localization_score(upl_block)
        baseline_score = band_localization_score(baseline_block)

        figure_path = os.path.join(
            args.out_dir,
            f"sample{args.sample_index}_seg{segment_id}_"
            f"len{item['upl_len']}-{item['baseline_len']}.png",
        )
        plot_pair(
            upl_block,
            baseline_block,
            item,
            upl_score,
            baseline_score,
            figure_path,
        )
        segment_summaries.append(
            {
                "segment_index": segment_id,
                "upl_raw_length": item["upl_len"],
                "baseline_raw_length": item["baseline_len"],
                "upl_n_comp": item["upl_spec"][3],
                "baseline_n_comp": item["baseline_spec"][3],
                "upl_norm_offset": upl_score,
                "baseline_norm_offset": baseline_score,
            }
        )

        if args.per_head and rank == 0:
            plot_per_head(
                upl_per_head,
                "MemoSight + UPL: per-head locality",
                os.path.join(
                    args.out_dir,
                    f"sample{args.sample_index}_seg{segment_id}_perhead_MemoSight_UPL.png",
                ),
                args.row_normalize,
            )
            plot_per_head(
                baseline_per_head,
                "LightThinker + contiguous: per-head locality",
                os.path.join(
                    args.out_dir,
                    f"sample{args.sample_index}_seg{segment_id}_perhead_LightThinker.png",
                ),
                args.row_normalize,
            )

    metadata = {
        "regime": regime,
        "sample_index": args.sample_index,
        "memosight_checkpoint": str(Path(args.model_path).resolve()),
        "lightthinker_checkpoint": str(Path(baseline_model_path).resolve()),
        "memosight_config": str(Path(args.compress_config).resolve()),
        "lightthinker_config": str(Path(baseline_config_path).resolve()),
        "memosight_sequence_length": upl_sample["seq_len"],
        "lightthinker_sequence_length": baseline_sample["seq_len"],
        "memosight_layers": upl_result["layers"],
        "lightthinker_layers": baseline_result["layers"],
        "functional_metrics": metrics,
        "segments": segment_summaries,
    }
    metrics_path = os.path.join(
        args.out_dir, f"sample{args.sample_index}_functional_metrics.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    summary_path = os.path.join(args.out_dir, f"sample{args.sample_index}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as stream:
        stream.write(f"regime: {regime}\n")
        stream.write(
            f"continuation NLL: MemoSight={metrics['global_nll_upl']:.6f} "
            f"LightThinker={metrics['global_nll_contiguous']:.6f} "
            f"delta={metrics['global_delta']:+.6f}\n"
        )
        for item in segment_summaries:
            stream.write(
                f"segment #{item['segment_index']}: "
                f"MemoSight offset={item['upl_norm_offset']:.6f}, "
                f"LightThinker offset={item['baseline_norm_offset']:.6f}\n"
            )
    print(f"[saved] {metrics_path}")
    print(f"[saved] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
