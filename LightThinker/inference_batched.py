"""
Batched (batch>1) inference for the MemoSight / LightThinker compression decoder.

This is a PARALLEL implementation to the single-sequence engine in `inference.py`.
The bs=1 loops in `inference.py` (`_sentence_level_generate`, `_sentence_level_mtp_register_generate`)
are left untouched and used as the correctness ORACLE.

Design (see plan): per-sample bookkeeping arrays + one LEFT-PADDED batched KV cache of
width L_pad = max_s valid_len[s]. Physical compression is preserved: on a compression step
(or MTP ragged-accept) we gather-rebuild the cache keeping only each sample's live columns,
so attention compute + peak memory actually shrink -- which is what the speedup measurement needs.

Scope of this file:
    - update_attention_method == 'local'
    - rolling_rope == False, compress_prompt == False
    - Vanilla AR, MemoSight AR (with EPL), and MemoSight SD (with EPL)
    - greedy decoding with the same repetition-penalty rule as inference.py
    - attn_implementation forced to 'sdpa'
MTP (`_sentence_level_mtp_register_generate_batched`) is added in a later step.

Everything else asserts out so we never silently produce wrong numbers.
"""

import os
import sys
import time
import json
import argparse
import hashlib
import random
import statistics
from pathlib import Path
import torch
import jsonlines
from typing import *
from tqdm import tqdm

# reuse everything from the single-sequence engine / project
from config import Config
from tokenizer import Tokenizer
from inference import (
    InferenceUtils,
    get_model_and_tokenizer,
    generate as oracle_generate,
    AttentionUtils,
    TokenUtils,
    MTPStats,
)
from dataset_reader import (
    AIME2026Reader,
    BBHReader,
    GPQAReader,
    GSM8KReader,
    HMMTFeb2025Reader,
    MMLUReader,
    Reader,
)
from utils import str2bool


def _penalized_argmax(
    logits: torch.Tensor,
    histories: List[List[int]],
    tokenizer: Tokenizer,
    repetition_penalty: float,
    extras: Optional[List[List[int]]] = None,
) -> List[int]:
    """Greedy batched counterpart of InferenceUtils.get_predicted_token_ids."""
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    result = []
    special_ids = set(tokenizer.all_special_ids)
    for row in range(logits.shape[0]):
        row_logits = logits[row].clone()
        if repetition_penalty != 1.0:
            seen = set(histories[row])
            if extras is not None:
                seen.update(extras[row])
            seen.difference_update(special_ids)
            if seen:
                token_ids = torch.as_tensor(
                    sorted(seen), dtype=torch.long, device=row_logits.device
                )
                selected = row_logits.index_select(0, token_ids)
                selected = torch.where(
                    selected < 0,
                    selected * repetition_penalty,
                    selected / repetition_penalty,
                )
                row_logits.index_copy_(0, token_ids, selected)
        result.append(int(torch.argmax(row_logits).item()))
    return result


# --------------------------------------------------------------------------- #
# Per-sample control state (parallels the scalars in _sentence_level_generate) #
# --------------------------------------------------------------------------- #
class BatchState:
    """Per-sample bookkeeping for the *active* working set.

    All arrays are Python lists indexed by the WORKING row (0..B_active-1); `orig_index`
    maps a working row back to its original sample id so outputs land in the right slot.
    Finished samples are removed from the working set (active-row compaction).
    """

    def __init__(self, orig_index: List[int], prompt_ids: List[List[int]],
                 valid_len: List[int], last_pos: List[int], started_at: float):
        n = len(orig_index)
        self.original_batch_size = n
        self.orig_index: List[int] = list(orig_index)        # working row -> original sample id
        self.prompt_ids: List[List[int]] = prompt_ids        # per working row (shown prompt ids)
        self.out_ids: List[List[int]] = [[] for _ in range(n)]  # committed output ids per working row
        self.valid_len: List[int] = list(valid_len)          # #real tokens in cache per row
        self.last_pos: List[int] = list(last_pos)            # last position_id emitted per row
        # local-compression bookkeeping (logical coords, 0..valid_len)
        self.local_start_off: List[int] = list(valid_len)    # where current CoT begins (= end of prompt initially)
        self.van_cot_start: List[int] = list(last_pos)       # position where current CoT segment starts
        for i in range(n):
            self.van_cot_start[i] = last_pos[i] + 1           # first generated position
        self.cot_start: List[int] = [value + 1 for value in last_pos]
        self.step_count: List[int] = [0] * n                 # committed new tokens per row
        self.compression_steps: List[int] = [0] * n

        # Fixed-size per-request accounting survives active-row compaction.
        self.started_at = started_at
        self.finish_reason: List[Optional[str]] = [None] * n
        self.request_completion_s: List[Optional[float]] = [None] * n
        self.peak_valid_kv_tokens: List[int] = list(valid_len)
        self.peak_total_valid_kv_tokens = sum(valid_len)
        self.peak_physical_kv_slots = n * (max(valid_len) if valid_len else 0)
        self._done: Dict[int, List[int]] = {}

    def __len__(self):
        return len(self.orig_index)

    def observe_kv(self, valid_len: Optional[List[int]] = None,
                   physical_width: Optional[int] = None):
        lengths = list(self.valid_len if valid_len is None else valid_len)
        if len(lengths) != len(self.orig_index):
            raise ValueError("KV accounting length does not match active rows")
        for row, length in enumerate(lengths):
            original = self.orig_index[row]
            self.peak_valid_kv_tokens[original] = max(
                self.peak_valid_kv_tokens[original], int(length)
            )
        self.peak_total_valid_kv_tokens = max(
            self.peak_total_valid_kv_tokens, sum(int(value) for value in lengths)
        )
        width = max(lengths, default=0) if physical_width is None else int(physical_width)
        self.peak_physical_kv_slots = max(
            self.peak_physical_kv_slots, len(lengths) * width
        )

    def mark_finished(self, row: int, reason: str):
        original = self.orig_index[row]
        if self.finish_reason[original] is None:
            self.finish_reason[original] = reason
            self.request_completion_s[original] = time.perf_counter() - self.started_at


# --------------------------------------------------------------------------- #
# Batched KV cache with gather-based compaction (replaces KVUtils.reduce_cache) #
# --------------------------------------------------------------------------- #
class BatchKVUtils:
    def __init__(self):
        from transformers import DynamicCache
        self.past_key_values = DynamicCache()

    def get_cache(self):
        return self.past_key_values

    @property
    def width(self) -> int:
        if len(self.past_key_values.key_cache) == 0:
            return 0
        return self.past_key_values.key_cache[0].shape[2]

    @torch.no_grad()
    def rebuild(self, row_keep: List[int], col_keep: List[List[int]], device):
        """Gather-rebuild the cache: keep working rows `row_keep`; for kept row i keep cache
        columns `col_keep[i]` (ordered). Produces a LEFT-PADDED [R, H, L_new, D] cache where
        each row's kept columns are RIGHT-aligned (newest at the far right) and pad slots
        (index 0, masked by attention) sit on the left.

        Returns new_valid_len: List[int] for the kept rows.
        """
        pkv = self.past_key_values
        n_layers = len(pkv.key_cache)
        assert n_layers > 0
        R = len(row_keep)
        new_valid_len = [len(col_keep[i]) for i in range(R)]
        L_new = max(new_valid_len) if R > 0 else 0

        # gather_index[R, L_new]: right-aligned real cols, left pad -> index 0
        gather_index = torch.zeros((R, L_new), dtype=torch.long, device=device)
        for i in range(R):
            cols = col_keep[i]
            n = len(cols)
            if n > 0:
                gather_index[i, L_new - n:] = torch.tensor(cols, dtype=torch.long, device=device)

        row_keep_t = torch.tensor(row_keep, dtype=torch.long, device=device)
        for l in range(n_layers):
            K = pkv.key_cache[l]                       # [B, H, L_old, D]
            V = pkv.value_cache[l]
            B0, Hh, L_old, Dd = K.shape
            K = K.index_select(0, row_keep_t)          # [R, H, L_old, D]
            V = V.index_select(0, row_keep_t)
            idx = gather_index[:, None, :, None].expand(R, Hh, L_new, Dd)
            pkv.key_cache[l] = torch.gather(K, 2, idx).contiguous()
            pkv.value_cache[l] = torch.gather(V, 2, idx).contiguous()
        pkv._seen_tokens = L_new
        return new_valid_len


# --------------------------------------------------------------------------- #
# Batched additive attention masks ([B,1,W,L_pad+W]) -- local method          #
# --------------------------------------------------------------------------- #
class BatchAttnUtils:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device
        self.min_dtype = torch.finfo(dtype).min

    @torch.no_grad()
    def build_local_step(
        self,
        L_pad: int,
        W: int,
        valid_len: List[int],          # per row, real tokens currently in cache
        step_keep: List[int],          # per row, #real step tokens (right-aligned in W)
        comp_indicator: List[Optional[Tuple[int, int, int]]],  # per row: (text_start_off, text_end_off, continue_rel) or None
        reg_count: int = 0,            # MTP: trailing reg_count step tokens are registers (block-right-aligned)
    ) -> torch.Tensor:
        """Build additive mask; 0 visible, min_dtype hidden.
        Column layout: [0:L_pad] = cache (left-padded per row), [L_pad:L_pad+W] = new query block.
        comp_indicator offsets are LOGICAL (relative to the row's real-token start).
        """
        R = len(valid_len)
        mv = self.min_dtype
        mask = torch.zeros((R, 1, W, L_pad + W), dtype=self.dtype, device=self.device)

        base_tri = torch.triu(
            torch.full((W, W), mv, dtype=self.dtype, device=self.device), diagonal=1
        )
        for b in range(R):
            pad_left_kv = L_pad - valid_len[b]           # masked cache columns on the left
            pad_left_q = W - step_keep[b]                # dead query rows on top (left pad of step)
            # 1. causal on new-query block
            mask[b, 0, :, L_pad:L_pad + W] = base_tri
            # 1b. in-block LEFT PAD: when W>1 (some other row compressed), this row's step is
            #     right-aligned; its leading (W-step_keep) block columns are PAD tokens and must
            #     be hidden from ALL query rows, else the real token attends pad via causality.
            if pad_left_q > 0:
                mask[b, 0, :, L_pad:L_pad + pad_left_q] = mv
            # 2. left-pad of KV columns -> hidden
            if pad_left_kv > 0:
                mask[b, 0, :, 0:pad_left_kv] = mv
            # 3. dead (left-pad) query rows: keep them from being all -inf via diagonal guard below;
            #    also they must not see real content (irrelevant, discarded) -- leave as-is.
            # 4. compression indicator: continue-token (and after) must not attend the raw CoT text.
            ind = comp_indicator[b]
            if ind is not None:
                text_start_off, text_end_off, continue_rel = ind
                # absolute cache columns of the CoT text span
                real_base = L_pad - valid_len[b]
                text_start_abs = real_base + text_start_off
                text_end_abs = real_base + text_end_off
                # query rows from `continue_rel` onward (within the real step block) mask the text.
                # real step block occupies query rows [pad_left_q, W); continue is at pad_left_q + continue_rel.
                q_from = pad_left_q + continue_rel
                if text_end_abs > text_start_abs and q_from < W:
                    mask[b, 0, q_from:, text_start_abs:text_end_abs] = mv
            # 4b. MTP register-vs-register: registers are the trailing reg_count real step
            #     tokens -> block cols [W-reg_count, W). A register query must NOT attend the
            #     OTHER registers of the same step (parallels inference.py:667-672).
            if reg_count > 0:
                reg_lo = W - reg_count
                for q in range(reg_lo, W):
                    for kk in range(reg_lo, W):
                        if kk != q:
                            mask[b, 0, q, L_pad + kk] = mv
            # 5. _unmask_unattended guard: every query row must attend >=1 key. Force self-diagonal
            #    on the new block (row q attends its own new-block column q). Real rows already do via
            #    causal; this also covers dead/left-pad rows harmlessly.
            diag = torch.arange(W, device=self.device)
            mask[b, 0, diag, L_pad + diag] = 0.0
        return mask


# --------------------------------------------------------------------------- #
# Batched greedy argmax (parallels InferenceUtils.get_predicted_token_ids)     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batched_argmax(model_output, idx: int = -1) -> torch.Tensor:
    logits = model_output.logits            # [B, seq, vocab]
    return torch.argmax(logits[:, idx, :], dim=-1)   # [B]


# --------------------------------------------------------------------------- #
# Batched prefill (compress_prompt == False)                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batched_prefill(
    model,
    tokenizer: Tokenizer,
    comp_config: Config,
    system_prompt: str,
    questions: List[str],
    kv_utils: BatchKVUtils,
    attn_utils: BatchAttnUtils,
    device: str,
    repetition_penalty: float = 1.0,
    max_prompt_len: Optional[int] = None,
) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
    """Returns (pred[B], prompt_ids_per_sample, real_len_per_sample)."""
    B = len(questions)
    per_ids: List[List[int]] = []
    for q in questions:
        prompt = tokenizer.bos_token + comp_config.template_cfg['complete'].format(
            system=system_prompt, question=q
        )
        ids = tokenizer.tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
        per_ids.append(ids)

    real_len = [len(x) for x in per_ids]
    if max_prompt_len is not None and any(length > max_prompt_len for length in real_len):
        raise ValueError(
            f"rendered prompt exceeds max_prompt_len={max_prompt_len}; lengths={real_len}"
        )
    Lp = max(real_len)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = torch.full((B, Lp), pad_id, dtype=torch.long, device=device)
    position_ids = torch.zeros((B, Lp), dtype=torch.long, device=device)
    for b in range(B):
        n = real_len[b]
        input_ids[b, Lp - n:] = torch.tensor(per_ids[b], dtype=torch.long, device=device)
        position_ids[b, Lp - n:] = torch.arange(n, dtype=torch.long, device=device)

    # attention mask [B,1,Lp,Lp]: causal + left-pad hidden + diagonal guard
    mv = attn_utils.min_dtype
    mask = torch.zeros((B, 1, Lp, Lp), dtype=attn_utils.dtype, device=device)
    base_tri = torch.triu(torch.full((Lp, Lp), mv, dtype=attn_utils.dtype, device=device), diagonal=1)
    for b in range(B):
        mask[b, 0] = base_tri
        pad_left = Lp - real_len[b]
        if pad_left > 0:
            mask[b, 0, :, 0:pad_left] = mv
    diag = torch.arange(Lp, device=device)
    mask[:, 0, diag, diag] = 0.0

    model_output = model(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=kv_utils.get_cache(),
        use_cache=True,
        return_dict=True,
    )
    pred = torch.as_tensor(
        _penalized_argmax(
            model_output.logits[:, -1, :], per_ids, tokenizer, repetition_penalty
        ),
        dtype=torch.long,
        device=device,
    )
    return pred, per_ids, real_len


# --------------------------------------------------------------------------- #
# Vanilla / LightThinker batched loop (parallels _sentence_level_generate)     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _sentence_level_generate_batched(
    model,
    tokenizer: Tokenizer,
    comp_config: Config,
    max_new_tokens: int,
    kv_utils: BatchKVUtils,
    attn_utils: BatchAttnUtils,
    state: BatchState,
    pred: torch.Tensor,          # [B] first predicted token from prefill
    device: str,
    use_EPL: bool = False,
    repetition_penalty: float = 1.0,
    compression_enabled: bool = True,
    force_split_schedule: Optional[List[set]] = None,   # TEST-ONLY: per orig-sample set of step_counts to force a split
) -> None:
    eos_id = tokenizer.eos_token_id
    split_id = comp_config.split_token_id
    continue_id = comp_config.continue_token_id
    pad_id = eos_id if eos_id is not None else 0

    pred_list: List[int] = pred.tolist()

    while len(state) > 0:
        R = len(state)
        L_pad = kv_utils.width

        # ---- 1. plan per-row step tokens ----
        step_tokens: List[List[int]] = []
        is_comp: List[bool] = []
        comp_indicator: List[Optional[Tuple[int, int, int]]] = []
        for b in range(R):
            p = pred_list[b]
            if compression_enabled and p == split_id:
                cot_length = state.last_pos[b] + 2 - (
                    state.cot_start[b] if use_EPL else state.van_cot_start[b]
                )
                comp_tokens = comp_config.get_output_comp_token_id(cot_length=cot_length)
                tokens = [split_id] + list(comp_tokens) + [continue_id]
                step_tokens.append(tokens)
                is_comp.append(True)
                # indicator: text span = [local_start_off, split_off+1); continue is last real step token.
                split_off = state.valid_len[b]              # split goes right after current real tokens
                text_start_off = state.local_start_off[b]
                text_end_off = split_off + 1
                continue_rel = len(tokens) - 1              # continue is the last token in the block
                comp_indicator.append((text_start_off, text_end_off, continue_rel))
            else:
                step_tokens.append([p])
                is_comp.append(False)
                comp_indicator.append(None)

        step_keep = [len(t) for t in step_tokens]
        W = max(step_keep)

        # ---- 2. input_ids / position_ids [R, W], right-aligned ----
        input_ids = torch.full((R, W), pad_id, dtype=torch.long, device=device)
        position_ids = torch.zeros((R, W), dtype=torch.long, device=device)
        for b in range(R):
            toks = step_tokens[b]
            wk = len(toks)
            input_ids[b, W - wk:] = torch.tensor(toks, dtype=torch.long, device=device)
            base = state.last_pos[b]
            if use_EPL and is_comp[b]:
                cot_end = base + 2
                compression_count = wk - 2
                step = (cot_end - state.cot_start[b]) / max(compression_count, 1)
                compressed_positions = [
                    state.cot_start[b] + int(k * step + step / 2)
                    for k in range(compression_count)
                ]
                row_positions = [cot_end - 1] + compressed_positions + [cot_end]
                position_ids[b, W - wk:] = torch.as_tensor(
                    row_positions, dtype=torch.long, device=device
                )
            else:
                position_ids[b, W - wk:] = torch.arange(
                    base + 1, base + 1 + wk, dtype=torch.long, device=device
                )
            # dead query rows (left pad): give them a benign position (0); masked out anyway.

        # ---- 3. mask ----
        valid_len = list(state.valid_len)
        mask = attn_utils.build_local_step(L_pad, W, valid_len, step_keep, comp_indicator)

        # ---- 4. forward (cache grows by W) ----
        model_output = model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=kv_utils.get_cache(),
            use_cache=True,
            return_dict=True,
        )
        state.observe_kv(
            [valid_len[b] + step_keep[b] for b in range(R)],
            physical_width=L_pad + W,
        )

        # ---- 5. next-token predictions (last real col = rightmost) ----
        histories = [
            state.prompt_ids[b] + state.out_ids[b] + [step_tokens[b][0]]
            for b in range(R)
        ]
        next_pred = _penalized_argmax(
            model_output.logits[:, -1, :], histories, tokenizer, repetition_penalty
        )

        # ---- 6. commit tokens, update per-row bookkeeping ----
        # After forward, cache width = L_pad + W. For row b, the W new columns occupy
        # [L_pad, L_pad+W); its real step tokens are right-aligned at [L_pad + (W-wk), L_pad+W).
        newly_finished_rows: List[int] = []
        finish_reasons: Dict[int, str] = {}
        for b in range(R):
            toks = step_tokens[b]
            wk = len(toks)
            # Output string parity with oracle: it appends only the DRIVER token
            # (`predicted_token_id`, i.e. the split token or the plain token), NOT the
            # internal comp/continue tokens (inference.py:1442). And it counts loop
            # iterations, not tokens (inference.py:1595).
            state.out_ids[b].append(toks[0])
            state.step_count[b] += 1
            if is_comp[b]:
                state.compression_steps[state.orig_index[b]] += 1
                if use_EPL:
                    state.last_pos[b] = state.last_pos[b] + 2
                    state.cot_start[b] = state.last_pos[b] + 1
                else:
                    state.last_pos[b] = state.last_pos[b] + wk
                    state.van_cot_start[b] = state.last_pos[b] + 1
            else:
                state.last_pos[b] = state.last_pos[b] + 1

        # ---- 6b. TEST-ONLY forced split injection (deterministic per orig sample) ----
        if force_split_schedule is not None:
            for b in range(R):
                if state.step_count[b] in force_split_schedule[state.orig_index[b]]:
                    next_pred[b] = split_id

        # ---- 7. compaction: build row_keep + col_keep, then rebuild ----
        # Determine per-row termination on the NEXT predicted token / length.
        row_keep: List[int] = []
        col_keep: List[List[int]] = []
        keep_state_idx: List[int] = []
        for b in range(R):
            wk = step_keep[b]
            real_new_start = L_pad + (W - wk)            # abs col of this row's first real new token
            real_base = L_pad - valid_len[b]             # abs col of this row's first real old token
            if is_comp[b]:
                # keep [real_base, local_start_abs) U [split_abs+1, cache_end)
                local_start_abs = real_base + state.local_start_off[b]
                split_abs = real_new_start               # split is first real new token
                kept = list(range(real_base, local_start_abs)) + \
                       list(range(split_abs + 1, L_pad + W))
                # update valid_len / local_start after compression
                new_vlen = (local_start_abs - real_base) + (L_pad + W - (split_abs + 1))
                state.valid_len[b] = new_vlen
                state.local_start_off[b] = new_vlen      # next CoT begins at the (new) end
            else:
                # keep all old real cols + the single new real col
                kept = list(range(real_base, L_pad)) + list(range(real_new_start, L_pad + W))
                state.valid_len[b] = valid_len[b] + wk    # wk == 1 here

            # termination check on next predicted token
            np_tok = next_pred[b]
            finished = False
            if np_tok == eos_id:
                finished = True
                finish_reasons[b] = "eos"
            elif state.step_count[b] >= max_new_tokens:
                finished = True
                finish_reasons[b] = "length_limit"
            elif state.valid_len[b] + 4 >= attn_utils_max_len(kv_utils, attn_utils):
                finished = True
                finish_reasons[b] = "cache_limit"

            if finished:
                if np_tok == eos_id and len(state.out_ids[b]) < max_new_tokens:
                    state.out_ids[b].append(np_tok)
                newly_finished_rows.append(b)
            else:
                row_keep.append(b)
                col_keep.append(kept)
                keep_state_idx.append(b)

        # flush finished rows to a side store on the state object
        if newly_finished_rows:
            _flush_finished(state, newly_finished_rows, finish_reasons)

        if len(row_keep) == 0:
            break

        # rebuild cache to keep only surviving rows + compressed columns
        new_valid = kv_utils.rebuild(row_keep, col_keep, device)

        # compact the state arrays to the surviving rows (in row_keep order)
        _compact_state(state, keep_state_idx, new_valid)
        state.observe_kv(new_valid, physical_width=kv_utils.width)
        pred_list = [next_pred[b] for b in keep_state_idx]


# ---- state compaction helpers (active-row removal) ----
def attn_utils_max_len(kv_utils, attn_utils) -> int:
    # max_length stored on attn_utils
    return getattr(attn_utils, "max_length", 10**9)


def _flush_finished(state: BatchState, rows: List[int], reasons: Optional[Dict[int, str]] = None):
    for b in rows:
        state._done[state.orig_index[b]] = state.out_ids[b]
        state.mark_finished(b, (reasons or {}).get(b, "unknown"))


def _compact_state(state: BatchState, keep_idx: List[int], new_valid: List[int]):
    state.orig_index = [state.orig_index[i] for i in keep_idx]
    state.prompt_ids = [state.prompt_ids[i] for i in keep_idx]
    state.out_ids = [state.out_ids[i] for i in keep_idx]
    state.last_pos = [state.last_pos[i] for i in keep_idx]
    state.van_cot_start = [state.van_cot_start[i] for i in keep_idx]
    state.cot_start = [state.cot_start[i] for i in keep_idx]
    state.local_start_off = [state.local_start_off[i] for i in keep_idx]
    state.step_count = [state.step_count[i] for i in keep_idx]
    state.valid_len = list(new_valid)


# --------------------------------------------------------------------------- #
# MemoSight (MTP register + self-speculative) batched loop                     #
# parallels _sentence_level_mtp_register_generate (inference.py:1608)          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _sentence_level_mtp_register_generate_batched(
    model, tokenizer, comp_config, max_new_tokens: int,
    kv_utils: BatchKVUtils, attn_utils: BatchAttnUtils, state: BatchState,
    pred: torch.Tensor, device: str, register_count: int,
    mtp_stats: Optional[List[Optional[MTPStats]]] = None,
    use_EPL: bool = False,
    repetition_penalty: float = 1.0,
) -> None:
    eos_id = tokenizer.eos_token_id
    split_id = comp_config.split_token_id
    continue_id = comp_config.continue_token_id
    reg_id = comp_config.register_token_id
    pad_id = eos_id if eos_id is not None else 0
    rc = register_count
    draft_len = rc + 1
    assert rc >= 1, "MTP batched requires register_count>=1"
    max_len = attn_utils.max_length
    pred_list: List[int] = pred.tolist()

    def _stat(b):
        return mtp_stats[state.orig_index[b]] if mtp_stats is not None else None

    while len(state) > 0:
        R = len(state)
        L_pad = kv_utils.width

        # ---- phase 1: plan step tokens (driver [+comp/continue] + registers) ----
        step_tokens, is_comp, comp_ind, n_noreg = [], [], [], []
        for b in range(R):
            p = pred_list[b]
            state.out_ids[b].append(p)                     # commit driver (parallels inference.py:1666)
            if p == split_id:
                cot_length = state.last_pos[b] + 2 - (
                    state.cot_start[b] if use_EPL else state.van_cot_start[b]
                )
                comp = list(comp_config.get_output_comp_token_id(cot_length=cot_length))
                toks = [split_id] + comp + [continue_id] + [reg_id] * rc
                is_comp.append(True)
                n_noreg.append(1 + len(comp) + 1)
                split_off = state.valid_len[b]
                # oracle update_attention_local_for_mtp_register: n_prefix includes the trailing
                # registers, so comp_end_r == new_length-1 -> only the LAST step token (last register)
                # masks the CoT text. So the text-mask query-row start = last step token = len(toks)-1.
                comp_ind.append((state.local_start_off[b], split_off + 1, len(toks) - 1))
            else:
                toks = [p] + [reg_id] * rc
                is_comp.append(False)
                n_noreg.append(1)
                comp_ind.append(None)
            step_tokens.append(toks)
        step_keep = [len(t) for t in step_tokens]
        W = max(step_keep)

        input_ids = torch.full((R, W), pad_id, dtype=torch.long, device=device)
        position_ids = torch.zeros((R, W), dtype=torch.long, device=device)
        for b in range(R):
            wk = step_keep[b]; base = state.last_pos[b]
            input_ids[b, W - wk:] = torch.tensor(step_tokens[b], dtype=torch.long, device=device)
            if use_EPL and is_comp[b]:
                nn = n_noreg[b]
                cot_end = base + 2
                compression_count = nn - 2
                step = (cot_end - state.cot_start[b]) / max(compression_count, 1)
                compressed_positions = [
                    state.cot_start[b] + int(k * step + step / 2)
                    for k in range(compression_count)
                ]
                non_register_positions = [cot_end - 1] + compressed_positions + [cot_end]
                register_positions = list(range(cot_end + 1, cot_end + 1 + rc))
                position_ids[b, W - wk:] = torch.as_tensor(
                    non_register_positions + register_positions,
                    dtype=torch.long,
                    device=device,
                )
            else:
                position_ids[b, W - wk:] = torch.arange(
                    base + 1, base + 1 + wk, dtype=torch.long, device=device
                )

        valid_len = list(state.valid_len)
        mask = attn_utils.build_local_step(L_pad, W, valid_len, step_keep, comp_ind, reg_count=rc)
        mo = model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                   past_key_values=kv_utils.get_cache(), use_cache=True, return_dict=True)
        state.observe_kv(
            [valid_len[b] + step_keep[b] for b in range(R)],
            physical_width=L_pad + W,
        )

        # ---- phase 2: drafts (anchor + registers = last draft_len cols, uniform) ----
        drafts = [[] for _ in range(R)]
        histories = [state.prompt_ids[b] + state.out_ids[b] for b in range(R)]
        for j in range(draft_len):
            logits = mo.logits[:, -(draft_len - j), :]
            picked = _penalized_argmax(
                logits,
                histories,
                tokenizer,
                repetition_penalty,
                extras=[list(tokens) for tokens in drafts],
            )
            for b, token_id in enumerate(picked):
                drafts[b].append(token_id)

        # ---- phase 2b: rebuild #1 -> remove registers + CoT + in-block pad ----
        row_keep = list(range(R)); col_keep = []
        for b in range(R):
            real_base = L_pad - valid_len[b]
            wk = step_keep[b]; real_new_start = L_pad + (W - wk)
            nn = n_noreg[b]
            if is_comp[b]:
                local_start_abs = real_base + state.local_start_off[b]
                split_abs = real_new_start
                kept = list(range(real_base, local_start_abs)) + list(range(split_abs + 1, real_new_start + nn))
                new_vlen = (local_start_abs - real_base) + (nn - 1)
                state.local_start_off[b] = new_vlen
            else:
                kept = list(range(real_base, L_pad)) + [real_new_start]
                new_vlen = valid_len[b] + 1
            col_keep.append(kept)
            split_pos = state.last_pos[b] + 1                # split is the first new token
            state.valid_len[b] = new_vlen
            if is_comp[b]:
                state.compression_steps[state.orig_index[b]] += 1
                if use_EPL:
                    state.last_pos[b] = split_pos + 1
                    state.cot_start[b] = state.last_pos[b] + 1
                else:
                    state.last_pos[b] += nn
                    # Match the single-request MemoSight target convention.
                    state.van_cot_start[b] = split_pos + 2
            else:
                state.last_pos[b] += 1
        kv_utils.rebuild(row_keep, col_keep, device)
        L_pad2 = kv_utils.width
        valid_len2 = list(state.valid_len)
        state.observe_kv(valid_len2, physical_width=L_pad2)

        # ---- phase 3: classify control; decide verify ----
        is_control = [drafts[b][0] == split_id or drafts[b][0] == eos_id for b in range(R)]
        need_verify = any((not is_control[b]) for b in range(R))

        predicted = [None] * R
        accepted_count = [1] * R
        keep_n = [0] * R    # #verify-block tokens to keep in cache per row

        if need_verify:
            v_pos = torch.zeros((R, draft_len), dtype=torch.long, device=device)
            v_ids = torch.zeros((R, draft_len), dtype=torch.long, device=device)
            for b in range(R):
                v_ids[b] = torch.tensor(drafts[b], dtype=torch.long, device=device)
                base = state.last_pos[b]
                v_pos[b] = torch.arange(base + 1, base + 1 + draft_len, dtype=torch.long, device=device)
            v_mask = attn_utils.build_local_step(L_pad2, draft_len, valid_len2,
                                                 [draft_len] * R, [None] * R, reg_count=0)
            vo = model(input_ids=v_ids, attention_mask=v_mask, position_ids=v_pos,
                       past_key_values=kv_utils.get_cache(), use_cache=True, return_dict=True)
            state.observe_kv(
                [valid_len2[b] + draft_len for b in range(R)],
                physical_width=L_pad2 + draft_len,
            )
            vpreds = [[] for _ in range(R)]
            for j in range(draft_len):
                logits = vo.logits[:, -(draft_len - j), :]
                picked = _penalized_argmax(
                    logits,
                    histories,
                    tokenizer,
                    repetition_penalty,
                    extras=[tokens[:j + 1] for tokens in drafts],
                )
                for b, token_id in enumerate(picked):
                    vpreds[b].append(token_id)

            for b in range(R):
                st = _stat(b)
                if is_control[b]:
                    predicted[b] = drafts[b][0]; accepted_count[b] = 1; keep_n[b] = 0
                    if st is not None:
                        st.record_control_step()
                    continue
                acc = 1
                for i in range(draft_len - 1):
                    if vpreds[b][i] == drafts[b][i + 1]:
                        acc += 1
                    else:
                        break
                if st is not None:
                    st.record_verify(gamma=draft_len - 1, raw_accepted_len=acc)
                control_token = None; extra = []
                for tok in drafts[b][1:acc]:
                    if tok == split_id or tok == eos_id:
                        control_token = tok; break
                    extra.append(tok)
                effective = 1 + len(extra)
                keep_n[b] = effective
                state.out_ids[b].append(drafts[b][0])
                state.out_ids[b].extend(extra)
                accepted_count[b] = effective
                predicted[b] = control_token if control_token is not None else vpreds[b][acc - 1]
                state.last_pos[b] += effective

            # rebuild #2: keep first keep_n[b] verify-block cols per row (ragged)
            rk2 = list(range(R)); ck2 = []
            for b in range(R):
                real_cols = list(range(L_pad2 - valid_len2[b], L_pad2))
                keep_cols = list(range(L_pad2, L_pad2 + keep_n[b]))
                ck2.append(real_cols + keep_cols)
                state.valid_len[b] = valid_len2[b] + keep_n[b]
            kv_utils.rebuild(rk2, ck2, device)
            state.observe_kv(state.valid_len, physical_width=kv_utils.width)
        else:
            for b in range(R):
                predicted[b] = drafts[b][0]; accepted_count[b] = 1
                st = _stat(b)
                if st is not None:
                    st.record_control_step()

        for b in range(R):
            st = _stat(b)
            if st is not None:
                st.record_step(
                    forward_passes=(1 if is_control[b] else 2),
                    committed=accepted_count[b],
                    is_comp=is_comp[b],
                )

        # ---- phase 4: termination + active-row compaction ----
        keep_idx, row_keep_final, col_keep_final = [], [], []
        newly_finished = []
        finish_reasons: Dict[int, str] = {}
        cur_w = kv_utils.width
        for b in range(R):
            state.step_count[b] += accepted_count[b]
            np_tok = predicted[b]
            output_length = len(state.out_ids[b])
            if np_tok == eos_id:
                finished = True
                finish_reasons[b] = "eos"
            elif output_length >= max_new_tokens:
                finished = True
                finish_reasons[b] = "length_limit"
            elif state.valid_len[b] + draft_len + 4 >= max_len:
                finished = True
                finish_reasons[b] = "cache_limit"
            else:
                finished = False
            if finished:
                if np_tok == eos_id and len(state.out_ids[b]) < max_new_tokens:
                    state.out_ids[b].append(np_tok)
                newly_finished.append(b)
            else:
                keep_idx.append(b)
                row_keep_final.append(b)
                col_keep_final.append(list(range(cur_w - state.valid_len[b], cur_w)))
        if newly_finished:
            _flush_finished(state, newly_finished, finish_reasons)
        if not keep_idx:
            break
        new_valid = kv_utils.rebuild(row_keep_final, col_keep_final, device)
        _compact_state(state, keep_idx, new_valid)
        state.observe_kv(new_valid, physical_width=kv_utils.width)
        pred_list = [predicted[b] for b in keep_idx]


# --------------------------------------------------------------------------- #
# Public batched generate                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batched_generate(
    model,
    tokenizer: Tokenizer,
    comp_config: Config,
    system_prompt: str,
    questions: List[str],
    max_new_tokens: int,
    max_length: int,
    device: str,
    dtype,
    update_attention_method: str = "local",
    use_EPL: bool = False,
    spec_decode: bool = False,
    repetition_penalty: float = 1.0,
    vanilla: bool = False,
    mtp_draft_len: Optional[int] = None,
    mtp_stats: Optional[List[Optional[MTPStats]]] = None,
    force_split_schedule: Optional[List[set]] = None,   # TEST-ONLY (vanilla path)
    _debug_return_ids: bool = False,                     # TEST-ONLY
    return_details: bool = False,
) -> List[str]:
    """Returns decoded output string per sample (in the original order)."""
    assert update_attention_method == "local", "batched path currently supports local only"
    assert getattr(model.config, "_attn_implementation", "sdpa") != "flash_attention_2", \
        "batched 4D masks require sdpa/eager, not flash_attention_2"
    use_mtp = bool(comp_config.mtp_cfg) and spec_decode and not vanilla
    if spec_decode and not comp_config.mtp_cfg:
        # parity with oracle: spec_decode on a non-MTP config falls back to plain sentence loop
        use_mtp = False

    B = len(questions)
    kv_utils = BatchKVUtils()
    attn_utils = BatchAttnUtils(dtype=dtype, device=device)
    attn_utils.max_length = max_length

    started_at = time.perf_counter()
    pred, prompt_ids, real_len = batched_prefill(
        model, tokenizer, comp_config, system_prompt, questions, kv_utils, attn_utils,
        device, repetition_penalty=repetition_penalty,
        max_prompt_len=max_length - max_new_tokens,
    )
    last_pos = [n - 1 for n in real_len]
    state = BatchState(
        orig_index=list(range(B)),
        prompt_ids=prompt_ids,
        valid_len=list(real_len),
        last_pos=last_pos,
        started_at=started_at,
    )
    state.observe_kv(real_len, physical_width=kv_utils.width)

    if comp_config.output_comp_level != "sentence":
        raise NotImplementedError("batched path currently supports output_comp_level='sentence'")

    if use_mtp:
        rc = int(mtp_draft_len) if mtp_draft_len is not None else int(comp_config.mtp_cfg.get("max_offset", 2))
        rc = max(1, rc)
        _sentence_level_mtp_register_generate_batched(
            model, tokenizer, comp_config, max_new_tokens, kv_utils, attn_utils, state, pred, device,
            register_count=rc, mtp_stats=mtp_stats, use_EPL=use_EPL,
            repetition_penalty=repetition_penalty,
        )
    else:
        _sentence_level_generate_batched(
            model, tokenizer, comp_config, max_new_tokens, kv_utils, attn_utils, state, pred, device,
            use_EPL=use_EPL, repetition_penalty=repetition_penalty,
            compression_enabled=not vanilla,
            force_split_schedule=force_split_schedule,
        )

    # gather outputs in original order
    done: Dict[int, List[int]] = getattr(state, "_done", {})
    # any rows still in state at loop end (shouldn't normally happen) also flush
    for i in range(len(state)):
        done.setdefault(state.orig_index[i], state.out_ids[i])
        state.mark_finished(i, "unknown")
    ordered_ids = [done.get(i, [])[:max_new_tokens] for i in range(B)]
    if _debug_return_ids:
        return ordered_ids
    outputs: List[str] = []
    for i in range(B):
        outputs.append(tokenizer.decode(ordered_ids[i]))
    if return_details:
        return dict(
            outputs=outputs,
            output_ids=ordered_ids,
            finish_reasons=[value or "unknown" for value in state.finish_reason],
            request_completion_s=[float(value or 0.0) for value in state.request_completion_s],
            peak_valid_kv_tokens=list(state.peak_valid_kv_tokens),
            peak_total_valid_kv_tokens=int(state.peak_total_valid_kv_tokens),
            peak_physical_kv_slots=int(state.peak_physical_kv_slots),
            compression_steps=list(state.compression_steps),
        )
    return outputs


# --------------------------------------------------------------------------- #
# Fair speedup benchmark (parallels eval_dataset)                              #
# --------------------------------------------------------------------------- #
def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: List[float]) -> Dict[str, float]:
    return dict(
        mean=(statistics.fmean(values) if values else 0.0),
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
        max=(max(values) if values else 0.0),
    )


def _load_sample_ids(reader: Reader, dataset_name: str,
                     manifest_path: Optional[str], limit: Optional[int]) -> List[int]:
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if dataset_name not in manifest:
            raise ValueError(f"sample manifest has no entry for {dataset_name}")
        sample_ids = [int(value) for value in manifest[dataset_name]]
    else:
        sample_ids = list(range(len(reader)))
    if limit is not None:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError("the selected sample set is empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample manifest contains duplicate ids")
    invalid = [value for value in sample_ids if value < 0 or value >= len(reader)]
    if invalid:
        raise ValueError(f"sample ids out of range: {invalid[:10]}")
    return sample_ids


def _prompt_hash(reader: Reader, tokenizer: Tokenizer, comp_config: Config,
                 sample_ids: List[int]) -> str:
    system_prompt = reader.get_system_prompt()
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        rendered = tokenizer.bos_token + comp_config.template_cfg['complete'].format(
            system=system_prompt, question=reader.get_prompt(idx=sample_id)
        )
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(rendered.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@torch.no_grad()
def bench_dataset(
    model, tokenizer, reader: Reader, comp_config: Config,
    dataset_name: str, batch_size: int, max_new_tokens: int, max_prompt_len: int,
    device: str, dtype, bench_mode: str, use_EPL: bool, spec_decode: bool,
    repetition_penalty: float, repeats: int, warmup_batches: int,
    warmup_batch_size: int, warmup_max_new_tokens: int,
    sample_ids: List[int], output_dir: str, seed: int,
) -> Dict:
    """Measure one method/batch condition and retain lossless per-request data."""
    if repeats < 1 or batch_size < 1:
        raise ValueError("repeats and batch_size must be positive")
    vanilla = bench_mode == "vanilla_ar"
    system_prompt = reader.get_system_prompt()
    on_cuda = device == "cuda" and torch.cuda.is_available()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Warmup is a separate invocation and never removes samples from measurement.
    warm_ids = sample_ids[:min(warmup_batch_size, len(sample_ids))]
    for _ in range(warmup_batches):
        warm_questions = [reader.get_prompt(idx=value) for value in warm_ids]
        warm_stats = [MTPStats() for _ in warm_ids] if spec_decode else None
        batched_generate(
            model=model, tokenizer=tokenizer, comp_config=comp_config,
            system_prompt=system_prompt, questions=warm_questions,
            max_new_tokens=min(warmup_max_new_tokens, max_new_tokens),
            max_length=min(warmup_max_new_tokens, max_new_tokens) + max_prompt_len,
            device=device, dtype=dtype, update_attention_method="local",
            use_EPL=use_EPL, spec_decode=spec_decode,
            repetition_penalty=repetition_penalty, vanilla=vanilla,
            mtp_stats=warm_stats, return_details=True,
        )
    if on_cuda:
        torch.cuda.synchronize()

    repeat_reports = []
    repeat_records = []
    reference_outputs: Optional[Dict[int, List[int]]] = None
    repeat_output_mismatches = []
    chunks = [sample_ids[start:start + batch_size]
              for start in range(0, len(sample_ids), batch_size)]

    for repeat_index in range(repeats):
        random.seed(seed)
        torch.manual_seed(seed)
        records = []
        total_time = 0.0
        peak_allocated = 0
        peak_reserved = 0
        peak_total_valid = 0
        peak_physical = 0
        pbar = tqdm(chunks, desc=f"{dataset_name}/{bench_mode}/bs{batch_size}/r{repeat_index + 1}")
        for batch_index, chunk in enumerate(pbar):
            questions = [reader.get_prompt(idx=value) for value in chunk]
            mtp_stats = [MTPStats() for _ in chunk] if spec_decode else None
            if on_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            details = batched_generate(
                model=model, tokenizer=tokenizer, comp_config=comp_config,
                system_prompt=system_prompt, questions=questions,
                max_new_tokens=max_new_tokens,
                max_length=max_new_tokens + max_prompt_len,
                device=device, dtype=dtype, update_attention_method="local",
                use_EPL=use_EPL, spec_decode=spec_decode,
                repetition_penalty=repetition_penalty, vanilla=vanilla,
                mtp_stats=mtp_stats, return_details=True,
            )
            if on_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            total_time += elapsed
            if on_cuda:
                peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
            peak_total_valid = max(peak_total_valid, details["peak_total_valid_kv_tokens"])
            peak_physical = max(peak_physical, details["peak_physical_kv_slots"])

            for local_index, sample_id in enumerate(chunk):
                output = details["outputs"][local_index]
                output_ids = details["output_ids"][local_index]
                model_answer = reader.extract_answer(output)
                ground_truth = reader.get_answer(sample_id)
                correct, _ = reader.compare_answer(model_answer, ground_truth, sample_id)
                mtp_summary = mtp_stats[local_index].summary() if mtp_stats else None
                records.append(dict(
                    dataset=dataset_name,
                    bench_mode=bench_mode,
                    batch_size=batch_size,
                    actual_batch_size=len(chunk),
                    batch_index=batch_index,
                    repeat=repeat_index,
                    sample_id=sample_id,
                    output_ids=output_ids,
                    output_text=output,
                    output_length=len(output_ids),
                    extracted_answer=model_answer,
                    ground_truth=ground_truth,
                    correct=bool(correct),
                    finish_reason=details["finish_reasons"][local_index],
                    request_completion_s=details["request_completion_s"][local_index],
                    peak_valid_kv_tokens=details["peak_valid_kv_tokens"][local_index],
                    compression_steps=details["compression_steps"][local_index],
                    mtp_stats=mtp_summary,
                ))

        record_path = output_path / f"{bench_mode}.bs{batch_size}.{dataset_name}.repeat{repeat_index + 1}.jsonl"
        with jsonlines.open(str(record_path), "w") as writer:
            writer.write_all(records)

        outputs_by_id = {item["sample_id"]: item["output_ids"] for item in records}
        if reference_outputs is None:
            reference_outputs = outputs_by_id
            mismatch_count = 0
        else:
            mismatch_count = sum(
                outputs_by_id.get(sample_id) != reference_outputs.get(sample_id)
                for sample_id in sample_ids
            )
        repeat_output_mismatches.append(mismatch_count)
        repeat_records.append(str(record_path.resolve()))

        lengths = [item["output_length"] for item in records]
        completions = [item["request_completion_s"] for item in records]
        per_sample_kv = [item["peak_valid_kv_tokens"] for item in records]
        finish_counts: Dict[str, int] = {}
        for item in records:
            reason = item["finish_reason"]
            finish_counts[reason] = finish_counts.get(reason, 0) + 1
        total_tokens = sum(lengths)
        spec_proposed = sum((item["mtp_stats"] or {}).get("spec_proposed", 0) for item in records)
        spec_accepted = sum((item["mtp_stats"] or {}).get("spec_accepted", 0) for item in records)
        decode_steps = sum((item["mtp_stats"] or {}).get("decode_steps", 0) for item in records)
        repeat_reports.append(dict(
            repeat=repeat_index,
            num_samples=len(records),
            actual_batch_sizes=[len(chunk) for chunk in chunks],
            total_committed_tokens=total_tokens,
            tokens_per_sec=(total_tokens / total_time if total_time else 0.0),
            total_time_s=total_time,
            requests_per_sec=(len(records) / total_time if total_time else 0.0),
            accuracy=(sum(item["correct"] for item in records) / len(records)),
            output_length=_distribution(lengths),
            length_limit_fraction=(finish_counts.get("length_limit", 0) / len(records)),
            cache_limit_fraction=(finish_counts.get("cache_limit", 0) / len(records)),
            finish_reason_counts=finish_counts,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            peak_total_valid_kv_tokens=peak_total_valid,
            peak_physical_kv_slots=peak_physical,
            peak_valid_kv_tokens=_distribution(per_sample_kv),
            request_completion_s=dict(p50=_percentile(completions, 0.50),
                                      p95=_percentile(completions, 0.95)),
            spec_proposed=spec_proposed,
            spec_accepted=spec_accepted,
            spec_accept_rate=(spec_accepted / spec_proposed if spec_proposed else 0.0),
            committed_tokens_per_decode_step=(total_tokens / decode_steps if decode_steps else 0.0),
            sample_records_path=str(record_path.resolve()),
        ))

    def median_field(name: str) -> float:
        return statistics.median(float(item[name]) for item in repeat_reports)

    return dict(
        schema_version=2,
        dataset=dataset_name,
        bench_mode=bench_mode,
        batch_size=batch_size,
        num_samples=len(sample_ids),
        sample_ids=sample_ids,
        prompt_sha256=_prompt_hash(reader, tokenizer, comp_config, sample_ids),
        repeats=repeats,
        repeat_reports=repeat_reports,
        repeat_sample_records_paths=repeat_records,
        sample_records_path=repeat_records[0],
        repeat_output_mismatches=repeat_output_mismatches,
        tokens_per_sec=median_field("tokens_per_sec"),
        tokens_per_sec_range=[min(item["tokens_per_sec"] for item in repeat_reports),
                              max(item["tokens_per_sec"] for item in repeat_reports)],
        total_time_s=median_field("total_time_s"),
        total_time_s_range=[min(item["total_time_s"] for item in repeat_reports),
                            max(item["total_time_s"] for item in repeat_reports)],
        requests_per_sec=median_field("requests_per_sec"),
        accuracy=median_field("accuracy"),
        output_length=dict(
            mean=statistics.median(item["output_length"]["mean"] for item in repeat_reports),
            p50=statistics.median(item["output_length"]["p50"] for item in repeat_reports),
            p95=statistics.median(item["output_length"]["p95"] for item in repeat_reports),
            max=statistics.median(item["output_length"]["max"] for item in repeat_reports),
        ),
        length_limit_fraction=median_field("length_limit_fraction"),
        cache_limit_fraction=median_field("cache_limit_fraction"),
        finish_reason_counts=repeat_reports[0]["finish_reason_counts"],
        peak_allocated_bytes=max(item["peak_allocated_bytes"] for item in repeat_reports),
        peak_reserved_bytes=max(item["peak_reserved_bytes"] for item in repeat_reports),
        peak_total_valid_kv_tokens=max(item["peak_total_valid_kv_tokens"] for item in repeat_reports),
        peak_physical_kv_slots=max(item["peak_physical_kv_slots"] for item in repeat_reports),
        peak_valid_kv_tokens=dict(
            mean=statistics.median(item["peak_valid_kv_tokens"]["mean"] for item in repeat_reports),
            p50=statistics.median(item["peak_valid_kv_tokens"]["p50"] for item in repeat_reports),
            p95=statistics.median(item["peak_valid_kv_tokens"]["p95"] for item in repeat_reports),
            max=statistics.median(item["peak_valid_kv_tokens"]["max"] for item in repeat_reports),
        ),
        request_completion_s=dict(
            p50=statistics.median(item["request_completion_s"]["p50"] for item in repeat_reports),
            p95=statistics.median(item["request_completion_s"]["p95"] for item in repeat_reports),
        ),
        spec_accept_rate=median_field("spec_accept_rate"),
        committed_tokens_per_decode_step=median_field("committed_tokens_per_decode_step"),
    )


def _bench_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--model_type', type=str, default='qwen', choices=['qwen', 'llama'])
    p.add_argument('--tokenizer_path', type=str, default=None)
    p.add_argument('--compress_config', type=str, required=True)
    p.add_argument('--bos_token', type=str, default="<|im_start|>")
    p.add_argument('--eos_token', type=str, default="<|im_end|>")
    p.add_argument('--bench_mode', type=str, default='vanilla_ar',
                   choices=['vanilla_ar', 'memosight_ar', 'memosight_sd',
                            'vanilla', 'lightthinker', 'memosight'])
    p.add_argument('--spec_decode', type=str2bool, default=False)
    p.add_argument('--use_EPL', type=str2bool, default=False)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_new_tokens', type=int, default=10240)
    p.add_argument('--max_prompt_len', type=int, default=1100)
    p.add_argument('--warmup_batches', type=int, default=1)
    p.add_argument('--warmup_batch_size', type=int, default=None)
    p.add_argument('--warmup_max_new_tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--repetition_penalty', type=float, default=1.1)
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float32'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--sample_manifest', type=str, default=None)
    p.add_argument('--output_tag', type=str, default='outputs/batch_samples')
    p.add_argument('--datasets', type=str, nargs='+', default=['gsm8k'],
                   choices=['mmlu', 'gsm8k', 'gpqa', 'bbh', 'aime_2026', 'hmmt_feb_2025'])
    p.add_argument('--bench_report', type=str, default=None)
    # unused-but-accepted for get_model_and_tokenizer parity
    p.add_argument('--model_tag', type=str, default=None)
    p.add_argument('--ckpt', type=int, default=None)
    return p.parse_args()


def main():
    args = _bench_parser()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    aliases = {'vanilla': 'vanilla_ar', 'lightthinker': 'memosight_ar',
               'memosight': 'memosight_sd'}
    args.bench_mode = aliases.get(args.bench_mode, args.bench_mode)
    expected_spec = args.bench_mode == 'memosight_sd'
    if args.spec_decode != expected_spec:
        raise ValueError(
            f"bench_mode={args.bench_mode} requires spec_decode={str(expected_spec).lower()}"
        )
    if args.bench_mode.startswith('memosight') and not args.use_EPL:
        raise ValueError("formal MemoSight AR/SD benchmarking requires --use_EPL true")
    if args.bench_mode == 'vanilla_ar' and args.use_EPL:
        raise ValueError("Vanilla AR requires --use_EPL false")
    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "batch benchmark requires exactly one visible GPU; set CUDA_VISIBLE_DEVICES"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    args.vanilla = args.bench_mode == 'vanilla_ar'
    comp_config = Config.from_file(args.compress_config)
    if args.bench_mode == 'memosight_sd' and not comp_config.mtp_cfg:
        raise ValueError("MemoSight SD requires an mtp block in compress_config")
    model, tokenizer = get_model_and_tokenizer(args, comp_config)
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer.tokenizer):
        model.resize_token_embeddings(len(tokenizer.tokenizer))

    model.eval()
    all_tasks = {
        "mmlu": MMLUReader,
        "gsm8k": GSM8KReader,
        "gpqa": GPQAReader,
        "bbh": BBHReader,
        "aime_2026": AIME2026Reader,
        "hmmt_feb_2025": HMMTFeb2025Reader,
    }
    reports = []
    for name in args.datasets:
        reader = all_tasks[name]()
        sample_ids = _load_sample_ids(reader, name, args.sample_manifest, args.limit)
        rep = bench_dataset(
            model=model, tokenizer=tokenizer, reader=reader, comp_config=comp_config,
            dataset_name=name, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            max_prompt_len=args.max_prompt_len, device=device, dtype=dtype,
            bench_mode=args.bench_mode, use_EPL=args.use_EPL,
            spec_decode=args.spec_decode, repetition_penalty=args.repetition_penalty,
            repeats=args.repeats, warmup_batches=args.warmup_batches,
            warmup_batch_size=(args.warmup_batch_size or args.batch_size),
            warmup_max_new_tokens=args.warmup_max_new_tokens,
            sample_ids=sample_ids, output_dir=args.output_tag, seed=args.seed,
        )
        print(json.dumps(rep, ensure_ascii=False))
        reports.append(rep)
    if args.bench_report:
        report_path = Path(args.bench_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = reports[0] if len(reports) == 1 else dict(reports=reports)
        payload["run_config"] = dict(
            model_path=str(Path(args.model_path).resolve()),
            tokenizer_path=str(Path(args.tokenizer_path).resolve()),
            compress_config=str(Path(args.compress_config).resolve()),
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            max_prompt_len=args.max_prompt_len,
            repetition_penalty=args.repetition_penalty,
            use_EPL=args.use_EPL,
            spec_decode=args.spec_decode,
            seed=args.seed,
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[bench] report written to {args.bench_report}")


if __name__ == '__main__':
    main()
