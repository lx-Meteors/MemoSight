"""LazyEviction adapter for the local Qwen3 DynamicCache decode path.

The recurrence tracking and MRI-centric score follow the authors' released
implementation (Halo-949/LazyEviction, commit 051d663).  The official Qwen2
patch expands GQA KV heads to all query heads before caching.  This adapter
keeps Qwen3's native GQA cache layout and marks a KV token important when any
query head in its GQA group crosses ``alpha``.  This preserves the algorithm
while avoiding an artificial query-head-sized KV cache.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from transformers import DynamicCache


@dataclass(frozen=True)
class LazyEvictionConfig:
    budget: int = 4915
    window_size: int = 363
    alpha: float = 0.0001

    def validate(self) -> None:
        if self.budget <= self.window_size:
            raise ValueError("LazyEviction budget must be greater than window_size")
        if self.window_size <= 0:
            raise ValueError("LazyEviction window_size must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("LazyEviction alpha must be in [0, 1]")


class LazyEvictionCompressor:
    """Per-sample recurrence state and lagged KV eviction coordinator."""

    def __init__(self, config: LazyEvictionConfig):
        config.validate()
        self.config = config
        self.last_recurrent: Dict[int, torch.Tensor] = {}
        self.max_recurrence_interval: Dict[int, torch.Tensor] = {}
        self.decode_step = 0
        self.compression_count = 0
        self.evicted_tokens = 0
        self.peak_cache_tokens = 0

    @staticmethod
    def _attention_layers(model) -> Iterable:
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise TypeError("LazyEviction requires the local Qwen3 model layout")
        return layers

    def enable_query_capture(self, model) -> None:
        # The local attention implementation already exposes the RoPE-applied
        # queries for R-KV. LazyEviction uses the same read-only capture hook.
        for layer in self._attention_layers(model):
            layer.self_attn.rkv_capture_queries = True
            layer.self_attn.rkv_query_window_size = 1

    @staticmethod
    def _grouped_attention_weights(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-KV-head attention, retaining importance from any GQA head."""
        batch_size, query_heads, query_length, head_dim = query_states.shape
        kv_heads = key_states.shape[1]
        if query_heads % kv_heads != 0:
            raise ValueError(
                f"query heads ({query_heads}) must be divisible by KV heads ({kv_heads})"
            )
        group_size = query_heads // kv_heads
        grouped_queries = query_states.reshape(
            batch_size,
            kv_heads,
            group_size,
            query_length,
            head_dim,
        )
        logits = torch.matmul(
            grouped_queries,
            key_states.unsqueeze(2).transpose(-1, -2),
        ) / math.sqrt(head_dim)
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        return weights.max(dim=2).values

    def _record_recurrence(
        self,
        layer_index: int,
        weights: torch.Tensor,
    ) -> None:
        batch_size, kv_heads, _, cache_length = weights.shape
        previous_last = self.last_recurrent.get(layer_index)
        previous_mri = self.max_recurrence_interval.get(layer_index)

        if previous_last is None:
            last = torch.zeros(
                (batch_size, kv_heads, cache_length),
                dtype=torch.int32,
                device=weights.device,
            )
            mri = torch.zeros_like(last)
        else:
            previous_length = previous_last.shape[-1]
            if previous_length > cache_length:
                raise RuntimeError(
                    f"LazyEviction state is longer than layer {layer_index} cache: "
                    f"{previous_length} > {cache_length}"
                )
            new_tokens = cache_length - previous_length
            if new_tokens:
                last_padding = torch.full(
                    (batch_size, kv_heads, new_tokens),
                    self.decode_step - 1,
                    dtype=torch.int32,
                    device=weights.device,
                )
                mri_padding = torch.zeros_like(last_padding)
                last = torch.cat((previous_last, last_padding), dim=-1)
                mri = torch.cat((previous_mri, mri_padding), dim=-1)
            else:
                last = previous_last
                mri = previous_mri

        old_last = last.clone()
        important = weights.squeeze(2) >= self.config.alpha
        last[important] = self.decode_step
        mri = torch.maximum(mri, last - old_last)
        self.last_recurrent[layer_index] = last
        self.max_recurrence_interval[layer_index] = mri

    def _select_layer(
        self,
        layer_index: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_length = key_states.shape[-2]
        window_size = self.config.window_size
        candidate_length = cache_length - window_size
        keep_from_candidates = self.config.budget - window_size
        if candidate_length <= keep_from_candidates:
            return key_states, value_states

        last = self.last_recurrent[layer_index]
        mri = self.max_recurrence_interval[layer_index]
        delta = self.decode_step - last

        # Match the released implementation's two sigmoid terms exactly.
        h1 = 2.0 * torch.sigmoid(
            -(1.0 + delta.to(torch.float32) / (mri.to(torch.float32) + 1e-5))
        )
        h2 = torch.where(
            mri == 0,
            torch.zeros_like(h1),
            2.0 * torch.sigmoid(1.0 - mri.to(torch.float32)),
        )
        importance = h1 + h2
        candidate_scores = importance[:, :, :candidate_length]
        top_indices = candidate_scores.topk(keep_from_candidates, dim=-1).indices

        gather_index = top_indices.unsqueeze(-1).expand(
            -1, -1, -1, key_states.shape[-1]
        )
        retained_keys = key_states[:, :, :candidate_length, :].gather(2, gather_index)
        retained_values = value_states[:, :, :candidate_length, :].gather(2, gather_index)
        recent_keys = key_states[:, :, -window_size:, :]
        recent_values = value_states[:, :, -window_size:, :]

        retained_last = last[:, :, :candidate_length].gather(2, top_indices)
        retained_mri = mri[:, :, :candidate_length].gather(2, top_indices)
        self.last_recurrent[layer_index] = torch.cat(
            (retained_last, last[:, :, -window_size:]), dim=-1
        )
        self.max_recurrence_interval[layer_index] = torch.cat(
            (retained_mri, mri[:, :, -window_size:]), dim=-1
        )
        return (
            torch.cat((retained_keys, recent_keys), dim=2),
            torch.cat((retained_values, recent_values), dim=2),
        )

    @torch.no_grad()
    def after_decode_forward(self, model, cache: DynamicCache) -> bool:
        self.decode_step += 1
        cache_length = cache.get_seq_length()
        self.peak_cache_tokens = max(self.peak_cache_tokens, cache_length)

        layers = list(self._attention_layers(model))
        if len(cache.key_cache) != len(layers):
            raise RuntimeError(
                f"LazyEviction layer mismatch: {len(cache.key_cache)} KV layers "
                f"vs {len(layers)} model layers"
            )
        for layer_index, layer in enumerate(layers):
            attention = layer.self_attn
            queries = getattr(attention, "rkv_last_query_states", None)
            if queries is None:
                raise RuntimeError(
                    "LazyEviction query capture is empty; use the local Qwen3 SDPA/eager path"
                )
            attention.rkv_last_query_states = None
            weights = self._grouped_attention_weights(
                queries.detach(), cache.key_cache[layer_index]
            )
            self._record_recurrence(layer_index, weights)

        if cache_length <= self.config.budget:
            return False
        if self.decode_step % self.config.window_size != 0:
            return False

        original_length = cache_length
        for layer_index in range(len(cache.key_cache)):
            keys, values = self._select_layer(
                layer_index,
                cache.key_cache[layer_index],
                cache.value_cache[layer_index],
            )
            cache.key_cache[layer_index] = keys
            cache.value_cache[layer_index] = values

        compacted_length = cache.key_cache[0].shape[-2]
        if compacted_length != self.config.budget:
            raise RuntimeError(
                f"LazyEviction produced {compacted_length} tokens, "
                f"expected {self.config.budget}"
            )
        cache._seen_tokens = compacted_length
        self.compression_count += 1
        self.evicted_tokens += original_length - compacted_length
        return True

    def summary(self) -> dict:
        return {
            "budget": self.config.budget,
            "window_size": self.config.window_size,
            "alpha": self.config.alpha,
            "decode_steps": self.decode_step,
            "compression_count": self.compression_count,
            "evicted_tokens": self.evicted_tokens,
            "peak_kv_tokens": self.peak_cache_tokens,
            "gqa_pooling": "max_attention_per_kv_group",
        }
