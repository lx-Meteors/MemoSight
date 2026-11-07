"""Training-free R-KV cache compression for the local Qwen3 inference path.

The scoring code follows the official HuggingFace implementation in
Zefan-Cai/R-KV.  This adapter keeps compression outside the model forward so
that MemoSight's existing model implementation and checkpoints remain usable.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from transformers import DynamicCache


def compute_attention_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    pooling: str = "max",
) -> torch.Tensor:
    """Compute GQA-aware query/key scores as in the official R-KV code."""
    batch_size, query_heads, query_length, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    if query_heads % kv_heads != 0:
        raise ValueError(
            f"query heads ({query_heads}) must be divisible by KV heads ({kv_heads})"
        )

    query_group_size = query_heads // kv_heads
    if query_group_size == 1:
        return torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    grouped_queries = query_states.reshape(
        batch_size,
        kv_heads,
        query_group_size,
        query_length,
        head_dim,
    )
    grouped_keys = key_states.unsqueeze(2)
    scores = torch.matmul(grouped_queries, grouped_keys.transpose(3, 4)) / math.sqrt(head_dim)
    if pooling == "max":
        return scores.max(dim=2).values
    if pooling == "mean":
        return scores.mean(dim=2)
    raise ValueError(f"Unsupported GQA pooling method: {pooling}")


def calculate_similarity(
    key_states: torch.Tensor,
    threshold: float = 0.5,
    retain_ratio: float = 0.2,
    retain_direction: str = "last",
) -> torch.Tensor:
    """Return the official redundancy score for every KV head and token."""
    _, _, sequence_length, _ = key_states.shape
    normalized_keys = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    similarity = torch.matmul(normalized_keys, normalized_keys.transpose(-1, -2))
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=key_states.device)
    similarity.masked_fill_(diagonal.view(1, 1, sequence_length, sequence_length), 0.0)
    similarity_mask = similarity > threshold

    retain_count = max(1, int(sequence_length * retain_ratio))
    positions = torch.arange(sequence_length, device=key_states.device).view(1, 1, 1, -1)
    indices = torch.where(similarity_mask, positions, torch.zeros_like(similarity_mask, dtype=torch.long))

    if retain_direction == "last":
        retained_neighbor = indices.max(dim=-1).values
    elif retain_direction == "first":
        # This intentionally mirrors the released implementation.  Its zero
        # sentinel means rows without an earlier match retain position zero.
        retained_neighbor = indices.min(dim=-1).values
    elif retain_direction == "last_percent":
        retained_neighbor = torch.topk(indices, k=retain_count, dim=-1).values[:, :, :, 0]
    elif retain_direction == "first_percent":
        retained_neighbor = torch.topk(
            indices, k=retain_count, dim=-1, largest=False
        ).values[:, :, :, -1]
    else:
        raise ValueError(f"Unsupported retain direction: {retain_direction}")

    similarity.scatter_(-1, retained_neighbor.unsqueeze(-1), 0)
    return similarity.mean(dim=-2).softmax(dim=-1)


@dataclass(frozen=True)
class RKVConfig:
    budget: int = 1536
    window_size: int = 8
    kernel_size: int = 7
    mix_lambda: float = 0.1
    retain_ratio: float = 0.2
    retain_direction: str = "last"
    similarity_threshold: float = 0.5
    buffer_size: int = 128

    def validate(self) -> None:
        if self.budget <= self.window_size:
            raise ValueError("R-KV budget must be greater than its observation window")
        if self.window_size <= 0 or self.buffer_size <= 0 or self.kernel_size <= 0:
            raise ValueError("R-KV window, buffer, and pooling-kernel sizes must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("R-KV pooling-kernel size must be odd")
        if not 0.0 <= self.mix_lambda <= 1.0:
            raise ValueError("R-KV mix lambda must be in [0, 1]")
        if not 0.0 < self.retain_ratio <= 1.0:
            raise ValueError("R-KV retain ratio must be in (0, 1]")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("R-KV similarity threshold must be in [-1, 1]")
        if self.retain_direction not in {"last", "first"}:
            raise ValueError("R-KV retain direction must be 'last' or 'first'")


class RKVCacheCompressor:
    """Per-sample R-KV state and DynamicCache compaction coordinator."""

    def __init__(self, config: RKVConfig):
        config.validate()
        self.config = config
        self.query_cache: Dict[int, torch.Tensor] = {}
        self.buffered_tokens = 0
        self.compression_count = 0
        self.evicted_tokens = 0
        self.peak_cache_tokens = 0

    @staticmethod
    def _attention_layers(model) -> Iterable:
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise TypeError("R-KV adapter currently requires the local Qwen3 model layout")
        return layers

    def enable_query_capture(self, model) -> None:
        for layer in self._attention_layers(model):
            layer.self_attn.rkv_capture_queries = True
            layer.self_attn.rkv_query_window_size = self.config.window_size

    def observe_queries(self, model) -> None:
        """Collect the last alpha RoPE-applied queries from the completed forward."""
        for layer_index, layer in enumerate(self._attention_layers(model)):
            attention = layer.self_attn
            queries = getattr(attention, "rkv_last_query_states", None)
            if queries is None:
                raise RuntimeError(
                    "R-KV query capture is empty; use the local Qwen3 SDPA/eager attention implementation"
                )
            queries = queries.detach()
            previous = self.query_cache.get(layer_index)
            if previous is not None:
                queries = torch.cat((previous, queries), dim=2)
            self.query_cache[layer_index] = queries[:, :, -self.config.window_size :, :]
            attention.rkv_last_query_states = None

    def _select_layer(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
    ):
        cache_length = key_states.shape[-2]
        window_size = min(self.config.window_size, cache_length)
        if cache_length <= self.config.budget:
            return key_states, value_states

        candidate_length = cache_length - window_size
        keep_from_candidates = self.config.budget - window_size
        if keep_from_candidates > candidate_length:
            return key_states, value_states

        attention_logits = compute_attention_scores(query_states, key_states)
        importance = torch.softmax(
            attention_logits[:, :, -window_size:, :candidate_length],
            dim=-1,
            dtype=torch.float32,
        ).mean(dim=-2).to(query_states.dtype)
        importance = F.max_pool1d(
            importance,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size // 2,
            stride=1,
        )

        redundancy = calculate_similarity(
            key_states,
            threshold=self.config.similarity_threshold,
            retain_ratio=self.config.retain_ratio,
            retain_direction=self.config.retain_direction,
        )[:, :, :candidate_length]
        final_score = (
            importance * self.config.mix_lambda
            - redundancy * (1.0 - self.config.mix_lambda)
        )
        indices = final_score.topk(keep_from_candidates, dim=-1).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1])
        retained_keys = key_states[:, :, :candidate_length, :].gather(2, gather_index)
        retained_values = value_states[:, :, :candidate_length, :].gather(2, gather_index)
        return (
            torch.cat((retained_keys, key_states[:, :, -window_size:, :]), dim=2),
            torch.cat((retained_values, value_states[:, :, -window_size:, :]), dim=2),
        )

    @torch.no_grad()
    def after_decode_forward(self, model, cache: DynamicCache) -> bool:
        """Observe one decoded token and compact at each fixed buffer boundary."""
        self.observe_queries(model)
        cache_length = cache.get_seq_length()
        self.peak_cache_tokens = max(self.peak_cache_tokens, cache_length)
        self.buffered_tokens += 1
        if self.buffered_tokens < self.config.buffer_size:
            return False
        self.buffered_tokens = 0
        if cache_length <= self.config.budget:
            return False

        original_length = cache_length
        if len(cache.key_cache) != len(self.query_cache):
            raise RuntimeError(
                f"R-KV layer mismatch: {len(cache.key_cache)} KV layers vs "
                f"{len(self.query_cache)} query layers"
            )
        for layer_index in range(len(cache.key_cache)):
            keys, values = self._select_layer(
                cache.key_cache[layer_index],
                cache.value_cache[layer_index],
                self.query_cache[layer_index],
            )
            cache.key_cache[layer_index] = keys
            cache.value_cache[layer_index] = values

        compacted_length = cache.key_cache[0].shape[-2]
        if compacted_length != self.config.budget:
            raise RuntimeError(
                f"R-KV produced {compacted_length} tokens, expected {self.config.budget}"
            )
        # DynamicCache uses this counter as its physical append offset.  The
        # caller separately passes the original logical cache_position so RoPE
        # positions do not jump backwards after an eviction.
        cache._seen_tokens = compacted_length
        self.compression_count += 1
        self.evicted_tokens += original_length - compacted_length
        return True

    def summary(self) -> dict:
        return {
            "budget": self.config.budget,
            "buffer_size": self.config.buffer_size,
            "window_size": self.config.window_size,
            "kernel_size": self.config.kernel_size,
            "mix_lambda": self.config.mix_lambda,
            "retain_ratio": self.config.retain_ratio,
            "retain_direction": self.config.retain_direction,
            "similarity_threshold": self.config.similarity_threshold,
            "compression_count": self.compression_count,
            "evicted_tokens": self.evicted_tokens,
            "peak_kv_tokens": self.peak_cache_tokens,
        }
