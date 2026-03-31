"""SMAQ attention scoring over compressed history plus exact recent tokens — MLX.

Ported from PyTorch to MLX for Apple Silicon execution.

The compressed path uses SMAQ's asymmetric score computation for keys and
standard value dequantization for the weighted sum.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from smaq.kv_cache import dequantize_values
from smaq.quantizer import SMAQQuantized, SMAQQuantizer
from smaq.store import CompressedKVStore, FlatCache

MIN_HISTORY_FOR_SMAQ = 16


def compute_hybrid_attention(
    query: mx.array,
    store: CompressedKVStore,
    recent_k: Optional[mx.array],
    recent_v: Optional[mx.array],
    num_query_heads: int,
    scale: Optional[float] = None,
) -> mx.array:
    """Compute attention output combining compressed history and exact recent KV."""
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_SMAQ
    has_recent = recent_k is not None and recent_k.size > 0

    if not has_history and not has_recent:
        return mx.zeros((query.shape[0], num_query_heads, head_dim))

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        hist_scores = _quantized_scores(query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale)
        hist_values = dequantize_values(
            flat.value_data, flat.value_scales, flat.value_zeros,
            flat.value_bits, store.value_group_size
        )
        weights = mx.softmax(hist_scores, axis=-1)
        return _apply_weights(weights, hist_values, gqa_ratio, num_kv_heads)

    if not has_history and has_recent:
        return _attend_exact_only(query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale)

    hist_scores = _quantized_scores(query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale)
    recent_scores = _exact_scores(query, recent_k, gqa_ratio, num_kv_heads, scale)
    logits = mx.concatenate([hist_scores, recent_scores], axis=-1)
    weights = mx.softmax(logits, axis=-1)

    hist_len = hist_scores.shape[-1]
    hist_values = dequantize_values(
        flat.value_data, flat.value_scales, flat.value_zeros,
        flat.value_bits, store.value_group_size
    )

    out_hist = _apply_weights(weights[..., :hist_len], hist_values, gqa_ratio, num_kv_heads)
    out_recent = _apply_weights(weights[..., hist_len:], recent_v, gqa_ratio, num_kv_heads)
    return out_hist + out_recent


def _quantized_scores(
    query: mx.array,
    flat: FlatCache,
    quantizer: SMAQQuantizer,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> mx.array:
    """Compute logits against SMAQ-compressed historical keys."""
    B, D = query.shape[0], query.shape[-1]
    q = query.reshape(B, num_kv_heads, gqa_ratio, D)

    all_scores = []
    for head_idx in range(num_kv_heads):
        key_q = SMAQQuantized(
            indices=flat.key_q.indices[:, head_idx:head_idx+1, :],
            norms=flat.key_q.norms[:, head_idx:head_idx+1],
            bits=flat.key_q.bits,
        )
        q_head = q[:, head_idx:head_idx+1, :, :]
        scores = quantizer.attention_score(q_head, key_q, scale=scale)
        all_scores.append(scores)

    return mx.concatenate(all_scores, axis=1).reshape(B, num_query_heads, flat.num_tokens)


def _exact_scores(
    query: mx.array,
    recent_k: mx.array,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> mx.array:
    """Compute exact logits against the recent ring buffer."""
    B, D = query.shape[0], query.shape[-1]
    q = query.reshape(B, num_kv_heads, gqa_ratio, D)
    k = recent_k.transpose(0, 2, 1)[None, :, :, :]

    scores = (q.astype(mx.float32) @ k.astype(mx.float32)) * scale
    return scores.reshape(B, num_query_heads, recent_k.shape[-2])


def _apply_weights(
    weights: mx.array,
    values: mx.array,
    gqa_ratio: int,
    num_kv_heads: int,
) -> mx.array:
    """Apply attention weights to either compressed-history or exact values."""
    B = weights.shape[0]
    v = values[None, :, :, :]
    w = weights.reshape(B, num_kv_heads, gqa_ratio, weights.shape[-1])
    out = (w.astype(mx.float32) @ v.astype(mx.float32))
    return out.reshape(B, num_kv_heads * gqa_ratio, values.shape[-1])


def _attend_exact_only(
    query: mx.array,
    recent_k: mx.array,
    recent_v: mx.array,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> mx.array:
    """Exact attention over the recent ring buffer only."""
    scores = _exact_scores(query, recent_k, gqa_ratio, num_kv_heads, scale)
    weights = mx.softmax(scores, axis=-1)
    return _apply_weights(weights, recent_v, gqa_ratio, num_kv_heads)
