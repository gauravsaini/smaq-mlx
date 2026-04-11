"""SMAQ attention for MLX — SDPA with SMAQ-compressed KV cache.

Ported from PyTorch to MLX for Apple Silicon execution.

Uses MLX native operations for all computation. The SMAQ metric is applied
during key encoding, and attention scores are computed using the shaped
metric without materializing full dequantized keys.
"""

import math
from typing import Optional

import mlx.core as mx

from smaq_mlx.quantizer import SMAQQuantized, SMAQQuantizer
from smaq_mlx.kv_cache import dequantize_values


def smaq_sdpa(
    queries: mx.array,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    """SMAQ SDPA with compressed key attention.

    Args:
        queries: (B, n_q_heads, T_q, D) query tensor
        cache: SMAQKVCache with compressed keys
        scale: attention scale factor
        mask: optional attention mask

    Returns:
        output: (B, n_q_heads, T_q, D) attention output
    """
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = cache.key_quantized.indices.shape[1] if cache.key_quantized else 0
    n_repeats = n_q_heads // n_kv_heads if n_kv_heads > 0 else 1

    scores_parts = []
    values_parts = []

    # Compressed history scores
    if cache.key_quantized is not None:
        q = queries.reshape(B, n_kv_heads, n_repeats, T_q, D)
        k_indices = cache.key_quantized.indices
        k_norms = cache.key_quantized.norms

        # Dequantize keys for attention
        k_hat = _dequantize_keys_from_cache(cache)
        k_expanded = k_hat[:, None, :, :, :]

        scores = (q.astype(mx.float32) @ k_expanded.astype(mx.float32).transpose(0, 1, 2, 4, 3)) * scale
        scores = scores.reshape(B, n_q_heads, T_q, k_hat.shape[-2])
        scores_parts.append(scores)

    # Exact buffer scores
    if cache.key_buffer is not None:
        q = queries.reshape(B, n_kv_heads, n_repeats, T_q, D)
        k_buf = cache.key_buffer[:, None, :, :, :]
        scores = (q.astype(mx.float32) @ k_buf.astype(mx.float32).transpose(0, 1, 2, 4, 3)) * scale
        scores = scores.reshape(B, n_q_heads, T_q, cache.key_buffer.shape[-2])
        scores_parts.append(scores)

    if not scores_parts:
        return mx.zeros((B, n_q_heads, T_q, D))

    scores = mx.concatenate(scores_parts, axis=-1)

    # Apply mask
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    # Softmax
    weights = mx.softmax(scores, axis=-1, precise=True)

    # Value output
    output_parts = []
    col_offset = 0

    if cache.value_data is not None:
        n_quant = cache.value_data.shape[-2]
        w_quant = weights[..., col_offset: col_offset + n_quant]
        v_dequant = dequantize_values(
            cache.value_data, cache.value_scales, cache.value_zeros,
            cache.value_bits, cache.value_group_size
        )
        v_expanded = v_dequant[:, None, :, :]
        out = (w_quant.astype(mx.float32) @ v_expanded.astype(mx.float32))
        output_parts.append(out.reshape(B, n_q_heads, T_q, D))
        col_offset += n_quant

    if cache.value_buffer is not None:
        n_buf = cache.value_buffer.shape[-2]
        w_buf = weights[..., col_offset: col_offset + n_buf]
        v_buf = cache.value_buffer[:, None, :, :]
        out = (w_buf.astype(mx.float32) @ v_buf.astype(mx.float32))
        output_parts.append(out.reshape(B, n_q_heads, T_q, D))

    if not output_parts:
        return mx.zeros((B, n_q_heads, T_q, D))

    return sum(output_parts)


def _dequantize_keys_from_cache(cache) -> mx.array:
    """Dequantize keys from SMAQ cache for attention computation."""
    return cache.key_quantizer.dequantize(cache.key_quantized)
