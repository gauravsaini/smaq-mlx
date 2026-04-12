"""SMAQ attention for MLX — SDPA with SMAQ-compressed KV cache.

Ported from PyTorch to MLX for Apple Silicon execution.

Uses MLX native operations for all computation. The SMAQ metric is applied
during key encoding, and attention scores are computed using the shaped
metric without materializing full dequantized keys.
"""

import mlx.core as mx


def smaq_sdpa(
    queries: mx.array,
    cache,
    scale: float,
    mask=None,
    require_true_compressed: bool = False,
) -> mx.array:
    """SMAQ SDPA over compressed history + exact recent tail."""
    cap = getattr(cache, "capabilities", None)
    used_compressed = getattr(cache, "key_quantized", None) is not None
    if require_true_compressed and cap is not None:
        if not used_compressed or not cap.compressed_history or cap.compressed_history_shadow_only:
            raise RuntimeError(
                f"True compressed mode requested, but cache is shadow-only: {getattr(cache, 'report', lambda: {})()}"
            )

    scores = cache.attention_scores(queries, scale=scale)
    if scores.shape[-1] == 0:
        return mx.zeros((*queries.shape[:-1], queries.shape[-1]))

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
    return cache.attend(weights)
