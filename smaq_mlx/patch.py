"""Monkey-patch for mlx-lm — transparent SMAQ integration.

When applied, this patches:
1. `make_prompt_cache` → returns SMAQKVCache objects instead of KVCache
2. `scaled_dot_product_attention` → handles SMAQ cache objects correctly

Usage (auto — via patched mlx-lm __init__.py):
    # After running: python -m smaq.install
    SMAQ_ENABLED=1 python -m mlx_lm.generate --model ... --prompt "hello"

Usage (manual — in your own scripts):
    from smaq_mlx.patch import apply
    apply()
    # Then use mlx_lm normally — SMAQ caches are used automatically

Usage (per-call — create SMAQ caches yourself):
    from smaq_mlx.patch import make_smaq_prompt_cache
    caches = make_smaq_prompt_cache(model)
    mlx_lm.generate(model, tokenizer, "hello", prompt_cache=caches)
"""

import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models.cache as _cache
import mlx_lm.models.base as _base

from smaq_mlx.kv_cache import SMAQKVCache

# Preserve originals
_original_make_prompt_cache = _cache.make_prompt_cache
_original_sdpa = _base.scaled_dot_product_attention
_patched = False


def make_smaq_prompt_cache(
    model: nn.Module,
    key_bits: int = 4,
    value_bits: int = 4,
    Sigma_q: Optional[mx.array] = None,
    **kwargs,
):
    """Create SMAQ KV caches for all layers of the model.

    Drop-in replacement for mlx_lm.models.cache.make_prompt_cache().
    Returns a list of SMAQKVCache objects instead of KVCache objects.
    """
    # If the model has its own cache factory, defer to it
    # (some models like Mamba have non-KV caches)
    if hasattr(model, "make_cache"):
        original_caches = model.make_cache()
        # Only replace standard KV caches, leave special caches alone
        result = []
        for i, c in enumerate(original_caches):
            if isinstance(c, (_cache.KVCache, _cache.RotatingKVCache)):
                head_dim = _get_head_dim(model, i)
                result.append(SMAQKVCache(
                    head_dim=head_dim,
                    Sigma_q=Sigma_q,
                    key_bits=key_bits,
                    value_bits=value_bits,
                    layer_idx=i,
                ))
            else:
                result.append(c)
        return result

    num_layers = len(model.layers)
    head_dim = _get_head_dim(model, 0)

    return [
        SMAQKVCache(
            head_dim=head_dim,
            Sigma_q=Sigma_q,
            key_bits=key_bits,
            value_bits=value_bits,
            layer_idx=i,
        )
        for i in range(num_layers)
    ]


def _get_head_dim(model, layer_idx: int) -> int:
    """Extract head_dim from a model layer."""
    layer = model.layers[layer_idx]
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        if hasattr(attn, "head_dim"):
            return attn.head_dim
        # Fallback: derive from hidden_size / num_heads
        if hasattr(attn, "hidden_size") and hasattr(attn, "num_heads"):
            return attn.hidden_size // attn.num_heads
    # Default fallback
    return 128


def _patched_make_prompt_cache(model, max_kv_size=None, **kwargs):
    """Patched make_prompt_cache that uses SMAQ when enabled."""
    smaq_enabled = os.environ.get("SMAQ_ENABLED", "0") == "1"
    if not smaq_enabled:
        return _original_make_prompt_cache(model, max_kv_size=max_kv_size)

    key_bits = int(os.environ.get("SMAQ_KEY_BITS", "4"))
    value_bits = int(os.environ.get("SMAQ_VALUE_BITS", "4"))
    return make_smaq_prompt_cache(
        model,
        key_bits=key_bits,
        value_bits=value_bits,
    )


def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None, **kwargs):
    """Patched SDPA — routes SMAQ caches to standard SDPA on full-precision KVs.

    SMAQKVCache stores full-precision keys/values for exact attention
    while also tracking quantized copies for memory estimation. The SDPA
    just uses the full-precision path that mlx-lm normally would.
    """
    if isinstance(cache, SMAQKVCache):
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks,
        )
    return _original_sdpa(queries, keys, values, cache, scale, mask, sinks=sinks, **kwargs)


def apply():
    """Activate the SMAQ monkey-patches. Idempotent."""
    global _patched
    if _patched:
        return
    _cache.make_prompt_cache = _patched_make_prompt_cache
    _base.scaled_dot_product_attention = _patched_sdpa
    _patched = True
    # Announce once
    key_bits = os.environ.get("SMAQ_KEY_BITS", "4")
    value_bits = os.environ.get("SMAQ_VALUE_BITS", "4")
    print(f"[SMAQ] Patched mlx-lm — key_bits={key_bits}, value_bits={value_bits}")


def revert():
    """Remove all patches."""
    global _patched
    _cache.make_prompt_cache = _original_make_prompt_cache
    _base.scaled_dot_product_attention = _original_sdpa
    _patched = False
