"""Monkey-patch for mlx-lm's SDPA dispatch — SMAQ integration.

Supports SMAQ KV cache by intercepting the SDPA call and routing to
SMAQ attention when the cache is an SMAQKVCache instance.

Usage:
    from smaq.patch import apply as smaq_patch_apply
    smaq_patch_apply()

    # Then load your model normally — SMAQ caches will be used automatically
"""

import mlx.core as mx
import mlx_lm.models.base as _base

from smaq.kv_cache import SMAQKVCache
from smaq.attention_smaq import smaq_sdpa

_original_sdpa = _base.scaled_dot_product_attention
_patched = False


def _patched_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
    """Patched SDPA that routes SMAQ caches to SMAQ attention."""
    if isinstance(cache, SMAQKVCache):
        return smaq_sdpa(queries, cache, scale, mask)
    return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)


def apply():
    """Activate the SMAQ SDPA patch. Idempotent."""
    global _patched
    if _patched:
        return
    _base.scaled_dot_product_attention = _patched_sdpa
    _patched = True


def revert():
    """Remove the patch."""
    global _patched
    _base.scaled_dot_product_attention = _original_sdpa
    _patched = False
