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

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

import mlx.core as mx
try:
    import mlx.nn as nn
except ImportError:  # pragma: no cover - import guard for non-MLX test envs
    nn = None

try:
    import mlx_lm.models.cache as _cache
    import mlx_lm.models.base as _base
except ImportError:  # pragma: no cover - import guard for non-mlx_lm envs
    _cache = None
    _base = None

from smaq_mlx.layout import infer_model_layout_adapter
from smaq_mlx.kv_cache import SMAQKVCache

# Preserve originals
_original_make_prompt_cache = getattr(_cache, "make_prompt_cache", None)
_original_sdpa = getattr(_base, "scaled_dot_product_attention", None)
_patched = False
_runtime_config = {
    "enabled": None,
    "key_bits": None,
    "value_bits": None,
    "mode": None,
    "strict_benchmark": None,
    "require_true_compressed": None,
}


def _normalize_config(config: Any = None, **overrides) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if config is not None:
        if is_dataclass(config):
            data.update(asdict(config))
        elif isinstance(config, dict):
            data.update(config)
        else:
            for key in (
                "enabled",
                "key_bits",
                "value_bits",
                "mode",
                "strict_benchmark",
                "require_true_compressed",
            ):
                if hasattr(config, key):
                    data[key] = getattr(config, key)
    data.update({k: v for k, v in overrides.items() if v is not None})
    return data


def configure(config: Any = None, **overrides):
    """Set in-process runtime configuration for the patched mlx-lm hooks."""
    normalized = _normalize_config(config, **overrides)
    for key in _runtime_config:
        if key in normalized:
            _runtime_config[key] = normalized[key]


def clear_configuration():
    """Reset all in-process runtime configuration overrides."""
    for key in _runtime_config:
        _runtime_config[key] = None


def current_configuration() -> dict[str, Any]:
    """Return current in-process runtime configuration overrides."""
    return dict(_runtime_config)


def _resolve_bool(config_key: str, env_key: str, default: bool) -> bool:
    value = _runtime_config.get(config_key)
    if value is not None:
        return bool(value)
    return os.environ.get(env_key, "1" if default else "0") == "1"


def _resolve_int(config_key: str, env_key: str, default: int) -> int:
    value = _runtime_config.get(config_key)
    if value is not None:
        return int(value)
    return int(os.environ.get(env_key, str(default)))


def _resolve_str(config_key: str, env_key: str, default: str) -> str:
    value = _runtime_config.get(config_key)
    if value is not None:
        return str(value)
    return os.environ.get(env_key, default)


def make_smaq_prompt_cache(
    model: nn.Module,
    key_bits: int = 4,
    value_bits: int = 4,
    Sigma_q: Optional[mx.array] = None,
    mode: str = "hybrid",
    strict_benchmark: bool = False,
    layout_adapter=None,
    **kwargs,
):
    """Create SMAQ KV caches for all layers of the model.

    Drop-in replacement for mlx_lm.models.cache.make_prompt_cache().
    Returns a list of SMAQKVCache objects instead of KVCache objects.
    """
    layout_adapter = layout_adapter or infer_model_layout_adapter(model)

    if hasattr(model, "make_cache") and _cache is not None:
        original_caches = model.make_cache()
        result = []
        for i, c in enumerate(original_caches):
            if isinstance(c, (_cache.KVCache, _cache.RotatingKVCache)):
                head_dim = _get_head_dim(model, i)
                result.append(
                    SMAQKVCache(
                        head_dim=head_dim,
                        Sigma_q=Sigma_q,
                        key_bits=key_bits,
                        value_bits=value_bits,
                        layer_idx=i,
                        layout_adapter=layout_adapter,
                        mode=mode,
                        strict_benchmark=strict_benchmark,
                    )
                )
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
            layout_adapter=layout_adapter,
            mode=mode,
            strict_benchmark=strict_benchmark,
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
    smaq_enabled = _resolve_bool("enabled", "SMAQ_ENABLED", False)
    if not smaq_enabled:
        if _original_make_prompt_cache is None:
            raise RuntimeError("mlx-lm not installed in current environment")
        return _original_make_prompt_cache(model, max_kv_size=max_kv_size)

    key_bits = _resolve_int("key_bits", "SMAQ_KEY_BITS", 4)
    value_bits = _resolve_int("value_bits", "SMAQ_VALUE_BITS", 4)
    mode = _resolve_str("mode", "SMAQ_CACHE_MODE", "hybrid")
    strict = _resolve_bool("strict_benchmark", "SMAQ_BENCHMARK_STRICT", False)
    return make_smaq_prompt_cache(
        model,
        key_bits=key_bits,
        value_bits=value_bits,
        mode=mode,
        strict_benchmark=strict,
    )


def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None, **kwargs):
    """Patched SDPA — route SMAQ caches to compressed-history attention."""
    if isinstance(cache, SMAQKVCache):
        from smaq_mlx.attention_smaq import smaq_sdpa

        require_true = _resolve_bool(
            "require_true_compressed", "SMAQ_REQUIRE_TRUE_COMPRESSED", False
        )
        return smaq_sdpa(
            queries,
            cache,
            scale=scale,
            mask=mask,
            require_true_compressed=require_true,
        )
    if _original_sdpa is None:
        raise RuntimeError("mlx_lm not installed; cannot use original SDPA")
    return _original_sdpa(queries, keys, values, cache, scale, mask, sinks=sinks, **kwargs)


def apply(
    config: Any = None,
    *,
    enabled: Optional[bool] = None,
    key_bits: Optional[int] = None,
    value_bits: Optional[int] = None,
    mode: Optional[str] = None,
    strict_benchmark: Optional[bool] = None,
    require_true_compressed: Optional[bool] = None,
):
    """Activate the SMAQ monkey-patches. Idempotent."""
    global _patched
    configure(
        config,
        enabled=enabled,
        key_bits=key_bits,
        value_bits=value_bits,
        mode=mode,
        strict_benchmark=strict_benchmark,
        require_true_compressed=require_true_compressed,
    )
    if _patched:
        return
    if _cache is None or _base is None:
        raise RuntimeError("mlx-lm not installed in current environment")
    _cache.make_prompt_cache = _patched_make_prompt_cache
    _base.scaled_dot_product_attention = _patched_sdpa
    _patched = True
    # Announce once
    key_bits = _resolve_int("key_bits", "SMAQ_KEY_BITS", 4)
    value_bits = _resolve_int("value_bits", "SMAQ_VALUE_BITS", 4)
    print(f"[SMAQ] Patched mlx-lm — key_bits={key_bits}, value_bits={value_bits}")


def revert():
    """Remove all patches."""
    global _patched
    _cache.make_prompt_cache = _original_make_prompt_cache
    _base.scaled_dot_product_attention = _original_sdpa
    _patched = False
    clear_configuration()
