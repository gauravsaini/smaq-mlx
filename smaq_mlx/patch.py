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

from smaq_mlx.backends import available_backends, dispatch_sdpa, make_prompt_cache_for_backend

# Preserve originals
_original_make_prompt_cache = getattr(_cache, "make_prompt_cache", None)
_original_sdpa = getattr(_base, "scaled_dot_product_attention", None)
_patched = False
_runtime_config = {
    "enabled": None,
    "backend": None,
    "key_bits": None,
    "value_bits": None,
    "mode": None,
    "strict_benchmark": None,
    "require_true_compressed": None,
    "polarquant_bits": None,
    "polarquant_key_seed": None,
    "polarquant_value_seed": None,
    "turboquant_bits": None,
    "turboquant_seed": None,
    "turboquant_fused": None,
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
                "backend",
                "key_bits",
                "value_bits",
                "mode",
                "strict_benchmark",
                "require_true_compressed",
                "polarquant_bits",
                "polarquant_key_seed",
                "polarquant_value_seed",
                "turboquant_bits",
                "turboquant_seed",
                "turboquant_fused",
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


def _resolve_float(config_key: str, env_key: str, default: float) -> float:
    value = _runtime_config.get(config_key)
    if value is not None:
        return float(value)
    return float(os.environ.get(env_key, str(default)))


def _resolve_str(config_key: str, env_key: str, default: str) -> str:
    value = _runtime_config.get(config_key)
    if value is not None:
        return str(value)
    return os.environ.get(env_key, default)


def make_prompt_cache(
    model: nn.Module,
    backend: str = "smaq",
    key_bits: int = 4,
    value_bits: int = 4,
    Sigma_q: Optional[mx.array] = None,
    mode: str = "hybrid",
    strict_benchmark: bool = False,
    layout_adapter=None,
    **kwargs,
):
    """Create backend-managed prompt caches for all compatible layers."""
    if _cache is None:
        raise RuntimeError("mlx-lm not installed in current environment")
    config = {
        "backend": backend,
        "key_bits": key_bits,
        "value_bits": value_bits,
        "Sigma_q": Sigma_q,
        "mode": mode,
        "strict_benchmark": strict_benchmark,
        "layout_adapter": layout_adapter,
        **kwargs,
    }
    return make_prompt_cache_for_backend(
        backend,
        model,
        cache_module=_cache,
        config=config,
    )


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
    """Backward-compatible SMAQ cache helper."""
    return make_prompt_cache(
        model,
        backend="smaq",
        key_bits=key_bits,
        value_bits=value_bits,
        Sigma_q=Sigma_q,
        mode=mode,
        strict_benchmark=strict_benchmark,
        layout_adapter=layout_adapter,
        **kwargs,
    )


def _patched_make_prompt_cache(model, max_kv_size=None, **kwargs):
    """Patched make_prompt_cache that uses SMAQ when enabled."""
    smaq_enabled = _resolve_bool("enabled", "SMAQ_ENABLED", False)
    if not smaq_enabled:
        if _original_make_prompt_cache is None:
            raise RuntimeError("mlx-lm not installed in current environment")
        return _original_make_prompt_cache(model, max_kv_size=max_kv_size)

    backend = _resolve_str("backend", "SMAQ_BACKEND", "smaq")
    key_bits = _resolve_int("key_bits", "SMAQ_KEY_BITS", 4)
    value_bits = _resolve_int("value_bits", "SMAQ_VALUE_BITS", 4)
    mode = _resolve_str("mode", "SMAQ_CACHE_MODE", "hybrid")
    strict = _resolve_bool("strict_benchmark", "SMAQ_BENCHMARK_STRICT", False)
    polarquant_bits = _resolve_float("polarquant_bits", "POLARQUANT_BITS", 3.0)
    polarquant_key_seed = _resolve_int("polarquant_key_seed", "POLARQUANT_KEY_SEED", 42)
    polarquant_value_seed = _resolve_int("polarquant_value_seed", "POLARQUANT_VALUE_SEED", 43)
    turboquant_bits = _resolve_int("turboquant_bits", "TURBOQUANT_BITS", 3)
    turboquant_seed = _resolve_int("turboquant_seed", "TURBOQUANT_SEED", 42)
    turboquant_fused = _resolve_bool("turboquant_fused", "TURBOQUANT_FUSED", True)
    return make_prompt_cache(
        model,
        backend=backend,
        key_bits=key_bits,
        value_bits=value_bits,
        mode=mode,
        strict_benchmark=strict,
        polarquant_bits=polarquant_bits,
        polarquant_key_seed=polarquant_key_seed,
        polarquant_value_seed=polarquant_value_seed,
        turboquant_bits=turboquant_bits,
        turboquant_seed=turboquant_seed,
        turboquant_fused=turboquant_fused,
    )


def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None, **kwargs):
    """Patched SDPA — route supported backend caches to their runtime backend."""
    dispatch_config = {
        "backend": _resolve_str("backend", "SMAQ_BACKEND", "smaq"),
        "require_true_compressed": _resolve_bool(
            "require_true_compressed", "SMAQ_REQUIRE_TRUE_COMPRESSED", False
        ),
        "key_bits": _resolve_int("key_bits", "SMAQ_KEY_BITS", 4),
        "value_bits": _resolve_int("value_bits", "SMAQ_VALUE_BITS", 4),
        "polarquant_bits": _resolve_float("polarquant_bits", "POLARQUANT_BITS", 3.0),
        "polarquant_key_seed": _resolve_int("polarquant_key_seed", "POLARQUANT_KEY_SEED", 42),
        "polarquant_value_seed": _resolve_int("polarquant_value_seed", "POLARQUANT_VALUE_SEED", 43),
        "turboquant_bits": _resolve_int("turboquant_bits", "TURBOQUANT_BITS", 3),
        "turboquant_seed": _resolve_int("turboquant_seed", "TURBOQUANT_SEED", 42),
        "turboquant_fused": _resolve_bool("turboquant_fused", "TURBOQUANT_FUSED", True),
    }
    dispatched = dispatch_sdpa(
        queries,
        keys,
        values,
        cache,
        scale=scale,
        mask=mask,
        sinks=sinks,
        config=dispatch_config,
        original_sdpa=_original_sdpa,
        **kwargs,
    )
    if dispatched is not None:
        return dispatched
    if _original_sdpa is None:
        raise RuntimeError("mlx_lm not installed; cannot use original SDPA")
    return _original_sdpa(queries, keys, values, cache, scale, mask, sinks=sinks, **kwargs)


def apply(
    config: Any = None,
    *,
    enabled: Optional[bool] = None,
    backend: Optional[str] = None,
    key_bits: Optional[int] = None,
    value_bits: Optional[int] = None,
    mode: Optional[str] = None,
    strict_benchmark: Optional[bool] = None,
    require_true_compressed: Optional[bool] = None,
    polarquant_bits: Optional[int] = None,
    polarquant_key_seed: Optional[int] = None,
    polarquant_value_seed: Optional[int] = None,
    turboquant_bits: Optional[int] = None,
    turboquant_seed: Optional[int] = None,
    turboquant_fused: Optional[bool] = None,
):
    """Activate the SMAQ monkey-patches. Idempotent."""
    global _patched
    configure(
        config,
        enabled=enabled,
        backend=backend,
        key_bits=key_bits,
        value_bits=value_bits,
        mode=mode,
        strict_benchmark=strict_benchmark,
        require_true_compressed=require_true_compressed,
        polarquant_bits=polarquant_bits,
        polarquant_key_seed=polarquant_key_seed,
        polarquant_value_seed=polarquant_value_seed,
        turboquant_bits=turboquant_bits,
        turboquant_seed=turboquant_seed,
        turboquant_fused=turboquant_fused,
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
    backend = _resolve_str("backend", "SMAQ_BACKEND", "smaq")
    print(
        f"[SMAQ] Patched mlx-lm — backend={backend}, "
        f"available_backends={','.join(available_backends())}, "
        f"key_bits={key_bits}, value_bits={value_bits}"
    )


def revert():
    """Remove all patches."""
    global _patched
    _cache.make_prompt_cache = _original_make_prompt_cache
    _base.scaled_dot_product_attention = _original_sdpa
    _patched = False
    clear_configuration()
