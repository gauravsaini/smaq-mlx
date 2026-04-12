"""Public API for production-style MLX KV runtime usage."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import mlx_lm

from smaq_mlx.backends import available_backends, backend_matrix
from smaq_mlx.patch import (
    apply,
    clear_configuration,
    configure,
    current_configuration,
    make_prompt_cache,
    make_smaq_prompt_cache,
    revert,
)


@dataclass
class MLXRuntimeConfig:
    """User-facing runtime configuration for MLX KV runtime backends."""

    enabled: bool = True
    backend: str = "smaq"
    key_bits: int = 4
    value_bits: int = 4
    mode: str = "hybrid"
    strict_benchmark: bool = False
    require_true_compressed: bool = False
    polarquant_bits: Optional[float] = None
    polarquant_key_seed: int = 42
    polarquant_value_seed: int = 43
    rotorquant_bits: Optional[int] = None
    rotorquant_key_seed: int = 42
    rotorquant_value_seed: int = 43
    folded_turbo_bits: Optional[int] = None
    folded_turbo_key_seed: int = 42
    folded_turbo_value_seed: int = 43
    folded_smaq_c: float = 5.0
    turboquant_bits: Optional[int] = None
    turboquant_seed: int = 42
    turboquant_fused: bool = True


SMAQConfig = MLXRuntimeConfig


def enable_backend(config: Optional[MLXRuntimeConfig] = None, **overrides):
    """Enable the configured MLX runtime backend for the current Python process."""
    config = config or MLXRuntimeConfig()
    apply(config, **overrides)


def enable_smaq(config: Optional[MLXRuntimeConfig] = None, **overrides):
    """Backward-compatible alias for enabling the runtime backend."""
    enable_backend(config, **overrides)


def disable_backend():
    """Disable runtime hooks for the current Python process."""
    revert()


def disable_smaq():
    """Backward-compatible alias for disabling runtime hooks."""
    disable_backend()


def get_runtime_configuration() -> dict:
    """Return active in-process runtime configuration."""
    return current_configuration()


def get_smaq_configuration() -> dict:
    """Backward-compatible alias for runtime configuration."""
    return get_runtime_configuration()


@contextmanager
def backend_enabled(config: Optional[MLXRuntimeConfig] = None, **overrides) -> Iterator[MLXRuntimeConfig]:
    """Context manager for temporary backend enablement."""
    resolved = config or MLXRuntimeConfig()
    enable_backend(resolved, **overrides)
    try:
        yield resolved
    finally:
        disable_backend()
        clear_configuration()


@contextmanager
def smaq_enabled(config: Optional[MLXRuntimeConfig] = None, **overrides) -> Iterator[MLXRuntimeConfig]:
    """Backward-compatible alias for the backend context manager."""
    with backend_enabled(config, **overrides) as resolved:
        yield resolved


def generate(model, tokenizer, prompt, config: Optional[MLXRuntimeConfig] = None, **kwargs):
    """Run mlx_lm.generate with the selected runtime backend enabled."""
    resolved = config or MLXRuntimeConfig()
    with backend_enabled(resolved):
        if "prompt_cache" not in kwargs:
            kwargs["prompt_cache"] = make_prompt_cache(
                model,
                backend=resolved.backend,
                key_bits=resolved.key_bits,
                value_bits=resolved.value_bits,
                mode=resolved.mode,
                strict_benchmark=resolved.strict_benchmark,
                polarquant_bits=resolved.polarquant_bits,
                polarquant_key_seed=resolved.polarquant_key_seed,
                polarquant_value_seed=resolved.polarquant_value_seed,
                rotorquant_bits=resolved.rotorquant_bits,
                rotorquant_key_seed=resolved.rotorquant_key_seed,
                rotorquant_value_seed=resolved.rotorquant_value_seed,
                folded_turbo_bits=resolved.folded_turbo_bits,
                folded_turbo_key_seed=resolved.folded_turbo_key_seed,
                folded_turbo_value_seed=resolved.folded_turbo_value_seed,
                folded_smaq_c=resolved.folded_smaq_c,
                turboquant_bits=resolved.turboquant_bits,
                turboquant_seed=resolved.turboquant_seed,
                turboquant_fused=resolved.turboquant_fused,
            )
        return mlx_lm.generate(model, tokenizer, prompt=prompt, **kwargs)


__all__ = [
    "MLXRuntimeConfig",
    "SMAQConfig",
    "available_backends",
    "backend_matrix",
    "backend_enabled",
    "disable_backend",
    "disable_smaq",
    "enable_backend",
    "enable_smaq",
    "generate",
    "get_runtime_configuration",
    "get_smaq_configuration",
    "make_prompt_cache",
    "make_smaq_prompt_cache",
    "smaq_enabled",
]
