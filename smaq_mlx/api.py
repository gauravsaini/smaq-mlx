"""Public API for production-style SMAQ-MLX usage."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import mlx_lm

from smaq_mlx.patch import (
    apply,
    clear_configuration,
    configure,
    current_configuration,
    make_smaq_prompt_cache,
    revert,
)


@dataclass
class SMAQConfig:
    """User-facing runtime configuration for SMAQ-MLX."""

    enabled: bool = True
    key_bits: int = 4
    value_bits: int = 4
    mode: str = "hybrid"
    strict_benchmark: bool = False
    require_true_compressed: bool = False


def enable_smaq(config: Optional[SMAQConfig] = None, **overrides):
    """Enable SMAQ hooks for the current Python process."""
    config = config or SMAQConfig()
    apply(config, **overrides)


def disable_smaq():
    """Disable SMAQ hooks for the current Python process."""
    revert()


def get_smaq_configuration() -> dict:
    """Return active in-process SMAQ configuration."""
    return current_configuration()


@contextmanager
def smaq_enabled(config: Optional[SMAQConfig] = None, **overrides) -> Iterator[SMAQConfig]:
    """Context manager for temporary SMAQ enablement."""
    resolved = config or SMAQConfig()
    enable_smaq(resolved, **overrides)
    try:
        yield resolved
    finally:
        disable_smaq()
        clear_configuration()


def generate(model, tokenizer, prompt, config: Optional[SMAQConfig] = None, **kwargs):
    """Run mlx_lm.generate with SMAQ enabled for this process."""
    resolved = config or SMAQConfig()
    with smaq_enabled(resolved):
        if "prompt_cache" not in kwargs:
            kwargs["prompt_cache"] = make_smaq_prompt_cache(
                model,
                key_bits=resolved.key_bits,
                value_bits=resolved.value_bits,
                mode=resolved.mode,
                strict_benchmark=resolved.strict_benchmark,
            )
        return mlx_lm.generate(model, tokenizer, prompt=prompt, **kwargs)
