"""Backend-agnostic SMAQ core contracts for MLX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheCapabilities:
    """Runtime cache/attention behavior summary."""

    strategy_name: str
    metric_name: str
    quantization_name: str
    compressed_history: bool
    compressed_history_shadow_only: bool
    values_compressed: bool
    decode_uses_compressed_keys: bool
    decode_uses_compressed_values: bool


@dataclass(frozen=True)
class LayoutInfo:
    """Normalized KV layout info for a model/layer."""

    adapter_name: str
    effective_head_dim: int
    observed_key_dim: int
    observed_value_dim: int
    unified_kv: bool = False
    note: str = ""


class CalibrationProvider:
    """Calibration state source for per-layer Sigma_q."""

    def get_sigma_q(self, layer_idx: int, head_dim: int, device: Any | None = None) -> Any | None:
        return None


class IdentityCalibrationProvider(CalibrationProvider):
    """No calibration. Identity metric."""

    pass


class StaticCalibrationProvider(CalibrationProvider):
    """Simple lookup by layer index or layer name."""

    def __init__(self, per_layer: dict[Any, Any]):
        self.per_layer = dict(per_layer)

    def get_sigma_q(self, layer_idx: int, head_dim: int, device: Any | None = None) -> Any | None:
        if layer_idx in self.per_layer:
            return self.per_layer[layer_idx]
        return self.per_layer.get(f"layer_{layer_idx}")
