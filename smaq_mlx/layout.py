"""Model layout adapters for Qwen/Gemma and generic MLX models."""

from __future__ import annotations

from typing import Any

from smaq_mlx.core import LayoutInfo


class ModelLayoutAdapter:
    """Normalize KV shapes for a specific model family."""

    name = "generic"

    def resolve_head_dim(self, model: Any, layer_idx: int) -> int:
        layer = model.layers[layer_idx]
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "head_dim"):
            return int(attn.head_dim)
        if attn is not None and hasattr(attn, "hidden_size") and hasattr(attn, "num_heads"):
            return int(attn.hidden_size // attn.num_heads)
        return 128

    def normalize_kv(self, keys: Any, values: Any, expected_head_dim: int | None = None) -> LayoutInfo:
        key_dim = int(keys.shape[-1])
        value_dim = int(values.shape[-1])
        unified = expected_head_dim is not None and key_dim != expected_head_dim
        return LayoutInfo(
            adapter_name=self.name,
            effective_head_dim=key_dim,
            observed_key_dim=key_dim,
            observed_value_dim=value_dim,
            unified_kv=unified,
            note="generic",
        )


class QwenLayoutAdapter(ModelLayoutAdapter):
    name = "qwen"


class GemmaLayoutAdapter(ModelLayoutAdapter):
    name = "gemma"

    def normalize_kv(self, keys: Any, values: Any, expected_head_dim: int | None = None) -> LayoutInfo:
        info = super().normalize_kv(keys, values, expected_head_dim=expected_head_dim)
        return LayoutInfo(
            adapter_name=self.name,
            effective_head_dim=info.observed_key_dim,
            observed_key_dim=info.observed_key_dim,
            observed_value_dim=info.observed_value_dim,
            unified_kv=True if expected_head_dim is not None and info.observed_key_dim != expected_head_dim else info.unified_kv,
            note="gemma_unified_kv" if expected_head_dim is not None and info.observed_key_dim != expected_head_dim else "gemma",
        )


def infer_model_layout_adapter(model: Any) -> ModelLayoutAdapter:
    """Infer adapter from model type/name."""
    joined = f"{type(model).__module__} {type(model).__name__}".lower()
    if "gemma" in joined:
        return GemmaLayoutAdapter()
    if "qwen" in joined:
        return QwenLayoutAdapter()
    return ModelLayoutAdapter()
