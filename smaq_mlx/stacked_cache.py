"""Experimental stacked cache backends for MLX.

Current experiment:
- stage 1: TurboQuant-style lossy pre-quantization
- stage 2: SMAQ compressed-history storage and decode

This is intentionally a research prototype, not a production path.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from smaq_mlx.core import CacheCapabilities
from smaq_mlx.kv_cache import SMAQKVCache


class TurboSMAQCascadeCache:
    """Experimental two-stage compression cache.

    Flow:
    1. approximate new tokens with TurboQuant
    2. feed the approximated tokens into SMAQ as the served cache format

    This is a true two-stage quantization experiment, but not a fused
    double-packed production cache.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        key_bits: int = 4,
        value_bits: int = 4,
        turboquant_bits: int = 3,
        turboquant_seed: int = 42,
        layer_idx: int = 0,
        layout_adapter=None,
        mode: str = "hybrid",
        strict_benchmark: bool = True,
        Sigma_q: Optional[mx.array] = None,
    ):
        from turboquant_mlx.cache import TurboQuantKVCache

        self.turbo_cache = TurboQuantKVCache(
            bits=turboquant_bits,
            k_bits=turboquant_bits,
            v_bits=turboquant_bits,
            seed=turboquant_seed,
            fused=False,
        )
        self.smaq_cache = SMAQKVCache(
            head_dim=head_dim,
            Sigma_q=Sigma_q,
            key_bits=key_bits,
            value_bits=value_bits,
            layer_idx=layer_idx,
            layout_adapter=layout_adapter,
            mode=mode,
            strict_benchmark=strict_benchmark,
        )
        self.turboquant_bits = turboquant_bits
        self._capabilities = CacheCapabilities(
            strategy_name="stacked_turbo_smaq",
            metric_name="smaq_metric_after_turboquant",
            quantization_name="turboquant_then_smaq",
            compressed_history=True,
            compressed_history_shadow_only=False,
            values_compressed=True,
            decode_uses_compressed_keys=True,
            decode_uses_compressed_values=True,
        )

    def __getattr__(self, name):
        return getattr(self.smaq_cache, name)

    @property
    def capabilities(self) -> CacheCapabilities:
        return self._capabilities

    @property
    def offset(self) -> int:
        return self.smaq_cache.offset

    @property
    def key_quantized(self):
        return self.smaq_cache.key_quantized

    def report(self) -> dict[str, int | str | bool]:
        return self.capability_report()

    def capability_report(self) -> dict[str, int | str | bool]:
        inner = self.smaq_cache.capability_report()
        inner.update(
            {
                "strategy_name": self._capabilities.strategy_name,
                "metric_name": self._capabilities.metric_name,
                "quantization_name": self._capabilities.quantization_name,
                "turboquant_bits": self.turboquant_bits,
                "offset": self.offset,
            }
        )
        return inner

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if keys.ndim == 3:
            keys = keys[None, ...]
        if values.ndim == 3:
            values = values[None, ...]
        num_new = keys.shape[-2]

        tq_keys, tq_values = self.turbo_cache.update_and_fetch(keys, values)
        approx_keys = tq_keys[..., -num_new:, :]
        approx_values = tq_values[..., -num_new:, :]
        return self.smaq_cache.update_and_fetch(approx_keys, approx_values)

    def memory_bytes(self, include_shadow: bool = False) -> dict[str, int]:
        smaq_mem = self.smaq_cache.memory_bytes(include_shadow=include_shadow)
        tq_mem = int(getattr(self.turbo_cache, "nbytes", 0))
        total = tq_mem + smaq_mem["total"]
        return {
            "turboquant_stage": tq_mem,
            "smaq_stage": smaq_mem["total"],
            "shadow": smaq_mem.get("shadow", 0),
            "total": total,
        }

    @property
    def nbytes(self):
        return self.memory_bytes()["total"]

    @property
    def nbytes_equivalent_fp16(self):
        uncompressed = int(getattr(self.turbo_cache, "uncompressed_nbytes", 0))
        if uncompressed:
            return uncompressed
        return self.smaq_cache.nbytes_equivalent_fp16
