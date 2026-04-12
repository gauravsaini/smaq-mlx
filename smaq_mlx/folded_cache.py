"""Single-cache folded TurboQuant + SMAQ backend.

Keys are quantized with a TurboQuant-style scalar codec after folding in the
SMAQ metric; values stay on the plain TurboQuant-style scalar path. The cache
stays single-stream, so memory tracks the codec footprint rather than a coarse
+ fine double-cache design.
"""

from __future__ import annotations

import math

import mlx.core as mx

from smaq_mlx.core import CacheCapabilities
from smaq_mlx.folded_turboquant import FoldedTurboQuantizer


class FoldedTurboSMAQKVCache:
    step = 256

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        key_seed: int = 42,
        value_seed: int = 43,
        smaq_c: float = 5.0,
    ):
        self.bits = int(bits)
        self.head_dim = int(head_dim)
        self.offset = 0
        self.key_quantizer = FoldedTurboQuantizer(
            dim=self.head_dim,
            bits=self.bits,
            seed=key_seed,
            Sigma_q=None,
            c=smaq_c,
        )
        self.value_quantizer = FoldedTurboQuantizer(
            dim=self.head_dim,
            bits=self.bits,
            seed=value_seed,
            Sigma_q=None,
            c=smaq_c,
        )
        self._key_packed = None
        self._key_norms = None
        self._value_packed = None
        self._value_norms = None
        self._stored_capacity = 0
        self._packed_dim = math.ceil(self.head_dim / max(1, (32 // self.bits)))
        self._capabilities = CacheCapabilities(
            strategy_name="folded_turbo_smaq",
            metric_name="folded_smaq_metric",
            quantization_name="single_cache_turbo_scalar",
            compressed_history=True,
            compressed_history_shadow_only=False,
            values_compressed=True,
            decode_uses_compressed_keys=True,
            decode_uses_compressed_values=True,
        )

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def metric_fitted(self) -> bool:
        return bool(getattr(self.key_quantizer, "metric_fitted", False))

    def capability_report(self):
        return {
            "strategy_name": self._capabilities.strategy_name,
            "metric_name": self._capabilities.metric_name,
            "quantization_name": self._capabilities.quantization_name,
            "compressed_history": self._capabilities.compressed_history,
            "compressed_history_shadow_only": self._capabilities.compressed_history_shadow_only,
            "values_compressed": self._capabilities.values_compressed,
            "decode_uses_compressed_keys": self._capabilities.decode_uses_compressed_keys,
            "decode_uses_compressed_values": self._capabilities.decode_uses_compressed_values,
            "metric_fitted": self.metric_fitted,
            "offset": int(self.offset),
        }

    def _expand_storage(self, bsz, n_kv_heads, new_steps, dtype):
        alloc_steps = ((self.offset + new_steps + self.step - 1) // self.step) * self.step
        shape = (bsz, n_kv_heads, alloc_steps)
        if self._key_packed is not None and self.offset > 0:
            old_k = self._key_packed[..., :self.offset, :]
            old_kn = self._key_norms[..., :self.offset]
            old_v = self._value_packed[..., :self.offset, :]
            old_vn = self._value_norms[..., :self.offset]
            self._key_packed = mx.concatenate([old_k, mx.zeros((*shape, self._packed_dim), dtype=mx.uint32)], axis=2)
            self._key_norms = mx.concatenate([old_kn, mx.zeros(shape, dtype=dtype)], axis=2)
            self._value_packed = mx.concatenate([old_v, mx.zeros((*shape, self._packed_dim), dtype=mx.uint32)], axis=2)
            self._value_norms = mx.concatenate([old_vn, mx.zeros(shape, dtype=dtype)], axis=2)
        else:
            self._key_packed = mx.zeros((*shape, self._packed_dim), dtype=mx.uint32)
            self._key_norms = mx.zeros(shape, dtype=dtype)
            self._value_packed = mx.zeros((*shape, self._packed_dim), dtype=mx.uint32)
            self._value_norms = mx.zeros(shape, dtype=dtype)
        self._stored_capacity = int(self._key_packed.shape[2])

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        bsz, n_kv_heads, steps, dim = keys.shape
        if dim != self.head_dim or values.shape[-1] != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got keys={keys.shape}, values={values.shape}")
        previous = self.offset
        if self._key_packed is None or (previous + steps) > self._stored_capacity:
            self._expand_storage(bsz, n_kv_heads, steps, keys.dtype)

        key_packed, key_norms = self.key_quantizer.quantize(keys.reshape(-1, dim))
        value_packed, value_norms = self.value_quantizer.quantize(values.reshape(-1, dim))
        self._key_packed[..., previous:previous + steps, :] = key_packed.reshape(bsz, n_kv_heads, steps, self._packed_dim)
        self._key_norms[..., previous:previous + steps] = key_norms.reshape(bsz, n_kv_heads, steps).astype(keys.dtype)
        self._value_packed[..., previous:previous + steps, :] = value_packed.reshape(bsz, n_kv_heads, steps, self._packed_dim)
        self._value_norms[..., previous:previous + steps] = value_norms.reshape(bsz, n_kv_heads, steps).astype(values.dtype)
        self.offset += steps
        return self.materialize(dtype=keys.dtype)

    def materialize(self, dtype=mx.float16):
        if self._key_packed is None or self.offset == 0:
            empty = mx.zeros((0, 0, 0, self.head_dim), dtype=dtype)
            return empty, empty
        bsz, n_kv_heads = int(self._key_packed.shape[0]), int(self._key_packed.shape[1])
        key_hat = self.key_quantizer.dequantize(
            self._key_packed[..., :self.offset, :].reshape(-1, self._packed_dim),
            self._key_norms[..., :self.offset].reshape(-1),
        ).reshape(bsz, n_kv_heads, self.offset, self.head_dim)
        value_hat = self.value_quantizer.dequantize(
            self._value_packed[..., :self.offset, :].reshape(-1, self._packed_dim),
            self._value_norms[..., :self.offset].reshape(-1),
        ).reshape(bsz, n_kv_heads, self.offset, self.head_dim)
        return key_hat.astype(dtype), value_hat.astype(dtype)

    def fit_metric_from_queries(self, queries: mx.array):
        if self.metric_fitted or queries.shape[-1] != self.head_dim:
            return
        q = queries.astype(mx.float32).reshape(-1, self.head_dim)
        if q.shape[0] < 8:
            return
        key_fp = None
        if self._key_packed is not None and self.offset > 0:
            key_fp = self.key_quantizer.dequantize(
                self._key_packed[..., :self.offset, :].reshape(-1, self._packed_dim),
                self._key_norms[..., :self.offset].reshape(-1),
            )
        self.key_quantizer.fit(q)
        if key_fp is not None:
            repacked, renorms = self.key_quantizer.quantize(key_fp)
            bsz, n_kv_heads = int(self._key_packed.shape[0]), int(self._key_packed.shape[1])
            self._key_packed[..., :self.offset, :] = repacked.reshape(bsz, n_kv_heads, self.offset, self._packed_dim)
            self._key_norms[..., :self.offset] = renorms.reshape(bsz, n_kv_heads, self.offset).astype(self._key_norms.dtype)

    def memory_bytes(self):
        if self._key_packed is None:
            return {"compressed_keys": 0, "compressed_values": 0, "total": 0}
        key_bytes = int(self._key_packed[..., :self.offset, :].nbytes + self._key_norms[..., :self.offset].nbytes)
        value_bytes = int(self._value_packed[..., :self.offset, :].nbytes + self._value_norms[..., :self.offset].nbytes)
        total = key_bytes + value_bytes
        return {"compressed_keys": key_bytes, "compressed_values": value_bytes, "total": total}

    @property
    def nbytes(self):
        return int(self.memory_bytes()["total"])

    @property
    def uncompressed_nbytes(self):
        if self._key_packed is None:
            return 0
        bsz = int(self._key_packed.shape[0])
        n_kv_heads = int(self._key_packed.shape[1])
        return bsz * n_kv_heads * int(self.offset) * int(self.head_dim) * 4

    @property
    def compression_ratio(self):
        return float(self.uncompressed_nbytes / self.nbytes) if self.nbytes else 0.0

    def size(self):
        return self.offset

    def empty(self):
        return self._key_packed is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, N, return_array=False, window_size=None):
        from mlx_lm.models.base import create_causal_mask

        offset = self.offset
        if window_size is not None:
            return create_causal_mask(N, offset, window_size=window_size)
        if N == 1:
            return None
        if return_array:
            return create_causal_mask(N, offset, window_size=window_size)
        return "causal"

    @property
    def state(self):
        if self._key_packed is None:
            return []
        return [
            self._key_packed[..., :self.offset, :],
            self._key_norms[..., :self.offset],
            self._value_packed[..., :self.offset, :],
            self._value_norms[..., :self.offset],
        ]

    @state.setter
    def state(self, value):
        if value is not None and value:
            self._key_packed, self._key_norms, self._value_packed, self._value_norms = value
            self.offset = int(self._key_packed.shape[2])
            self._stored_capacity = self.offset

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.bits, self.head_dim, int(self.metric_fitted))))

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0])
        self.bits = int(float(value[1]))
        self.head_dim = int(value[2])
