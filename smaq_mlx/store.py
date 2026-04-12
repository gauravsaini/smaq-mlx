"""SMAQ compressed KV store — MLX implementation.

Ported from PyTorch to MLX for Apple Silicon execution.

Historical tokens are quantized in chunks and flattened lazily on first read,
mirroring the TurboQuant storage pattern.
"""

from typing import NamedTuple, Optional

import mlx.core as mx

from smaq_mlx.core import CacheCapabilities
from smaq_mlx.kv_cache import quantize_values, dequantize_values
from smaq_mlx.quantizer import SMAQQuantized, SMAQQuantizer


class FlatCache(NamedTuple):
    """Flattened view for fast read access."""

    key_q: SMAQQuantized
    value_data: mx.array
    value_scales: mx.array
    value_zeros: mx.array
    value_bits: int
    num_tokens: int


class CompressedKVStore:
    """Chunked SMAQ KV store with lazy flattening — MLX version."""

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        Sigma_q: Optional[mx.array] = None,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = min(value_group_size, head_dim)
        self.layer_idx = layer_idx

        self.quantizer = SMAQQuantizer(
            dim=head_dim,
            Sigma_q=Sigma_q,
            bits=key_bits,
        )

        self._key_chunks: list[SMAQQuantized] = []
        self._value_data_chunks: list[mx.array] = []
        self._value_scales_chunks: list[mx.array] = []
        self._value_zeros_chunks: list[mx.array] = []
        self._chunk_lengths: list[int] = []
        self._flat: Optional[FlatCache] = None
        self._capabilities = CacheCapabilities(
            strategy_name="smaq_store",
            metric_name="smaq_metric",
            quantization_name="scalar_group",
            compressed_history=True,
            compressed_history_shadow_only=False,
            values_compressed=True,
            decode_uses_compressed_keys=True,
            decode_uses_compressed_values=True,
        )

    @property
    def num_tokens(self) -> int:
        return sum(self._chunk_lengths)

    @property
    def num_chunks(self) -> int:
        return len(self._chunk_lengths)

    def append_chunk(self, key: mx.array, value: mx.array):
        """Quantize and append a contiguous KV chunk."""
        # Handle different input shapes:
        # (T, H, D) -> chunk_len = T
        # (B, H, T, D) -> chunk_len = T
        if key.ndim == 3:
            chunk_len = key.shape[0]
            k = key[None, :, :, :]
        elif key.ndim == 4:
            chunk_len = key.shape[-2]
            k = key
        else:
            k = key.reshape(1, 1, -1, self.head_dim)
            chunk_len = k.shape[-2]

        if value.ndim == 3:
            v = value[None, :, :, :]
        elif value.ndim == 4:
            v = value
        else:
            v = value.reshape(1, 1, -1, self.head_dim)

        key_q = self.quantizer.quantize(k)
        v_data, v_scales, v_zeros, _ = quantize_values(
            v, bits=self.value_bits, group_size=self.value_group_size
        )

        self._key_chunks.append(key_q)
        self._value_data_chunks.append(v_data)
        self._value_scales_chunks.append(v_scales)
        self._value_zeros_chunks.append(v_zeros)
        self._chunk_lengths.append(chunk_len)
        self._flat = None

    def get_flat_cache(self) -> Optional[FlatCache]:
        """Return a cached flattened view of the compressed history."""
        if not self._key_chunks:
            return None
        if self._flat is not None:
            return self._flat

        flat_kq = _concat_key_q(self._key_chunks)
        flat_vd = mx.concatenate(self._value_data_chunks, axis=-2)
        flat_vs = mx.concatenate(self._value_scales_chunks, axis=-2)
        flat_vz = mx.concatenate(self._value_zeros_chunks, axis=-2)

        self._flat = FlatCache(
            key_q=flat_kq,
            value_data=flat_vd,
            value_scales=flat_vs,
            value_zeros=flat_vz,
            value_bits=self.value_bits,
            num_tokens=self.num_tokens,
        )
        return self._flat

    @property
    def capabilities(self) -> CacheCapabilities:
        return self._capabilities

    def report(self) -> dict[str, int | str | bool]:
        return {
            "strategy_name": self._capabilities.strategy_name,
            "metric_name": self._capabilities.metric_name,
            "quantization_name": self._capabilities.quantization_name,
            "compressed_history": self._capabilities.compressed_history,
            "compressed_history_shadow_only": self._capabilities.compressed_history_shadow_only,
            "values_compressed": self._capabilities.values_compressed,
            "decode_uses_compressed_keys": self._capabilities.decode_uses_compressed_keys,
            "decode_uses_compressed_values": self._capabilities.decode_uses_compressed_values,
            "num_tokens": self.num_tokens,
            "num_chunks": self.num_chunks,
        }

    def memory_bytes(self) -> int:
        total = 0
        for key_q in self._key_chunks:
            total += key_q.indices.size
            total += key_q.norms.size * 2
        for v_data in self._value_data_chunks:
            total += v_data.size
        for v_scales in self._value_scales_chunks:
            total += v_scales.size * 2
        for v_zeros in self._value_zeros_chunks:
            total += v_zeros.size * 2
        return total

    def reset(self):
        self._key_chunks.clear()
        self._value_data_chunks.clear()
        self._value_scales_chunks.clear()
        self._value_zeros_chunks.clear()
        self._chunk_lengths.clear()
        self._flat = None


def _concat_key_q(chunks: list[SMAQQuantized]) -> SMAQQuantized:
    """Concatenate flattened key chunks along token dimension."""
    return SMAQQuantized(
        indices=mx.concatenate([chunk.indices for chunk in chunks], axis=-2),
        norms=mx.concatenate([chunk.norms for chunk in chunks], axis=-1),
        bits=chunks[0].bits,
    )
