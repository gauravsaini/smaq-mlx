"""SMAQ KV cache — MLX implementation.

Drop-in replacement for mlx-lm's KVCache that compresses keys using SMAQ
spectral metric-aware quantization and values using group quantization.

Implements the same interface as mlx-lm's KVCache:
  - update_and_fetch(keys, values) -> (keys, values)
  - offset property
  - state property
"""

import math
from typing import Optional

import mlx.core as mx

from smaq.quantizer import SMAQQuantized, SMAQQuantizer


def unpack_values(data: mx.array, bits: int) -> mx.array:
    """Unpack bit-packed value tensors into per-element values."""
    if bits == 2:
        v0 = data & 0x03
        v1 = (data >> 2) & 0x03
        v2 = (data >> 4) & 0x03
        v3 = (data >> 6) & 0x03
        return mx.stack([v0, v1, v2, v3], axis=-1).reshape(*data.shape[:-1], data.shape[-1] * 4)
    if bits == 4:
        v0 = data & 0x0F
        v1 = (data >> 4) & 0x0F
        return mx.stack([v0, v1], axis=-1).reshape(*data.shape[:-1], data.shape[-1] * 2)
    return data


def quantize_values(
    v: mx.array,
    bits: int = 2,
    group_size: int = 32,
) -> tuple[mx.array, mx.array, mx.array, int]:
    """Groupwise asymmetric quantization for value vectors.

    Returns: (data, scales, zeros, bits)
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    if d % group_size != 0:
        raise ValueError(f"head_dim {d} must be divisible by group_size {group_size}")

    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)
    v_min = v_grouped.min(axis=-1, keepdims=True)
    v_max = v_grouped.max(axis=-1, keepdims=True)

    n_levels = (2 ** bits) - 1
    scale = mx.maximum((v_max - v_min) / n_levels, 1e-10)
    zero = v_min

    v_q = mx.clip(mx.round((v_grouped - zero) / scale), 0, n_levels).astype(mx.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    if bits == 2:
        if d % 4 != 0:
            raise ValueError(f"head_dim {d} must be divisible by 4 for 2-bit packing")
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
        v_q_flat = packed
    elif bits == 4:
        if d % 2 != 0:
            raise ValueError(f"head_dim {d} must be divisible by 2 for 4-bit packing")
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
        v_q_flat = packed

    return (v_q_flat, scale.squeeze(-1), zero.squeeze(-1), bits)


def dequantize_values(
    data: mx.array,
    scales: mx.array,
    zeros: mx.array,
    bits: int,
    group_size: int = 32,
) -> mx.array:
    """Reconstruct quantized values."""
    vals_per = 4 if bits == 2 else 2 if bits == 4 else 1
    packed_d = data.shape[-1]
    d = packed_d * vals_per

    unpacked = unpack_values(data, bits).astype(mx.float32)
    n_groups = d // group_size

    unpacked = unpacked.reshape(*unpacked.shape[:-1], n_groups, group_size)
    scales_exp = scales[..., None]
    zeros_exp = zeros[..., None]
    return (unpacked * scales_exp + zeros_exp).reshape(*unpacked.shape[:-2], d)


class SMAQKVCache:
    """Drop-in KV cache with SMAQ-compressed keys + group-quantized values.

    Implements the mlx-lm KVCache interface for seamless integration:
      - update_and_fetch(keys, values) -> (keys, values)
      - offset property (used by RoPE)
      - state property (used by mlx-lm generation)
    """

    step = 256

    def __init__(
        self,
        head_dim: int,
        Sigma_q: Optional[mx.array] = None,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.layer_idx = layer_idx

        self.key_quantizer = SMAQQuantizer(
            dim=head_dim,
            Sigma_q=Sigma_q,
            bits=key_bits,
        )

        # mlx-lm interface attributes
        self.offset = 0
        self.seq_len = 0

        # Compressed storage
        self.key_quantized: Optional[SMAQQuantized] = None
        self.value_data: Optional[mx.array] = None
        self.value_scales: Optional[mx.array] = None
        self.value_zeros: Optional[mx.array] = None

        # Exact buffer for recent tokens
        self.key_buffer: Optional[mx.array] = None
        self.value_buffer: Optional[mx.array] = None

        # Full dequantized buffers for drop-in compatibility
        self._key_dequant: Optional[mx.array] = None
        self._value_dequant: Optional[mx.array] = None

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize new KV pairs, store compressed, return dequantized for SDPA.

        This is the main interface called by mlx-lm's attention layers.
        Keys are SMAQ-quantized, values are group-quantized.
        Dequantized versions are returned for the attention computation.
        """
        prev = self.offset
        num_steps = keys.shape[2]
        self.offset += num_steps
        self.seq_len = self.offset

        # SMAQ quantize keys
        new_key_q = self.key_quantizer.quantize(keys)
        # Group quantize values
        new_v_data, new_v_scales, new_v_zeros, _ = quantize_values(
            values, bits=self.value_bits, group_size=self.value_group_size
        )

        # Dequantize for return
        new_k_hat = self.key_quantizer.dequantize(new_key_q)
        new_v_hat = dequantize_values(
            new_v_data, new_v_scales, new_v_zeros,
            self.value_bits, self.value_group_size
        )

        # Append to compressed storage
        if self.key_quantized is None:
            self.key_quantized = new_key_q
            self.value_data = new_v_data
            self.value_scales = new_v_scales
            self.value_zeros = new_v_zeros
            self._key_dequant = new_k_hat
            self._value_dequant = new_v_hat
        else:
            self.key_quantized = SMAQQuantized(
                indices=mx.concatenate([self.key_quantized.indices, new_key_q.indices], axis=-2),
                norms=mx.concatenate([self.key_quantized.norms, new_key_q.norms], axis=-1),
                bits=new_key_q.bits,
            )
            self.value_data = mx.concatenate([self.value_data, new_v_data], axis=-2)
            self.value_scales = mx.concatenate([self.value_scales, new_v_scales], axis=-2)
            self.value_zeros = mx.concatenate([self.value_zeros, new_v_zeros], axis=-2)
            self._key_dequant = mx.concatenate([self._key_dequant, new_k_hat], axis=-2)
            self._value_dequant = mx.concatenate([self._value_dequant, new_v_hat], axis=-2)

        return self._key_dequant, self._value_dequant

    @property
    def state(self):
        """Return state for mlx-lm compatibility."""
        return [self._key_dequant, self._value_dequant] if self._key_dequant is not None else []

    @state.setter
    def state(self, v):
        pass

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        if self._key_dequant is not None:
            self._key_dequant = self._key_dequant[..., :self.offset, :]
            self._value_dequant = self._value_dequant[..., :self.offset, :]
        return n

    def empty(self):
        return self.key_quantized is None

    def prefill(self, keys: mx.array, values: mx.array):
        """Capture a full prefill segment (legacy interface)."""
        self.update_and_fetch(keys, values)

    def append(self, key: mx.array, value: mx.array):
        """Append a decode token (legacy interface)."""
        self.update_and_fetch(key, value)

    def attention_scores(self, query: mx.array, scale: Optional[float] = None) -> mx.array:
        """Compute attention scores against compressed history."""
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)
        if self.key_quantized is None:
            return mx.zeros((*query.shape[:-1], 0))
        return self.key_quantizer.attention_score(query, self.key_quantized, scale=scale)

    def attend(self, attn_weights: mx.array) -> mx.array:
        """Apply attention weights to compressed values."""
        if self.value_data is None:
            return mx.zeros((*attn_weights.shape[:-1], self.head_dim))
        v_dequant = dequantize_values(
            self.value_data, self.value_scales, self.value_zeros,
            self.value_bits, self.value_group_size
        )
        return (attn_weights.astype(mx.float32) @ v_dequant)

    def memory_bytes(self) -> dict[str, int]:
        """Estimate memory usage of the cache."""
        info = {"quantized_keys": 0, "quantized_values": 0, "total": 0}

        if self.key_quantized is not None:
            info["quantized_keys"] += self.key_quantized.indices.size
            info["quantized_keys"] += self.key_quantized.norms.size * 2

        if self.value_data is not None:
            info["quantized_values"] += self.value_data.size
            info["quantized_values"] += self.value_scales.size * 2
            info["quantized_values"] += self.value_zeros.size * 2

        info["total"] = info["quantized_keys"] + info["quantized_values"]
        return info

    def get_seq_length(self) -> int:
        return self.offset
