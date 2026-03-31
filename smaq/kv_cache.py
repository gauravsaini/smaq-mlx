"""SMAQ KV cache — MLX implementation.

Ported from PyTorch to MLX for Apple Silicon execution.

Keys use the SMAQ quantizer; values use standard group quantization so the
runtime can slot into the same decode pattern as TurboQuant.
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
    """Drop-in KV cache with SMAQ-compressed history plus an exact recent buffer.

    MLX version — uses pre-allocation with step=256 like turboquant-mlx.
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

        self.seq_len = 0
        self.key_quantized: Optional[SMAQQuantized] = None
        self.value_data: Optional[mx.array] = None
        self.value_scales: Optional[mx.array] = None
        self.value_zeros: Optional[mx.array] = None
        self.value_bits: int = value_bits
        self.key_buffer: Optional[mx.array] = None
        self.value_buffer: Optional[mx.array] = None
        self._offset = 0

    def prefill(self, keys: mx.array, values: mx.array):
        """Capture a full prefill segment."""
        seq_len = keys.shape[-2]
        self.seq_len = seq_len

        if seq_len <= self.buffer_size:
            self.key_buffer = keys
            self.value_buffer = values
            return

        n_quant = seq_len - self.buffer_size
        keys_to_quant = keys[..., :n_quant, :]
        values_to_quant = values[..., :n_quant, :]
        self.key_buffer = keys[..., n_quant:, :]
        self.value_buffer = values[..., n_quant:, :]
        self.key_quantized = self.key_quantizer.quantize(keys_to_quant)
        v_data, v_scales, v_zeros, _ = quantize_values(
            values_to_quant, bits=self.value_bits, group_size=self.value_group_size
        )
        self.value_data = v_data
        self.value_scales = v_scales
        self.value_zeros = v_zeros

    def append(self, key: mx.array, value: mx.array):
        """Append a decode token into the exact ring and flush as needed."""
        self.seq_len += key.shape[-2]

        if self.key_buffer is not None:
            self.key_buffer = mx.concatenate([self.key_buffer, key], axis=-2)
            self.value_buffer = mx.concatenate([self.value_buffer, value], axis=-2)
        else:
            self.key_buffer = key
            self.value_buffer = value

        if self.key_buffer.shape[-2] > self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        n_flush = self.key_buffer.shape[-2] - self.buffer_size
        keys_flush = self.key_buffer[..., :n_flush, :]
        values_flush = self.value_buffer[..., :n_flush, :]

        self.key_buffer = self.key_buffer[..., n_flush:, :]
        self.value_buffer = self.value_buffer[..., n_flush:, :]

        new_key_q = self.key_quantizer.quantize(keys_flush)
        new_v_data, new_v_scales, new_v_zeros, _ = quantize_values(
            values_flush, bits=self.value_bits, group_size=self.value_group_size
        )

        if self.key_quantized is None:
            self.key_quantized = new_key_q
            self.value_data = new_v_data
            self.value_scales = new_v_scales
            self.value_zeros = new_v_zeros
            return

        self.key_quantized = SMAQQuantized(
            indices=mx.concatenate([self.key_quantized.indices, new_key_q.indices], axis=-2),
            norms=mx.concatenate([self.key_quantized.norms, new_key_q.norms], axis=-1),
            bits=new_key_q.bits,
        )
        self.value_data = mx.concatenate([self.value_data, new_v_data], axis=-2)
        self.value_scales = mx.concatenate([self.value_scales, new_v_scales], axis=-2)
        self.value_zeros = mx.concatenate([self.value_zeros, new_v_zeros], axis=-2)

    def attention_scores(self, query: mx.array, scale: Optional[float] = None) -> mx.array:
        """Compute attention scores against compressed history and exact buffer."""
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores_parts = []
        if self.key_quantized is not None:
            scores_parts.append(
                self.key_quantizer.attention_score(query, self.key_quantized, scale=scale)
            )

        if self.key_buffer is not None:
            scores_parts.append(
                (query.astype(mx.float32) @ self.key_buffer.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
            )

        if not scores_parts:
            shape = (*query.shape[:-2], query.shape[-2], 0)
            return mx.zeros(shape)

        return mx.concatenate(scores_parts, axis=-1)

    def attend(self, attn_weights: mx.array) -> mx.array:
        """Apply attention weights to compressed values and exact buffer values."""
        output_parts = []
        col_offset = 0

        if self.value_data is not None:
            n_quant = self.value_data.shape[-2]
            w_quant = attn_weights[..., col_offset: col_offset + n_quant]
            v_dequant = dequantize_values(
                self.value_data, self.value_scales, self.value_zeros,
                self.value_bits, self.value_group_size
            )
            output_parts.append(w_quant.astype(mx.float32) @ v_dequant)
            col_offset += n_quant

        if self.value_buffer is not None:
            n_buf = self.value_buffer.shape[-2]
            w_buf = attn_weights[..., col_offset: col_offset + n_buf]
            output_parts.append(w_buf.astype(mx.float32) @ self.value_buffer)

        if not output_parts:
            return mx.zeros((*attn_weights.shape[:-1], self.head_dim))
        return sum(output_parts)

    def memory_bytes(self) -> dict[str, int]:
        """Estimate memory usage of the cache."""
        info = {"quantized_keys": 0, "quantized_values": 0, "buffer": 0, "total": 0}

        if self.key_quantized is not None:
            info["quantized_keys"] += self.key_quantized.indices.size
            info["quantized_keys"] += self.key_quantized.norms.size * 2

        if self.value_data is not None:
            info["quantized_values"] += self.value_data.size
            info["quantized_values"] += self.value_scales.size * 2
            info["quantized_values"] += self.value_zeros.size * 2

        if self.key_buffer is not None:
            info["buffer"] += self.key_buffer.size * 2
        if self.value_buffer is not None:
            info["buffer"] += self.value_buffer.size * 2

        info["total"] = info["quantized_keys"] + info["quantized_values"] + info["buffer"]
        return info

    def get_seq_length(self) -> int:
        return self.seq_len
