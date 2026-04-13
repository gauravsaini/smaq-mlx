"""SMAQ KV cache — MLX implementation.

Drop-in replacement for mlx-lm's KVCache that compresses keys using SMAQ
spectral metric-aware quantization and values using group quantization.

Implements the same interface as mlx-lm's KVCache:
  - update_and_fetch(keys, values) -> (keys, values)
  - offset property
  - state property

Design:
- During prefill: store full keys/values for exact attention, also quantize for tracking
- During decode: compute attention against quantized keys via custom SDPA
- The patch.py intercepts SDPA to use quantized attention when possible
"""

import math
from typing import Optional

import mlx.core as mx

from smaq_mlx.core import CacheCapabilities
from smaq_mlx.layout import ModelLayoutAdapter
from smaq_mlx.quantizer import SMAQQuantized, SMAQQuantizer


def _create_causal_mask(N, offset, window_size=None):
    """Build a causal attention mask."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


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
    """Drop-in KV cache with compressed-history attention.

    True compressed path:
    - historical tokens stored as quantized keys/values
    - recent tail kept exact in ring buffer
    - SDPA uses compressed history + exact tail

    Compatibility path:
    - optional full-precision shadow cache is still tracked for mlx-lm state
      and debugging, but benchmark/reporting can ignore it.
    """

    def __init__(
        self,
        head_dim: int,
        Sigma_q: Optional[mx.array] = None,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        layer_idx: int = 0,
        layout_adapter: ModelLayoutAdapter | None = None,
        mode: str = "hybrid",
        strict_benchmark: bool = False,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.layer_idx = layer_idx
        self.layout_adapter = layout_adapter or ModelLayoutAdapter()
        self.mode = mode
        self.strict_benchmark = strict_benchmark

        self._sigma_q = Sigma_q
        self._layout_info = None
        self.key_quantizer: Optional[SMAQQuantizer] = None

        self.offset = 0
        self.seq_len = 0

        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.key_quantized: Optional[SMAQQuantized] = None
        self.value_data: Optional[mx.array] = None
        self.value_scales: Optional[mx.array] = None
        self.value_zeros: Optional[mx.array] = None
        self.key_buffer: Optional[mx.array] = None
        self.value_buffer: Optional[mx.array] = None
        self._linear_attn_state: list = [None, None]
        self._capabilities = self._build_capabilities()

    def _build_capabilities(self) -> CacheCapabilities:
        compressed = self.mode in ("hybrid", "true_compressed")
        shadow = self.mode == "shadow"
        return CacheCapabilities(
            strategy_name="smaq_mlx_cache",
            metric_name="smaq_metric",
            quantization_name="scalar_group",
            compressed_history=compressed,
            compressed_history_shadow_only=shadow,
            values_compressed=True,
            decode_uses_compressed_keys=compressed,
            decode_uses_compressed_values=compressed,
        )

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
            "head_dim": self.head_dim,
            "buffer_size": self.buffer_size,
            "offset": self.offset,
        }

    def _normalize_io(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if keys.ndim == 3:
            keys = keys[None, ...]
        if values.ndim == 3:
            values = values[None, ...]
        return keys, values

    def _ensure_layout(self, keys: mx.array, values: mx.array):
        if self._layout_info is not None:
            return
        info = self.layout_adapter.normalize_kv(keys, values, expected_head_dim=self.head_dim)
        self._layout_info = info
        self.head_dim = info.effective_head_dim
        if self._sigma_q is None:
            self._sigma_q = mx.eye(self.head_dim)
        self.key_quantizer = SMAQQuantizer(
            dim=self.head_dim,
            Sigma_q=self._sigma_q,
            bits=self.key_bits,
        )
        self.value_group_size = min(self.value_group_size, values.shape[-1])

    def _append_shadow(self, keys: mx.array, values: mx.array):
        """Legacy shadow append — now a no-op.

        The shadow cache was the root cause of the memory leak: it grew
        at full FP16 rate alongside the compressed store, tripling memory.
        KV reconstruction now happens on-demand from compressed + buffer.
        """
        pass

    def _append_compressed(self, keys: mx.array, values: mx.array):
        if self.key_quantizer is None:
            raise RuntimeError("SMAQKVCache not initialized")
        key_q = self.key_quantizer.quantize(keys)
        val_q = quantize_values(values, bits=self.value_bits, group_size=self.value_group_size)
        if self.key_quantized is None:
            self.key_quantized = key_q
            self.value_data = val_q[0]
            self.value_scales = val_q[1]
            self.value_zeros = val_q[2]
            return
        self.key_quantized = SMAQQuantized(
            indices=mx.concatenate([self.key_quantized.indices, key_q.indices], axis=-2),
            norms=mx.concatenate([self.key_quantized.norms, key_q.norms], axis=-1),
            bits=self.key_quantized.bits,
        )
        self.value_data = mx.concatenate([self.value_data, val_q[0]], axis=-2)
        self.value_scales = mx.concatenate([self.value_scales, val_q[1]], axis=-2)
        self.value_zeros = mx.concatenate([self.value_zeros, val_q[2]], axis=-2)

    def _flush_buffer_if_needed(self):
        if self.key_buffer is None or self.key_buffer.shape[-2] <= self.buffer_size:
            return
        n_flush = self.key_buffer.shape[-2] - self.buffer_size
        flush_k = self.key_buffer[..., :n_flush, :]
        flush_v = self.value_buffer[..., :n_flush, :]
        self._append_compressed(flush_k, flush_v)
        self.key_buffer = self.key_buffer[..., n_flush:, :]
        self.value_buffer = self.value_buffer[..., n_flush:, :]

    def __getitem__(self, idx):
        """Support cache[0], cache[1] for linear_attn layers (Qwen3.5)."""
        return self._linear_attn_state[idx]

    def __len__(self):
        """Return the cache sequence length for mlx-lm attention mask creation."""
        return self.offset

    def __bool__(self):
        """Always truthy so 'cache = cache or make_cache()' works."""
        return True

    @property
    def lengths(self):
        return None

    def __setitem__(self, idx, value):
        """Support cache[0] = x, cache[1] = y for linear_attn layers."""
        self._linear_attn_state[idx] = value
        
    def advance(self, offset: int = 1):
        """Advance cache offset."""
        # Typically the offset is automatically managed in update_and_fetch,
        # but some layers explicitly call advance. We just ignore if already accounted for,
        # or implement it if needed. mlx_lm uses it for specific architectures.
        pass

    def make_mask(self, N, return_array=False, window_size=None):
        """Create attention mask compatible with mlx-lm's expectation."""
        if N == 1:
            return None
        if return_array or (window_size and N > window_size):
            return _create_causal_mask(N, self.offset, window_size=window_size)
        return "causal"

    def _reconstruct_kv(self) -> tuple[mx.array, mx.array]:
        """Reconstruct full KV arrays from compressed history + exact buffer.

        This replaces the old shadow cache approach. Instead of keeping a
        separate full-precision copy (which was the memory leak), we
        reconstruct on demand from the already-stored compressed data.
        """
        parts_k = []
        parts_v = []

        # Compressed history
        if self.key_quantized is not None:
            k_hat = self.key_quantizer.dequantize(self.key_quantized)
            v_hat = dequantize_values(
                self.value_data, self.value_scales, self.value_zeros,
                self.value_bits, self.value_group_size,
            )
            parts_k.append(k_hat)
            parts_v.append(v_hat)

        # Exact buffer tail
        if self.key_buffer is not None:
            parts_k.append(self.key_buffer)
            parts_v.append(self.value_buffer)

        if not parts_k:
            return None, None

        keys = mx.concatenate(parts_k, axis=-2) if len(parts_k) > 1 else parts_k[0]
        values = mx.concatenate(parts_v, axis=-2) if len(parts_v) > 1 else parts_v[0]
        return keys, values

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Store KV pairs and build compressed-history view."""
        keys, values = self._normalize_io(keys, values)
        self._ensure_layout(keys, values)

        self.offset += keys.shape[-2]
        self.seq_len = self.offset

        if self.key_buffer is None:
            self.key_buffer = keys
            self.value_buffer = values
        else:
            self.key_buffer = mx.concatenate([self.key_buffer, keys], axis=-2)
            self.value_buffer = mx.concatenate([self.value_buffer, values], axis=-2)
        self._flush_buffer_if_needed()

        # Reconstruct full KV from compressed + buffer (no shadow)
        rec_k, rec_v = self._reconstruct_kv()
        return rec_k, rec_v

    @property
    def state(self):
        """Return state for mlx-lm compatibility."""
        k, v = self._reconstruct_kv()
        return [k, v] if k is not None else []

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
        """Trim n tokens from the end of the cache.

        Properly trims compressed history and/or exact buffer.
        """
        n = min(self.offset, n)
        if n == 0:
            return 0

        self.offset -= n
        self.seq_len = self.offset
        remaining_to_trim = n

        # Trim from buffer first (most recent tokens)
        if self.key_buffer is not None:
            buf_len = self.key_buffer.shape[-2]
            trim_from_buf = min(remaining_to_trim, buf_len)
            if trim_from_buf == buf_len:
                self.key_buffer = None
                self.value_buffer = None
            else:
                self.key_buffer = self.key_buffer[..., :buf_len - trim_from_buf, :]
                self.value_buffer = self.value_buffer[..., :buf_len - trim_from_buf, :]
            remaining_to_trim -= trim_from_buf

        # If we still need to trim, trim from compressed history
        if remaining_to_trim > 0 and self.key_quantized is not None:
            comp_len = self.key_quantized.indices.shape[-2]
            new_comp_len = max(0, comp_len - remaining_to_trim)
            if new_comp_len == 0:
                self.key_quantized = None
                self.value_data = None
                self.value_scales = None
                self.value_zeros = None
            else:
                self.key_quantized = SMAQQuantized(
                    indices=self.key_quantized.indices[..., :new_comp_len, :],
                    norms=self.key_quantized.norms[..., :new_comp_len],
                    bits=self.key_quantized.bits,
                )
                self.value_data = self.value_data[..., :new_comp_len, :]
                self.value_scales = self.value_scales[..., :new_comp_len, :]
                self.value_zeros = self.value_zeros[..., :new_comp_len, :]

        return n

    def empty(self):
        return self.key_buffer is None and self.key_quantized is None

    def prefill(self, keys: mx.array, values: mx.array):
        """Capture a full prefill segment (legacy interface)."""
        self.update_and_fetch(keys, values)

    def append(self, key: mx.array, value: mx.array):
        """Append a decode token (legacy interface)."""
        self.update_and_fetch(key, value)

    def attention_scores(self, query: mx.array, scale: Optional[float] = None) -> mx.array:
        """Compute attention scores against compressed history + recent exact tail."""
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)
        B, n_q_heads, T_q, D = query.shape
        n_kv_heads = None
        if self.key_quantized is not None:
            n_kv_heads = self.key_quantized.indices.shape[1]
        elif self.key_buffer is not None:
            n_kv_heads = self.key_buffer.shape[1]
        else:
            n_kv_heads = n_q_heads
        gqa_ratio = max(1, n_q_heads // max(1, n_kv_heads))
        q_grouped = query.reshape(B, n_kv_heads, gqa_ratio, T_q, D)

        scores = []
        if self.key_quantized is not None:
            head_scores = []
            for head_idx in range(n_kv_heads):
                key_q = SMAQQuantized(
                    indices=self.key_quantized.indices[:, head_idx, :, :],
                    norms=self.key_quantized.norms[:, head_idx, :],
                    bits=self.key_quantized.bits,
                )
                q_head = q_grouped[:, head_idx, :, :, :]
                head_scores.append(self.key_quantizer.attention_score(q_head, key_q, scale=scale))
            scores.append(mx.concatenate(head_scores, axis=1))
        if self.key_buffer is not None:
            buf_scores = mx.einsum(
                "bhgtd,bhnd->bhgtn",
                q_grouped.astype(mx.float32),
                self.key_buffer.astype(mx.float32),
            ) * scale
            scores.append(buf_scores.reshape(B, n_q_heads, T_q, self.key_buffer.shape[-2]))
        if not scores:
            return mx.zeros((*query.shape[:-1], 0))
        return mx.concatenate(scores, axis=-1)

    def attend(self, attn_weights: mx.array) -> mx.array:
        """Apply attention weights to compressed values + exact tail."""
        B, n_q_heads, T_q, _ = attn_weights.shape
        n_kv_heads = None
        if self.value_data is not None:
            n_kv_heads = self.value_data.shape[1]
        elif self.value_buffer is not None:
            n_kv_heads = self.value_buffer.shape[1]
        else:
            n_kv_heads = n_q_heads
        gqa_ratio = max(1, n_q_heads // max(1, n_kv_heads))
        weights_grouped = attn_weights.reshape(B, n_kv_heads, gqa_ratio, T_q, attn_weights.shape[-1])

        outputs = []
        offset = 0
        if self.value_data is not None:
            n_quant = self.value_data.shape[-2]
            w_quant = weights_grouped[..., offset : offset + n_quant]
            v_dequant = dequantize_values(
                self.value_data, self.value_scales, self.value_zeros, self.value_bits, self.value_group_size
            )
            outputs.append(mx.einsum("bhgtn,bhnd->bhgtd", w_quant.astype(mx.float32), v_dequant.astype(mx.float32)))
            offset += n_quant
        if self.value_buffer is not None:
            n_buf = self.value_buffer.shape[-2]
            w_buf = weights_grouped[..., offset : offset + n_buf]
            outputs.append(mx.einsum("bhgtn,bhnd->bhgtd", w_buf.astype(mx.float32), self.value_buffer.astype(mx.float32)))
        if not outputs:
            return mx.zeros((*attn_weights.shape[:-1], self.head_dim))
        out = sum(outputs)
        return out.reshape(B, n_q_heads, T_q, self.head_dim)

    def memory_bytes(self, include_shadow: bool = False) -> dict[str, int]:
        """Estimate memory usage of compressed path.

        include_shadow=True adds full-precision compatibility shadow.
        """
        info = {
            "compressed_keys": 0,
            "compressed_values": 0,
            "exact_buffer": 0,
            "shadow": 0,
            "total": 0,
        }

        if self.key_quantized is not None:
            info["compressed_keys"] += self.key_quantized.indices.nbytes
            info["compressed_keys"] += self.key_quantized.norms.nbytes

        if self.value_data is not None:
            info["compressed_values"] += self.value_data.nbytes
            info["compressed_values"] += self.value_scales.nbytes
            info["compressed_values"] += self.value_zeros.nbytes

        if self.key_buffer is not None:
            info["exact_buffer"] += self.key_buffer.nbytes
        if self.value_buffer is not None:
            info["exact_buffer"] += self.value_buffer.nbytes

        info["total"] = info["compressed_keys"] + info["compressed_values"] + info["exact_buffer"]
        return info

    @property
    def nbytes(self):
        """Memory usage in bytes (for compatibility with mlx-lm)."""
        return self.memory_bytes()["total"]

    @property
    def nbytes_equivalent_fp16(self):
        """FP16 equivalent memory for comparison (what the uncompressed KV would cost)."""
        if self.offset == 0:
            return 0
        # Estimate from compressed + buffer token counts
        n_tokens = self.offset
        # Infer n_kv_heads from buffer or compressed store
        n_kv_heads = 1
        bsz = 1
        if self.key_buffer is not None:
            bsz = self.key_buffer.shape[0] if self.key_buffer.ndim >= 4 else 1
            n_kv_heads = self.key_buffer.shape[-3]
        elif self.key_quantized is not None:
            bsz = self.key_quantized.indices.shape[0] if self.key_quantized.indices.ndim >= 4 else 1
            n_kv_heads = self.key_quantized.indices.shape[-3]
        # FP16 = 2 bytes per element, keys + values
        return bsz * n_kv_heads * n_tokens * self.head_dim * 2 * 2

    def capability_report(self) -> dict[str, int | str | bool]:
        return {
            "strategy_name": self.capabilities.strategy_name,
            "metric_name": self.capabilities.metric_name,
            "quantization_name": self.capabilities.quantization_name,
            "compressed_history": self.capabilities.compressed_history,
            "compressed_history_shadow_only": self.capabilities.compressed_history_shadow_only,
            "values_compressed": self.capabilities.values_compressed,
            "decode_uses_compressed_keys": self.capabilities.decode_uses_compressed_keys,
            "decode_uses_compressed_values": self.capabilities.decode_uses_compressed_values,
            "layout_adapter": getattr(self._layout_info, "adapter_name", "pending"),
            "effective_head_dim": getattr(self._layout_info, "effective_head_dim", self.head_dim),
            "offset": self.offset,
        }

    def get_seq_length(self) -> int:
        return self.offset

    def to_quantized(self, group_size=64, bits=4):
        """Compatibility stub — SMAQ cache is already compressed."""
        return self

    @classmethod
    def from_state(cls, state, meta_state):
        """Reconstruct from saved state (compatibility)."""
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj
