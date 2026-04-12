"""RotorQuant cache integration for mlx-lm."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from smaq_mlx.rotor_ops import MV_DIM
from smaq_mlx.rotorquant import RotorQuantMSE


class RotorQuantKVCache:
    """RotorQuant-compressed KV cache for mlx-lm.

    This is a cache-friendly MSE path today: both keys and values are compressed
    with the rotor quantizer, then dequantized on fetch for standard attention.
    """

    step = 256

    def __init__(
        self,
        bits: int = 3,
        head_dim: int = 128,
        key_seed: int = 42,
        value_seed: int = 43,
    ):
        self.rotor_bits = int(bits)
        self.head_dim = int(head_dim)
        self.offset = 0
        self.key_quantizer = RotorQuantMSE(d=self.head_dim, bit_width=self.rotor_bits, seed=key_seed)
        self.value_quantizer = RotorQuantMSE(d=self.head_dim, bit_width=self.rotor_bits, seed=value_seed)
        self._key_indices = None
        self._key_norms = None
        self._value_indices = None
        self._value_norms = None
        self._stored_capacity = 0
        self._coeff_dim = ((self.head_dim + 2) // 3) * MV_DIM
        self._values_per_int = max(1, 32 // self.rotor_bits)
        self._packed_dim = math.ceil(self._coeff_dim / self._values_per_int)

    @staticmethod
    def _to_numpy(array: mx.array, dtype=np.float32) -> np.ndarray:
        return np.asarray(array.tolist(), dtype=dtype)

    def _pack_indices(self, indices: np.ndarray) -> np.ndarray:
        padded = np.zeros((*indices.shape[:-1], self._packed_dim, self._values_per_int), dtype=np.uint32)
        padded.reshape(*indices.shape[:-1], -1)[..., : indices.shape[-1]] = indices.astype(np.uint32)
        shifts = (np.arange(self._values_per_int, dtype=np.uint32) * self.rotor_bits).reshape(
            *((1,) * (padded.ndim - 1)), self._values_per_int
        )
        return np.sum(padded << shifts, axis=-1, dtype=np.uint32)

    def _unpack_indices(self, packed: np.ndarray) -> np.ndarray:
        shifts = (np.arange(self._values_per_int, dtype=np.uint32) * self.rotor_bits).reshape(
            *((1,) * packed.ndim), self._values_per_int
        )
        expanded = np.expand_dims(packed.astype(np.uint32), axis=-1)
        mask = np.uint32((1 << self.rotor_bits) - 1)
        unpacked = ((expanded >> shifts) & mask).reshape(*packed.shape[:-1], -1)
        return unpacked[..., : self._coeff_dim].astype(np.uint8)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        bsz, n_kv_heads, steps, dim = keys.shape
        previous = self.offset
        key_np = self._to_numpy(keys, dtype=np.float32).reshape(-1, dim)
        value_np = self._to_numpy(values, dtype=np.float32).reshape(-1, dim)
        key_quant = self.key_quantizer.quantize(key_np)
        value_quant = self.value_quantizer.quantize(value_np)
        key_indices = self._pack_indices(key_quant.indices.reshape(bsz, n_kv_heads, steps, -1))
        key_norms = np.array(key_quant.norms, dtype=np.float32).reshape(bsz, n_kv_heads, steps, 1)
        value_indices = self._pack_indices(value_quant.indices.reshape(bsz, n_kv_heads, steps, -1))
        value_norms = np.array(value_quant.norms, dtype=np.float32).reshape(bsz, n_kv_heads, steps, 1)

        if self._key_indices is None or (previous + steps) > self._stored_capacity:
            self._expand_storage(bsz, n_kv_heads, steps, keys.dtype)

        self._key_indices[..., previous:previous + steps, :] = mx.array(key_indices, dtype=mx.uint32)
        self._key_norms[..., previous:previous + steps, :] = mx.array(key_norms, dtype=keys.dtype)
        self._value_indices[..., previous:previous + steps, :] = mx.array(value_indices, dtype=mx.uint32)
        self._value_norms[..., previous:previous + steps, :] = mx.array(value_norms, dtype=values.dtype)
        self.offset += steps

        all_key_idx = self._unpack_indices(self._to_numpy(self._key_indices[..., :self.offset, :], dtype=np.uint32)).reshape(
            -1, self._coeff_dim
        )
        all_key_norms = self._to_numpy(self._key_norms[..., :self.offset, :], dtype=np.float32).reshape(-1)
        all_value_idx = self._unpack_indices(
            self._to_numpy(self._value_indices[..., :self.offset, :], dtype=np.uint32)
        ).reshape(-1, self._coeff_dim)
        all_value_norms = self._to_numpy(self._value_norms[..., :self.offset, :], dtype=np.float32).reshape(-1)

        deq_keys = self.key_quantizer.dequantize(all_key_idx, all_key_norms)
        deq_values = self.value_quantizer.dequantize(all_value_idx, all_value_norms)
        deq_keys = mx.array(deq_keys.reshape(bsz, n_kv_heads, self.offset, dim), dtype=keys.dtype)
        deq_values = mx.array(deq_values.reshape(bsz, n_kv_heads, self.offset, dim), dtype=values.dtype)
        return deq_keys, deq_values

    def _expand_storage(self, bsz, n_kv_heads, new_steps, dtype):
        alloc_steps = ((self.step + new_steps - 1) // self.step) * self.step
        shape = (bsz, n_kv_heads, alloc_steps)
        if self._key_indices is not None and self.offset > 0:
            old_k = self._key_indices[..., :self.offset, :]
            old_kn = self._key_norms[..., :self.offset, :]
            old_v = self._value_indices[..., :self.offset, :]
            old_vn = self._value_norms[..., :self.offset, :]
            self._key_indices = mx.concatenate([old_k, mx.zeros((*shape, self._packed_dim), dtype=mx.uint32)], axis=2)
            self._key_norms = mx.concatenate([old_kn, mx.zeros((*shape, 1), dtype=dtype)], axis=2)
            self._value_indices = mx.concatenate([old_v, mx.zeros((*shape, self._packed_dim), dtype=mx.uint32)], axis=2)
            self._value_norms = mx.concatenate([old_vn, mx.zeros((*shape, 1), dtype=dtype)], axis=2)
        else:
            self._key_indices = mx.zeros((*shape, self._packed_dim), dtype=mx.uint32)
            self._key_norms = mx.zeros((*shape, 1), dtype=dtype)
            self._value_indices = mx.zeros((*shape, self._packed_dim), dtype=mx.uint32)
            self._value_norms = mx.zeros((*shape, 1), dtype=dtype)
        self._stored_capacity = self._key_indices.shape[2]

    def size(self):
        return self.offset

    def empty(self):
        return self._key_indices is None

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
        if self._key_indices is None:
            return []
        return [
            self._key_indices[..., :self.offset, :],
            self._key_norms[..., :self.offset, :],
            self._value_indices[..., :self.offset, :],
            self._value_norms[..., :self.offset, :],
        ]

    @state.setter
    def state(self, value):
        if value is not None and value:
            (self._key_indices, self._key_norms, self._value_indices, self._value_norms) = value
            self.offset = self._key_indices.shape[2]
            self._stored_capacity = self._key_indices.shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.rotor_bits, self.head_dim)))

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0])
        self.rotor_bits = int(float(value[1]))
        self.head_dim = int(value[2])

    @property
    def nbytes(self):
        if self._key_indices is None:
            return 0
        arrays = [
            self._key_indices[..., :self.offset, :],
            self._key_norms[..., :self.offset, :],
            self._value_indices[..., :self.offset, :],
            self._value_norms[..., :self.offset, :],
        ]
        return int(sum(arr.nbytes for arr in arrays))

    @property
    def uncompressed_nbytes(self):
        if self._key_indices is None:
            return 0
        bsz = int(self._key_indices.shape[0])
        n_kv_heads = int(self._key_indices.shape[1])
        return bsz * n_kv_heads * int(self.offset) * int(self.head_dim) * 4

    @property
    def compression_ratio(self):
        return float(self.uncompressed_nbytes / self.nbytes) if self.nbytes else 0.0
