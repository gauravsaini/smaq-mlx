"""SMAQ capture module — MLX implementation.

Ported from PyTorch to MLX for Apple Silicon execution.

Keeps decode cheap by buffering recent exact KV tokens and only flushing
older chunks into the compressed SMAQ store.
"""

from typing import Optional, TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from smaq_mlx.store import CompressedKVStore


class RingBuffer:
    """Fixed-size ring buffer for recent exact KV tokens."""

    def __init__(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: type = mx.float32,
    ):
        self.capacity = capacity
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self._k = mx.zeros((capacity, num_kv_heads, head_dim), dtype=dtype)
        self._v = mx.zeros((capacity, num_kv_heads, head_dim), dtype=dtype)
        self._pos = 0
        self._total_written = 0

    @property
    def size(self) -> int:
        return self._pos

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(
        self,
        key: mx.array,
        value: mx.array,
        num_tokens: int,
    ) -> Optional[tuple[mx.array, mx.array]]:
        """Append tokens and return overflow when the ring wraps."""
        overflow_k_parts = []
        overflow_v_parts = []
        offset = 0
        remaining = num_tokens

        while remaining > 0:
            space = self.capacity - self._pos
            if space <= 0:
                overflow_k_parts.append(self._k[: self._pos])
                overflow_v_parts.append(self._v[: self._pos])
                self._pos = 0
                space = self.capacity

            n_write = min(remaining, space)
            self._k[self._pos : self._pos + n_write] = key[offset: offset + n_write]
            self._v[self._pos : self._pos + n_write] = value[offset: offset + n_write]
            self._pos += n_write
            offset += n_write
            remaining -= n_write

        self._total_written += num_tokens

        if overflow_k_parts:
            return mx.concatenate(overflow_k_parts, axis=0), mx.concatenate(overflow_v_parts, axis=0)
        return None

    def drain(self) -> Optional[tuple[mx.array, mx.array]]:
        """Return buffered tokens and reset."""
        if self._pos == 0:
            return None
        k = self._k[: self._pos]
        v = self._v[: self._pos]
        self._pos = 0
        return k, v

    def peek(self) -> Optional[tuple[mx.array, mx.array]]:
        """Read current buffered tokens without draining."""
        if self._pos == 0:
            return None
        return self._k[: self._pos], self._v[: self._pos]

    def reset(self):
        self._pos = 0
        self._total_written = 0


class KVCaptureEngine:
    """Bulk capture and decode ingestion for the SMAQ compressed store."""

    def __init__(
        self,
        store: "CompressedKVStore",
        ring_capacity: int = 128,
        dtype: type = mx.float32,
    ):
        self.store = store
        self.ring = RingBuffer(
            capacity=ring_capacity,
            num_kv_heads=store.num_kv_heads,
            head_dim=store.head_dim,
            dtype=dtype,
        )
        self._prefill_done = False

    @property
    def total_compressed_tokens(self) -> int:
        return self.store.num_tokens

    @property
    def total_buffered_tokens(self) -> int:
        return self.ring.size

    @property
    def total_tokens(self) -> int:
        return self.total_compressed_tokens + self.total_buffered_tokens

    def ingest_prefill(self, key: mx.array, value: mx.array, num_tokens: int):
        """Ingest a prefill block, leaving only the recent tail exact."""
        if num_tokens <= self.ring.capacity:
            self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            n_compress = num_tokens - self.ring.capacity
            self.store.append_chunk(key[:n_compress], value[:n_compress])
            self.ring.write(key[n_compress:num_tokens], value[n_compress:num_tokens], self.ring.capacity)
        self._prefill_done = True

    def ingest_decode(self, key: mx.array, value: mx.array, num_tokens: int):
        """Append cheap decode tokens and flush ring overflow to the store."""
        overflow = self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        if overflow is not None:
            self.store.append_chunk(*overflow)

    def flush(self):
        """Force-flush the ring buffer into compressed storage."""
        data = self.ring.drain()
        if data is not None:
            self.store.append_chunk(*data)

    def reset(self):
        self.ring.reset()
        self.store.reset()
        self._prefill_done = False
