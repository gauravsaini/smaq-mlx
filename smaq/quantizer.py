"""SMAQ scalar quantizer — fast deployment path, MLX implementation.

Ported from PyTorch to MLX for Apple Silicon execution.

This module provides a per-dimension scalar quantizer that applies the SMAQ
spectral metric before quantising each coordinate independently. It mirrors
TurboQuant's quantizer interface but replaces the random rotation with the
calibrated query-aware metric.

Note: The paper's main experiments use block vector quantization (k-means
with 256 centroids in 8D blocks) — see block_vq.py. This scalar path is a
faster deployment alternative.
"""

import math
from typing import NamedTuple, Optional

import mlx.core as mx

from smaq.ssf import build_smaq_metric


class SMAQQuantized(NamedTuple):
    """Bit-packed SMAQ key representation."""

    indices: mx.array
    norms: mx.array
    bits: int


def _pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack scalar centroid indices into uint32 words."""
    vals_per_word = 32 // bits
    batch_shape = indices.shape[:-1]
    d = indices.shape[-1]

    padded_d = ((d + vals_per_word - 1) // vals_per_word) * vals_per_word
    if padded_d > d:
        pad_width = [(0, 0)] * (indices.ndim - 1) + [(0, padded_d - d)]
        indices = mx.pad(indices.astype(mx.uint32), pad_width, constant_values=0)
    else:
        indices = indices.astype(mx.uint32)

    reshaped = indices.reshape(*batch_shape, -1, vals_per_word)
    shifts = mx.array([i * bits for i in range(vals_per_word)], dtype=mx.uint32)
    return mx.sum((reshaped & ((1 << bits) - 1)) << shifts, axis=-1)


def _unpack_indices(packed: mx.array, bits: int, d: int) -> mx.array:
    """Unpack scalar centroid indices from uint32 words."""
    vals_per_word = 32 // bits
    mask = (1 << bits) - 1
    shifts = mx.array([i * bits for i in range(vals_per_word)], dtype=mx.uint32)
    unpacked = ((packed[..., None] >> shifts) & mask).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :d]


def _normal_ppf(probs: mx.array) -> mx.array:
    """Inverse standard normal CDF using MLX erfinv."""
    return math.sqrt(2.0) * mx.erfinv(2.0 * probs - 1.0)


class SMAQQuantizer:
    """Spectral Metric-Aware Quantizer — MLX version.

    Keys are normalized, mapped by the shaped Mahalanobis metric E,
    quantized with a scalar codebook, and reconstructed with E_inv.
    """

    def __init__(
        self,
        dim: int,
        Sigma_q: Optional[mx.array] = None,
        bits: int = 3,
        E: Optional[mx.array] = None,
        E_inv: Optional[mx.array] = None,
    ):
        self.dim = dim
        self.bits = bits

        if E is None or E_inv is None:
            if Sigma_q is None:
                Sigma_q = mx.eye(dim)
            E, E_inv = build_smaq_metric(Sigma_q.astype(mx.float32), c=5.0)

        self.E = E.astype(mx.float32)
        self.E_inv = E_inv.astype(mx.float32)

        probs = mx.linspace(0.0, 1.0, (2 ** bits) + 2)[1:-1]
        centroids = _normal_ppf(probs).astype(mx.float32)

        boundaries_p = mx.linspace(0.0, 1.0, (2 ** bits) + 1)
        clipped = mx.maximum(mx.minimum(boundaries_p, 1.0 - 1e-6), 1e-6)
        boundaries = _normal_ppf(clipped).astype(mx.float32)
        boundaries = mx.concatenate([mx.array([-float("inf")]), boundaries[1:-1], mx.array([float("inf")])])

        self.centroids = centroids
        self.boundaries = boundaries
        self.decision_boundaries = boundaries[1:-1]

        mx.eval(self.E, self.E_inv, self.centroids, self.boundaries, self.decision_boundaries)

    def fit(self, cal_keys: mx.array, cal_queries: mx.array):
        mu = mx.mean(cal_queries, axis=0)
        q_c = cal_queries - mu
        Sigma_q = (q_c.T @ q_c) / max(1, q_c.shape[0])
        E, E_inv = build_smaq_metric(Sigma_q.astype(mx.float32), c=5.0)
        self.E = E.astype(mx.float32)
        self.E_inv = E_inv.astype(mx.float32)
        mx.eval(self.E, self.E_inv)

    def rotate_query(self, query: mx.array) -> mx.array:
        """Project queries into the inverse metric space."""
        return (query.astype(mx.float32) @ self.E_inv.T).astype(query.dtype)

    def quantize(self, k: mx.array) -> SMAQQuantized:
        """Compress key vectors of shape (..., dim)."""
        norms = mx.linalg.norm(k, axis=-1)
        k_unit = k / (norms[..., None] + 1e-10)
        y = k_unit.astype(mx.float32) @ self.E.T

        result = mx.zeros(y.shape, dtype=mx.uint8)
        for i in range(self.decision_boundaries.size):
            result = result + (y > self.decision_boundaries[i]).astype(mx.uint8)

        packed = _pack_indices(result, self.bits)
        return SMAQQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: SMAQQuantized) -> mx.array:
        """Reconstruct approximate keys from the packed representation."""
        indices = _unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        k_hat = (y_hat.astype(mx.float32) @ self.E_inv.T)
        return (k_hat * q.norms[..., None]).astype(self.E.dtype)

    def attention_score(
        self,
        query: mx.array,
        quantized_key: SMAQQuantized,
        scale: Optional[float] = None,
    ) -> mx.array:
        """Compute <query, key_hat> without materializing full dequantized keys."""
        query_rot = self.rotate_query(query)
        indices = _unpack_indices(quantized_key.indices, quantized_key.bits, self.dim)
        y_hat = self.centroids[indices]
        scores = (query_rot.astype(mx.float32) @ y_hat.astype(mx.float32).T)
        scores = scores * quantized_key.norms[..., None, :]

        if scale is not None:
            scores = scores * scale
        return scores.astype(query.dtype)

    def __call__(self, k: mx.array) -> mx.array:
        """Quantize and immediately dequantize for smoke tests."""
        return self.dequantize(self.quantize(k))
