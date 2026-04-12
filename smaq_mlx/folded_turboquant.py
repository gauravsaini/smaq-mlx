"""Folded TurboQuant quantizer.

This backend keeps a single TurboQuant-style scalar codec, but folds SMAQ's
spectral shaping into the encode/decode transform for keys. The storage format
stays "one cache, one codec"; only the geometry seen by the codec changes.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx

from smaq_mlx.ssf import build_smaq_metric


def walsh_hadamard_transform(x: mx.array) -> mx.array:
    """Orthogonal Walsh-Hadamard transform on the last dimension."""
    dim = int(x.shape[-1])
    if dim & (dim - 1):
        raise ValueError(f"Hadamard transform requires power-of-two dim, got {dim}")

    y = x.astype(mx.float32)
    block = 1
    while block < dim:
        y = y.reshape(*y.shape[:-1], -1, block * 2)
        left = y[..., :, :block]
        right = y[..., :, block:]
        y = mx.concatenate([left + right, left - right], axis=-1) * (1.0 / math.sqrt(2.0))
        y = y.reshape(*x.shape[:-1], dim)
        block *= 2
    return y


def random_diagonal_sign(dim: int, seed: int = 42) -> mx.array:
    key = mx.random.key(seed)
    mask = mx.random.bernoulli(p=0.5, shape=(dim,), key=key)
    return mx.where(mask, mx.array(1.0), mx.array(-1.0)).astype(mx.float32)


def randomized_hadamard_transform(x: mx.array, signs: mx.array) -> mx.array:
    return walsh_hadamard_transform(x * signs)


def inverse_randomized_hadamard(x: mx.array, signs: mx.array) -> mx.array:
    return walsh_hadamard_transform(x) * signs


def _pack_indices(indices: mx.array, bits: int) -> mx.array:
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


def _unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    vals_per_word = 32 // bits
    mask = (1 << bits) - 1
    shifts = mx.array([i * bits for i in range(vals_per_word)], dtype=mx.uint32)
    unpacked = ((packed[..., None] >> shifts) & mask).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :dim]


def _compute_gaussian_codebook(bits: int) -> mx.array:
    codebooks = {
        1: [-0.7979, 0.7979],
        2: [-1.5104, -0.4528, 0.4528, 1.5104],
        3: [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520],
        4: [-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
            0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326],
    }
    if bits not in codebooks:
        raise ValueError(f"Unsupported bit width: {bits}. Use 1-4.")
    return mx.array(codebooks[bits], dtype=mx.float32)


class FoldedTurboQuantizer:
    """TurboQuant-style scalar codec with optional SMAQ metric folding."""

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        n_kv_heads: int = 1,
        seed: int = 42,
        Sigma_q: Optional[mx.array] = None,
        c: float = 5.0,
    ):
        self.dim = int(dim)
        self.bits = int(bits)
        self.n_kv_heads = int(n_kv_heads)
        self.seed = int(seed)
        self.c = float(c)
        self.signs = random_diagonal_sign(self.dim, seed=self.seed)
        self.centroids = _compute_gaussian_codebook(self.bits)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0
        self.scale = 1.0 / math.sqrt(self.dim)
        self.metric_fitted = False
        self.Sigma_q = Sigma_q
        self.set_metric(Sigma_q)

    def set_metric(self, Sigma_q: Optional[mx.array]):
        self.Sigma_q = Sigma_q
        if Sigma_q is None:
            self.dynamic_scale = mx.full((self.n_kv_heads, 1, self.dim), self.scale, dtype=mx.float32)
            self.dynamic_boundaries = self.boundaries[:, None, None, None] * self.dynamic_scale[None, :, :, :]
            self.dynamic_centroids = self.centroids[:, None, None, None] * self.dynamic_scale[None, :, :, :]
            self.metric_fitted = False
        else:
            I = mx.eye(self.dim, dtype=mx.float32)
            H_matrix = randomized_hadamard_transform(I, self.signs)
            
            scales = []
            for h in range(self.n_kv_heads):
                # 0. Build the SMAQ shaped metric to prevent condition-number collapse
                E, _ = build_smaq_metric(Sigma_q[h], c=self.c)
                # Shaped covariance 
                M = E @ E.T
                
                # 1. Normalize and rotate the shaped covariance
                M = M.astype(mx.float32)
                trace = mx.sum(mx.diag(M))
                normalized_M = M / mx.maximum(trace, 1e-8)
                M_rot = H_matrix @ normalized_M @ H_matrix.T
                
                # 2. Extract the true analytical variance in the rotated space
                V = mx.maximum(mx.diag(M_rot), 1e-8)
                scales.append(mx.sqrt(V))
                
            # Compute optimal per-channel bounds using true standard deviation
            self.dynamic_scale = mx.stack(scales)[:, None, :]  # Shape: [n_kv_heads, 1, dim]
            self.dynamic_boundaries = self.boundaries[:, None, None, None] * self.dynamic_scale[None, :, :, :]
            self.dynamic_centroids = self.centroids[:, None, None, None] * self.dynamic_scale[None, :, :, :]
            self.metric_fitted = True
            
        mx.eval(self.dynamic_scale, self.dynamic_boundaries, self.dynamic_centroids, self.signs)

    def fit(self, calibration_queries: mx.array):
        # queries shape: [n_kv_heads, N, dim]
        q = calibration_queries.astype(mx.float32)
        q_centered = q - mx.mean(q, axis=1, keepdims=True)
        # Vectorized covariance per head
        Sigma_q = (q_centered.transpose(0, 2, 1) @ q_centered) / max(1, q_centered.shape[1])
        self.set_metric(Sigma_q)

    def quantize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = x.astype(mx.float32)
        norms = mx.linalg.norm(x, axis=-1)
        safe_norms = mx.maximum(norms, 1e-8)
        x_unit = x / safe_norms[..., None]
        
        # 1. Hadamard Rotation standard
        x_rot = randomized_hadamard_transform(x_unit, self.signs)
        
        # 2. Dynamic Codebook Normalization: mathematically safe index selection 
        #    using per-channel pre-scaled Gaussian boundaries directly.
        indices = mx.zeros(x_rot.shape, dtype=mx.uint8)
        for i in range(len(self.boundaries)):
            indices = indices + (x_rot > self.dynamic_boundaries[i]).astype(mx.uint8)
        packed = _pack_indices(indices, self.bits)
        return packed, norms

    def dequantize(self, packed: mx.array, norms: mx.array) -> mx.array:
        indices = _unpack_indices(packed, self.bits, self.dim).astype(mx.int32)
        
        # 1 & 2. Reconstruct using the dynamic dimension-scaled codebook
        # We index from standard centroids then scale, which is arithmetically equivalent 
        # to self.dynamic_centroids[indices, d] but preserves MLX fast inner-indexing logic.
        y_hat = self.centroids[indices]
        x_shaped_hat = y_hat * self.dynamic_scale
        
        # 3. Inverse Hadamard
        x_unit_hat = inverse_randomized_hadamard(x_shaped_hat, self.signs)
        
        return (x_unit_hat * norms[..., None]).astype(mx.float32)
