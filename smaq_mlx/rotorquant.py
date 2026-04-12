"""RotorQuant-inspired scalar compressor for MLX KV caches.

This is the cache-friendly MSE path: rotor decorrelation in Cl(3,0), scalar
quantization over the rotated multivector coefficients, and inverse rotor
reconstruction. It gives us a real RotorQuant runtime row without waiting for the
full fused Metal/QJL stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smaq_mlx.rotor_ops import (
    MV_DIM,
    embed_vectors,
    extract_vectors,
    make_random_rotor,
    nearest_centroid_indices,
    optimal_centroids,
    rotor_sandwich,
)


@dataclass
class RotorQuantized:
    indices: np.ndarray
    norms: np.ndarray
    bit_width: int


class RotorQuantMSE:
    """MSE-style RotorQuant suitable for cache compression."""

    def __init__(self, d: int, bit_width: int, seed: int = 42):
        self.d = d
        self.bit_width = int(bit_width)
        self.n_groups = (d + 2) // 3
        self.centroids = optimal_centroids(self.bit_width, max(self.n_groups * MV_DIM, 64))

        rng = np.random.default_rng(seed)
        self.rotors_s = np.empty(self.n_groups, dtype=np.float32)
        self.rotors_b12 = np.empty(self.n_groups, dtype=np.float32)
        self.rotors_b13 = np.empty(self.n_groups, dtype=np.float32)
        self.rotors_b23 = np.empty(self.n_groups, dtype=np.float32)
        for g in range(self.n_groups):
            rotor = make_random_rotor(rng)
            self.rotors_s[g] = rotor[0]
            self.rotors_b12[g] = rotor[4]
            self.rotors_b13[g] = rotor[5]
            self.rotors_b23[g] = rotor[6]

    def _apply_rotors(self, mv: np.ndarray) -> np.ndarray:
        result = np.empty_like(mv)
        for g in range(self.n_groups):
            result[:, g] = rotor_sandwich(
                self.rotors_s[g],
                self.rotors_b12[g],
                self.rotors_b13[g],
                self.rotors_b23[g],
                mv[:, g],
            )
        return result

    def _unapply_rotors(self, mv: np.ndarray) -> np.ndarray:
        result = np.empty_like(mv)
        for g in range(self.n_groups):
            result[:, g] = rotor_sandwich(
                self.rotors_s[g],
                -self.rotors_b12[g],
                -self.rotors_b13[g],
                -self.rotors_b23[g],
                mv[:, g],
            )
        return result

    def quantize(self, x: np.ndarray) -> RotorQuantized:
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        x = x.astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=-1).astype(np.float32)
        safe_norms = np.where(norms > 1e-10, norms, 1.0).astype(np.float32)
        x_unit = x / safe_norms[:, np.newaxis]

        mv, _ = embed_vectors(x_unit)
        mv_rot = self._apply_rotors(mv)
        flat = mv_rot.reshape(mv_rot.shape[0], -1)
        indices = nearest_centroid_indices(flat, self.centroids).astype(np.uint8)
        if single:
            indices = indices[0]
            norms = np.array(norms[0], dtype=np.float32)
        return RotorQuantized(indices=indices, norms=norms, bit_width=self.bit_width)

    def dequantize(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        single = indices.ndim == 1
        if single:
            indices = indices[np.newaxis, :]
            norms = np.array([norms], dtype=np.float32)
        values = self.centroids[indices]
        mv = values.reshape(values.shape[0], self.n_groups, MV_DIM)
        mv_recon = self._unapply_rotors(mv)
        x = extract_vectors(mv_recon, self.d)
        x = x * norms[:, np.newaxis]
        if single:
            return x[0]
        return x
