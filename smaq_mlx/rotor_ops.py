"""RotorQuant helper math for MLX runtime integration.

This module keeps the RotorQuant-specific geometric machinery small and local so
the runtime backend can use it without depending on the historical reference repo.
The implementation here follows the Cl(3,0) rotor path from turboquant_plus, but
uses a compact MSE-oriented flow that is suitable for cache integration.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

MV_DIM = 8


def gp_rotor_mv(s, p12, p13, p23, x):
    """Sparse geometric product: rotor * multivector."""
    r = np.empty_like(x)
    r[..., 0] = s * x[..., 0] - p12 * x[..., 4] - p13 * x[..., 5] - p23 * x[..., 6]
    r[..., 1] = s * x[..., 1] + p12 * x[..., 2] + p13 * x[..., 3] + p23 * x[..., 7]
    r[..., 2] = s * x[..., 2] - p12 * x[..., 1] + p23 * x[..., 3] - p13 * x[..., 7]
    r[..., 3] = s * x[..., 3] - p13 * x[..., 1] - p23 * x[..., 2] + p12 * x[..., 7]
    r[..., 4] = s * x[..., 4] + p12 * x[..., 0]
    r[..., 5] = s * x[..., 5] + p13 * x[..., 0]
    r[..., 6] = s * x[..., 6] + p23 * x[..., 0]
    r[..., 7] = s * x[..., 7] - p23 * x[..., 1] + p13 * x[..., 2] - p12 * x[..., 3]
    return r


def rotor_sandwich(s, p12, p13, p23, x):
    """Apply R x R~ using two sparse rotor-multivector products."""
    temp = gp_rotor_mv(s, p12, p13, p23, x)
    return gp_rotor_mv(s, -p12, -p13, -p23, temp)


def make_random_rotor(rng: np.random.Generator) -> np.ndarray:
    """Generate a normalized sparse rotor in Cl(3,0)."""
    bivector = rng.standard_normal(3)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    norm = np.linalg.norm(bivector)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    direction = bivector / norm
    half = angle / 2.0
    rotor = np.zeros(8, dtype=np.float32)
    rotor[0] = np.cos(half)
    rotor[4] = np.sin(half) * direction[0]
    rotor[5] = np.sin(half) * direction[1]
    rotor[6] = np.sin(half) * direction[2]
    rotor_norm = np.sqrt(rotor[0] ** 2 + rotor[4] ** 2 + rotor[5] ** 2 + rotor[6] ** 2)
    return rotor / rotor_norm


def embed_vectors(vectors: np.ndarray) -> tuple[np.ndarray, int]:
    """Embed vectors into Cl(3,0) multivectors, padding to groups of 3."""
    original_dim = vectors.shape[-1]
    pad = (3 - original_dim % 3) % 3
    if pad > 0:
        vectors = np.pad(vectors, [(0, 0)] * (vectors.ndim - 1) + [(0, pad)])
    n_groups = vectors.shape[-1] // 3
    grouped = vectors.reshape(*vectors.shape[:-1], n_groups, 3)
    mv = np.zeros((*grouped.shape[:-1], MV_DIM), dtype=vectors.dtype)
    mv[..., 1] = grouped[..., 0]
    mv[..., 2] = grouped[..., 1]
    mv[..., 3] = grouped[..., 2]
    return mv, original_dim


def extract_vectors(multivectors: np.ndarray, original_dim: int) -> np.ndarray:
    """Extract Euclidean vectors from Cl(3,0) multivectors."""
    vectors = np.stack([multivectors[..., 1], multivectors[..., 2], multivectors[..., 3]], axis=-1)
    vectors = vectors.reshape(*multivectors.shape[:-2], -1)
    return vectors[..., :original_dim]


@lru_cache(maxsize=64)
def optimal_centroids(bit_width: int, d_eff: int) -> np.ndarray:
    """Approximate optimal scalar codebook for post-rotation coefficients.

    We avoid a SciPy dependency by using a sampled Lloyd iteration on the Gaussian
    approximation of the rotated coefficient distribution.
    """
    n_centroids = 1 << bit_width
    sigma = 1.0 / math.sqrt(d_eff)
    rng = np.random.default_rng(1000 + bit_width * 17 + d_eff)
    samples = rng.normal(loc=0.0, scale=sigma, size=200000).astype(np.float32)

    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * d_eff))
        return np.array([-c, c], dtype=np.float32)
    if bit_width == 2:
        return (np.array([-1.51, -0.453, 0.453, 1.51], dtype=np.float32) / math.sqrt(d_eff)).astype(np.float32)

    quantiles = np.linspace(0.0, 1.0, n_centroids + 2, dtype=np.float32)[1:-1]
    centroids = np.quantile(samples, quantiles).astype(np.float32)
    for _ in range(20):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        indices = np.searchsorted(boundaries, samples)
        new_centroids = centroids.copy()
        for idx in range(n_centroids):
            members = samples[indices == idx]
            if members.size:
                new_centroids[idx] = float(members.mean())
        centroids = new_centroids
    return np.sort(centroids.astype(np.float32))


def nearest_centroid_indices(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return np.searchsorted(boundaries, values.ravel()).reshape(values.shape)
