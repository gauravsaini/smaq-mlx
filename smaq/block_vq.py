"""SMAQ Block Vector Quantizer — MLX implementation.

Ported from PyTorch to MLX for Apple Silicon execution.

This implements the block VQ pipeline described in the SMAQ paper:
k-means in the SMAQ-shaped metric space with pre-decoded centroids
for zero-FLOP decode.

The paper's main results (Table 1) use this quantizer:
  - 8D blocks, 256 centroids = 1 bit/dim
  - Log-compressed spectral shaping with c=5.0

For a faster scalar deployment path, see quantizer.py.
"""

import math
from typing import NamedTuple, Optional

import mlx.core as mx

from smaq.ssf import build_smaq_metric


class BlockVQQuantized(NamedTuple):
    """Quantized representation from SMAQ block VQ."""

    indices: mx.array
    n_blocks: int
    block_dim: int


def _kmeans(
    data: mx.array,
    n_centroids: int,
    n_iters: int = 20,
    seed: int = 42,
) -> mx.array:
    """K-means++ initialisation followed by Lloyd iterations — MLX version."""
    n, d = data.shape
    mx.random.seed(seed)

    if n <= n_centroids:
        out = mx.zeros((n_centroids, d), dtype=data.dtype)
        out[:n] = data
        return out

    # K-means++ seeding
    indices = [mx.random.randint(0, n, (1,)).item()]
    for _ in range(n_centroids - 1):
        centroids_so_far = data[mx.array(indices)]
        dists = mx.linalg.norm(
            data[:, None, :] - centroids_so_far[None, :, :], axis=-1
        ).min(axis=1)
        probs = dists ** 2
        denom = probs.sum()
        if denom > 0:
            probs = probs / denom
        else:
            probs = mx.full((n,), 1.0 / n)
        idx = mx.random.categorical(mx.log(probs + 1e-10)).item()
        indices.append(idx)

    centroids = mx.array(data[mx.array(indices)])

    # Lloyd iterations
    for _ in range(n_iters):
        dists = mx.linalg.norm(
            data[:, None, :] - centroids[None, :, :], axis=-1
        )
        assignments = mx.argmin(dists, axis=1)
        new_centroids = mx.zeros_like(centroids)
        counts = mx.zeros(n_centroids)

        for i in range(n_centroids):
            mask = (assignments == i).astype(mx.float32)
            count = mask.sum()
            counts = counts.at[mx.array([i])].add(count)
            if count > 0:
                new_centroids = new_centroids.at[mx.array([i])].add(
                    (mask[:, None] * data).sum(axis=0) / count
                )

        mask = counts > 0
        centroids = mx.where(mask[:, None], new_centroids, centroids)

    return centroids


class SMAQBlockVQ:
    """Block vector quantizer matching the SMAQ paper — MLX version.

    Keys of dimension head_dim are partitioned into n_blocks blocks of
    block_dim dimensions. Each block is transformed by the per-block SMAQ
    metric E, quantized via k-means lookup, and reconstructed using
    pre-decoded centroids (E_inv @ centroid).

    The pre-decoded centroid design means that dequantize is a pure table
    lookup with zero extra FLOPs beyond standard VQ.
    """

    def __init__(
        self,
        head_dim: int,
        block_dim: int = 8,
        n_centroids: int = 256,
        c: float = 5.0,
    ):
        if head_dim % block_dim != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by block_dim={block_dim}"
            )

        self.head_dim = head_dim
        self.block_dim = block_dim
        self.n_blocks = head_dim // block_dim
        self.n_centroids = n_centroids
        self.c = c

        eye = mx.eye(block_dim)
        self.E_blocks = mx.tile(eye[None, :, :], (self.n_blocks, 1, 1))
        self.E_inv_blocks = mx.tile(eye[None, :, :], (self.n_blocks, 1, 1))
        self.centroids = mx.zeros((self.n_blocks, n_centroids, block_dim))
        self.decoded_centroids = mx.zeros((self.n_blocks, n_centroids, block_dim))
        self._fitted = False

    @property
    def bits_per_dim(self) -> float:
        return math.log2(self.n_centroids) / self.block_dim

    def fit(
        self,
        calibration_keys: mx.array,
        calibration_queries: mx.array,
        kmeans_iters: int = 20,
        seed: int = 42,
    ) -> "SMAQBlockVQ":
        """Calibrate the quantizer from data.

        Computes per-block query covariance Sigma_q, builds the SMAQ metric
        E via log-compressed spectral shaping, runs k-means in the shaped
        space, and stores pre-decoded centroids E_inv @ centroid.
        """
        N = calibration_keys.shape[0]
        E_list, E_inv_list, cent_list, dec_list = [], [], [], []

        for bi in range(self.n_blocks):
            bj = slice(bi * self.block_dim, (bi + 1) * self.block_dim)

            q_block = calibration_queries[:, bj].astype(mx.float32)
            Sigma_q = (q_block.T @ q_block) / N

            E, E_inv = build_smaq_metric(Sigma_q, c=self.c)
            E_list.append(E)
            E_inv_list.append(E_inv)

            k_shaped = calibration_keys[:, bj].astype(mx.float32) @ E.T

            cents = _kmeans(k_shaped, self.n_centroids, kmeans_iters, seed + bi)
            cent_list.append(cents)

            dec_list.append(cents @ E_inv.T)

        self.E_blocks = mx.stack(E_list)
        self.E_inv_blocks = mx.stack(E_inv_list)
        self.centroids = mx.stack(cent_list)
        self.decoded_centroids = mx.stack(dec_list)
        self._fitted = True
        mx.eval(self.E_blocks, self.E_inv_blocks, self.centroids, self.decoded_centroids)
        return self

    def quantize(self, k: mx.array) -> BlockVQQuantized:
        """Encode key vectors into block VQ indices."""
        batch_shape = k.shape[:-1]
        k_flat = k.reshape(-1, self.head_dim).astype(mx.float32)

        block_indices = []
        for bi in range(self.n_blocks):
            bj = slice(bi * self.block_dim, (bi + 1) * self.block_dim)
            k_shaped = k_flat[:, bj] @ self.E_blocks[bi].T
            dists = mx.linalg.norm(
                k_shaped[:, None, :] - self.centroids[bi][None, :, :], axis=-1
            )
            block_indices.append(mx.argmin(dists, axis=-1))

        indices = mx.stack(block_indices, axis=-1)
        return BlockVQQuantized(
            indices=indices.reshape(*batch_shape, self.n_blocks),
            n_blocks=self.n_blocks,
            block_dim=self.block_dim,
        )

    def dequantize(self, q: BlockVQQuantized) -> mx.array:
        """Reconstruct keys from block VQ indices — pure table lookup."""
        batch_shape = q.indices.shape[:-1]
        idx = q.indices.reshape(-1, self.n_blocks).astype(mx.uint32)

        blocks = []
        for bi in range(self.n_blocks):
            blocks.append(self.decoded_centroids[bi][idx[:, bi]])

        k_hat = mx.concatenate(blocks, axis=-1)
        return k_hat.reshape(*batch_shape, self.head_dim)

    def attention_score(
        self,
        query: mx.array,
        quantized_key: BlockVQQuantized,
        scale: Optional[float] = None,
    ) -> mx.array:
        """Compute attention scores <query, k_hat>."""
        k_hat = self.dequantize(quantized_key)
        scores = query.astype(mx.float32) @ k_hat.astype(mx.float32).T
        if scale is not None:
            scores = scores * scale
        return scores.astype(query.dtype)

    def logit_mse(self, queries: mx.array, keys: mx.array) -> float:
        """Compute held-out logit MSE: E[(q^T (k - k_hat))^2]."""
        q = self.quantize(keys)
        k_hat = self.dequantize(q)
        delta = keys.astype(mx.float32) - k_hat.astype(mx.float32)
        return ((queries.astype(mx.float32) * delta).sum(axis=-1) ** 2).mean().item()

    def __call__(self, k: mx.array) -> mx.array:
        """Quantize and immediately dequantize (for smoke tests)."""
        return self.dequantize(self.quantize(k))
