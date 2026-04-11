"""Spectral shaping utilities for SMAQ — MLX implementation.

Ported from PyTorch to MLX for Apple Silicon execution.
"""

import mlx.core as mx


def ssf_log(eigvals: mx.array, c: float = 5.0) -> mx.array:
    """Apply the log-compressed spectral shaping function from the SMAQ paper.

    The output is volume-normalized so the metric changes shape without
    introducing a global scale term.
    """
    shaped = mx.log1p(c * mx.maximum(eigvals, 0))
    log_shaped = mx.log(mx.maximum(shaped, 1e-8))
    log_shaped = log_shaped - mx.mean(log_shaped)
    return mx.exp(log_shaped)


def build_smaq_metric(Sigma_q: mx.array, c: float = 5.0) -> tuple[mx.array, mx.array]:
    """Construct the shaped metric matrix E and its inverse.

    Args:
        Sigma_q: Query covariance matrix (d, d)
        c: Log-compression parameter (default 5.0)

    Returns:
        (E, E_inv) where E = V @ diag(sqrt(f(lambda))) @ V^T
    """
    evals, evecs = mx.linalg.eigh(Sigma_q, stream=mx.cpu)
    shaped_evals = ssf_log(evals, c)
    sqrt_diag = mx.diag(mx.sqrt(shaped_evals))
    inv_sqrt_diag = mx.diag(1.0 / mx.sqrt(shaped_evals))
    E = evecs @ sqrt_diag @ evecs.T
    E_inv = evecs @ inv_sqrt_diag @ evecs.T
    return E, E_inv
