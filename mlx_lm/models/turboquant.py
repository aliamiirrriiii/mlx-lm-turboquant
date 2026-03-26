"""
TurboQuant: KV cache compression via random rotation + Lloyd-Max quantization + QJL.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
from scipy import integrate


def _gaussian_approx_pdf(x: float, d: int) -> float:
    """Gaussian approximation N(0, 1/d) for rotated coordinate distribution. Accurate for d >= 64."""
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[mx.array, mx.array]:
    """
    Solve Lloyd-Max optimal scalar quantizer for the coordinate distribution
    after random rotation of a d-dimensional unit vector.

    Args:
        d: vector dimension
        bits: number of quantization bits
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        centroids: mx.array of shape (2^bits,) — sorted optimal centroids
        boundaries: mx.array of shape (2^bits - 1,) — decision boundaries
    """
    n_levels = 2 ** bits
    pdf = lambda x: _gaussian_approx_pdf(x, d)
    sigma = 1.0 / math.sqrt(d)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = integrate.quad(pdf, a, b)
            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        mx.array(centroids, dtype=mx.float32),
        mx.array(boundaries, dtype=mx.float32),
    )


def generate_rotation_matrix(d: int, seed: int = 42) -> mx.array:
    """Generate random orthogonal rotation matrix via QR decomposition of Gaussian matrix."""
    rng = np.random.RandomState(seed)
    G = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(G)
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign[np.newaxis, :]
    return mx.array(Q, dtype=mx.float16)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: int = 42) -> mx.array:
    """Generate random Gaussian projection matrix for QJL. Shape (m, d)."""
    if m is None:
        m = d
    rng = np.random.RandomState(seed)
    S = rng.randn(m, d).astype(np.float32)
    return mx.array(S, dtype=mx.float16)
