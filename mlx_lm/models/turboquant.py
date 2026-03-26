"""
TurboQuant: KV cache compression via random rotation + Lloyd-Max quantization + QJL.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)
"""

import math
from dataclasses import dataclass, field
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


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack integer indices into uint32 arrays. indices shape: (..., d), values in [0, 2^bits).

    Handles cross-word boundary spanning when bit_offset + bits > 32.
    """
    *batch_dims, d = indices.shape
    n_ints = (d * bits + 31) // 32

    packed = mx.zeros((*batch_dims, n_ints), dtype=mx.uint32)
    for i in range(d):
        bit_pos = i * bits
        int_idx = bit_pos // 32
        bit_offset = bit_pos % 32
        val = indices[..., i].astype(mx.uint32)
        # Write bits that fit in the current word
        packed[..., int_idx] = packed[..., int_idx] | (val << bit_offset)
        # Handle overflow into the next word
        overflow = bit_offset + bits - 32
        if overflow > 0 and int_idx + 1 < n_ints:
            packed[..., int_idx + 1] = packed[..., int_idx + 1] | (val >> (bits - overflow))
    return packed


def unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack uint32 arrays back to integer indices. Returns shape (..., dim).

    Handles cross-word boundary spanning when bit_offset + bits > 32.
    """
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    result = []
    for i in range(dim):
        bit_pos = i * bits
        int_idx = bit_pos // 32
        bit_offset = bit_pos % 32
        val = packed[..., int_idx] >> bit_offset
        # Handle overflow from the next word
        overflow = bit_offset + bits - 32
        if overflow > 0:
            val = val | (packed[..., int_idx + 1] << (bits - overflow))
        result.append(val & mask)
    return mx.stack(result, axis=-1)


def pack_signs(signs: mx.array) -> mx.array:
    """Pack sign values (+1/-1) into uint32 bit arrays. signs shape: (..., d)."""
    *batch_dims, d = signs.shape
    n_ints = (d + 31) // 32
    bits = (signs > 0).astype(mx.uint32)
    packed = mx.zeros((*batch_dims, n_ints), dtype=mx.uint32)
    for i in range(d):
        int_idx = i // 32
        bit_offset = i % 32
        packed[..., int_idx] = packed[..., int_idx] | (bits[..., i] << bit_offset)
    return packed


def unpack_signs(packed: mx.array, dim: int) -> mx.array:
    """Unpack uint32 bit arrays back to sign values (+1/-1). Returns shape (..., dim)."""
    result = []
    for i in range(dim):
        int_idx = i // 32
        bit_offset = i % 32
        bit = (packed[..., int_idx] >> bit_offset) & mx.array(1, dtype=mx.uint32)
        sign = mx.where(bit, mx.array(1.0), mx.array(-1.0))
        result.append(sign)
    return mx.stack(result, axis=-1)


# ---------------------------------------------------------------------------
# TurboQuant Encode / Decode — Pure MLX fallback path
# ---------------------------------------------------------------------------


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant compression."""
    head_dim: int
    bits: int = 3
    seed: int = 42

    # Computed at init
    mse_bits: int = field(init=False)
    Pi_k: mx.array = field(init=False, repr=False)
    Pi_v: mx.array = field(init=False, repr=False)
    S: mx.array = field(init=False, repr=False)
    centroids_k: mx.array = field(init=False, repr=False)
    boundaries_k: mx.array = field(init=False, repr=False)
    centroids_v: mx.array = field(init=False, repr=False)
    boundaries_v: mx.array = field(init=False, repr=False)

    def __post_init__(self):
        self.mse_bits = max(self.bits - 1, 1)
        self.Pi_k = generate_rotation_matrix(self.head_dim, seed=self.seed)
        self.Pi_v = generate_rotation_matrix(self.head_dim, seed=self.seed + 1)
        self.S = generate_qjl_matrix(self.head_dim, m=self.head_dim, seed=self.seed + 2)
        self.centroids_k, self.boundaries_k = solve_lloyd_max(self.head_dim, self.mse_bits)
        self.centroids_v, self.boundaries_v = solve_lloyd_max(self.head_dim, self.bits)


def turboquant_encode(
    x: mx.array,
    config: TurboQuantConfig,
    mode: str = "key",
) -> dict:
    """
    Encode vectors using TurboQuant.

    Args:
        x: input tensor (B, n_heads, seq_len, head_dim)
        config: TurboQuantConfig
        mode: "key" (MSE + QJL) or "value" (MSE only, full bits)

    Returns:
        dict with packed idx, qjl (keys only), rnorm (keys only), vnorm
    """
    B, n_heads, seq_len, d = x.shape

    # Normalize to unit sphere
    vnorm = mx.linalg.norm(x, axis=-1, keepdims=True).astype(mx.float16)
    x_unit = x / (vnorm.astype(x.dtype) + 1e-8)

    if mode == "key":
        Pi = config.Pi_k
        centroids = config.centroids_k
        mse_bits = config.mse_bits
    else:
        Pi = config.Pi_v
        centroids = config.centroids_v
        mse_bits = config.bits

    # Stage 1: Rotate and quantize
    y = x_unit @ Pi.astype(x.dtype).T

    # Find nearest centroid per coordinate
    c = centroids.astype(x.dtype)
    diffs = mx.abs(mx.expand_dims(y, -1) - c)  # (B, H, S, d, n_levels)
    indices = mx.argmin(diffs, axis=-1).astype(mx.uint32)  # (B, H, S, d)
    y_hat = c[indices]  # (B, H, S, d)

    # Pack indices
    packed_idx = pack_indices(indices.reshape(-1, d), bits=mse_bits)
    packed_idx = packed_idx.reshape(B, n_heads, seq_len, -1)

    result = {"idx": packed_idx, "vnorm": vnorm}

    if mode == "key":
        # Stage 2: QJL on residual
        x_hat = y_hat @ Pi.astype(x.dtype)  # inverse rotation
        residual = x_unit - x_hat
        rnorm = mx.linalg.norm(residual, axis=-1, keepdims=True).astype(mx.float16)

        S = config.S.astype(x.dtype)
        projected = residual @ S.T  # (B, H, S, m)
        signs = mx.sign(projected)
        signs = mx.where(signs == 0, mx.array(1.0, dtype=signs.dtype), signs)

        packed_qjl = pack_signs(signs.reshape(-1, d))
        packed_qjl = packed_qjl.reshape(B, n_heads, seq_len, -1)

        result["qjl"] = packed_qjl
        result["rnorm"] = rnorm

    return result


def turboquant_decode_values(
    encoded: dict,
    config: TurboQuantConfig,
) -> mx.array:
    """
    Decode value vectors from compressed representation.

    Args:
        encoded: dict from turboquant_encode with mode="value"
        config: TurboQuantConfig

    Returns:
        Reconstructed values (B, n_heads, seq_len, head_dim)
    """
    packed_idx = encoded["idx"]
    vnorm = encoded["vnorm"]
    B, n_heads, seq_len, _ = packed_idx.shape
    d = config.head_dim

    # Unpack indices
    flat_packed = packed_idx.reshape(-1, packed_idx.shape[-1])
    indices = unpack_indices(flat_packed, bits=config.bits, dim=d)
    indices = indices.reshape(B, n_heads, seq_len, d)

    # Lookup centroids and inverse rotate
    c = config.centroids_v
    y_hat = c[indices].astype(mx.float16)
    Pi_v = config.Pi_v.astype(mx.float16)
    x_hat = y_hat @ Pi_v  # inverse rotation: y = x @ Pi.T, so x = y @ Pi

    # Rescale by original norm
    return x_hat * vnorm.astype(mx.float16)
