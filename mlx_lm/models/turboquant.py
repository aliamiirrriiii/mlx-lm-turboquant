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
    """Pack integer indices into uint32 arrays. Vectorized — no Python loops."""
    *batch_dims, d = indices.shape
    vals_per_word = 32 // bits
    indices_u32 = indices.astype(mx.uint32)

    if d % vals_per_word == 0 and bits in (1, 2, 4, 8, 16):
        # Fast path: bits evenly divides 32, no cross-word overflow
        n_words = d // vals_per_word
        reshaped = indices_u32.reshape(*batch_dims, n_words, vals_per_word)
        shifts = mx.arange(0, 32, bits, dtype=mx.uint32)  # [0, bits, 2*bits, ...]
        return mx.sum(reshaped << shifts, axis=-1)

    # General path for 3-bit etc: pad to next multiple of vals_per_word to avoid overflow
    # Strategy: store vals_per_word values per uint32, waste a few bits
    # For 3-bit: 10 values per word (30 bits used, 2 wasted)
    n_ints = (d + vals_per_word - 1) // vals_per_word
    pad = n_ints * vals_per_word - d
    if pad > 0:
        indices_u32 = mx.pad(indices_u32, [(0, 0)] * len(batch_dims) + [(0, pad)])
    reshaped = indices_u32.reshape(*batch_dims, n_ints, vals_per_word)
    shifts = mx.arange(0, vals_per_word * bits, bits, dtype=mx.uint32)
    return mx.sum(reshaped << shifts, axis=-1)


def unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack uint32 arrays back to integer indices. Vectorized — no Python loops."""
    vals_per_word = 32 // bits
    mask_val = (1 << bits) - 1

    if dim % vals_per_word == 0 and bits in (1, 2, 4, 8, 16):
        # Fast path: expand each word into vals_per_word values
        n_words = dim // vals_per_word
        shifts = mx.arange(0, 32, bits, dtype=mx.uint32)  # [0, bits, 2*bits, ...]
        expanded = mx.expand_dims(packed, -1) >> shifts  # (..., n_words, vpw)
        expanded = expanded & mx.array(mask_val, dtype=mx.uint32)
        return expanded.reshape(*packed.shape[:-1], dim)

    # General path for 3-bit etc: vals_per_word values per uint32 (matches pack)
    vals_per_word = 32 // bits
    shifts = mx.arange(0, vals_per_word * bits, bits, dtype=mx.uint32)
    mask = mx.array(mask_val, dtype=mx.uint32)
    expanded = (mx.expand_dims(packed, -1) >> shifts) & mask
    n_total = packed.shape[-1] * vals_per_word
    expanded = expanded.reshape(*packed.shape[:-1], n_total)
    return expanded[..., :dim]


def pack_signs(signs: mx.array) -> mx.array:
    """Pack sign values (+1/-1) into uint32 bit arrays. Vectorized."""
    *batch_dims, d = signs.shape
    n_words = (d + 31) // 32
    bits = (signs > 0).astype(mx.uint32)
    # Pad to multiple of 32
    pad = n_words * 32 - d
    if pad > 0:
        bits = mx.pad(bits, [(0, 0)] * len(batch_dims) + [(0, pad)])
    bits = bits.reshape(*batch_dims, n_words, 32)
    shifts = mx.arange(32, dtype=mx.uint32)
    return mx.sum(bits << shifts, axis=-1)


def unpack_signs(packed: mx.array, dim: int) -> mx.array:
    """Unpack uint32 bit arrays back to sign values (+1/-1). Vectorized."""
    *batch_dims, n_words = packed.shape
    shifts = mx.arange(32, dtype=mx.uint32)
    expanded = (mx.expand_dims(packed, -1) >> shifts) & mx.array(1, dtype=mx.uint32)
    expanded = expanded.reshape(*batch_dims, n_words * 32)
    expanded = expanded[..., :dim]
    return mx.where(expanded, mx.array(1.0), mx.array(-1.0))


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
    # For small codebooks (4 or 8 entries), broadcast is fine but we flatten
    # the spatial dims to keep the intermediate tensor manageable
    c = centroids.astype(x.dtype)
    orig_shape = y.shape
    flat_y = y.reshape(-1, d)  # (N, d)
    # (N, d, 1) - (n_levels,) -> (N, d, n_levels)
    diffs = mx.abs(mx.expand_dims(flat_y, -1) - c)
    indices = mx.argmin(diffs, axis=-1).astype(mx.uint32)  # (N, d)
    y_hat = c[indices]  # (N, d)
    indices = indices.reshape(orig_shape)
    y_hat = y_hat.reshape(orig_shape)

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


def turboquant_inner_product(
    queries: mx.array,
    compressed_keys: dict,
    config: TurboQuantConfig,
) -> mx.array:
    """
    Compute unbiased inner product estimates: <q, k> for each query-key pair.

    Uses: alpha_k * [<Pi@q, codebook[idx]> + gamma * sqrt(pi/2) / m * <S@q, qjl>]

    Args:
        queries: (B, n_heads, L_q, head_dim)
        compressed_keys: dict from turboquant_encode with mode="key"
        config: TurboQuantConfig

    Returns:
        scores: (B, n_heads, L_q, L_k)
    """
    packed_idx = compressed_keys["idx"]
    packed_qjl = compressed_keys["qjl"]
    rnorm = compressed_keys["rnorm"]
    vnorm = compressed_keys["vnorm"]

    B, n_heads, L_k, _ = packed_idx.shape
    _, _, L_q, d = queries.shape
    m = d  # QJL projection dimension = head_dim

    dtype = queries.dtype
    Pi = config.Pi_k.astype(dtype)
    S = config.S.astype(dtype)
    c = config.centroids_k.astype(dtype)

    # Unpack key indices: (B*H*L_k, d) -> (B, H, L_k, d)
    flat_idx = packed_idx.reshape(-1, packed_idx.shape[-1])
    key_indices = unpack_indices(flat_idx, bits=config.mse_bits, dim=d)
    key_indices = key_indices.reshape(B, n_heads, L_k, d)

    # Unpack QJL signs: (B*H*L_k, m) -> (B, H, L_k, m)
    flat_qjl = packed_qjl.reshape(-1, packed_qjl.shape[-1])
    key_signs = unpack_signs(flat_qjl, dim=m)
    key_signs = key_signs.reshape(B, n_heads, L_k, m)

    # Pre-rotate query: q_rot = q @ Pi^T (same rotation as encode)
    q_rot = queries @ Pi.T

    # Term 1: <q_rot, codebook[idx]> for each query-key pair
    key_centroids = c[key_indices]  # (B, H, L_k, d)
    # (B, H, L_q, d) @ (B, H, d, L_k) -> (B, H, L_q, L_k)
    term1 = q_rot @ key_centroids.swapaxes(-2, -1)

    # Term 2: QJL correction
    # q_proj = q @ S^T -> (B, H, L_q, m)
    q_proj = queries @ S.T
    # (B, H, L_q, m) @ (B, H, m, L_k) -> (B, H, L_q, L_k)
    qjl_dot = q_proj @ key_signs.swapaxes(-2, -1)

    correction_scale = math.sqrt(math.pi / 2) / m
    # rnorm: (B, H, L_k, 1) -> (B, H, 1, L_k)
    rnorm_t = rnorm.astype(dtype).swapaxes(-2, -1)
    term2 = correction_scale * rnorm_t * qjl_dot

    # Scale by key vector norm: vnorm (B, H, L_k, 1) -> (B, H, 1, L_k)
    vnorm_t = vnorm.astype(dtype).swapaxes(-2, -1)

    return vnorm_t * (term1 + term2)


# ---------------------------------------------------------------------------
# Metal Kernels
# ---------------------------------------------------------------------------

def _make_encode_kernel():
    """Fused encode kernel: normalize -> rotate -> quantize -> pack -> (QJL for keys)."""
    if not mx.metal.is_available():
        return None

    source = """
        uint vec_id = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;

        if (vec_id >= N) return;

        // Pointers for this vector
        const device T* vec_in = input + vec_id * D;
        device uint32_t* idx_out = out_idx + vec_id * IDX_STRIDE;
        device T* vnorm_out = out_vnorm + vec_id;
        device T* rnorm_out = out_rnorm + vec_id;

        // Zero the packed outputs (all threads contribute via atomic OR)
        for (uint i = tid; i < IDX_STRIDE; i += THREADS) {
            idx_out[i] = 0;
        }

        device uint32_t* qjl_out = out_qjl + vec_id * QJL_STRIDE;
        if (IS_KEY) {
            for (uint i = tid; i < QJL_STRIDE; i += THREADS) {
                qjl_out[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Step 1: Compute vector norm
        float norm_sq = 0.0f;
        for (uint i = tid; i < D; i += THREADS) {
            float v = static_cast<float>(vec_in[i]);
            norm_sq += v * v;
        }
        norm_sq = simd_sum(norm_sq);
        float vnorm = sqrt(norm_sq);
        float inv_norm = (vnorm > 1e-8f) ? (1.0f / vnorm) : 0.0f;

        if (tid == 0) {
            vnorm_out[0] = static_cast<T>(vnorm);
        }

        // Step 2: Rotate normalized vector: y[i] = sum_j Pi[i,j] * x_norm[j]
        // Step 3: Quantize each coordinate to nearest centroid and pack
        // Step 4: Also compute y_hat (quantized rotated) for residual
        float residual_norm_sq = 0.0f;

        for (uint i = tid; i < D; i += THREADS) {
            // Rotate: y_i = dot(Pi[i,:], x_norm)
            float y_i = 0.0f;
            for (uint j = 0; j < D; j++) {
                y_i += static_cast<float>(Pi[i * D + j]) * static_cast<float>(vec_in[j]) * inv_norm;
            }

            // Find nearest centroid
            uint best_c = 0;
            float best_dist = 1e10f;
            for (uint c = 0; c < N_LEVELS; c++) {
                float dist = abs(y_i - centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }

            // Pack index bits (atomic OR for thread safety)
            uint bit_pos = i * BITS;
            uint word = bit_pos / 32;
            uint offset = bit_pos % 32;
            atomic_fetch_or_explicit(
                (device atomic_uint*)&idx_out[word],
                (best_c << offset), memory_order_relaxed);
            // Handle overflow across word boundary
            if (offset + BITS > 32) {
                uint overflow_bits = offset + BITS - 32;
                atomic_fetch_or_explicit(
                    (device atomic_uint*)&idx_out[word + 1],
                    (best_c >> (BITS - overflow_bits)), memory_order_relaxed);
            }

            if (IS_KEY) {
                // Compute x_hat[i] = dot(Pi^T[i,:], y_hat) = dot(Pi[:,i], y_hat)
                // But we need the full y_hat vector for this...
                // Instead, store y_hat_i = centroids[best_c] for use in residual
                // We'll compute the residual in rotated space: r_rot = y - y_hat
                // ||r||^2 = ||r_rot||^2 (rotation preserves norm)
                float r_rot_i = y_i - centroids[best_c];
                residual_norm_sq += r_rot_i * r_rot_i;

                // QJL: project residual through S and take sign
                // proj_i = dot(S[i,:], residual) = dot(S[i,:], Pi^T @ r_rot)
                // Since S and Pi are both random, we can project r_rot directly
                // through (S @ Pi^T) which is also random Gaussian
                // For simplicity and correctness, project r_rot through S
                // (this changes the projection matrix but preserves unbiasedness)
                float proj_i = 0.0f;
                for (uint j = 0; j < D; j++) {
                    // We need residual in original space for exact match with fallback
                    // residual_orig[k] = sum_j Pi^T[k,j] * r_rot[j] = sum_j Pi[j,k] * r_rot[j]
                    // proj_i = sum_k S[i,k] * residual_orig[k]
                    //        = sum_k S[i,k] * sum_j Pi[j,k] * r_rot[j]
                    // This is a double loop which is too expensive per-thread.
                    // Instead we project in rotated space: use S @ Pi^T as the projection
                    proj_i += static_cast<float>(S[i * D + j]) * static_cast<float>(Pi[j * D + tid]);
                    // This doesn't work either — we need the full r_rot vector.
                }
                // Fallback: just compute sign of S[i,:] @ r_rot directly
                // We need all r_rot values, but each thread only computes one.
                // This requires shared memory or a different approach.
            }
        }

        if (IS_KEY) {
            residual_norm_sq = simd_sum(residual_norm_sq);
            if (tid == 0) {
                rnorm_out[0] = static_cast<T>(sqrt(residual_norm_sq));
            }
        }
    """

    # The full QJL sign computation needs all residual values simultaneously,
    # which is hard in a single-pass kernel without threadgroup memory.
    # Use a practical two-kernel approach: kernel 1 does rotate+quantize+pack+norms,
    # kernel 2 does QJL sign projection.
    # For now, return None to use pure MLX fallback for encode.
    # The attention_scores kernel is where the real speedup is.
    return None


def _make_attention_scores_kernel():
    """
    Compute attention scores directly on compressed keys.

    Each thread processes one query against one key position.
    The query rotation and projection are done in pure MLX (fast matmul),
    then this kernel does the per-key scoring in packed space.
    """
    if not mx.metal.is_available():
        return None

    source = """
        // Grid: (N_Q * N_K, 1, 1) — one thread per (query, key) pair
        // Or better: grid.x = query index, grid.y = key block
        uint q_idx = thread_position_in_grid.x;
        uint k_idx = thread_position_in_grid.y;
        uint tid = thread_position_in_threadgroup.x;

        if (q_idx >= N_Q || k_idx >= N_K) return;

        // q_rot and q_proj are pre-computed (rotated/projected queries)
        const device T* q_rot_ptr = q_rot + q_idx * D;
        const device T* q_proj_ptr = q_proj + q_idx * D;

        const device uint32_t* k_idx_ptr = keys_idx + k_idx * IDX_STRIDE;
        const device uint32_t* k_qjl_ptr = keys_qjl + k_idx * QJL_STRIDE;

        // Term 1: dot(q_rot, codebook[idx]) — iterate over coordinates
        // Packing: vals_per_word values per uint32, no cross-word overflow
        constexpr uint VPW = 32 / BITS;
        float term1 = 0.0f;
        for (uint i = 0; i < D; i++) {
            uint w = i / VPW;
            uint pos_in_word = i % VPW;
            uint offset = pos_in_word * BITS;
            uint mask = (1u << BITS) - 1u;
            uint c_idx = (k_idx_ptr[w] >> offset) & mask;
            term1 += static_cast<float>(q_rot_ptr[i]) * centroids[c_idx];
        }

        // Term 2: QJL popcount dot product
        // Convert q_proj to sign bits, then XOR with stored signs
        float qjl_dot = 0.0f;
        for (uint w = 0; w < QJL_STRIDE; w++) {
            uint32_t k_word = k_qjl_ptr[w];
            uint32_t q_word = 0;
            // Build q_proj sign bits for this word
            for (uint b = 0; b < 32 && (w * 32 + b) < D; b++) {
                float q_val = static_cast<float>(q_proj_ptr[w * 32 + b]);
                if (q_val >= 0.0f) q_word |= (1u << b);
            }
            uint32_t xor_val = q_word ^ k_word;
            int disagree = popcount(xor_val);
            int bits_used = min(32u, D - w * 32);
            qjl_dot += static_cast<float>(bits_used - 2 * disagree);
        }

        float alpha = static_cast<float>(keys_vnorm[k_idx]);
        float gamma = static_cast<float>(keys_rnorm[k_idx]);
        float corr_scale = static_cast<float>(correction_scale[0]);
        float correction = corr_scale * gamma * qjl_dot;
        float score = alpha * (term1 + correction);

        out[q_idx * N_K + k_idx] = score;
    """

    return mx.fast.metal_kernel(
        name="turboquant_attn_scores",
        input_names=["q_rot", "q_proj", "keys_idx", "keys_qjl",
                      "keys_vnorm", "keys_rnorm", "centroids", "correction_scale"],
        output_names=["out"],
        source=source,
    )


def _make_dequantize_values_kernel():
    """Fused value dequantization: unpack indices -> lookup centroids -> inverse rotate -> scale."""
    if not mx.metal.is_available():
        return None

    source = """
        uint vec_id = thread_position_in_grid.x;
        uint coord = thread_position_in_grid.y;

        if (vec_id >= N || coord >= D) return;

        const device uint32_t* idx_ptr = values_idx + vec_id * IDX_STRIDE;

        // Unpack index for this coordinate
        uint bit_pos = coord * BITS;
        uint word = bit_pos / 32;
        uint offset = bit_pos % 32;
        uint mask = (1u << BITS) - 1u;
        uint c_idx = (idx_ptr[word] >> offset) & mask;
        if (offset + BITS > 32) {
            uint overflow = offset + BITS - 32;
            c_idx |= (idx_ptr[word + 1] << (BITS - overflow)) & mask;
        }

        // y_hat[coord] = centroids[c_idx]
        // x_hat[coord_out] = sum_j Pi_v^T[coord_out, j] * y_hat[j] = sum_j Pi_v[j, coord_out] * y_hat[j]
        // But we only have y_hat[coord], not the full vector.
        // Each thread computes one ROTATED coordinate; we need cooperative inverse rotation.

        // Each grid position is (vec_id, output_coord).
        // output[coord] = vnorm * sum_j Pi_v[j, coord] * centroids[idx[j]]
        // Packing: vals_per_word values per uint32, no cross-word overflow
        constexpr uint VPW = 32 / BITS;  // vals per word
        float val = 0.0f;
        for (uint j = 0; j < D; j++) {
            uint w = j / VPW;
            uint pos_in_word = j % VPW;
            uint off = pos_in_word * BITS;
            uint m = (1u << BITS) - 1u;
            uint cj = (idx_ptr[w] >> off) & m;
            val += static_cast<float>(Pi_v[j * D + coord]) * centroids[cj];
        }

        float vnorm = static_cast<float>(values_vnorm[vec_id]);
        out[vec_id * D + coord] = static_cast<T>(val * vnorm);
    """

    return mx.fast.metal_kernel(
        name="turboquant_dequant_values",
        input_names=["values_idx", "values_vnorm", "Pi_v", "centroids"],
        output_names=["out"],
        source=source,
    )


# Initialize kernels at module load
_attn_scores_kernel = _make_attention_scores_kernel()
_dequant_values_kernel = _make_dequantize_values_kernel()


# ---------------------------------------------------------------------------
# Metal wrapper functions
# ---------------------------------------------------------------------------

def turboquant_inner_product_metal(
    queries: mx.array,
    compressed_keys: dict,
    config: TurboQuantConfig,
) -> mx.array:
    """Compute attention scores using Metal kernel on compressed keys."""
    if _attn_scores_kernel is None:
        return turboquant_inner_product(queries, compressed_keys, config)

    packed_idx = compressed_keys["idx"]
    packed_qjl = compressed_keys["qjl"]
    rnorm = compressed_keys["rnorm"]
    vnorm = compressed_keys["vnorm"]

    B, n_heads, L_k, idx_stride = packed_idx.shape
    _, _, L_q, d = queries.shape
    m = d
    qjl_stride = packed_qjl.shape[-1]

    dtype = queries.dtype
    Pi = config.Pi_k.astype(dtype)
    S = config.S.astype(dtype)

    # Pre-rotate and pre-project queries in MLX (fast matmul)
    q_rot = queries @ Pi.T      # (B, H, L_q, d)
    q_proj = queries @ S.T      # (B, H, L_q, d)

    # Flatten for kernel
    N_Q = B * n_heads * L_q
    N_K = L_k
    flat_q_rot = q_rot.reshape(N_Q, d).astype(mx.float16)
    flat_q_proj = q_proj.reshape(N_Q, d).astype(mx.float16)

    # We need per-head key data. Reshape keys to (B*H, L_k, ...)
    flat_k_idx = packed_idx.reshape(B * n_heads, L_k, idx_stride)
    flat_k_qjl = packed_qjl.reshape(B * n_heads, L_k, qjl_stride)
    flat_k_vnorm = vnorm.reshape(B * n_heads, L_k)
    flat_k_rnorm = rnorm.reshape(B * n_heads, L_k)

    corr_scale_arr = mx.array([math.sqrt(math.pi / 2) / m], dtype=mx.float32)

    # Process each (batch, head) group
    all_scores = []
    for bh in range(B * n_heads):
        k_idx_bh = flat_k_idx[bh]       # (L_k, idx_stride)
        k_qjl_bh = flat_k_qjl[bh]       # (L_k, qjl_stride)
        k_vnorm_bh = flat_k_vnorm[bh].astype(mx.float16)  # (L_k,)
        k_rnorm_bh = flat_k_rnorm[bh].astype(mx.float16)  # (L_k,)

        # Queries for this (batch, head): depends on L_q per head
        q_start = bh * L_q
        q_end = q_start + L_q
        q_rot_bh = flat_q_rot[q_start:q_end]    # (L_q, d)
        q_proj_bh = flat_q_proj[q_start:q_end]  # (L_q, d)

        outputs = _attn_scores_kernel(
            inputs=[q_rot_bh, q_proj_bh, k_idx_bh, k_qjl_bh,
                    k_vnorm_bh, k_rnorm_bh, config.centroids_k, corr_scale_arr],
            template=[
                ("T", mx.float16),
                ("D", d),
                ("BITS", config.mse_bits),
                ("IDX_STRIDE", idx_stride),
                ("QJL_STRIDE", qjl_stride),
                ("N_Q", L_q),
                ("N_K", N_K),
            ],
            grid=(L_q, N_K, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(L_q, N_K)],
            output_dtypes=[mx.float32],
        )
        all_scores.append(outputs[0])

    scores = mx.stack(all_scores, axis=0)  # (B*H, L_q, L_k)
    return scores.reshape(B, n_heads, L_q, L_k)


def turboquant_decode_values_metal(
    encoded: dict,
    config: TurboQuantConfig,
) -> mx.array:
    """Decode values using Metal kernel."""
    if _dequant_values_kernel is None:
        return turboquant_decode_values(encoded, config)

    packed_idx = encoded["idx"]
    vnorm = encoded["vnorm"]
    B, n_heads, seq_len, idx_stride = packed_idx.shape
    d = config.head_dim

    N = B * n_heads * seq_len
    flat_idx = packed_idx.reshape(N, idx_stride)
    flat_vnorm = vnorm.reshape(N).astype(mx.float16)

    outputs = _dequant_values_kernel(
        inputs=[flat_idx, flat_vnorm, config.Pi_v, config.centroids_v],
        template=[
            ("T", mx.float16),
            ("D", d),
            ("BITS", config.bits),
            ("IDX_STRIDE", idx_stride),
            ("N", N),
        ],
        grid=(N, d, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(N, d)],
        output_dtypes=[mx.float16],
    )

    return outputs[0].reshape(B, n_heads, seq_len, d)


# ---------------------------------------------------------------------------
# Auto-dispatch: Metal if available, else pure MLX
# ---------------------------------------------------------------------------

def encode(x: mx.array, config: TurboQuantConfig, mode: str = "key") -> dict:
    """Encode vectors. Uses pure MLX (Metal encode kernel deferred to v2)."""
    return turboquant_encode(x, config, mode=mode)


def decode_values(encoded: dict, config: TurboQuantConfig) -> mx.array:
    """Decode values with auto Metal dispatch."""
    if _dequant_values_kernel is not None:
        return turboquant_decode_values_metal(encoded, config)
    return turboquant_decode_values(encoded, config)


def inner_product(
    queries: mx.array, compressed: dict, config: TurboQuantConfig
) -> mx.array:
    """Compute inner products. Uses vectorized pure MLX (faster than Metal per-head loop)."""
    return turboquant_inner_product(queries, compressed, config)
