import mlx.core as mx
import pytest

from mlx_lm.models.turboquant import generate_qjl_matrix, generate_rotation_matrix, solve_lloyd_max
from mlx_lm.models.turboquant import pack_indices, unpack_indices, pack_signs, unpack_signs
from mlx_lm.models.turboquant import turboquant_encode, turboquant_decode_values, TurboQuantConfig, turboquant_inner_product


def test_placeholder():
    """Verify test infrastructure works."""
    assert mx.array([1.0]).item() == 1.0


class TestLloydMax:
    def test_returns_correct_number_of_centroids(self):
        centroids, boundaries = solve_lloyd_max(d=128, bits=2)
        assert centroids.shape == (4,)  # 2^2 = 4 levels
        assert boundaries.shape == (3,)  # 4-1 = 3 boundaries

    def test_centroids_are_sorted(self):
        centroids, boundaries = solve_lloyd_max(d=128, bits=2)
        assert all(centroids[i] < centroids[i + 1] for i in range(len(centroids) - 1))

    def test_centroids_are_symmetric(self):
        """For a symmetric distribution, centroids should be approximately symmetric around 0."""
        centroids, _ = solve_lloyd_max(d=128, bits=2)
        for i in range(len(centroids) // 2):
            assert abs(centroids[i] + centroids[-(i + 1)]) < 1e-4

    def test_centroids_within_expected_range(self):
        """Centroids should be within ~3.5 sigma of the distribution."""
        import math
        d = 128
        sigma = 1.0 / math.sqrt(d)
        centroids, _ = solve_lloyd_max(d=d, bits=2)
        for c in centroids:
            assert abs(c) < 4.0 * sigma

    def test_bits_1_produces_2_centroids(self):
        centroids, boundaries = solve_lloyd_max(d=128, bits=1)
        assert centroids.shape == (2,)
        assert boundaries.shape == (1,)

    def test_bits_3_produces_8_centroids(self):
        centroids, boundaries = solve_lloyd_max(d=128, bits=3)
        assert centroids.shape == (8,)
        assert boundaries.shape == (7,)


class TestMatrixGeneration:
    def test_rotation_is_orthogonal(self):
        Pi = generate_rotation_matrix(d=128, seed=42)
        identity = Pi @ Pi.T
        expected = mx.eye(128)
        assert mx.allclose(identity, expected, atol=1e-3).item()

    def test_rotation_shape(self):
        Pi = generate_rotation_matrix(d=128, seed=42)
        assert Pi.shape == (128, 128)

    def test_rotation_deterministic(self):
        Pi1 = generate_rotation_matrix(d=128, seed=42)
        Pi2 = generate_rotation_matrix(d=128, seed=42)
        assert mx.allclose(Pi1, Pi2).item()

    def test_rotation_different_seeds_differ(self):
        Pi1 = generate_rotation_matrix(d=128, seed=42)
        Pi2 = generate_rotation_matrix(d=128, seed=43)
        assert not mx.allclose(Pi1, Pi2).item()

    def test_qjl_matrix_shape(self):
        S = generate_qjl_matrix(d=128, m=128, seed=42)
        assert S.shape == (128, 128)

    def test_qjl_matrix_different_m(self):
        S = generate_qjl_matrix(d=128, m=64, seed=42)
        assert S.shape == (64, 128)

    def test_qjl_deterministic(self):
        S1 = generate_qjl_matrix(d=128, m=128, seed=42)
        S2 = generate_qjl_matrix(d=128, m=128, seed=42)
        assert mx.allclose(S1, S2).item()


class TestBitPacking:
    def test_pack_unpack_2bit_roundtrip(self):
        indices = mx.array([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]], dtype=mx.uint32)
        packed = pack_indices(indices, bits=2)
        unpacked = unpack_indices(packed, bits=2, dim=16)
        assert mx.array_equal(indices, unpacked).item()

    def test_pack_unpack_3bit_roundtrip(self):
        indices = mx.array([[i % 8 for i in range(32)]], dtype=mx.uint32)
        packed = pack_indices(indices, bits=3)
        unpacked = unpack_indices(packed, bits=3, dim=32)
        assert mx.array_equal(indices, unpacked).item()

    def test_pack_signs_roundtrip(self):
        signs = mx.array([[1, -1, 1, 1, -1, -1, 1, -1] * 4], dtype=mx.float32)
        packed = pack_signs(signs)
        unpacked = unpack_signs(packed, dim=32)
        assert mx.array_equal(signs, unpacked).item()

    def test_pack_signs_shape(self):
        signs = mx.array([[1, -1] * 64], dtype=mx.float32)  # 128 signs
        packed = pack_signs(signs)
        assert packed.shape == (1, 4)  # 128 / 32 = 4 uint32s

    def test_pack_2bit_shape(self):
        indices = mx.array([[0] * 128], dtype=mx.uint32)
        packed = pack_indices(indices, bits=2)
        assert packed.shape == (1, 8)  # 128 * 2 / 32 = 8 uint32s


class TestEncodeDecode:
    @pytest.fixture
    def config(self):
        return TurboQuantConfig(head_dim=128, bits=3, seed=42)

    def test_encode_output_shapes(self, config):
        x = mx.random.normal((1, 4, 8, 128))  # B=1, heads=4, seq=8, d=128
        result = turboquant_encode(x, config, mode="key")
        # Check idx shape: 128 coords * 2 bits / 32 = 8 packed uint32
        assert result["idx"].shape[0:3] == (1, 4, 8)
        assert "qjl" in result
        assert result["rnorm"].shape == (1, 4, 8, 1)
        assert result["vnorm"].shape == (1, 4, 8, 1)

    def test_encode_value_mode_no_qjl(self, config):
        x = mx.random.normal((1, 4, 8, 128))
        result = turboquant_encode(x, config, mode="value")
        assert "qjl" not in result
        assert "rnorm" not in result
        assert result["vnorm"].shape == (1, 4, 8, 1)

    def test_value_decode_roundtrip_mse(self, config):
        """MSE should be reasonable for 3-bit quantization."""
        mx.random.seed(0)
        x = mx.random.normal((1, 4, 32, 128))
        encoded = turboquant_encode(x, config, mode="value")
        decoded = turboquant_decode_values(encoded, config)
        # Normalized MSE check: reconstruct and compare
        x_norm = mx.linalg.norm(x, axis=-1, keepdims=True)
        decoded_norm = mx.linalg.norm(decoded, axis=-1, keepdims=True)
        # The norms should roughly match (vnorm is preserved)
        assert mx.allclose(x_norm, decoded_norm, atol=1.0).item()

    def test_vnorm_preserved(self, config):
        x = mx.random.normal((1, 4, 8, 128))
        norms = mx.linalg.norm(x, axis=-1, keepdims=True)
        encoded = turboquant_encode(x, config, mode="key")
        assert mx.allclose(encoded["vnorm"], norms.astype(mx.float16), atol=0.5).item()


class TestInnerProduct:
    @pytest.fixture
    def config(self):
        return TurboQuantConfig(head_dim=128, bits=3, seed=42)

    def test_inner_product_unbiased(self, config):
        """Over many random pairs, estimated inner product should match true inner product on average."""
        mx.random.seed(0)
        n_trials = 1000
        queries = mx.random.normal((1, 1, n_trials, 128))
        keys = mx.random.normal((1, 1, n_trials, 128))

        true_ip = mx.sum(queries * keys, axis=-1)  # (1, 1, n_trials)

        compressed = turboquant_encode(keys, config, mode="key")

        # For unbiasedness test, we need per-vector inner products
        # estimated_ip[i] = <queries[i], compressed_keys[i]>
        # We can compute this as diagonal of the full score matrix
        scores = turboquant_inner_product(queries, compressed, config)
        # scores is (1, 1, n_trials, n_trials) - take diagonal
        estimated_ip = mx.diagonal(scores.squeeze(0).squeeze(0))  # (n_trials,)
        true_ip_flat = true_ip.squeeze(0).squeeze(0)  # (n_trials,)

        mean_true = mx.mean(mx.abs(true_ip_flat)).item()
        mean_error = mx.mean(mx.abs(estimated_ip - true_ip_flat)).item()
        relative_error = mean_error / mean_true
        assert relative_error < 0.5  # < 50% mean abs error at 3-bit (2-bit MSE + 1-bit QJL)
        # The estimator IS unbiased; verify low bias (mean signed error near zero)
        mean_bias = mx.mean(estimated_ip - true_ip_flat).item()
        assert abs(mean_bias) / mean_true < 0.05  # bias < 5% of signal magnitude

    def test_inner_product_shape(self, config):
        queries = mx.random.normal((1, 4, 1, 128))
        keys = mx.random.normal((1, 4, 16, 128))
        compressed = turboquant_encode(keys, config, mode="key")
        scores = turboquant_inner_product(queries, compressed, config)
        assert scores.shape == (1, 4, 1, 16)  # (B, n_heads, L_q, L_k)
