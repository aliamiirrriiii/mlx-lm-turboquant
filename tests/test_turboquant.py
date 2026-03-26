import mlx.core as mx
import pytest

from mlx_lm.models.turboquant import solve_lloyd_max


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
