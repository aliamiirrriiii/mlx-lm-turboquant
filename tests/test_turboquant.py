import mlx.core as mx
import pytest


def test_placeholder():
    """Verify test infrastructure works."""
    assert mx.array([1.0]).item() == 1.0
