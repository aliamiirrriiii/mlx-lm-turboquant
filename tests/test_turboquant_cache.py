import mlx.core as mx
import pytest
from mlx_lm.models.cache import TurboQuantKVCache, KVCache
from mlx_lm.models.base import scaled_dot_product_attention


class TestTurboQuantKVCache:
    def test_init(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        assert cache.bits == 3
        assert cache.offset == 0
        assert cache.empty()

    def test_update_and_fetch_returns_self_tuple(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 8, 128))
        values = mx.random.normal((1, 4, 8, 128))
        result_k, result_v = cache.update_and_fetch(keys, values)
        assert isinstance(result_k, TurboQuantKVCache)
        assert isinstance(result_v, TurboQuantKVCache)
        assert cache.offset == 8

    def test_shape_property(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 8, 128))
        values = mx.random.normal((1, 4, 8, 128))
        cache.update_and_fetch(keys, values)
        assert cache.shape == (1, 4, 8, 128)

    def test_incremental_update(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        k1 = mx.random.normal((1, 4, 8, 128))
        v1 = mx.random.normal((1, 4, 8, 128))
        cache.update_and_fetch(k1, v1)
        assert cache.offset == 8

        k2 = mx.random.normal((1, 4, 1, 128))
        v2 = mx.random.normal((1, 4, 1, 128))
        cache.update_and_fetch(k2, v2)
        assert cache.offset == 9

    def test_is_trimmable(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        assert cache.is_trimmable()

    def test_trim(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 10, 128))
        values = mx.random.normal((1, 4, 10, 128))
        cache.update_and_fetch(keys, values)
        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache.offset == 7

    def test_make_mask_single_token(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 8, 128))
        values = mx.random.normal((1, 4, 8, 128))
        cache.update_and_fetch(keys, values)
        mask = cache.make_mask(N=1, return_array=False, window_size=None)
        assert mask is None  # N=1, no mask needed

    def test_meta_state_roundtrip(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 8, 128))
        values = mx.random.normal((1, 4, 8, 128))
        cache.update_and_fetch(keys, values)
        ms = cache.meta_state
        assert isinstance(ms, tuple)
        assert len(ms) == 4  # offset, bits, seed, head_dim

    def test_empty_check(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        assert cache.empty()
        keys = mx.random.normal((1, 4, 1, 128))
        values = mx.random.normal((1, 4, 1, 128))
        cache.update_and_fetch(keys, values)
        assert not cache.empty()

    def test_get_compressed_keys(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 8, 128))
        values = mx.random.normal((1, 4, 8, 128))
        cache.update_and_fetch(keys, values)
        ck = cache.get_compressed_keys()
        assert "idx" in ck
        assert "qjl" in ck
        assert "rnorm" in ck
        assert "vnorm" in ck
        assert ck["idx"].shape[2] == 8  # seq_len

    def test_get_compressed_values(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 8, 128))
        values = mx.random.normal((1, 4, 8, 128))
        cache.update_and_fetch(keys, values)
        cv = cache.get_compressed_values()
        assert "idx" in cv
        assert "vnorm" in cv
        assert cv["idx"].shape[2] == 8

    def test_kvcache_to_turboquant(self):
        kv = KVCache()
        keys = mx.random.normal((1, 4, 8, 128))
        values = mx.random.normal((1, 4, 8, 128))
        kv.update_and_fetch(keys, values)
        tq = kv.to_turboquant(bits=3, seed=42)
        assert isinstance(tq, TurboQuantKVCache)
        assert tq.offset == 8

    def test_nbytes_less_than_standard(self):
        keys = mx.random.normal((1, 4, 64, 128))
        values = mx.random.normal((1, 4, 64, 128))

        std = KVCache()
        std.update_and_fetch(keys, values)

        tq = TurboQuantKVCache(bits=3, seed=42)
        tq.update_and_fetch(keys, values)

        assert tq.nbytes < std.nbytes


class TestTurboQuantSDPA:
    def test_attention_output_shape(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 16, 128))
        values = mx.random.normal((1, 4, 16, 128))
        cache.update_and_fetch(keys, values)

        queries = mx.random.normal((1, 4, 1, 128))
        output = scaled_dot_product_attention(
            queries, cache, cache, cache=cache, scale=128**-0.5, mask=None
        )
        assert output.shape == (1, 4, 1, 128)

    def test_attention_no_nan(self):
        """Output should not contain NaN."""
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 32, 128))
        values = mx.random.normal((1, 4, 32, 128))
        cache.update_and_fetch(keys, values)

        queries = mx.random.normal((1, 4, 1, 128))
        output = scaled_dot_product_attention(
            queries, cache, cache, cache=cache, scale=128**-0.5, mask=None
        )
        assert not mx.any(mx.isnan(output)).item()

    def test_attention_cosine_similarity(self):
        """TurboQuant attention should be somewhat close to uncompressed attention."""
        mx.random.seed(0)
        keys = mx.random.normal((1, 4, 32, 128))
        values = mx.random.normal((1, 4, 32, 128))
        queries = mx.random.normal((1, 4, 1, 128))
        scale = 128**-0.5

        ref = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=None
        )

        cache = TurboQuantKVCache(bits=3, seed=42)
        cache.update_and_fetch(keys, values)
        tq = scaled_dot_product_attention(
            queries, cache, cache, cache=cache, scale=scale, mask=None
        )

        cos_sim = mx.sum(ref * tq) / (mx.linalg.norm(ref) * mx.linalg.norm(tq) + 1e-8)
        assert cos_sim.item() > 0.85  # > 85% similarity at 3-bit

    def test_sinks_raises_error(self):
        cache = TurboQuantKVCache(bits=3, seed=42)
        keys = mx.random.normal((1, 4, 8, 128))
        values = mx.random.normal((1, 4, 8, 128))
        cache.update_and_fetch(keys, values)
        queries = mx.random.normal((1, 4, 1, 128))
        with pytest.raises(ValueError, match="TurboQuant"):
            scaled_dot_product_attention(
                queries, cache, cache, cache=cache, scale=0.1,
                mask=None, sinks=mx.array([0])
            )
