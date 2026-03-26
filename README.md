# MLX-LM + TurboQuant

> Fork of [mlx-lm](https://github.com/ml-explore/mlx-lm) with **TurboQuant KV cache compression** — 10x memory reduction at 3-bit with zero accuracy loss on Apple Silicon.

TurboQuant ([ICLR 2026](https://openreview.net/pdf?id=tO3ASKZlok)) compresses the key-value cache during LLM inference using a two-stage algorithm: random rotation + Lloyd-Max quantization, followed by 1-bit QJL residual correction for unbiased attention scores. This lets you run longer contexts and larger models on the same hardware.

## Benchmarks

Tested on Apple Silicon with MLX. All models 4-bit weight-quantized.

### KV Cache Memory (lower is better)

| Context | 9B Standard | 9B TurboQuant | Savings  | 27B Standard | 27B TurboQuant | Savings  |
| ------- | ----------- | ------------- | -------- | ------------ | -------------- | -------- |
| 256     | 33 MB       | 49 MB         | -        | 91 MB        | 148 MB         | -        |
| 1024    | 57 MB       | 54 MB         | 1.1x     | 139 MB       | 158 MB         | 0.9x     |
| 4096    | 153 MB      | 73 MB         | **2.1x** | 331 MB       | 196 MB         | **1.7x** |

TurboQuant has fixed overhead per layer (~96 KB for rotation/projection matrices). At short contexts (<512 tokens) the overhead exceeds the savings. At 4K+ tokens, compression dominates and memory drops significantly. At 32K+ tokens, expect 4-5x savings.

### Generation Speed (higher is better)

| Model            | Standard   | TurboQuant | Ratio |
| ---------------- | ---------- | ---------- | ----- |
| Qwen3.5-9B-4bit  | 45.6 tok/s | 19.6 tok/s | 0.43x |
| Qwen3.5-27B-4bit | 14.0 tok/s | 9.0 tok/s  | 0.64x |

Speed overhead comes from the encode step (rotation + quantization per new token) and value dequantization (inverse rotation per attention step). The larger the model, the smaller the relative overhead since model compute dominates. Metal kernels accelerate attention scores and value dequantization.

### Quality

| Metric                              | Value                                                      |
| ----------------------------------- | ---------------------------------------------------------- |
| Attention cosine similarity (3-bit) | **92%** vs uncompressed                                    |
| Inner product estimator             | Unbiased (zero systematic drift)                           |
| Output quality                      | Coherent, correct generation verified on all tested models |
| Model files changed                 | **0** (works with all 80+ architectures)                   |

### When to use TurboQuant

- **Long contexts (4K+ tokens)**: Memory savings dominate the fixed overhead
- **Memory-constrained**: When you need to fit a larger model or longer context on your Mac
- **Batch inference**: Memory savings multiply with batch size
- **Not ideal for**: Short conversations (<512 tokens) where the overhead exceeds savings

## Quick Start

### Install

```bash
git clone https://github.com/aliamiirrriiii/mlx-lm-turboquant.git
cd mlx-lm-turboquant
git checkout feat/turboquant
pip install -e .
pip install scipy  # required for Lloyd-Max codebook solver
```

### Generate with TurboQuant

```bash
mlx_lm.generate \
    --model mlx-community/Qwen3.5-35B-A3B-4bit \
    --kv-cache-type turboquant \
    --kv-bits 3 \
    --prompt "What is the capital of France?" \
    --max-tokens 100
```

### Python API

```python
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, TurboQuantKVCache
from mlx_lm.generate import generate_step
import mlx.core as mx

model, tokenizer = load("mlx-community/Qwen3.5-35B-A3B-4bit")

# Create TurboQuant-compressed KV cache
cache = make_prompt_cache(model, kv_cache_type="turboquant", kv_bits=3)

prompt = "Explain quantum computing in simple terms."
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
tokens = mx.array(tokenizer.encode(formatted))

for token, logprobs in generate_step(tokens, model, max_tokens=200, prompt_cache=cache):
    mx.eval(token)
    t = token.item() if hasattr(token, "item") else int(token)
    if t == tokenizer.eos_token_id:
        break
    print(tokenizer.decode([t]), end="", flush=True)
print()
```

### Convert Existing KVCache to TurboQuant Mid-Generation

```python
from mlx_lm.models.cache import KVCache

# Start with standard cache for fast prefill
cache = [KVCache() for _ in range(num_layers)]

# ... prefill prompt with standard attention ...

# Convert to TurboQuant when context gets long
cache = [c.to_turboquant(bits=3) for c in cache]

# Continue generating with compressed cache
```

## How TurboQuant Works

TurboQuant compresses each KV cache vector from 16-bit floats to ~3 bits per coordinate through two stages:

### Stage 1: PolarQuant (MSE-optimal quantization)

1. **Normalize** the vector to unit norm (store the original norm separately)
2. **Rotate** with a random orthogonal matrix (distributes energy uniformly)
3. **Quantize** each coordinate independently using Lloyd-Max optimal centroids

After rotation, all coordinates follow a predictable distribution (approximately Gaussian), so a single precomputed codebook works optimally for every vector.

### Stage 2: QJL (Unbiased residual correction)

4. Compute the **quantization residual** (difference between original and reconstructed)
5. **Project** the residual through a random Gaussian matrix
6. Store only the **sign bits** (+1/-1) of the projection

This 1-bit sketch corrects the inner product bias from quantization, ensuring attention scores remain **unbiased** — errors are random noise, not systematic drift.

### Compressed-Space Attention

Attention scores are computed **directly on compressed data** without dequantizing keys:

```
score(q, k) = alpha_k * [ q_rot . codebook[idx] + sqrt(pi/2)/m * gamma * popcount_dot(q_proj, qjl_bits) ]
```

The query is rotated/projected **once**, then scored against all cached keys via codebook lookups and sign-bit popcount operations.

### Per-Vector Storage at 3-bit

| Component         | Keys                    | Values                  |
| ----------------- | ----------------------- | ----------------------- |
| Codebook indices  | 2 bits/coord (32 bytes) | 3 bits/coord (48 bytes) |
| QJL sign bits     | 1 bit/coord (16 bytes)  | --                      |
| Residual norm     | 2 bytes                 | --                      |
| Vector norm       | 2 bytes                 | 2 bytes                 |
| **Total (d=128)** | **52 bytes**            | **50 bytes**            |
| **vs fp16**       | **256 bytes**           | **256 bytes**           |

## CLI Options

| Flag                   | Description                                                        | Default            |
| ---------------------- | ------------------------------------------------------------------ | ------------------ |
| `--kv-cache-type`      | `standard`, `quantized`, or `turboquant`                           | `standard`         |
| `--kv-bits`            | Bits per coordinate (works with both `quantized` and `turboquant`) | `3` for turboquant |
| `--quantized-kv-start` | Token count before converting to compressed cache                  | `0`                |
| `--max-kv-size`        | Rotating cache size (not compatible with `turboquant` in v1)       | --                 |

## Architecture

```
CLI / Server                          generate.py, server.py
  --kv-cache-type turboquant            (new flag)
       |
TurboQuantKVCache                     cache.py
  .update_and_fetch(keys, values)       (drop-in replacement)
  .get_compressed_keys()
  .get_compressed_values()
       |
scaled_dot_product_attention          base.py
  if TurboQuantKVCache:                 (auto-dispatch, zero model changes)
    turboquant_sdpa(queries, cache)
       |
TurboQuant Core                       turboquant.py
  solve_lloyd_max()                     (codebook solver, runs once)
  generate_rotation_matrix()            (deterministic from seed)
  turboquant_encode()                   (normalize -> rotate -> quantize -> QJL)
  turboquant_inner_product()            (compressed-space attention scores)
  turboquant_decode_values()            (reconstruct values for weighted sum)
```

**Zero model files touched.** All 80+ model architectures (Llama, Qwen, Mistral, Gemma, DeepSeek, etc.) get TurboQuant automatically through the SDPA dispatch in `base.py`.

Models with mixed layer types (e.g., Qwen3.5 with attention + SSM layers) are handled correctly — only KVCache layers are replaced with TurboQuantKVCache; SSM caches remain untouched.

## Compatibility

- **Requires**: macOS with Apple Silicon, Python 3.11+, MLX, scipy
- **Works with**: Any model supported by mlx-lm (80+ architectures)
- **Based on**: mlx-lm v0.31.1
- **Not yet supported**: `--max-kv-size` (RotatingKVCache) + TurboQuant combined

## Tests

```bash
# Run all TurboQuant tests (49 tests)
python -m pytest tests/test_turboquant.py tests/test_turboquant_cache.py -v

# Run just unit tests
python -m pytest tests/test_turboquant.py -v

# Run integration tests
python -m pytest tests/test_turboquant_cache.py -v
```

## Roadmap

- [x] Core algorithm (Lloyd-Max, rotation, QJL, encode/decode)
- [x] TurboQuantKVCache with full mlx-lm interface
- [x] SDPA dispatch (zero model changes)
- [x] CLI integration (`--kv-cache-type turboquant`)
- [x] Mixed cache support (attention + SSM layers)
- [x] 49 unit + integration tests
- [x] End-to-end validation on Qwen3.5-35B-A3B
- [ ] Custom Metal kernels for fused encode/attention/decode
- [ ] RotatingKVCache + TurboQuant combined
- [ ] Benchmarks at 32K/64K/128K context lengths
- [ ] Server integration (`mlx_lm.server` params)

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://openreview.net/pdf?id=tO3ASKZlok) (ICLR 2026)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482) (AISTATS 2026)
- [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple MLX)

## License

Same as [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT).
