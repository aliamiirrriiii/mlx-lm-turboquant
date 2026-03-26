"""
TurboQuant KV Cache Benchmark

Compares standard KVCache vs TurboQuant at various context lengths.
Measures: cache memory, generation speed, output quality.
"""

import time
import json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, TurboQuantKVCache, make_prompt_cache
from mlx_lm.generate import generate_step


def measure_cache_memory(model, tokenizer, cache_type, context_len, kv_bits=3):
    """Fill a cache to a specific context length and measure memory."""
    cache = make_prompt_cache(model, kv_cache_type=cache_type, kv_bits=kv_bits)

    # Generate dummy prompt to fill cache to desired length
    dummy = mx.ones((1,), dtype=mx.uint32) * tokenizer.encode("a")[0]
    dummy_batch = mx.broadcast_to(dummy, (context_len,))

    # Prefill in chunks
    chunk_size = min(512, context_len)
    for start in range(0, context_len, chunk_size):
        end = min(start + chunk_size, context_len)
        chunk = dummy_batch[start:end]
        model(chunk[None], cache=cache)
        mx.eval([c._keys_idx if hasattr(c, '_keys_idx') and c._keys_idx is not None
                 else (c.keys if hasattr(c, 'keys') and c.keys is not None else mx.zeros(1))
                 for c in cache])

    # Measure
    total_bytes = sum(c.nbytes for c in cache if hasattr(c, 'nbytes'))
    del cache
    return total_bytes


def measure_generation_speed(model, tokenizer, cache_type, prompt, n_tokens=50, kv_bits=3):
    """Measure actual generation speed with proper MLX sync."""
    cache = make_prompt_cache(model, kv_cache_type=cache_type, kv_bits=kv_bits)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(formatted)
    prompt_array = mx.array(tokens)

    gen_tokens = []
    # Warmup: generate 5 tokens without timing
    gen = generate_step(prompt_array, model, max_tokens=n_tokens + 5, prompt_cache=cache)

    for i in range(5):
        token, _ = next(gen)
        mx.eval(token)
        t = token.item() if hasattr(token, "item") else int(token)
        gen_tokens.append(t)

    # Timed generation: sync before and after each token
    mx.synchronize()
    t_start = time.perf_counter()

    for i in range(n_tokens):
        try:
            token, _ = next(gen)
            mx.eval(token)
            t = token.item() if hasattr(token, "item") else int(token)
            if t == tokenizer.eos_token_id:
                break
            gen_tokens.append(t)
        except StopIteration:
            break

    mx.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    timed_tokens = len(gen_tokens) - 5
    tok_per_sec = timed_tokens / elapsed if elapsed > 0 else 0

    cache_bytes = sum(c.nbytes for c in cache if hasattr(c, 'nbytes'))
    response = tokenizer.decode(gen_tokens[:80])

    del cache
    return tok_per_sec, cache_bytes, response, timed_tokens


def benchmark_model(model_name, prompt):
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")

    model, tokenizer = load(model_name)
    num_layers = len(model.layers)
    print(f"  Layers: {num_layers}")

    results = {"model": model_name, "layers": num_layers}

    # --- Memory benchmark at various context lengths ---
    print(f"\n  Memory Benchmark (cache size at various context lengths):")
    print(f"  {'Context':<10} {'Standard':>12} {'TurboQuant':>12} {'Ratio':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")

    mem_results = {}
    for ctx_len in [256, 1024, 4096]:
        try:
            std_bytes = measure_cache_memory(model, tokenizer, "standard", ctx_len)
            tq_bytes = measure_cache_memory(model, tokenizer, "turboquant", ctx_len)
            ratio = std_bytes / tq_bytes if tq_bytes > 0 else 0

            std_mb = std_bytes / 1024 / 1024
            tq_mb = tq_bytes / 1024 / 1024

            print(f"  {ctx_len:<10} {std_mb:>10.1f}MB {tq_mb:>10.1f}MB {ratio:>7.1f}x")
            mem_results[ctx_len] = {
                "standard_mb": round(std_mb, 2),
                "turboquant_mb": round(tq_mb, 2),
                "ratio": round(ratio, 2),
            }
        except Exception as e:
            print(f"  {ctx_len:<10} ERROR: {e}")
            mem_results[ctx_len] = {"error": str(e)}

    results["memory"] = mem_results

    # --- Speed benchmark ---
    print(f"\n  Speed Benchmark (tok/s, 50 tokens after warmup):")
    print(f"  {'Cache':<15} {'tok/s':>10} {'Cache MB':>10} {'Preview'}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*30}")

    speed_results = {}
    for cache_type in ["standard", "turboquant"]:
        try:
            tok_s, cache_bytes, response, n_tok = measure_generation_speed(
                model, tokenizer, cache_type, prompt, n_tokens=50
            )
            cache_mb = cache_bytes / 1024 / 1024
            preview = response[:60].replace('\n', ' ')

            print(f"  {cache_type:<15} {tok_s:>10.1f} {cache_mb:>10.1f} {preview}...")
            speed_results[cache_type] = {
                "tok_per_sec": round(tok_s, 1),
                "cache_mb": round(cache_mb, 2),
                "tokens_generated": n_tok,
            }
        except Exception as e:
            print(f"  {cache_type:<15} ERROR: {e}")
            speed_results[cache_type] = {"error": str(e)}

    results["speed"] = speed_results

    # Speed comparison
    if "standard" in speed_results and "turboquant" in speed_results:
        if "error" not in speed_results["standard"] and "error" not in speed_results["turboquant"]:
            std_s = speed_results["standard"]["tok_per_sec"]
            tq_s = speed_results["turboquant"]["tok_per_sec"]
            ratio = tq_s / std_s if std_s > 0 else 0
            print(f"\n  Speed ratio: {ratio:.2f}x ({std_s:.1f} -> {tq_s:.1f} tok/s)")
            results["speed_ratio"] = round(ratio, 2)

    del model, tokenizer
    mx.eval(mx.zeros(1))
    return results


def main():
    models = [
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        "mlx-community/Qwen3.5-27B-4bit",
    ]

    prompt = (
        "Explain the key differences between TCP and UDP protocols. "
        "Cover reliability, speed, use cases, and connection handling. "
        "Be thorough but concise."
    )

    all_results = {}
    for model_name in models:
        try:
            results = benchmark_model(model_name, prompt)
            all_results[model_name] = results
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")

    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name}: ERROR")
            continue
        short = model_name.split("/")[-1]
        print(f"{short}:")
        if "memory" in results:
            for ctx, m in results["memory"].items():
                if "error" not in m:
                    print(f"  {ctx} tokens: {m['standard_mb']:.1f}MB -> {m['turboquant_mb']:.1f}MB ({m['ratio']:.1f}x)")
        if "speed" in results:
            for ct, s in results["speed"].items():
                if "error" not in s:
                    print(f"  {ct}: {s['tok_per_sec']:.1f} tok/s")
        if "speed_ratio" in results:
            print(f"  Speed ratio: {results['speed_ratio']:.2f}x")
        print()

    with open("bench_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Results saved to bench_results.json")


if __name__ == "__main__":
    main()
