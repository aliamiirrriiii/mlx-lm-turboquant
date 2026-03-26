# Structure Map

## Top-level layout

```
mlx-lm-fork/
├── mlx_lm/                  # Main library package
│   ├── models/              # Model implementations
│   │   ├── turboquant.py    # [NEW] TurboQuant: Lloyd-Max solver + future KV compression
│   │   ├── base.py          # Base model class / SDPA dispatch
│   │   ├── cache.py         # KV cache implementations (incl. TurboQuantKVCache)
│   │   └── ...              # Other model architectures (llama, mistral, etc.)
│   └── ...
├── tests/
│   ├── test_turboquant.py   # TurboQuant tests (Lloyd-Max, encode/decode, inner product)
│   ├── test_turboquant_cache.py  # [NEW] TurboQuantKVCache tests (Task 7)
│   └── ...                  # Other test modules
├── docs/
│   ├── STRUCTURE_MAP.md     # This file
│   └── LESSONS_LEARNED.md   # Accumulated lessons from coding sessions
└── ...
```

## Key files

| Path                             | Purpose                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| `mlx_lm/models/turboquant.py`    | Lloyd-Max solver, bit packing, encode/decode, inner product (Task 6) |
| `mlx_lm/models/base.py`          | Base model; will need SDPA dispatch hooks for TurboQuant             |
| `mlx_lm/models/cache.py`         | KV caches incl. TurboQuantKVCache (Task 7) + KVCache.to_turboquant() |
| `tests/test_turboquant.py`       | TDD tests for TurboQuant components                                  |
| `tests/test_turboquant_cache.py` | TurboQuantKVCache tests (Task 7)                                     |

## Recently added/changed

- `mlx_lm/models/turboquant.py` — created (Task 2: Lloyd-Max solver); bit packing (Task 4); encode/decode fallback (Task 5); inner product estimator (Task 6)
- `tests/test_turboquant.py` — `TestLloydMax` (Task 2); `TestBitPacking` (Task 4); `TestEncodeDecode` (Task 5); `TestInnerProduct` (Task 6)
- `mlx_lm/models/cache.py` — added `TurboQuantKVCache` class + `KVCache.to_turboquant()` (Task 7)
- `tests/test_turboquant_cache.py` — created (Task 7: TurboQuantKVCache tests); `TestTurboQuantSDPA` (Task 8)
- `mlx_lm/models/base.py` — added `turboquant_scaled_dot_product_attention` + TurboQuant branch in `scaled_dot_product_attention` (Task 8)
- `docs/` — directory created; `STRUCTURE_MAP.md` and `LESSONS_LEARNED.md` added
