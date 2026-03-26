# Structure Map

## Top-level layout

```
mlx-lm-fork/
├── mlx_lm/                  # Main library package
│   ├── models/              # Model implementations
│   │   ├── turboquant.py    # [NEW] TurboQuant: Lloyd-Max solver + future KV compression
│   │   ├── base.py          # Base model class / SDPA dispatch
│   │   ├── cache.py         # KV cache implementations
│   │   └── ...              # Other model architectures (llama, mistral, etc.)
│   └── ...
├── tests/
│   ├── test_turboquant.py   # TurboQuant tests (Lloyd-Max, future tests)
│   └── ...                  # Other test modules
├── docs/
│   ├── STRUCTURE_MAP.md     # This file
│   └── LESSONS_LEARNED.md   # Accumulated lessons from coding sessions
└── ...
```

## Key files

| Path                          | Purpose                                                         |
| ----------------------------- | --------------------------------------------------------------- |
| `mlx_lm/models/turboquant.py` | Lloyd-Max solver, bit packing, encode/decode fallback (Task 5)  |
| `mlx_lm/models/base.py`       | Base model; will need SDPA dispatch hooks for TurboQuant        |
| `mlx_lm/models/cache.py`      | Existing KV cache; TurboQuantKVCache will extend this interface |
| `tests/test_turboquant.py`    | TDD tests for TurboQuant components                             |

## Recently added/changed

- `mlx_lm/models/turboquant.py` — created (Task 2: Lloyd-Max solver); bit packing (Task 4); encode/decode fallback (Task 5)
- `tests/test_turboquant.py` — `TestLloydMax` (Task 2); `TestBitPacking` (Task 4); `TestEncodeDecode` (Task 5)
- `docs/` — directory created; `STRUCTURE_MAP.md` and `LESSONS_LEARNED.md` added
