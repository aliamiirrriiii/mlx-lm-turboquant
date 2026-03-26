# Lessons Learned

### [2026-03-26] 3-bit Pack/Unpack Fails at Word Boundaries

**Problem**: `test_pack_unpack_3bit_roundtrip` failed — index at position 21 (bits=3, bit_pos=63) was corrupted because `bit_offset=31` and `bits=3` means the value spans two uint32 words (1 bit in word 1, 2 bits in word 2).
**Root Cause**: The initial implementation only wrote/read bits within a single uint32 word using `val << bit_offset`, silently dropping overflow bits when `bit_offset + bits > 32`.
**Solution**: Added explicit cross-word overflow handling: in `pack_indices`, write `overflow = bit_offset + bits - 32` overflow bits into `int_idx + 1`; in `unpack_indices`, OR in the low-order overflow bits from `int_idx + 1` before masking.
**Prevention**: When implementing any N-bit packing where N does not evenly divide 32 (e.g., 3-bit), always check whether indices can straddle word boundaries. Specifically test with `d` large enough to hit a boundary (e.g., 32 values at 3 bits hits bit_pos=63 which straddles words 1 and 2).

### [2026-03-26] scipy Not Pre-installed in mlx-env

**Problem**: `from scipy import integrate` raised `ModuleNotFoundError` when running tests in `~/mlx-env`.
**Root Cause**: scipy is not part of the base mlx-lm environment dependencies.
**Solution**: Installed with `~/mlx-env/bin/pip install scipy`.
**Prevention**: When a new module depends on scipy (or any non-standard scientific library), check whether it is present in the target venv before running tests. Add scipy to requirements if it is a permanent dependency of turboquant.py.
