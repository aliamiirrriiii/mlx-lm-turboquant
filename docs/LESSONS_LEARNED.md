# Lessons Learned

### [2026-03-26] scipy Not Pre-installed in mlx-env

**Problem**: `from scipy import integrate` raised `ModuleNotFoundError` when running tests in `~/mlx-env`.
**Root Cause**: scipy is not part of the base mlx-lm environment dependencies.
**Solution**: Installed with `~/mlx-env/bin/pip install scipy`.
**Prevention**: When a new module depends on scipy (or any non-standard scientific library), check whether it is present in the target venv before running tests. Add scipy to requirements if it is a permanent dependency of turboquant.py.
