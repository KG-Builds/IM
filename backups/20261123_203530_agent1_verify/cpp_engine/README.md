# C++ engine

Build from this directory:

```bash
python -m pip install pybind11 setuptools
python setup.py build_ext --inplace
```

This first version is a port scaffold and must be validated against the current Python simulator before research runs. The current implementation uses C++ RNG, so identical seeds do not produce identical streams to NumPy; compare deterministic model logic and distributional outcomes, or implement a shared RNG contract if exact per-run matching is required.
