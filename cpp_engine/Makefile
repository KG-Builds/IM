# Project-local build only.  No package installation, no system changes.
PYTHON ?= python3

.PHONY: build clean check

build:
	$(PYTHON) setup.py build_ext --inplace

check:
	$(PYTHON) -c "import sys; sys.path.insert(0, '.'); import simulation_engine as e; print('import ok, openmp=', e.openmp_enabled())"

clean:
	rm -rf build *.so
