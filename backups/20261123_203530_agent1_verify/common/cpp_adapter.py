"""Python <-> C++ adapter.

The C++ module directory is added to ``sys.path`` *before* importing the
extension (fixes the starter's import-order defect).  The persistent
``DiffusionEngine`` is built once per graph and reused for every replicate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

try:  # package import
    from .network import build_adjacency
except ImportError:  # top-level import with common/ on sys.path
    from network import build_adjacency  # type: ignore

_CPP_DIR = Path(__file__).resolve().parents[1] / "cpp_engine"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))

HAS_CPP = False
IMPORT_ERROR: str | None = None
try:
    import simulation_engine as _simulation_engine  # type: ignore

    HAS_CPP = True
except Exception as exc:  # pragma: no cover - environment dependent
    _simulation_engine = None
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def cpp_available() -> bool:
    return bool(HAS_CPP)


def cpp_import_error() -> str | None:
    return IMPORT_ERROR


def openmp_available() -> bool:
    return bool(HAS_CPP and _simulation_engine.openmp_enabled())


class CppSimulator:
    """Persistent graph-resident simulator.

    ``simulate`` and ``simulate_many`` return plain dicts with integer counts so
    the rest of the Python code is independent of the pybind11 result type.
    """

    def __init__(self, nodes: Sequence[int], G, n_threads: int = 0):
        if not HAS_CPP:
            raise ImportError(
                "The C++ simulation_engine extension is not importable. "
                f"Build it with `cd cpp_engine && python setup.py build_ext --inplace`. "
                f"Original error: {IMPORT_ERROR}"
            )
        self.nodes = [int(x) for x in nodes]
        self.n_threads = int(n_threads)
        self.index, self.adj, self.probs, self.polarity, self.strength = build_adjacency(G, self.nodes)
        self.engine = _simulation_engine.DiffusionEngine(
            self.nodes, self.adj, self.probs, self.polarity, self.strength
        )

    @property
    def num_nodes(self) -> int:
        return int(self.engine.num_nodes)

    @property
    def num_edges(self) -> int:
        return int(self.engine.num_edges)

    @staticmethod
    def _as_dict(result) -> dict:
        return {
            "spread_A": int(result.spread_A),
            "spread_B": int(result.spread_B),
            "ignorant": int(result.ignorant),
            "rounds": int(result.rounds),
        }

    def simulate(self, seeds_A: Iterable[int], seeds_B: Iterable[int], seed: int) -> dict:
        result = self.engine.simulate(
            [int(x) for x in seeds_A], [int(x) for x in seeds_B], int(seed)
        )
        return self._as_dict(result)

    def simulate_many(self, seeds_A: Iterable[int], seeds_B: Iterable[int],
                      seeds: Iterable[int], n_threads: int | None = None) -> list[dict]:
        threads = self.n_threads if n_threads is None else int(n_threads)
        results = self.engine.simulate_many(
            [int(x) for x in seeds_A], [int(x) for x in seeds_B],
            [int(s) for s in seeds], threads,
        )
        return [self._as_dict(r) for r in results]

    def simulate_batch(self, seeds_A: Sequence[Sequence[int]], seeds_B: Sequence[Sequence[int]],
                       seeds: Sequence[int], n_threads: int | None = None) -> list[dict]:
        threads = self.n_threads if n_threads is None else int(n_threads)
        results = self.engine.simulate_batch(
            [[int(x) for x in a] for a in seeds_A],
            [[int(x) for x in b] for b in seeds_B],
            [int(s) for s in seeds], threads,
        )
        return [self._as_dict(r) for r in results]

    def __call__(self, seeds_A, seeds_B, seed) -> dict:
        return self.simulate(seeds_A, seeds_B, seed)


def make_cpp_engine(nodes: Sequence[int], G, n_threads: int = 0) -> CppSimulator:
    return CppSimulator(nodes, G, n_threads=n_threads)


def make_cpp_simulator(nodes: Sequence[int], G, n_threads: int = 0):
    """Backward-compatible callable ``run(A, B, seed) -> dict``."""
    simulator = CppSimulator(nodes, G, n_threads=n_threads)
    return simulator.simulate
