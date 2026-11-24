"""Test helpers and path bootstrap."""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "common", ROOT / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from network import build_adjacency  # noqa: E402


def persistent_tmp_dir(prefix: str) -> Path:
    """Create a unique test directory that is deliberately NOT deleted.

    The project rules for this rerun forbid deletion. Test artefacts therefore
    accumulate under ``tests_tmp/`` and can be removed manually by the user.
    """
    base = ROOT / "tests_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_graph(nodes, edges):
    """Build a DiGraph from ``(u, v, probability, polarity, strength)`` tuples."""
    G = nx.DiGraph()
    G.add_nodes_from(int(n) for n in nodes)
    for u, v, prob, pol, strength in edges:
        u, v = int(u), int(v)
        pol, strength = int(pol), int(strength)
        G.add_edge(u, v, probability=float(prob), polarity=pol,
                   strength=strength, rating=pol * strength)
    return G


def adjacency(G, nodes):
    return build_adjacency(G, [int(n) for n in nodes])


def chain_graph(n=5, prob=1.0, polarity=1):
    nodes = list(range(n))
    edges = [(i, i + 1, prob, polarity, 5) for i in range(n - 1)]
    return make_graph(nodes, edges)


def two_into_one_graph(strength_a=8, strength_b=3):
    nodes = [0, 1, 2]
    edges = [
        (0, 2, 1.0, 1, strength_a),
        (1, 2, 1.0, 1, strength_b),
    ]
    return make_graph(nodes, edges)


def random_small_graph(n=30, m=60, seed=123):
    import numpy as np

    rng = np.random.default_rng(seed)
    nodes = list(range(n))
    pairs = set()
    while len(pairs) < m:
        u, v = rng.integers(0, n, size=2)
        if u == v:
            continue
        pairs.add((int(u), int(v)))
    probs = [0.1, 0.5, 0.9]
    edges = []
    for u, v in sorted(pairs):
        p = float(probs[int(rng.integers(0, 3))])
        pol = 1 if rng.random() < 0.5 else -1
        strength = int(rng.integers(1, 11))
        edges.append((u, v, p, pol, strength))
    return make_graph(nodes, edges)
