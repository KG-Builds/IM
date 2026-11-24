"""Memory-safe generator for the 5,000-node / 37,500-edge signed network.

The existing `network_generator.py` is preserved unchanged for the original
1,000-node network.  This module implements the same design at larger scale
without materialising all 12.5M candidate pairs:

* undirected Barabási–Albert backbone with an 8-node initial clique and m=7
* random undirected fill to exactly `n_edges` unique pairs
* each pair oriented uniformly at random
* sign positive with probability 0.5
* trust magnitude uniform integer 1..10
* probability mapping 1-3 -> 0.1, 4-7 -> 0.5, 8-10 -> 0.9

One CSV row is one directed **trust** arc.  The influence graph is the exact
reverse and is built by `common.network.load_influence_network`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def generate_large_synthetic_network(
    n_nodes: int = 5000,
    n_edges: int = 37500,
    seed: int = 42,
    sign_positive_probability: float = 0.5,
    initial_clique: int = 8,
    m: int = 7,
) -> pd.DataFrame:
    if n_nodes < initial_clique:
        raise ValueError("n_nodes must be >= initial_clique")
    if not (0.0 <= sign_positive_probability <= 1.0):
        raise ValueError("sign_positive_probability must be in [0,1]")

    rng = np.random.default_rng(seed)
    base = nx.complete_graph(initial_clique)
    backbone = nx.barabasi_albert_graph(
        n_nodes, m, seed=seed, initial_graph=base
    )
    pairs = {tuple(sorted((int(u), int(v)))) for u, v in backbone.edges()}
    if len(pairs) > n_edges:
        raise ValueError(
            f"BA backbone already has {len(pairs)} edges, more than requested {n_edges}"
        )
    max_pairs = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_pairs:
        raise ValueError(f"n_edges={n_edges} exceeds maximum {max_pairs}")

    while len(pairs) < n_edges:
        u, v = (int(x) for x in rng.integers(0, n_nodes, size=2))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        pairs.add((u, v))

    rows = []
    for u, v in sorted(pairs):
        if rng.random() < 0.5:
            src, dst = u, v
        else:
            src, dst = v, u
        positive = rng.random() < sign_positive_probability
        magnitude = int(rng.integers(1, 11))
        rating = magnitude if positive else -magnitude
        probability = 0.1 if magnitude <= 3 else (0.5 if magnitude <= 7 else 0.9)
        rows.append(
            (src, dst, rating, probability, magnitude, 1 if positive else -1)
        )

    df = pd.DataFrame(
        rows,
        columns=["source", "target", "rating", "probability", "strength", "polarity"],
    )
    # Hard invariants.
    if len(df) != n_edges:
        raise RuntimeError(f"Expected {n_edges} arcs, generated {len(df)}")
    if df.duplicated(["source", "target"]).any():
        raise RuntimeError("Duplicate directed arcs generated")
    if (df["source"] == df["target"]).any():
        raise RuntimeError("Self-loops generated")
    nodes = sorted(set(df["source"]) | set(df["target"]))
    if nodes != list(range(n_nodes)):
        raise RuntimeError("Node ids are not contiguous 0..n-1")
    if not (df["rating"] == df["polarity"] * df["strength"]).all():
        raise RuntimeError("rating != polarity * strength")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate the 5000-node / 37500-edge network")
    parser.add_argument("--output", required=True, help="CSV path inside the workspace")
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument("--edges", type=int, default=37500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing network: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_large_synthetic_network(
        n_nodes=args.nodes, n_edges=args.edges, seed=args.seed
    )
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} directed trust arcs to {output}")
    print(f"Nodes: {args.nodes}; edges: {args.edges}; seed: {args.seed}")


if __name__ == "__main__":
    main()
