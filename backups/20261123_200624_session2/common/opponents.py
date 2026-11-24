"""Opponent (Rumor B) seed strategies B1–B4.

All strategies produce a deterministic ranking of at most ``max_k`` nodes; the
K-seed set is the prefix of that ranking, so B seed sets are nested across K.
Exact logic is preserved from the starter implementation (the only in-workspace
source) and documented in AUDIT_REPORT.md §2.3.
"""
from __future__ import annotations

import random

import networkx as nx

try:
    from . import config
except ImportError:
    import config


def random_opponent_order(nodes, seed: int, max_k: int) -> list[int]:
    rng = random.Random(seed)
    return rng.sample(list(nodes), max_k)


def degree_opponent_order(g: nx.DiGraph, nodes, max_k: int) -> list[int]:
    return sorted(nodes, key=lambda u: (-g.out_degree(u), int(u)))[:max_k]


def positive_pagerank_opponent_order(g: nx.DiGraph, nodes, max_k: int) -> list[int]:
    pos = nx.DiGraph()
    pos.add_nodes_from(nodes)
    pos.add_edges_from(
        (u, v, {"weight": d["probability"]})
        for u, v, d in g.edges(data=True)
        if d["polarity"] > 0
    )
    if pos.number_of_edges():
        pr = nx.pagerank(pos, alpha=0.85, weight="weight", max_iter=200, tol=1e-10)
    else:
        pr = {u: 1.0 / len(nodes) for u in nodes}
    return sorted(nodes, key=lambda u: (-pr.get(u, 0.0), int(u)))[:max_k]


def community_opponent_order(g: nx.DiGraph, nodes, seed: int, max_k: int) -> list[int]:
    und = nx.Graph()
    und.add_nodes_from(nodes)
    und.add_edges_from((u, v) for u, v in g.edges() if u != v)
    if und.number_of_edges():
        communities = list(nx.community.louvain_communities(und, seed=seed, weight=None))
    else:
        communities = [{u} for u in nodes]
    largest = sorted(communities, key=lambda c: (-len(c), min(c)))[0]
    order = sorted(largest, key=lambda u: (-g.out_degree(u), int(u)))
    if len(order) < max_k:
        raise ValueError(
            f"Largest Louvain community has {len(order)} nodes, fewer than max K={max_k}"
        )
    return order[:max_k]


def opponent_orders(g: nx.DiGraph, nodes, scenario_id: str, seed: int | None = None, max_k: int = 50) -> list[int]:
    """Return the full ranking (length ``max_k``) for a scenario."""
    seed = config.MASTER_SEED if seed is None else int(seed)
    if scenario_id == "B1":
        order = random_opponent_order(nodes, config.opponent_seed(scenario_id), max_k)
    elif scenario_id == "B2":
        order = degree_opponent_order(g, nodes, max_k)
    elif scenario_id == "B3":
        order = positive_pagerank_opponent_order(g, nodes, max_k)
    elif scenario_id == "B4":
        order = community_opponent_order(g, nodes, seed, max_k)
    else:
        raise ValueError(f"Unknown scenario {scenario_id!r}")
    if len(order) != max_k:
        raise ValueError(f"Opponent order for {scenario_id} has length {len(order)}, expected {max_k}")
    return order


def opponent_seeds(g, nodes, scenario_id: str, k: int, seed: int | None = None) -> list[int]:
    return sorted(opponent_orders(g, nodes, scenario_id, seed=seed, max_k=k))


def describe_strategy(scenario_id: str) -> str:
    return {
        "B1": "Uniform random sample of 50 nodes (random.Random(810043)); K-prefix.",
        "B2": "Top-50 by out-degree descending, node id ascending; K-prefix.",
        "B3": "Top-50 by PageRank on the positive-only subgraph (edge weight = transmission probability, alpha=0.85), node id ascending on ties; K-prefix.",
        "B4": "Nodes of the largest Louvain community on the undirected sign-agnostic projection (seed 42), ordered by out-degree descending then node id; K-prefix.",
    }[scenario_id]
