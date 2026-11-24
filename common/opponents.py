"""Opponent B seed strategies B1-B4.

The implementations mirror the starter code exactly (ranking, graph, tie
breaking, seed derivation).  No scientific behaviour is changed here.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx

SCENARIO_IDS = ["B1", "B2", "B3", "B4"]
SCENARIO_NAMES = {
    "B1": "Random opponent",
    "B2": "Degree-based opponent",
    "B3": "Positive PageRank opponent",
    "B4": "Community-concentrated opponent",
}
SCENARIO_DESCRIPTIONS = {
    "B1": (
        "Uniform random sample of all nodes without replacement, using "
        "random.Random(master_seed + 810001). The K seeds are the first K "
        "elements of a fixed 50-node order, so orders are nested across K."
    ),
    "B2": (
        "Nodes sorted by out-degree descending, ties broken by ascending node id. "
        "The K seeds are the first K elements, nested across K."
    ),
    "B3": (
        "PageRank on the positive-only directed subgraph, edge weight = "
        "transmission probability, alpha=0.85, max_iter=200, tol=1e-10. "
        "Nodes sorted by PageRank descending, ties by ascending node id."
    ),
    "B4": (
        "Louvain communities on the undirected, unweighted projection with signs "
        "ignored (seed 42). The largest community (ties: smallest member id) is "
        "sorted by out-degree descending, ties by ascending node id."
    ),
}

B1_SEED_OFFSET = 810001


def _scenario_number(scenario_id: str) -> int:
    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"Unknown scenario {scenario_id!r}; expected one of {SCENARIO_IDS}")
    return int(scenario_id[1:])


def opponent_order(G: nx.DiGraph, nodes: Iterable[int], scenario_id: str,
                   seed: int = 42, max_k: int = 50) -> list[int]:
    """Return the fixed ranked opponent order (length ``max_k``) for a scenario."""
    nodes = sorted(int(n) for n in nodes)
    _scenario_number(scenario_id)
    if max_k > len(nodes):
        raise ValueError(f"max_k={max_k} exceeds number of nodes={len(nodes)}")

    if scenario_id == "B1":
        import random

        rng = random.Random(int(seed) + B1_SEED_OFFSET)
        return [int(u) for u in rng.sample(nodes, max_k)]

    if scenario_id == "B2":
        return sorted(nodes, key=lambda u: (-G.out_degree(u), int(u)))[:max_k]

    if scenario_id == "B3":
        pos = nx.DiGraph()
        pos.add_nodes_from(nodes)
        pos.add_edges_from(
            (u, v, {"weight": float(d["probability"])})
            for u, v, d in G.edges(data=True)
            if int(d["polarity"]) > 0
        )
        if pos.number_of_edges():
            pr = nx.pagerank(pos, alpha=0.85, weight="weight", max_iter=200, tol=1e-10)
        else:
            pr = {u: 1.0 / len(nodes) for u in nodes}
        return sorted(nodes, key=lambda u: (-float(pr.get(u, 0.0)), int(u)))[:max_k]

    # B4
    und = nx.Graph()
    und.add_nodes_from(nodes)
    und.add_edges_from((int(u), int(v)) for u, v in G.edges() if u != v)
    if und.number_of_edges():
        comms = list(nx.community.louvain_communities(und, seed=int(seed), weight=None))
    else:
        comms = [{u} for u in nodes]
    largest = sorted(comms, key=lambda c: (-len(c), min(c)))[0]
    order = sorted(largest, key=lambda u: (-G.out_degree(u), int(u)))
    if len(order) < max_k:
        raise ValueError(
            f"Largest community has {len(order)} nodes, fewer than max_k={max_k}; "
            "B4 cannot supply the requested opponent seeds."
        )
    return [int(u) for u in order[:max_k]]


def opponent_seed_set(order: list[int], k: int) -> list[int]:
    """First K elements of the fixed order (nested across K)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(order):
        raise ValueError(f"k={k} exceeds opponent order length={len(order)}")
    return [int(u) for u in order[:k]]


def all_opponent_orders(G: nx.DiGraph, nodes: Iterable[int], seed: int = 42,
                        max_k: int = 50) -> dict[str, list[int]]:
    return {sid: opponent_order(G, nodes, sid, seed=seed, max_k=max_k) for sid in SCENARIO_IDS}


def describe_scenario(scenario_id: str) -> dict:
    _scenario_number(scenario_id)
    return {
        "scenario_id": scenario_id,
        "scenario_name": SCENARIO_NAMES[scenario_id],
        "description": SCENARIO_DESCRIPTIONS[scenario_id],
    }
