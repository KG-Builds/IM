"""Candidate-pool generation for SSA.

The score is polarity-aware: outgoing positive edges add transmission
probability, outgoing negative edges subtract it.  Negative relationships are
never rewarded as positive influence.
"""
from __future__ import annotations

import random
from typing import Iterable

import networkx as nx

POOL_RATIO = 10
RANDOM_POOL_FRACTION = 0.30


def candidate_scores(G: nx.DiGraph, nodes: Iterable[int]) -> dict[int, float]:
    """Node-level polarity-aware score.

    score(u) = sum(prob of outgoing positive edges)
             - sum(prob of outgoing negative edges)

    This is the starter's design; it deliberately does not use target in-degree
    or distances.  Negative edges can only lower the score.
    """
    scores: dict[int, float] = {}
    for u in nodes:
        pos = 0.0
        neg = 0.0
        for v in G.successors(u):
            d = G[u][v]
            if int(d["polarity"]) > 0:
                pos += float(d["probability"])
            else:
                neg += float(d["probability"])
        scores[int(u)] = float(pos - neg)
    return scores


def make_candidate_pool(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int], k: int,
                        seed: int, pool_ratio: int = POOL_RATIO,
                        random_fraction: float = RANDOM_POOL_FRACTION,
                        scores: dict[int, float] | None = None):
    """Build the candidate pool and return ``(pool, scores, metadata)``.

    Target size is ``pool_ratio * k``.  ``int((1-random_fraction)*size)`` nodes
    are taken from the polarity-aware ranking and the remainder are sampled
    uniformly without replacement from the remaining eligible nodes.  Opponent
    seeds are excluded everywhere.  If fewer eligible nodes exist, the pool is
    truncated and the metadata records it; fewer than ``k`` candidates is an
    error.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not (0.0 <= random_fraction <= 1.0):
        raise ValueError("random_fraction must be in [0,1]")

    nodes = sorted(int(u) for u in nodes)
    B_set = {int(u) for u in B}
    eligible = [u for u in nodes if u not in B_set]
    if len(eligible) < k:
        raise ValueError(f"Only {len(eligible)} eligible nodes for k={k}")

    if scores is None:
        scores = candidate_scores(G, nodes)
    pool_size = int(pool_ratio * k)
    ranked = sorted(eligible, key=lambda u: (-scores.get(u, 0.0), int(u)))
    n_ranked_target = int(pool_size * (1.0 - random_fraction))
    ranked_part = ranked[: min(n_ranked_target, len(ranked))]

    n_random_target = pool_size - len(ranked_part)
    ranked_set = set(ranked_part)
    remaining = [u for u in eligible if u not in ranked_set]
    rng = random.Random(int(seed))
    n_random = min(n_random_target, len(remaining))
    random_part = rng.sample(remaining, n_random) if n_random > 0 else []

    pool: list[int] = []
    seen: set[int] = set()
    for u in ranked_part + random_part:
        if u not in seen:
            pool.append(int(u))
            seen.add(int(u))

    if len(pool) < k:
        # Fall back to any remaining eligible nodes (small-network safety).
        for u in eligible:
            if u not in seen:
                pool.append(int(u))
                seen.add(int(u))
                if len(pool) >= pool_size:
                    break
    if len(pool) < k:
        raise ValueError(f"Candidate pool has {len(pool)} nodes, fewer than k={k}")

    meta = {
        "pool_size_target": pool_size,
        "pool_size_actual": len(pool),
        "n_ranked": len(ranked_part),
        "n_random": len(random_part),
        "random_fraction_requested": float(random_fraction),
        "random_fraction_actual": (len(random_part) / len(pool)) if pool else 0.0,
        "n_eligible": len(eligible),
        "n_opponent_seeds": len(B_set),
        "truncated": len(pool) < pool_size,
        "seed": int(seed),
        "pool_ratio": int(pool_ratio),
    }
    return pool, scores, meta


def pool_signature(pool: Iterable[int]) -> str:
    """Stable short signature of a candidate pool (for checkpoint validation)."""
    try:
        from .utils import stable_seed
    except ImportError:  # top-level import with common/ on sys.path
        from utils import stable_seed  # type: ignore

    return f"{len(list(pool))}:{stable_seed(*sorted(int(u) for u in pool)):016x}"
