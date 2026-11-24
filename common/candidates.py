"""Candidate-pool generation for SSA.

Reference-aligned with the supplied six-objective notebook
(`old work/competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb`):

    score(u) = sum(prob of outgoing positive edges)
             - sum(prob of outgoing negative edges)

Ranking key (exactly the reference's four-level tie-break):

    (-net_signed_score, -positive_score, negative_score, node_id)

Pool composition:

    n_random = int(round(pool_size * random_fraction))
    n_ranked = pool_size - n_random
    pool     = top_ranked + rng.sample(ranked[n_ranked:], n_random)

The random part is sampled from the *ranked remainder*, not from node-id order.
This matters: it changes which nodes enter the 30 % exploration component.
"""
from __future__ import annotations

import random
from typing import Iterable

import networkx as nx

POOL_RATIO = 10
RANDOM_POOL_FRACTION = 0.30


def candidate_score_details(G: nx.DiGraph, nodes: Iterable[int]) -> dict[int, dict]:
    """Return positive, negative and net outgoing-probability scores per node."""
    details: dict[int, dict] = {}
    for u in nodes:
        u = int(u)
        pos = 0.0
        neg = 0.0
        for v in G.successors(u):
            d = G[u][v]
            p = float(d["probability"])
            polarity = int(d["polarity"])
            if polarity > 0:
                pos += p
            elif polarity < 0:
                neg += p
            # polarity == 0 (zero-rated reference edges) contributes nothing.
        details[u] = {
            "positive_outgoing_score": float(pos),
            "negative_outgoing_score": float(neg),
            "net_signed_candidate_score": float(pos - neg),
        }
    return details


def candidate_scores(G: nx.DiGraph, nodes: Iterable[int]) -> dict[int, float]:
    """Net polarity-aware score per node (compatibility helper)."""
    return {
        u: d["net_signed_candidate_score"]
        for u, d in candidate_score_details(G, nodes).items()
    }


def make_candidate_pool(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int], k: int,
                        seed: int, pool_ratio: int = POOL_RATIO,
                        random_fraction: float = RANDOM_POOL_FRACTION,
                        scores: dict[int, float] | None = None,
                        score_details: dict[int, dict] | None = None):
    """Build the candidate pool and return ``(pool, scores, metadata)``.

    Target size is ``pool_ratio * k``.  ``n_random = int(round(size *
    random_fraction))`` nodes are sampled from the ranked remainder; the other
    ``size - n_random`` are the top ranked nodes.  Opponent seeds are excluded
    everywhere.  If fewer eligible nodes exist, the pool is truncated and the
    metadata records it; fewer than ``k`` candidates is an error.

    ``scores`` is accepted for backward compatibility and returned unchanged if
    supplied; ranking always uses the full positive/negative details.
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

    if score_details is None:
        score_details = candidate_score_details(G, nodes)
    if scores is None:
        scores = {u: score_details[u]["net_signed_candidate_score"] for u in nodes}

    pool_size = int(pool_ratio * k)
    ranked = sorted(
        eligible,
        key=lambda u: (
            -score_details[u]["net_signed_candidate_score"],
            -score_details[u]["positive_outgoing_score"],
            score_details[u]["negative_outgoing_score"],
            int(u),
        ),
    )

    n_random_target = int(round(pool_size * random_fraction))
    n_ranked_target = pool_size - n_random_target
    ranked_part = ranked[: min(n_ranked_target, len(ranked))]
    remaining = ranked[len(ranked_part):]
    rng = random.Random(int(seed))
    n_random = min(n_random_target, len(remaining))
    random_part = rng.sample(remaining, n_random) if n_random > 0 else []

    pool: list[int] = []
    seen: set[int] = set()
    for u in ranked_part + random_part:
        if u not in seen:
            pool.append(int(u))
            seen.add(int(u))

    if len(pool) < pool_size:
        # Small-network safety: fill from the ranked remainder.  The reference
        # raises when pool_size > eligible; we truncate and record it instead.
        for u in remaining:
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
        "ranking": "net_then_positive_then_negative_then_id",
        "random_part_source": "ranked_remainder",
    }
    return pool, scores, meta


def pool_signature(pool: Iterable[int]) -> str:
    """Stable short signature of a candidate pool (for checkpoint validation)."""
    try:
        from .utils import stable_seed
    except ImportError:  # top-level import with common/ on sys.path
        from utils import stable_seed  # type: ignore

    return f"{len(list(pool))}:{stable_seed(*sorted(int(u) for u in pool)):016x}"
