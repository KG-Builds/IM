"""Polarity-aware candidate-pool construction.

Target size is 10*K: 70 % nodes with the best polarity-aware outgoing score and
30 % uniformly random exploration nodes, excluding opponent (B) seeds. The score
is ``sum(p | positive out-edges) - sum(p | negative out-edges)`` so negative
relationships never count as positive influence.
"""
from __future__ import annotations

import random

try:
    from . import config
except ImportError:
    import config


def candidate_scores(g, nodes) -> dict[int, float]:
    scores: dict[int, float] = {}
    for u in nodes:
        pos = 0.0
        neg = 0.0
        for v in g.successors(u):
            d = g[u][v]
            if d["polarity"] > 0:
                pos += d["probability"]
            else:
                neg += d["probability"]
        scores[int(u)] = pos - neg
    return scores


def pool_composition(k: int, n_ranked: int, n_random: int, pool_size: int) -> dict:
    return {
        "target_pool_size": int(config.POOL_RATIO * k),
        "ranked_requested": int(n_ranked),
        "random_requested": int(n_random),
        "actual_pool_size": int(pool_size),
        "random_fraction_requested": float(config.RANDOM_POOL_FRACTION),
        "rounding": "n_ranked = int(target * (1 - random_fraction)); random fills the remainder",
    }


def make_candidate_pool_detailed(g, nodes, b_seeds, k: int, seed: int | None = None) -> dict:
    k = int(k)
    if k <= 0:
        raise ValueError("K must be positive")
    b_set = set(int(x) for x in b_seeds)
    target_size = config.POOL_RATIO * k
    scores = candidate_scores(g, nodes)
    eligible = [int(u) for u in nodes if int(u) not in b_set]
    if len(eligible) < k:
        raise ValueError(f"Only {len(eligible)} eligible nodes for K={k}")

    ranked = sorted(eligible, key=lambda u: (-scores[u], u))
    n_ranked = int(target_size * (1.0 - config.RANDOM_POOL_FRACTION))
    ranked_part = ranked[:n_ranked]

    if seed is None:
        seed = config.candidate_pool_seed
    rng = random.Random(int(seed))
    ranked_set = set(ranked_part)
    remaining = [u for u in eligible if u not in ranked_set]
    n_random = min(target_size - len(ranked_part), len(remaining))
    random_part = rng.sample(remaining, n_random)

    pool = list(dict.fromkeys(ranked_part + random_part))
    if len(pool) < k:
        raise ValueError(f"Candidate pool too small ({len(pool)}) for K={k}")
    if b_set & set(pool):
        raise AssertionError("Opponent seeds leaked into the candidate pool")
    if len(pool) != len(set(pool)):
        raise AssertionError("Duplicate nodes in candidate pool")

    return {
        "pool": pool,
        "scores": scores,
        "ranked_pool": ranked_part,
        "random_pool": random_part,
        "composition": pool_composition(k, n_ranked, target_size - n_ranked, len(pool)),
    }


def make_candidate_pool(g, nodes, b_seeds, k: int, seed: int):
    """Compatibility wrapper returning ``(pool, score_map)``."""
    detail = make_candidate_pool_detailed(g, nodes, b_seeds, k, seed)
    return detail["pool"], detail["scores"]
