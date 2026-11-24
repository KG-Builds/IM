"""Bounded one-for-one Swap refinement.

Starter policy preserved: sample up to ``max_tests`` one-for-one replacement
pairs, evaluate all of them, and apply the single best strictly improving swap.
The procedure is elitist (equal or worse swaps are rejected), keeps exactly K
unique candidate nodes, and never introduces an opponent B seed.
"""
from __future__ import annotations

import random
from typing import Callable, Iterable

SWAP_MAX_TESTS = 100


def sampled_swap_refinement(
    initial: Iterable[int],
    pool: Iterable[int],
    B: Iterable[int],
    k: int,
    seeds: Iterable[int],
    variant: str,
    scorer: Callable,
    max_tests: int = SWAP_MAX_TESTS,
    random_seed: int = 1,
):
    """Return ``(selected_sorted, info)``.

    ``scorer(A, B, seeds, variant, k) -> float`` is the objective evaluator.
    """
    initial = [int(u) for u in initial]
    pool = [int(u) for u in pool]
    B_set = {int(b) for b in B}
    seeds = [int(s) for s in seeds]
    if len(initial) != k or len(set(initial)) != k:
        raise ValueError("initial solution must contain exactly K unique nodes")
    if not set(initial).issubset(set(pool)):
        raise ValueError("initial solution contains nodes outside the candidate pool")
    if set(initial) & B_set:
        raise ValueError("initial solution contains opponent B seeds")
    if max_tests < 0:
        raise ValueError("max_tests must be non-negative")

    rng = random.Random(int(random_seed))
    selected = list(initial)
    evaluations = 0

    def score(solution):
        nonlocal evaluations
        evaluations += 1
        return float(scorer(solution, B, seeds, variant, k))

    objective_before = score(selected)
    current = objective_before

    outsiders = [u for u in pool if u not in set(selected) and u not in B_set]
    total_pairs = len(selected) * len(outsiders)
    if total_pairs <= max_tests:
        pairs = [(old, new) for old in selected for new in outsiders]
    else:
        pair_set: set[tuple[int, int]] = set()
        # Bounded sampling; the pair space is finite and larger than max_tests.
        guard = 0
        while len(pair_set) < max_tests and guard < max_tests * 1000:
            pair_set.add((rng.choice(selected), rng.choice(outsiders)))
            guard += 1
        pairs = sorted(pair_set)

    best_score = current
    best_pair: tuple[int, int] | None = None
    for old, new in pairs:
        trial = [u for u in selected if u != old] + [new]
        value = score(trial)
        if value > best_score + 1e-12:
            best_score = value
            best_pair = (old, new)

    accepted = best_pair is not None
    if accepted:
        old, new = best_pair  # type: ignore[misc]
        selected.remove(old)
        selected.append(new)
        current = best_score

    if len(selected) != k or len(set(selected)) != k:
        raise RuntimeError("Swap refinement violated the exactly-K unique invariant")
    if not set(selected).issubset(set(pool)):
        raise RuntimeError("Swap refinement returned nodes outside the candidate pool")
    if set(selected) & B_set:
        raise RuntimeError("Swap refinement returned opponent B seeds")

    info = {
        "swap_tests": int(len(pairs)),
        "max_tests": int(max_tests),
        "accepted_swap": [int(best_pair[0]), int(best_pair[1])] if accepted else None,
        "accepted": bool(accepted),
        "objective_before": float(objective_before),
        "objective_after": float(current),
        "objective_gain": float(current - objective_before),
        "objective_evaluations": int(evaluations),
        "n_candidate_pairs_total": int(total_pairs),
        "random_seed": int(random_seed),
    }
    return sorted(selected), info
