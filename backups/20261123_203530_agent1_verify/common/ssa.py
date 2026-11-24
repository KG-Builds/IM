"""Discrete set-based Salp Swarm Algorithm (SSA).

The algorithm and equations are preserved from the starter implementation:
population 12, 10 iterations, mutation 0.20, ``c1 = 2*exp(-(4t/T)^2)``,
exploitation probability ``1 - c1/2``, leader/follower updates, best-so-far
elitism (the worst individual is replaced by the food source).

Every returned solution has exactly K unique candidate nodes, contains no
opponent seed, and is a subset of the candidate pool.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Iterable

import numpy as np
import pandas as pd

SSA_POPULATION = 12
SSA_ITERATIONS = 10
SSA_MUTATION_RATE = 0.20


def repair_solution(sequence: Iterable[int], pool: Iterable[int], k: int,
                    rng: random.Random) -> list[int]:
    """Deduplicate, keep only pool members, and pad to exactly K."""
    pool = [int(u) for u in pool]
    pool_set = set(pool)
    result: list[int] = []
    seen: set[int] = set()
    for u in sequence:
        u = int(u)
        if u in pool_set and u not in seen:
            result.append(u)
            seen.add(u)
    if len(result) < k:
        remaining = [u for u in pool if u not in seen]
        if len(remaining) < k - len(result):
            raise ValueError("Candidate pool too small to repair solution")
        result.extend(rng.sample(remaining, k - len(result)))
    return result[:k]


def _validate_inputs(pool: list[int], B: Iterable[int], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(pool) != len(set(pool)):
        raise ValueError("Candidate pool contains duplicates")
    if len(pool) < k:
        raise ValueError(f"Candidate pool size {len(pool)} < k={k}")
    if set(pool) & {int(b) for b in B}:
        raise ValueError("Candidate pool contains opponent B seeds")


def run_discrete_ssa(
    pool: Iterable[int],
    B: Iterable[int],
    k: int,
    seeds: Iterable[int],
    variant: str,
    scorer: Callable,
    population_size: int = SSA_POPULATION,
    iterations: int = SSA_ITERATIONS,
    mutation_rate: float = SSA_MUTATION_RATE,
    random_seed: int = 42,
):
    """Run SSA and return ``(food, food_score, history_df, info)``.

    ``scorer(A, B, seeds, variant, k) -> float`` is the objective evaluator.
    """
    pool = [int(u) for u in pool]
    B = [int(b) for b in B]
    seeds = [int(s) for s in seeds]
    _validate_inputs(pool, B, k)
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0,1]")

    evaluations = 0

    def score(solution):
        nonlocal evaluations
        evaluations += 1
        return float(scorer(solution, B, seeds, variant, k))

    rng = random.Random(int(random_seed))
    population = [rng.sample(pool, k) for _ in range(population_size)]
    scores = [score(s) for s in population]
    best_i = int(np.argmax(scores))
    food = list(population[best_i])
    food_score = float(scores[best_i])
    history = [{
        "iteration": 0,
        "best_objective_score": food_score,
        "mean_objective_score": float(np.mean(scores)),
        "worst_objective_score": float(np.min(scores)),
    }]

    for iteration in range(1, iterations + 1):
        c1 = 2.0 * math.exp(-((4.0 * iteration / iterations) ** 2))
        exploit = 1.0 - min(c1 / 2.0, 1.0)

        new_pop: list[list[int]] = []
        leader: list[int] = []
        current = population[0]
        for _pos in range(k):
            if rng.random() < exploit:
                choices = [u for u in food if u not in leader]
                if choices:
                    leader.append(rng.choice(choices))
                    continue
            choices = [u for u in current if u not in leader]
            if choices and rng.random() < 0.5:
                leader.append(rng.choice(choices))
            else:
                leader.append(rng.choice([u for u in pool if u not in leader]))
        new_pop.append(repair_solution(leader, pool, k, rng))

        for i in range(1, population_size):
            updated = list(population[i])
            previous = new_pop[i - 1]
            for pos in rng.sample(range(k), rng.randint(1, k)):
                used = set(updated[:pos] + updated[pos + 1:])
                choices = [u for u in previous if u not in used]
                if choices:
                    updated[pos] = rng.choice(choices)
            for pos in range(k):
                if rng.random() < mutation_rate:
                    used = set(updated[:pos] + updated[pos + 1:])
                    choices = [u for u in pool if u not in used]
                    if choices:
                        updated[pos] = rng.choice(choices)
            new_pop.append(repair_solution(updated, pool, k, rng))

        new_scores = [score(s) for s in new_pop]
        ib = int(np.argmax(new_scores))
        if new_scores[ib] > food_score:
            food = list(new_pop[ib])
            food_score = float(new_scores[ib])
        iw = int(np.argmin(new_scores))
        new_pop[iw] = list(food)
        population = new_pop

        history.append({
            "iteration": iteration,
            "best_objective_score": food_score,
            "mean_objective_score": float(np.mean(new_scores)),
            "worst_objective_score": float(np.min(new_scores)),
        })

    # Final invariant checks (fail loudly rather than returning a bad solution).
    if len(food) != k or len(set(food)) != k:
        raise RuntimeError("SSA returned a solution that is not exactly K unique nodes")
    if not set(food).issubset(set(pool)):
        raise RuntimeError("SSA returned nodes outside the candidate pool")
    if set(food) & set(B):
        raise RuntimeError("SSA returned opponent B seeds")

    info = {
        "population_size": int(population_size),
        "iterations": int(iterations),
        "mutation_rate": float(mutation_rate),
        "random_seed": int(random_seed),
        "objective_evaluations": int(evaluations),
        "best_objective_score": float(food_score),
    }
    return list(food), float(food_score), pd.DataFrame(history), info
