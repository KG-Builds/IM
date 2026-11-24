"""Pure-Python reference implementation of the signed competitive diffusion model.

This is the readable reference used to validate the C++ engine's *model logic*.
It deliberately uses Python's ``random.Random`` so its random stream is not
identical to the C++ ``std::mt19937_64`` stream; parity is checked with
deterministic graphs and distributional comparisons, never by demanding
seed-for-seed equality across different RNGs.

Model (finite-contact Maki-Thompson variant, synchronous rounds):

* states: 0 ignorant, 1 A-spreader, 2 B-spreader, 3 A-stifler, 4 B-stifler
* A and B seeds start as spreaders; overlap is rejected.
* Each round, every spreader picks one uniformly random outgoing neighbour it
  has not contacted before.
* If it has no uncontacted outgoing neighbour, or the chosen target is not
  ignorant at the start of the round, the spreader becomes a stifler of its own
  rumor (it does not transmit that round).
* Otherwise, with the edge transmission probability, it sends its rumor; on a
  negative edge the rumor is flipped (A->B, B->A).
* Proposals are resolved synchronously: highest edge strength wins; ties between
  different rumors are broken uniformly at random.
* Final spread_A = count of states 1 and 3; spread_B likewise; ignorant = state 0.
* ``rounds`` follows the reference notebook: it is the number of the terminal
  round in which no spreader remained, i.e. executed rounds + 1.
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence


def simulate_python(
    node_ids: Sequence[int],
    adj: Sequence[Sequence[int]],
    probs: Sequence[Sequence[float]],
    polarity: Sequence[Sequence[int]],
    strength: Sequence[Sequence[int]],
    seeds_A: Iterable[int],
    seeds_B: Iterable[int],
    seed: int,
) -> dict:
    node_ids = [int(x) for x in node_ids]
    n = len(node_ids)
    if not (len(adj) == len(probs) == len(polarity) == len(strength) == n):
        raise ValueError("adjacency arrays must all have length n")

    idx = {node: i for i, node in enumerate(node_ids)}
    state = [0] * n
    for node in seeds_A:
        node = int(node)
        if node not in idx:
            raise ValueError(f"Unknown A seed {node}")
        state[idx[node]] = 1
    for node in seeds_B:
        node = int(node)
        if node not in idx:
            raise ValueError(f"Unknown B seed {node}")
        if state[idx[node]] != 0:
            raise ValueError("A/B seed overlap")
        state[idx[node]] = 2

    rng = random.Random(int(seed))
    contacted = [[False] * len(adj[u]) for u in range(n)]
    rounds = 0
    max_rounds = len(adj) + sum(len(a) for a in adj) + 10
    max_rounds = max(max_rounds, 10)

    while True:
        old = list(state)
        next_state = list(state)
        any_spreader = False
        stiflers: list[int] = []
        proposals: list[list[tuple[int, int]]] = [[] for _ in range(n)]

        for u in range(n):
            if old[u] not in (1, 2):
                continue
            any_spreader = True
            available = [j for j in range(len(adj[u])) if not contacted[u][j]]
            if not available:
                stiflers.append(u)
                continue
            j = rng.choice(available)
            contacted[u][j] = True
            v = int(adj[u][j])
            if old[v] != 0:
                stiflers.append(u)
                continue
            if rng.random() < float(probs[u][j]):
                rumor = 1 if old[u] == 1 else 2
                if int(polarity[u][j]) < 0:
                    rumor = 2 if rumor == 1 else 1
                proposals[v].append((rumor, int(strength[u][j])))

        if not any_spreader:
            break

        for v in range(n):
            if not proposals[v]:
                continue
            max_s = max(s for _, s in proposals[v])
            contenders = sorted({r for r, s in proposals[v] if s == max_s})
            next_state[v] = rng.choice(contenders)

        for u in stiflers:
            next_state[u] = 3 if old[u] == 1 else 4

        state = next_state
        rounds += 1
        if rounds > max_rounds:  # pragma: no cover - safety guard
            raise RuntimeError("Python simulation exceeded finite-contact safety limit")

    A = sum(1 for s in state if s in (1, 3))
    B = sum(1 for s in state if s in (2, 4))
    I = sum(1 for s in state if s == 0)
    # Reference-notebook semantics: include the terminal empty round.
    return {"spread_A": int(A), "spread_B": int(B), "ignorant": int(I),
            "rounds": int(rounds + 1)}


def simulate_python_graph(G, nodes, seeds_A, seeds_B, seed: int) -> dict:
    """Convenience wrapper around a networkx DiGraph."""
    try:
        from .network import build_adjacency
    except ImportError:
        from network import build_adjacency  # type: ignore

    node_ids = [int(x) for x in nodes]
    index, adj, probs, pol, strength = build_adjacency(G, node_ids)
    return simulate_python(node_ids, adj, probs, pol, strength, seeds_A, seeds_B, seed)


def simulate_many_python(
    node_ids,
    adj,
    probs,
    polarity,
    strength,
    seeds_A,
    seeds_B,
    seeds: Iterable[int],
) -> list[dict]:
    return [
        simulate_python(node_ids, adj, probs, polarity, strength, seeds_A, seeds_B, int(s))
        for s in seeds
    ]
