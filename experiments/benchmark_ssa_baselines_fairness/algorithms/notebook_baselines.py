"""Seed-selection algorithms re-implemented from
``old work/competitive_im_no_celf_csv_persist.ipynb``.

The implementations preserve the notebook's selection logic, objectives,
random-seed formulas and tie-breaking.  Data-structure changes are limited to
making the same algorithm practical on the 5,000-node network; no scientific
behaviour is intentionally changed.
"""
from __future__ import annotations

import heapq
import math
import random
import time
from collections import deque
from typing import Callable, Iterable

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from common.ssa import run_discrete_ssa
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from common.ssa import run_discrete_ssa


# ---------------------------------------------------------------------------
# Random / Top Degree / Signed PageRank
# ---------------------------------------------------------------------------
def random_seeds(nodes: Iterable[int], B: Iterable[int], k: int,
                 scenario_number: int, master_seed: int) -> list[int]:
    """Exact notebook Random baseline.

    Notebook: random.Random(master_seed + 50000 + k*100 + scenario_number)
              .sample(eligible_nodes, k), then sorted.
    """
    B_set = {int(x) for x in B}
    eligible = [int(u) for u in nodes if int(u) not in B_set]
    rng = random.Random(int(master_seed) + 50000 + int(k) * 100 + int(scenario_number))
    return sorted(int(u) for u in rng.sample(eligible, int(k)))


def top_degree_seeds(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int],
                     k: int) -> list[int]:
    """Top-k eligible nodes by influence out-degree, ties by node id."""
    B_set = {int(x) for x in B}
    eligible = [int(u) for u in nodes if int(u) not in B_set]
    ranked = sorted(eligible, key=lambda u: (-G.out_degree(u), int(u)))
    return ranked[:int(k)]


def compute_signed_pagerank(G: nx.DiGraph, nodes: Iterable[int],
                            alpha: float = 0.85, max_iter: int = 200,
                            tol: float = 1e-10) -> dict[int, float]:
    """Two-layer signed PageRank from the notebook.

    Positive edge: preserve polarity. Negative edge: flip polarity.
    Transition weight = absolute original rating. Teleport only to the
    positive-polarity copy. Score = positive-layer mass - negative-layer mass.
    """
    nodes = [int(u) for u in nodes]
    H = nx.DiGraph()
    H.add_nodes_from((u, 1) for u in nodes)
    H.add_nodes_from((u, -1) for u in nodes)
    for u, v, data in G.edges(data=True):
        rating = int(data.get("rating", 0))
        weight = abs(rating)
        if weight == 0:
            continue
        if rating > 0:
            H.add_edge((u, 1), (v, 1), weight=weight)
            H.add_edge((u, -1), (v, -1), weight=weight)
        else:
            H.add_edge((u, 1), (v, -1), weight=weight)
            H.add_edge((u, -1), (v, 1), weight=weight)
    personalization = {node: 0.0 for node in H.nodes()}
    positive_mass = 1.0 / len(nodes)
    for u in nodes:
        personalization[(u, 1)] = positive_mass
    pr = nx.pagerank(
        H, alpha=alpha, personalization=personalization, weight="weight",
        max_iter=max_iter, tol=tol,
    )
    return {u: float(pr[(u, 1)] - pr[(u, -1)]) for u in nodes}


def signed_pagerank_seeds(nodes: Iterable[int], B: Iterable[int], k: int,
                          scores: dict[int, float]) -> list[int]:
    B_set = {int(x) for x in B}
    eligible = [int(u) for u in nodes if int(u) not in B_set]
    ranked = sorted(eligible, key=lambda u: (-scores[u], int(u)))
    return ranked[:int(k)]


# ---------------------------------------------------------------------------
# CRIEG (Hong et al., IEEE TCYB 2021) -- notebook implementation
# ---------------------------------------------------------------------------
def crieg_seeds(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int], k: int,
                num_sketches: int = 2000, random_seed: int = 42
                ) -> tuple[list[int], dict]:
    """Competitive Reverse Reachable sketches on the positive subgraph.

    The notebook stores each node's per-sketch weight in a dict.  This version
    stores the same weights in lists, which is exactly equivalent for the
    greedy coverage calculation and materially faster on 5,000 nodes.
    """
    nodes = [int(u) for u in nodes]
    B_set = {int(u) for u in B}
    eligible = [u for u in nodes if u not in B_set]
    if len(eligible) < int(k):
        raise ValueError("Eligible nodes fewer than k.")

    rev_adj: dict[int, list[tuple[int, float]]] = {u: [] for u in nodes}
    for u, v, data in G.edges(data=True):
        if int(data.get("polarity", 0)) > 0:
            prob = float(data.get("probability", 0.0))
            if prob > 0.0:
                rev_adj[v].append((u, prob))

    node_sketches: dict[int, list[tuple[int, float]]] = {u: [] for u in eligible}
    rng = random.Random(int(random_seed))
    nonempty_sketches = 0
    for s_idx in range(int(num_sketches)):
        target = rng.choice(eligible)
        dist = {target: 0}
        queue = deque([target])
        min_b_dist = None
        while queue:
            curr = queue.popleft()
            d_curr = dist[curr]
            if min_b_dist is not None and d_curr >= min_b_dist:
                continue
            for prev_node, prob in rev_adj[curr]:
                if prev_node not in dist and rng.random() < prob:
                    d_next = d_curr + 1
                    dist[prev_node] = d_next
                    if prev_node in B_set:
                        if min_b_dist is None or d_next < min_b_dist:
                            min_b_dist = d_next
                    else:
                        queue.append(prev_node)
        if min_b_dist is not None:
            nonempty_sketches += 1
        for node, d_val in dist.items():
            if node in B_set:
                continue
            if min_b_dist is None or d_val < min_b_dist:
                node_sketches[node].append((s_idx, 1.0))
            elif d_val == min_b_dist:
                node_sketches[node].append((s_idx, 0.5))

    covered_weight = [0.0] * int(num_sketches)
    selected: list[int] = []
    remaining = set(eligible)
    for _ in range(int(k)):
        best_node = None
        best_marginal = -1.0
        for u in remaining:
            gain = 0.0
            for s_idx, weight in node_sketches[u]:
                if weight > covered_weight[s_idx]:
                    gain += weight - covered_weight[s_idx]
            if gain > best_marginal or (
                gain == best_marginal and (best_node is None or int(u) < int(best_node))
            ):
                best_marginal = gain
                best_node = u
        if best_node is None:
            raise RuntimeError("CRIEG failed to select a node.")
        selected.append(int(best_node))
        remaining.remove(best_node)
        for s_idx, weight in node_sketches[best_node]:
            if weight > covered_weight[s_idx]:
                covered_weight[s_idx] = weight

    info = {
        "num_sketches": int(num_sketches),
        "nonempty_sketches": int(nonempty_sketches),
        "random_seed": int(random_seed),
        "eligible_nodes": int(len(eligible)),
    }
    return selected, info


# ---------------------------------------------------------------------------
# NCGE (Liang et al., ESWA 2025) -- centrality pool + lazy greedy
# ---------------------------------------------------------------------------
def compute_ncge_centrality_scores(G: nx.DiGraph, nodes: Iterable[int],
                                   alpha: float = 0.5, beta: float = 0.3,
                                   gamma: float = 0.2, k_approx: int = 200,
                                   seed: int = 42) -> dict[int, float]:
    nodes = [int(u) for u in nodes]
    n = G.number_of_nodes()
    out_deg = dict(G.out_degree())
    max_deg = max(out_deg.values()) or 1.0
    deg_norm = {u: d / max_deg for u, d in out_deg.items()}

    if n <= 10000:
        clo = nx.closeness_centrality(G)
        bet = nx.betweenness_centrality(G, k=min(int(k_approx), n), seed=int(seed))
    else:  # pragma: no cover - not used on the 5,000-node network
        pr = nx.pagerank(G, alpha=0.85, max_iter=50)
        max_pr = max(pr.values()) or 1.0
        clo = {u: v / max_pr for u, v in pr.items()}
        bet = nx.betweenness_centrality(G, k=min(100, n), seed=int(seed))

    max_clo = max(clo.values()) or 1.0
    max_bet = max(bet.values()) or 1.0
    return {
        u: float(
            alpha * deg_norm.get(u, 0.0)
            + beta * (clo.get(u, 0.0) / max_clo)
            + gamma * (bet.get(u, 0.0) / max_bet)
        )
        for u in nodes
    }


def ncge_candidate_pool(nodes: Iterable[int], B: Iterable[int], pool_size: int,
                        scores: dict[int, float]) -> list[int]:
    B_set = {int(x) for x in B}
    eligible = [int(u) for u in nodes if int(u) not in B_set]
    if int(pool_size) > len(eligible):
        raise ValueError("Not enough eligible nodes for NCGE pool.")
    ranked = sorted(eligible, key=lambda u: (-scores[u], int(u)))
    return ranked[:int(pool_size)]


# ---------------------------------------------------------------------------
# Signed-GA (Chakraborty et al., 2025) -- notebook implementation
# ---------------------------------------------------------------------------
def signed_ga_seeds(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int], k: int,
                    steps: int = 30, damping: float = 0.92,
                    ridge: float = 1e-6,
                    permc_spec: str = "MMD_AT_PLUS_A") -> tuple[list[int], dict]:
    nodes = [int(u) for u in nodes]
    B_set = {int(u) for u in B}
    node_list = sorted(nodes, key=lambda x: int(x))
    n = len(node_list)
    idx_map = {u: i for i, u in enumerate(node_list)}

    rows, cols, vals_signed = [], [], []
    in_abs_sum = np.zeros(n, dtype=float)
    in_neg_sum = np.zeros(n, dtype=float)
    for u, v, data in G.edges(data=True):
        i = idx_map[int(v)]
        j = idx_map[int(u)]
        pol = int(data.get("polarity", 0))
        prob = float(data.get("probability", 0.0))
        if pol == 0 or prob <= 0.0:
            continue
        w_ji = pol * prob
        rows.append(i)
        cols.append(j)
        vals_signed.append(w_ji)
        in_abs_sum[i] += abs(w_ji)
        if w_ji < 0:
            in_neg_sum[i] += abs(w_ji)

    W_signed = sp.csr_matrix((vals_signed, (rows, cols)), shape=(n, n))
    L = sp.diags(in_abs_sum + float(ridge)) - W_signed

    p_B = np.zeros(n, dtype=float)
    for b_node in B_set:
        p_B[idx_map[b_node]] = 1.0

    eligible_indices = np.array(
        [idx_map[u] for u in node_list if u not in B_set], dtype=int
    )
    if eligible_indices.size < int(k):
        raise ValueError("Eligible nodes fewer than k.")
    p_A = np.zeros(n, dtype=float)
    p_A[eligible_indices] = float(k) / len(eligible_indices)

    def project_simplex_eligible(v_elig: np.ndarray, budget: float) -> np.ndarray:
        u_sorted = np.sort(v_elig)[::-1]
        cssv = np.cumsum(u_sorted) - budget
        ind = np.arange(1, len(v_elig) + 1)
        cond = u_sorted - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        return np.maximum(v_elig - theta, 0.0)

    rhs_base = in_neg_sum.copy()
    step_size = float(k)
    solves = 0
    for _ in range(int(steps)):
        M = (L + sp.diags(p_A + p_B)).tocsc()
        # Sparse LU with the MMD_AT_PLUS_A ordering is an exact direct solve;
        # the ordering only reduces fill-in relative to the default COLAMD.
        lu = spla.splu(M, permc_spec=str(permc_spec))
        x_A = lu.solve(p_A + rhs_base)
        x_A = np.clip(x_A, 0.0, 1.0)
        lu_t = spla.splu(M.T.tocsc(), permc_spec=str(permc_spec))
        adj = lu_t.solve(np.ones(n, dtype=float))
        grad = (1.0 - x_A) * adj / float(n)
        p_A_elig = p_A[eligible_indices] + step_size * grad[eligible_indices]
        p_A[:] = 0.0
        p_A[eligible_indices] = project_simplex_eligible(p_A_elig, float(k))
        step_size *= float(damping)
        solves += 1

    eligible_nodes = [u for u in node_list if u not in B_set]
    ranked = sorted(eligible_nodes, key=lambda u: (-p_A[idx_map[u]], int(u)))
    info = {
        "steps": int(steps),
        "linear_solves": int(solves),
        "damping": float(damping),
        "ridge": float(ridge),
        "permc_spec": str(permc_spec),
        "eligible_nodes": int(len(eligible_nodes)),
    }
    return [int(x) for x in ranked[:int(k)]], info


# ---------------------------------------------------------------------------
# CELF -- exact notebook lazy-greedy logic, O1 fitness
# ---------------------------------------------------------------------------
def celf_lazy_greedy(pool: Iterable[int], B: Iterable[int], k: int,
                     scorer: Callable[[Iterable[int]], float],
                     progress_label: str = "CELF") -> tuple[list[int], list[dict], dict]:
    """Lazy greedy over a candidate pool.

    ``scorer(A)`` must return the notebook's O1 weighted fitness
    (``Spread_A + 1.5 * Blocked_B``) on the fixed optimization MC seeds.
    The heap/refresh logic is the notebook's exactly.
    """
    pool = list(dict.fromkeys(int(u) for u in pool))
    B = [int(x) for x in B]
    if len(pool) < int(k):
        raise ValueError(f"{progress_label}: candidate pool smaller than k.")

    cache: dict[tuple[int, ...], float] = {}
    evaluations = 0

    def score(solution: Iterable[int]) -> float:
        nonlocal evaluations
        key = tuple(sorted({int(u) for u in solution}))
        if key not in cache:
            evaluations += 1
            cache[key] = float(scorer(key))
        return cache[key]

    selected: list[int] = []
    current = score(selected)
    heap: list[tuple[float, int, int]] = []
    for node in pool:
        gain = score([node]) - current
        heapq.heappush(heap, (-gain, int(node), 0))

    history: list[dict] = []
    refreshes = 0
    while len(selected) < int(k):
        if not heap:
            raise RuntimeError(f"{progress_label}: CELF heap exhausted.")
        neg_gain, node, last_size = heapq.heappop(heap)
        if node in selected:
            continue
        if last_size != len(selected):
            gain = score(selected + [node]) - current
            heapq.heappush(heap, (-gain, int(node), len(selected)))
            refreshes += 1
            continue
        selected.append(int(node))
        current = score(selected)
        history.append({
            "step": len(selected),
            "node": int(node),
            "objective": float(current),
            "marginal_gain_at_acceptance": float(-neg_gain),
            "heap_refreshes": int(refreshes),
            "unique_fitness_evaluations": len(cache),
        })

    if len(selected) != int(k) or len(set(selected)) != int(k):
        raise RuntimeError(f"{progress_label}: invalid selected set.")
    if set(selected) & set(B):
        raise RuntimeError(f"{progress_label}: selected an opponent seed.")
    info = {
        "candidate_pool_size": len(pool),
        "objective_evaluations": int(evaluations),
        "unique_fitness_evaluations": int(len(cache)),
        "heap_refreshes": int(refreshes),
        "final_objective": float(current),
    }
    return selected, history, info


# ---------------------------------------------------------------------------
# SSA-Notebook: exact notebook SSA with O1 fitness and notebook seed schedule
# ---------------------------------------------------------------------------
def run_ssa_notebook(pool: Iterable[int], B: Iterable[int], k: int,
                     optimization_seeds: Iterable[int],
                     scorer: Callable, population: int, iterations: int,
                     mutation_rate: float, random_seed: int):
    return run_discrete_ssa(
        pool, B, k, optimization_seeds, "O1", scorer,
        population_size=int(population),
        iterations=int(iterations),
        mutation_rate=float(mutation_rate),
        random_seed=int(random_seed),
    )
