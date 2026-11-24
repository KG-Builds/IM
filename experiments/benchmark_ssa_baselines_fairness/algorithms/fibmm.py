"""Fairness-Aware Influence Blocking Maximization (FIBMM).

Paper
-----
Yu Xiao, Ding Liu, Lan Yang, Zhiwu Li,
"Fairness-Aware Influence Blocking Maximization in Social Networks",
IEEE Transactions on Computational Social Systems, 12(6), 2025.

Implemented components
----------------------
* Louvain communities with the paper's small-community merge rule.
* Community-aware reverse protector (CRP) sets sampled on a CIC surrogate.
* Two-hop lower-bound estimate used to size the CRP samples (Eqs. 14-17 and
  Algorithm 2).
* Two-stage node selection (Algorithm 1): overall saving-ratio greedy for
  ``k - floor(k*h)`` nodes, then maximin-fairness greedy for the remainder.
* Internal metrics Pi(S), tau(S), sigma(S) = (1-h) Pi(S) + h tau(S).

Documented adaptation
---------------------
The paper's CIC model is unsigned.  Our evaluated diffusion model is the
signed finite-contact Maki-Thompson model.  The FIBMM *selection* algorithm
therefore uses the paper's CIC surrogate on the positive-edge subgraph (only
positive trust arcs can carry the truth cascade).  Final performance is always
measured with the full signed model by the common external evaluator.  The
surrogate is never used to rank methods externally.

The martingale sample-size formula is implemented, but the total number of CRP
sets is capped for computational feasibility.  The cap and the realized sample
sizes are recorded for every configuration.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import scipy.sparse as sp


@dataclass
class FIBMMConfig:
    h: float = 0.9
    epsilon: float = 0.1
    l: int = 1
    community_min_fraction: float = 0.005
    community_seed: int = 42
    max_crp_samples: int = 600_000
    min_samples_per_community: int = 50
    crp_seed: int = 900000


def detect_communities(G: nx.DiGraph, nodes: Iterable[int],
                       min_fraction: float = 0.005,
                       seed: int = 42) -> list[set[int]]:
    """Louvain on the undirected unweighted influence projection.

    Communities smaller than ``min_fraction * n`` are merged into one
    community, following the paper's experimental setting.  The result is
    sorted by size descending and then by smallest node id, so the order is
    deterministic.
    """
    nodes = [int(u) for u in nodes]
    und = nx.Graph()
    und.add_nodes_from(nodes)
    und.add_edges_from((int(u), int(v)) for u, v in G.edges() if u != v)
    if und.number_of_edges() == 0:
        communities = [{u} for u in nodes]
    else:
        communities = list(nx.community.louvain_communities(
            und, seed=int(seed), weight=None
        ))
    threshold = max(1, int(round(float(min_fraction) * len(nodes))))
    small = [c for c in communities if len(c) < threshold]
    large = [c for c in communities if len(c) >= threshold]
    if small:
        merged: set[int] = set()
        for c in small:
            merged.update(int(u) for u in c)
        if merged:
            large.append(merged)
    if not large:
        large = [set(nodes)]
    return sorted(
        [set(int(u) for u in c) for c in large],
        key=lambda c: (-len(c), min(c)),
    )


def _positive_out(G: nx.DiGraph, nodes: Iterable[int]
                  ) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, list[tuple[int, float]]] = {int(u): [] for u in nodes}
    for u, v, data in G.edges(data=True):
        if int(data.get("polarity", 0)) > 0:
            p = float(data.get("probability", 0.0))
            if p > 0.0:
                out[int(u)].append((int(v), p))
    return out


def _positive_reverse(G: nx.DiGraph, nodes: Iterable[int]
                      ) -> dict[int, list[tuple[int, float]]]:
    rev: dict[int, list[tuple[int, float]]] = {int(u): [] for u in nodes}
    for u, v, data in G.edges(data=True):
        if int(data.get("polarity", 0)) > 0:
            p = float(data.get("probability", 0.0))
            if p > 0.0:
                rev[int(v)].append((int(u), p))
    return rev


def two_hop_lower_bound(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int],
                        k: int) -> dict:
    """Algorithm 2 of the paper: two-hop lower bound on saved nodes.

    ``LB`` is a lower bound on the expected number of saved nodes in the
    two-hop out-neighbourhood of the rumor seeds.  ``Q1`` and ``Q2`` are the
    first- and second-hop out-neighbours of the rumor seeds.
    """
    nodes = [int(u) for u in nodes]
    B_set = {int(u) for u in B}
    out = _positive_out(G, nodes)

    Q1: set[int] = set()
    for u in B_set:
        for v, _p in out[u]:
            if v not in B_set:
                Q1.add(v)
    Q2: set[int] = set()
    for u in Q1:
        for v, _p in out[u]:
            if v not in B_set and v not in Q1:
                Q2.add(v)
    two_hop = Q1 | Q2

    direct_B: dict[int, float] = {v: 0.0 for v in two_hop}
    for u in B_set:
        for v, p in out[u]:
            if v in direct_B:
                direct_B[v] += p

    kappa1: dict[int, float] = {v: float(direct_B.get(v, 0.0)) for v in two_hop}
    for u in Q1:
        base = float(direct_B.get(u, 0.0))
        if base <= 0.0:
            continue
        for v, p in out[u]:
            if v in kappa1:
                kappa1[v] += base * p

    kappa2: dict[int, float] = {}
    for v in Q1:
        extra = 1.0
        for u, p in out[v]:
            if u not in B_set:
                extra += p
        kappa2[v] = float(kappa1.get(v, 0.0) * extra)

    top_q1 = sorted(Q1, key=lambda u: (-kappa2.get(u, 0.0), int(u)))[: int(k)]
    selected_lb = set(top_q1)

    via_selected: dict[int, float] = {v: 0.0 for v in two_hop}
    for u in Q1:
        if u in selected_lb:
            continue
        base = float(direct_B.get(u, 0.0))
        if base <= 0.0:
            continue
        for v, p in out[u]:
            if v in via_selected:
                via_selected[v] += base * p

    kappa1_prime: dict[int, float] = {}
    for v in two_hop:
        if v in selected_lb:
            continue
        kappa1_prime[v] = float(direct_B.get(v, 0.0)) + float(via_selected[v])

    inf1 = float(sum(kappa1.values()))
    inf2 = float(sum(kappa1_prime.values()))
    LB = max(inf1 - inf2, 1.0)
    return {
        "Q1": sorted(Q1),
        "Q2": sorted(Q2),
        "two_hop": sorted(two_hop),
        "kappa1": kappa1,
        "kappa2": kappa2,
        "kappa1_prime": kappa1_prime,
        "inf1": inf1,
        "inf2": inf2,
        "LB": LB,
        "LB_seed_set": sorted(selected_lb),
    }


def sample_size_plan(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int],
                     k: int, communities: list[set[int]],
                     cfg: FIBMMConfig) -> dict:
    """Compute the paper's per-community CRP sample sizes, with a total cap."""
    nodes = [int(u) for u in nodes]
    B_set = {int(u) for u in B}
    lb = two_hop_lower_bound(G, nodes, B, k)
    two_hop = set(lb["two_hop"])
    n_two = len(two_hop)

    records = []
    for c_idx, community in enumerate(communities):
        n_q = len(community - B_set)
        if n_q <= 0:
            continue
        omega = (len(community & two_hop) / n_two) if n_two else 0.0
        if omega <= 0.0:
            continue
        k_q = min(float(n_q), max(1.0, omega * float(k)))
        lb_q = max(omega * float(lb["LB"]), 1e-9)
        alpha = math.sqrt(float(cfg.l) * math.log(n_q) + math.log(2.0))
        inside = max(
            (1.0 - 1.0 / math.e) * math.log(n_q / k_q)
            + float(cfg.l) * math.log(n_q) + math.log(2.0),
            1e-12,
        )
        beta = math.sqrt(inside)
        lam = (
            2.0 * n_q * ((1.0 - 1.0 / math.e) * alpha + beta) ** 2
            / (float(cfg.epsilon) ** 2)
        )
        theta = lam / lb_q
        records.append({
            "community_index": int(c_idx),
            "community_size": int(len(community)),
            "n_q": int(n_q),
            "omega": float(omega),
            "k_q": float(k_q),
            "LB_q": float(lb_q),
            "theta_raw": float(theta),
        })

    if not records:
        raise RuntimeError("FIBMM has no community with a positive rumor reach.")

    raw_total = sum(r["theta_raw"] for r in records)
    scale = min(1.0, float(cfg.max_crp_samples) / raw_total) if raw_total > 0 else 1.0
    for r in records:
        r["theta_scaled"] = max(
            int(cfg.min_samples_per_community),
            int(round(r["theta_raw"] * scale)),
        )

    # The minimum-per-community rule can push the total above the cap.
    while sum(r["theta_scaled"] for r in records) > int(cfg.max_crp_samples):
        largest = max(records, key=lambda r: r["theta_scaled"])
        if largest["theta_scaled"] <= int(cfg.min_samples_per_community):
            break
        largest["theta_scaled"] -= 1

    return {
        "LB": float(lb["LB"]),
        "inf1": float(lb["inf1"]),
        "inf2": float(lb["inf2"]),
        "Q1_size": len(lb["Q1"]),
        "Q2_size": len(lb["Q2"]),
        "two_hop_size": n_two,
        "LB_seed_set": list(lb["LB_seed_set"]),
        "theta_raw_total": float(raw_total),
        "scale": float(scale),
        "theta_total": int(sum(r["theta_scaled"] for r in records)),
        "records": records,
    }


def _sample_crp(v: int, rev_adj: dict[int, list[tuple[int, float]]],
                B_set: set[int], rng: random.Random) -> list[int]:
    """One community-aware reverse protector set on the positive CIC surrogate.

    Returns the nodes that reach ``v`` no later than any rumor seed in the
    sampled realization.  Returns ``[]`` when ``v`` is not reachable from any
    rumor seed (the paper's ``RC_i(v) = empty`` case).
    """
    dist = {v: 0}
    queue = deque([v])
    min_b_dist = None
    while queue:
        cur = queue.popleft()
        d = dist[cur]
        if min_b_dist is not None and d >= min_b_dist:
            continue
        for prev, p in rev_adj[cur]:
            if prev in dist:
                continue
            if rng.random() < p:
                nd = d + 1
                dist[prev] = nd
                if prev in B_set:
                    if min_b_dist is None or nd < min_b_dist:
                        min_b_dist = nd
                else:
                    queue.append(prev)
    if min_b_dist is None:
        return []
    return [int(u) for u, d in dist.items() if u not in B_set and d <= min_b_dist]


def build_crp_sets(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int],
                   communities: list[set[int]], plan: dict,
                   cfg: FIBMMConfig) -> dict:
    """Generate all CRP sets and build the coverage data structures."""
    nodes = [int(u) for u in nodes]
    B_set = {int(u) for u in B}
    rev = _positive_reverse(G, nodes)
    rng = random.Random(int(cfg.crp_seed))

    sets: list[np.ndarray] = []
    set_community: list[int] = []
    nonempty_counts: list[int] = []
    empty_counts: list[int] = []
    records_by_index = {int(r["community_index"]): r for r in plan["records"]}

    for c_idx, community in enumerate(communities):
        record = records_by_index.get(int(c_idx))
        if record is None:
            nonempty_counts.append(0)
            empty_counts.append(0)
            continue
        eligible = sorted(int(u) for u in community if u not in B_set)
        n_empty = 0
        n_nonempty = 0
        for _ in range(int(record["theta_scaled"])):
            v = rng.choice(eligible)
            crp = _sample_crp(v, rev, B_set, rng)
            if not crp:
                n_empty += 1
                continue
            arr = np.asarray(sorted(crp), dtype=np.int32)
            sets.append(arr)
            set_community.append(int(c_idx))
            n_nonempty += 1
        nonempty_counts.append(int(n_nonempty))
        empty_counts.append(int(n_empty))

    if not sets:
        raise RuntimeError("FIBMM generated no non-empty CRP sets.")

    n_sets = len(sets)
    rows = np.concatenate(sets)
    lengths = np.fromiter((len(a) for a in sets), dtype=np.int64, count=n_sets)
    cols = np.repeat(np.arange(n_sets, dtype=np.int32), lengths)
    node_sets = sp.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(len(nodes), n_sets),
    )
    set_community_arr = np.asarray(set_community, dtype=np.int32)

    gain_total = np.zeros(len(nodes), dtype=np.int32)
    gain_comm = np.zeros((len(communities), len(nodes)), dtype=np.int32)
    for j, arr in enumerate(sets):
        c = int(set_community_arr[j])
        gain_total[arr] += 1
        gain_comm[c, arr] += 1

    return {
        "sets": sets,
        "set_community": set_community_arr,
        "node_sets": node_sets,
        "nonempty_counts": np.asarray(nonempty_counts, dtype=np.int64),
        "empty_counts": np.asarray(empty_counts, dtype=np.int64),
        "gain_total": gain_total,
        "gain_comm": gain_comm,
        "covered": np.zeros(n_sets, dtype=bool),
        "covered_counts": np.zeros(len(communities), dtype=np.int64),
        "n_sets": int(n_sets),
        "total_nonempty": int(n_sets),
        "total_covered": 0,
    }


def _select_node(data: dict, available: np.ndarray, community_index: int | None,
                 node_count: int) -> int:
    if community_index is None:
        scores = np.where(available, data["gain_total"], -1)
    else:
        scores = np.where(available, data["gain_comm"][community_index], -1)
    best = int(np.argmax(scores))
    if scores[best] >= 0:
        return best
    # Fallback: no remaining node covers an uncovered set in the chosen
    # community.  Use the overall coverage gain, then the smallest id.
    scores = np.where(available, data["gain_total"], -1)
    best = int(np.argmax(scores))
    if scores[best] >= 0:
        return best
    candidates = np.flatnonzero(available)
    if candidates.size == 0:
        raise RuntimeError("FIBMM has no available node left.")
    return int(candidates[0])


def _apply_selection(data: dict, v: int, available: np.ndarray) -> None:
    available[v] = False
    ns = data["node_sets"]
    start, end = ns.indptr[v], ns.indptr[v + 1]
    covered = data["covered"]
    for j in ns.indices[start:end]:
        j = int(j)
        if covered[j]:
            continue
        covered[j] = True
        c = int(data["set_community"][j])
        data["covered_counts"][c] += 1
        data["total_covered"] += 1
        arr = data["sets"][j]
        data["gain_total"][arr] -= 1
        data["gain_comm"][c, arr] -= 1


def select_fibmm(G: nx.DiGraph, nodes: Iterable[int], B: Iterable[int], k: int,
                 cfg: FIBMMConfig,
                 communities: list[set[int]] | None = None) -> tuple[list[int], dict]:
    """Run FIBMM and return ``(S, info)``.

    ``S`` is exactly K unique eligible nodes.  ``info`` records the paper's
    internal fairness metrics, the sample-size plan, and the two-stage counts.
    """
    t0 = time.perf_counter()
    nodes = [int(u) for u in nodes]
    B_set = {int(u) for u in B}
    if int(k) > len(nodes) - len(B_set):
        raise ValueError("K exceeds eligible nodes.")

    if communities is None:
        communities = detect_communities(
            G, nodes, min_fraction=cfg.community_min_fraction,
            seed=cfg.community_seed,
        )
    plan = sample_size_plan(G, nodes, B, int(k), communities, cfg)
    data = build_crp_sets(G, nodes, B, communities, plan, cfg)

    available = np.ones(len(nodes), dtype=bool)
    available[list(B_set)] = False
    selected: list[int] = []
    # Paper: k - floor(k*h) overall-utility seeds, then fairness seeds.
    # Computing it this way avoids floating-point floor(0.9999999)=0 at K=10.
    n_stage1 = int(k) - int(math.floor(float(k) * float(cfg.h)))
    n_stage1 = max(0, min(int(k), n_stage1))
    stage1_selected: list[int] = []
    stage2_selected: list[int] = []

    for _ in range(n_stage1):
        v = _select_node(data, available, None, len(nodes))
        _apply_selection(data, v, available)
        selected.append(v)
        stage1_selected.append(v)

    while len(selected) < int(k):
        nonempty = data["nonempty_counts"]
        valid = np.flatnonzero(nonempty > 0)
        if valid.size:
            ratios = data["covered_counts"][valid] / nonempty[valid]
            c_star = int(valid[int(np.argmin(ratios))])
        else:
            c_star = None
        v = _select_node(data, available, c_star, len(nodes))
        _apply_selection(data, v, available)
        selected.append(v)
        stage2_selected.append(v)

    if len(selected) != int(k) or len(set(selected)) != int(k):
        raise RuntimeError("FIBMM returned an invalid seed set.")
    if set(selected) & B_set:
        raise RuntimeError("FIBMM selected an opponent seed.")

    nonempty = data["nonempty_counts"]
    covered = data["covered_counts"]
    ratios = np.ones(len(communities), dtype=float)
    for c in range(len(communities)):
        if nonempty[c] > 0:
            ratios[c] = float(covered[c] / nonempty[c])
    valid = np.flatnonzero(nonempty > 0)
    tau = float(ratios[valid].min()) if valid.size else 1.0
    pi_total = (
        float(data["total_covered"] / data["total_nonempty"])
        if data["total_nonempty"] else 0.0
    )
    sigma = (1.0 - float(cfg.h)) * pi_total + float(cfg.h) * tau

    community_rows = []
    for c, community in enumerate(communities):
        community_rows.append({
            "community_index": int(c),
            "community_size": int(len(community)),
            "nonempty_crp": int(nonempty[c]),
            "empty_crp": int(data["empty_counts"][c]),
            "covered_crp": int(covered[c]),
            "estimated_saving_ratio": float(ratios[c]),
        })

    info = {
        "algorithm": "FIBMM",
        "paper": "Xiao et al., IEEE TCSS 2025",
        "h": float(cfg.h),
        "epsilon": float(cfg.epsilon),
        "l": int(cfg.l),
        "k": int(k),
        "n_communities": int(len(communities)),
        "communities": [sorted(int(u) for u in c) for c in communities],
        "sample_plan": plan,
        "crp_sets": int(data["n_sets"]),
        "crp_total_nonempty": int(data["total_nonempty"]),
        "crp_total_covered": int(data["total_covered"]),
        "pi_total": float(pi_total),
        "tau": float(tau),
        "sigma": float(sigma),
        "community_metrics": community_rows,
        "stage1_count": int(len(stage1_selected)),
        "stage2_count": int(len(stage2_selected)),
        "stage1_seeds": [int(u) for u in stage1_selected],
        "stage2_seeds": [int(u) for u in stage2_selected],
        "selection_runtime_sec": float(time.perf_counter() - t0),
        "surrogate": "positive_edge_cic",
        "sample_cap": int(cfg.max_crp_samples),
        "crp_seed": int(cfg.crp_seed),
    }
    return [int(u) for u in selected], info
