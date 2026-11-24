
# ============================== CELL 3 ==============================
# ============================================================
# SECTION 1: IMPORTS, PATHS, AND CONFIGURATION (8 METHODS)
# ============================================================
import os
import json
import math
import time
import heapq
import random
import hashlib
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Auto-locate soc-sign-bitcoinotc (.csv.gz or .csv) anywhere in /kaggle/input or current folder
input_matches = list(Path("/kaggle/input").rglob("*bitcoinotc*.csv*")) if Path("/kaggle/input").exists() else []
local_matches = list(Path(".").glob("*bitcoinotc*.csv*"))
all_matches = sorted(input_matches + local_matches)

if not all_matches:
    raise FileNotFoundError(
        "Could not find any *bitcoinotc* CSV file. Available files in /kaggle/input: "
        + str([str(p) for p in Path("/kaggle/input").rglob("*")])
    )

DATA_FILE = all_matches[0]
OUTPUT_DIR = (
    Path("/kaggle/working/competitive_im_results_no_celf_csv_persist")
    if Path("/kaggle/working").exists()
    else Path("./competitive_im_results_no_celf_csv_persist")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 42
K_VALUES = [10, 20, 30, 40, 50]

CONFIG = {
    "k": 10,
    "candidate_pool_ratio": 10,
    "random_candidate_fraction": 0.30,
    "optimization_mc_runs": 15,
    "evaluation_mc_runs": 100,
    "ssa_population": 12,
    "ssa_iterations": 10,
    "ssa_mutation_rate": 0.20,
    "master_seed": MASTER_SEED,
    "resume": True,
}

# Experiment Toggles (CELF disabled for this run)
RUN_NCGE = True          # Liang et al. (2025) - Centrality Candidate Pool + Greedy
RUN_CRIEG = True         # Hong et al. (IEEE TCYB 2021) - Competitive Reverse Reachability
RUN_SIGNED_GA = True     # Chakraborty et al. (2025) - Signed Voter Equilibrium Gradient Ascent
RUN_CELF = False         # Disabled for this experiment
RUN_SSA = True           # Proposed: 70/30 Signed Pool + Discrete Salp Swarm Algorithm

print("DATA_FILE:", DATA_FILE)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("K_VALUES:", K_VALUES)

# ============================== CELL 5 ==============================
# ============================================================
# CELL 2: LOAD, VALIDATE AND MAP BITCOIN OTC RATINGS
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    header=None,
    names=["source", "target", "rating", "time"]
)

if df.empty:
    raise ValueError("The dataset is empty.")

# Convert all four columns to numeric values.
for col in ["source", "target", "rating", "time"]:
    df[col] = pd.to_numeric(df[col], errors="raise")

# Do not silently discard malformed or missing rows.
if df.isna().any().any():
    bad_counts = df.isna().sum()
    raise ValueError(
        "Missing values found; inspect these column counts before proceeding:\n"
        + bad_counts.to_string()
    )

if not np.isfinite(df[["source", "target", "rating", "time"]].to_numpy(dtype=float)).all():
    raise ValueError("Dataset contains non-finite numeric values.")

# IDs and ratings must be integer-valued. Timestamps may be decimals.
for col in ["source", "target", "rating"]:
    if not np.equal(df[col].to_numpy(dtype=float),
                    np.floor(df[col].to_numpy(dtype=float))).all():
        raise ValueError(f"Column '{col}' contains non-integer values.")
    df[col] = df[col].astype("int64")

df["time"] = df["time"].astype("float64")

if not df["rating"].between(-10, 10).all():
    raise ValueError("Ratings outside the expected range [-10, 10].")

# Map signed trust ratings to transmission probabilities.
# Positive relationships preserve the current rumour.
# Negative relationships transmit the opposite rumour after a successful attempt.
# Zero-rated relationships do not transmit.
conditions = [
    df["rating"].between(1, 3),
    df["rating"].between(4, 7),
    df["rating"].between(8, 10),
    df["rating"].between(-3, -1),
    df["rating"].between(-7, -4),
    df["rating"].between(-10, -8),
    df["rating"].eq(0)
]
categories = [
    "weak_positive", "medium_positive", "strong_positive",
    "weak_negative", "medium_negative", "strong_negative", "neutral"
]
probabilities = [0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.0]

df["strength_category"] = np.select(conditions, categories, default="invalid")
df["probability"] = np.select(conditions, probabilities, default=np.nan).astype("float64")
df["strength_level"] = df["rating"].abs().astype("int64")

if (df["strength_category"] == "invalid").any() or df["probability"].isna().any():
    raise ValueError("At least one rating was not mapped correctly.")

if not df["probability"].between(0.0, 1.0).all():
    raise ValueError("Invalid transmission probability detected.")

duplicate_pairs = int(df.duplicated(subset=["source", "target"]).sum())

print("Dataset loaded and validated successfully!")
print("Dataset shape:", df.shape)
print("Unique users:", len(set(df["source"]) | set(df["target"])))
print("Positive ratings:", int((df["rating"] > 0).sum()))
print("Negative ratings:", int((df["rating"] < 0).sum()))
print("Negative-edge category counts:")
display(df.loc[df["rating"] < 0, "strength_category"].value_counts())
print("Zero ratings:", int((df["rating"] == 0).sum()))
print("Duplicate source-target pairs:", duplicate_pairs)
print("Timestamp range:", (df["time"].min(), df["time"].max()))

print("\nRelationship category counts:")
display(df["strength_category"].value_counts())

print("\nRating distribution:")
display(df["rating"].value_counts().sort_index())

print("\nFirst five processed rows:")
display(df.head())

# ============================== CELL 7 ==============================
G = nx.DiGraph()
G.add_nodes_from(sorted(set(df["source"]).union(set(df["target"]))))
for row in df.itertuples(index=False):
    G.add_edge(int(row.target), int(row.source),
               original_rating=int(row.rating),
               category=row.strength_category,
               probability=float(row.probability),
               strength=int(row.strength_level),
               polarity=1 if int(row.rating) > 0 else (-1 if int(row.rating) < 0 else 0))

if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    raise ValueError("Empty graph; check dataset preprocessing.")

NODES = tuple(sorted(G.nodes()))
ADJ = {u: tuple(sorted(G.successors(u))) for u in NODES}
EDGE_PROB = {(u,v): float(G[u][v]["probability"]) for u,v in G.edges()}
EDGE_STRENGTH = {(u,v): int(G[u][v]["strength"]) for u,v in G.edges()}
EDGE_POLARITY = {(u,v): int(G[u][v]["polarity"]) for u,v in G.edges()}

print("Nodes:", G.number_of_nodes())
print("Directed edges:", G.number_of_edges())
print("Positive edges:", sum(d["probability"] > 0 for _,_,d in G.edges(data=True)))
print("Zero-probability edges:", sum(d["probability"] == 0 for _,_,d in G.edges(data=True)))
print("Negative edges with non-zero transmission probability:",
      sum(d["polarity"] == -1 and d["probability"] > 0 for _,_,d in G.edges(data=True)))

# ============================== CELL 9 ==============================
def competitive_maki_thompson(seeds_A, seeds_B, rng):
    """
    Competitive signed Maki-Thompson diffusion.

    Positive edge: successful transmission preserves the source rumour.
    Negative edge: successful transmission flips to the opposite rumour.
    Zero edge: probability 0, so it cannot transmit.

    Adopted nodes never switch rumours. Only ignorant targets can be activated.
    Directed edge orientation is preserved exactly as constructed in G.
    """
    A, B = set(seeds_A), set(seeds_B)
    if A & B:
        raise ValueError("A and B seed sets overlap.")
    if not A.issubset(G) or not B.issubset(G):
        raise ValueError("Unknown seed node.")

    # Initial seeds retain their assigned rumour, regardless of incident edge signs.
    states = {u: "I" for u in NODES}
    for u in sorted(A):
        states[u] = "A_S"
    for u in sorted(B):
        states[u] = "B_S"

    # Each spreader contacts each outgoing neighbour at most once.
    contacted = {u: set() for u in NODES}
    max_rounds = G.number_of_edges() + 1

    for round_no in range(1, max_rounds + 1):
        old = states
        new = old.copy()
        spreaders = [u for u in NODES if old[u] in ("A_S", "B_S")]
        if not spreaders:
            break

        proposals = defaultdict(list)
        stiflers = set()

        for source in spreaders:
            available = [v for v in ADJ[source] if v not in contacted[source]]
            if not available:
                stiflers.add(source)
                continue

            target = rng.choice(available)
            contacted[source].add(target)

            # Already-adopted nodes (A or B) cannot switch to another rumour.
            if old[target] != "I":
                stiflers.add(source)
                continue

            if rng.random() < EDGE_PROB[(source, target)]:
                source_rumour = "A" if old[source] == "A_S" else "B"
                polarity = EDGE_POLARITY[(source, target)]
                if polarity < 0:
                    transmitted_rumour = "B" if source_rumour == "A" else "A"
                else:
                    transmitted_rumour = source_rumour

                # Preserve the original strongest-absolute-trust tie-break.
                # If equally strong successful attempts propose different rumours,
                # choose uniformly among the tied proposals.
                proposals[target].append(
                    (transmitted_rumour, EDGE_STRENGTH[(source, target)])
                )

        for target, attempts in proposals.items():
            strongest = max(strength for _, strength in attempts)
            contenders = {
                rumour for rumour, strength in attempts
                if strength == strongest
            }
            winner = rng.choice(sorted(contenders))
            new[target] = winner + "_S"

        # Preserve existing Maki-Thompson stifling behaviour.
        for u in stiflers:
            if old[u] == "A_S":
                new[u] = "A_T"
            elif old[u] == "B_S":
                new[u] = "B_T"

        states = new
    else:
        raise RuntimeError("Simulation exceeded finite-contact safety bound.")

    spread_A = sum(x in ("A_S", "A_T") for x in states.values())
    spread_B = sum(x in ("B_S", "B_T") for x in states.values())
    ignorant = sum(x == "I" for x in states.values())

    if spread_A + spread_B + ignorant != len(NODES):
        raise AssertionError("Final state counts do not sum to graph size.")

    return {
        "spread_A": int(spread_A),
        "spread_B": int(spread_B),
        "net_advantage": int(spread_A - spread_B),
        "ignorant": int(ignorant),
        "rounds": int(round_no)
    }

def evaluate_seed_set(A, B, sim_seeds, baseline_B_cache={}):
    """
    Evaluate Rumour A seed set against fixed Rumour B seed set across sim_seeds.
    Also tracks baseline Rumour B spread (with A=empty) so blocked B nodes can be measured.
    """
    b_key = (tuple(sorted(set(B))), tuple(sim_seeds))
    if b_key not in baseline_B_cache:
        b_only_spreads = []
        for s in sim_seeds:
            rng = np.random.default_rng(int(s))
            res_b = competitive_maki_thompson([], B, rng=rng)
            b_only_spreads.append(res_b["spread_B"])
        baseline_B_cache[b_key] = float(np.mean(b_only_spreads))

    spreads_A = []
    spreads_B = []
    net_advs = []

    for s in sim_seeds:
        rng = np.random.default_rng(int(s))
        res = competitive_maki_thompson(A, B, rng=rng)
        sA = res["spread_A"]
        sB = res["spread_B"]
        spreads_A.append(sA)
        spreads_B.append(sB)
        net_advs.append(sA - sB)

    mean_A = float(np.mean(spreads_A))
    mean_B = float(np.mean(spreads_B))
    blocked_B = max(0.0, baseline_B_cache[b_key] - mean_B)

    return {
        "spread_A": mean_A,
        "spread_B": mean_B,
        "blocked_B": blocked_B,
        "net_advantage": float(np.mean(net_advs)),
        "weighted_fitness": 1.0 * mean_A + 1.5 * blocked_B,
    }


def fitness_net_advantage(A, B, sim_seeds, cache):
    """
    Weighted optimization objective:
        1.0 * mean(spread_A) + 1.5 * mean(blocked_B)
    """
    key = tuple(sorted(set(A)))
    if key not in cache:
        cache[key] = evaluate_seed_set(A, B, sim_seeds)["weighted_fitness"]
    return cache[key]




# ============================== CELL 11 ==============================
# Mapping and polarity sanity checks.
assert CONFIG["candidate_pool_ratio"] >= 1
assert CONFIG["evaluation_mc_runs"] >= 2

expected_probability = {
    0: 0.0,
    -1: 0.1, -2: 0.1, -3: 0.1,
    -4: 0.5, -5: 0.5, -6: 0.5, -7: 0.5,
    -8: 0.9, -9: 0.9, -10: 0.9,
    1: 0.1, 2: 0.1, 3: 0.1,
    4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5,
    8: 0.9, 9: 0.9, 10: 0.9
}
for rating, probability in expected_probability.items():
    rows = df.loc[df["rating"] == rating, "probability"]
    if not rows.empty:
        assert np.allclose(rows.to_numpy(), probability), (
            f"Incorrect probability mapping for rating {rating}"
        )

assert (df.loc[df["rating"] == 0, "probability"] == 0.0).all()
assert (df.loc[df["rating"] < 0, "probability"] > 0.0).all()

rng0 = random.Random(MASTER_SEED)
B0 = set(rng0.sample(list(NODES), CONFIG["k"]))
A0 = set(rng0.sample([u for u in NODES if u not in B0], CONFIG["k"]))
one = competitive_maki_thompson(A0, B0, random.Random(MASTER_SEED + 1))
print("Signed counter-influence preflight:", one)
assert one["spread_A"] >= CONFIG["k"] and one["spread_B"] >= CONFIG["k"]
assert one["spread_A"] + one["spread_B"] <= len(NODES)


# ============================== CELL 14 ==============================
def compute_ncge_centrality_scores(alpha=0.5, beta=0.3, gamma=0.2, k_approx=200):
    """
    Computes node centrality scores D_v = alpha*degree + beta*closeness + gamma*betweenness
    following NCGE (Liang et al., ESWA 2025, Algorithm 3).
    """
    n = G.number_of_nodes()
    out_deg = dict(G.out_degree())
    max_deg = max(out_deg.values()) or 1.0
    deg_norm = {u: d / max_deg for u, d in out_deg.items()}

    if n <= 10000:
        clo = nx.closeness_centrality(G)
        bet = nx.betweenness_centrality(G, k=min(k_approx, n), seed=MASTER_SEED)
    else:
        # Fast scalable proxy on very large graphs (e.g., Slashdot)
        pr = nx.pagerank(G, alpha=0.85, max_iter=50)
        max_pr = max(pr.values()) or 1.0
        clo = {u: v / max_pr for u, v in pr.items()}
        bet = nx.betweenness_centrality(G, k=min(100, n), seed=MASTER_SEED)

    max_clo = max(clo.values()) or 1.0
    max_bet = max(bet.values()) or 1.0

    return {
        u: float(
            alpha * deg_norm.get(u, 0.0)
            + beta * (clo.get(u, 0.0) / max_clo)
            + gamma * (bet.get(u, 0.0) / max_bet)
        )
        for u in NODES
    }


NCGE_CENTRALITY_SCORES = compute_ncge_centrality_scores()


def generate_weighted_candidate_pool(B, pool_size, random_fraction, random_seed):
    """
    70/30 Signed Weighted Candidate Pool: polarity-aware one-hop candidate pool.

    Score(u) = sum(probability on positive outgoing edges)
               - sum(probability on negative outgoing edges).
    """
    B = set(B)
    eligible = [u for u in NODES if u not in B]
    if pool_size > len(eligible):
        raise ValueError("Not enough eligible nodes.")

    score_details = {}
    for u in eligible:
        positive_score = 0.0
        negative_score = 0.0
        for v in ADJ[u]:
            probability = EDGE_PROB[(u, v)]
            polarity = EDGE_POLARITY[(u, v)]
            if polarity > 0:
                positive_score += probability
            elif polarity < 0:
                negative_score += probability

        score_details[u] = {
            "positive_outgoing_score": float(positive_score),
            "negative_outgoing_score": float(negative_score),
            "net_signed_candidate_score": float(positive_score - negative_score),
        }

    ranked = sorted(
        eligible,
        key=lambda u: (
            -score_details[u]["net_signed_candidate_score"],
            -score_details[u]["positive_outgoing_score"],
            score_details[u]["negative_outgoing_score"],
            int(u)
        )
    )
    n_random = int(round(pool_size * random_fraction))
    n_ranked = pool_size - n_random
    top_ranked = ranked[:n_ranked]
    remaining = ranked[n_ranked:]
    rng = random.Random(random_seed)
    random_nodes = rng.sample(remaining, n_random) if n_random else []
    pool = top_ranked + random_nodes

    if len(pool) != pool_size or len(set(pool)) != pool_size or set(pool) & B:
        raise AssertionError("Invalid Weighted candidate pool.")
    return pool, score_details


def generate_ncge_candidate_pool(B, pool_size):
    """
    NCGE Candidate Pool (Liang et al., ESWA 2025, Algorithm 3):
    Selects top-m eligible nodes ranked by network centrality D_v.
    """
    B = set(B)
    eligible = [u for u in NODES if u not in B]
    if pool_size > len(eligible):
        raise ValueError("Not enough eligible nodes.")

    score_details = {
        u: {"ncge_centrality_score": float(NCGE_CENTRALITY_SCORES[u])}
        for u in eligible
    }
    ranked = sorted(
        eligible,
        key=lambda u: (-NCGE_CENTRALITY_SCORES[u], int(u))
    )
    pool = ranked[:pool_size]

    if len(pool) != pool_size or len(set(pool)) != pool_size or set(pool) & B:
        raise AssertionError("Invalid NCGE candidate pool.")
    return pool, score_details


print("Weighted (70/30) and NCGE (Liang et al., 2025) candidate-pool generators ready.")

# ============================== CELL 16 ==============================
def top_out_degree_seeds(pool, k):
    """Top-k candidates by directed out-degree in the influence graph."""
    ranked = sorted(pool, key=lambda u: (-G.out_degree(u), u))
    return ranked[:k]


def compute_signed_pagerank(alpha=0.85, max_iter=200, tol=1e-10):
    """
    Two-layer signed PageRank:
      positive edge: preserve polarity
      negative edge: flip polarity
    Edge transition weight = absolute original rating.
    Return positive-layer score minus negative-layer score.
    """
    H = nx.DiGraph()
    positive_nodes = [(u, 1) for u in NODES]
    negative_nodes = [(u, -1) for u in NODES]
    H.add_nodes_from(positive_nodes)
    H.add_nodes_from(negative_nodes)

    for u, v, data in G.edges(data=True):
        rating = int(data["original_rating"])
        weight = abs(rating)
        if weight == 0:
            continue

        if rating > 0:
            H.add_edge((u, 1), (v, 1), weight=weight)
            H.add_edge((u, -1), (v, -1), weight=weight)
        else:
            H.add_edge((u, 1), (v, -1), weight=weight)
            H.add_edge((u, -1), (v, 1), weight=weight)

    # Teleport only to positive-polarity copies.
    personalization = {node: 0.0 for node in H.nodes()}
    positive_mass = 1.0 / len(NODES)
    for u in NODES:
        personalization[(u, 1)] = positive_mass

    pr = nx.pagerank(
        H,
        alpha=alpha,
        personalization=personalization,
        weight="weight",
        max_iter=max_iter,
        tol=tol
    )

    return {u: float(pr[(u, 1)] - pr[(u, -1)]) for u in NODES}


SIGNED_PAGERANK_PREPROCESSING_START = time.perf_counter()
SIGNED_PAGERANK_SCORES = compute_signed_pagerank()
SIGNED_PAGERANK_PREPROCESSING_RUNTIME_SEC = (
    time.perf_counter() - SIGNED_PAGERANK_PREPROCESSING_START
)


def signed_pagerank_seeds(pool, k):
    """Top-k candidates by positive-minus-negative signed PageRank."""
    ranked = sorted(
        pool,
        key=lambda u: (-SIGNED_PAGERANK_SCORES[u], u)
    )
    return ranked[:k]


print("Baseline ranking functions ready.")
print("Signed PageRank computed for", len(SIGNED_PAGERANK_SCORES), "nodes.")
print(
    "Signed PageRank one-time preprocessing:",
    round(SIGNED_PAGERANK_PREPROCESSING_RUNTIME_SEC, 4),
    "seconds"
)


# ============================== CELL 19 ==============================
# ============================================================
# SECTION 8 (ADDITION): CRIEG (HONG 2021) & SIGNED-GA (CHAKRABORTY 2025)
# ============================================================
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def crieg_seeds(B, k, num_sketches=2000, random_seed=42):
    """
    CRIEG (Hong et al., IEEE TCYB 2021)[cite: 1]:
    Builds Competitive Reverse Reachable (CRR) sketches on the positive subgraph[cite: 4],
    partitioning nodes into competitive-free D_g(u) (arriving strictly before B, weight=1.0)
    and competitive B_g(u) (tying with B, weight=0.5)[cite: 4], then greedily selects k
    nodes from all eligible nodes V \ B[cite: 3, 5].
    """
    rng = random.Random(random_seed)
    B_set = set(B)
    eligible = [u for u in NODES if u not in B_set]
    if len(eligible) < k:
        raise ValueError("Eligible nodes fewer than k.")

    rev_adj = {u: [] for u in NODES}
    for u, v, data in G.edges(data=True):
        if int(data.get("polarity", 0)) > 0:
            prob = float(data.get("probability", 0.0))
            if prob > 0.0:
                rev_adj[v].append((u, prob))

    node_gains = {u: {} for u in eligible}

    for s_idx in range(num_sketches):
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

        for node, d_val in dist.items():
            if node in B_set:
                continue
            if min_b_dist is None or d_val < min_b_dist:
                node_gains[node][s_idx] = 1.0
            elif d_val == min_b_dist:
                node_gains[node][s_idx] = 0.5

    selected = []
    covered_weight = [0.0] * num_sketches
    remaining = set(eligible)

    for _ in range(k):
        best_node = None
        best_marginal = -1.0
        for u in remaining:
            gain = 0.0
            for s_idx, w in node_gains[u].items():
                if w > covered_weight[s_idx]:
                    gain += (w - covered_weight[s_idx])
            if gain > best_marginal or (gain == best_marginal and (best_node is None or int(u) < int(best_node))):
                best_marginal = gain
                best_node = u

        selected.append(best_node)
        remaining.remove(best_node)
        for s_idx, w in node_gains[best_node].items():
            if w > covered_weight[s_idx]:
                covered_weight[s_idx] = w

    return [int(x) for x in selected]


def signed_ga_seeds(B, k, steps=30):
    """
    Signed Voter Gradient Ascent (Chakraborty et al., Feb 2025)[cite: 15, 19]:
    Solves the steady-state signed Voter equilibrium system [L + diag(p_A + p_B)] x_A* = p_A - 1^T W^-
    and projects gradient ascent steps onto the budget-k simplex, selecting the top-k allocated
    nodes from all eligible nodes V \ B[cite: 18, 19, 20].
    """
    B_set = set(B)
    node_list = sorted(NODES, key=lambda x: int(x))
    n = len(node_list)
    idx_map = {u: i for i, u in enumerate(node_list)}

    rows, cols, vals_signed, vals_abs = [], [], [], []
    in_abs_sum = np.zeros(n, dtype=float)
    in_neg_sum = np.zeros(n, dtype=float)

    for u, v, data in G.edges(data=True):
        i = idx_map[v]  # target node receiving influence
        j = idx_map[u]  # source node exerting influence
        pol = int(data.get("polarity", 0))
        prob = float(data.get("probability", 0.0))
        if pol == 0 or prob <= 0.0:
            continue
        w_ji = pol * prob
        rows.append(i)
        cols.append(j)
        vals_signed.append(w_ji)
        vals_abs.append(abs(w_ji))
        in_abs_sum[i] += abs(w_ji)
        if w_ji < 0:
            in_neg_sum[i] += abs(w_ji)  # - (1^T W^-)_i = sum_{w_ji < 0} |w_ji|

    W_signed = sp.csr_matrix((vals_signed, (rows, cols)), shape=(n, n))
    # Signed Laplacian L = diag(in_abs_sum) - W_signed[cite: 19]
    L = sp.diags(in_abs_sum + 1e-6) - W_signed

    p_B = np.zeros(n, dtype=float)
    for b_node in B_set:
        p_B[idx_map[b_node]] = 1.0

    eligible_indices = np.array([idx_map[u] for u in node_list if u not in B_set], dtype=int)
    p_A = np.zeros(n, dtype=float)
    p_A[eligible_indices] = float(k) / len(eligible_indices)

    def project_simplex_eligible(v_elig, budget):
        u_sorted = np.sort(v_elig)[::-1]
        cssv = np.cumsum(u_sorted) - budget
        ind = np.arange(1, len(v_elig) + 1)
        cond = u_sorted - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        return np.maximum(v_elig - theta, 0.0)

    rhs_base = in_neg_sum.copy()
    step_size = float(k)

    for _ in range(steps):
        M = (L + sp.diags(p_A + p_B)).tocsc()
        x_A = spla.spsolve(M, p_A + rhs_base)
        x_A = np.clip(x_A, 0.0, 1.0)
        # Adjoint solve for gradient: grad = (1/n) * (I - diag(x_A)) * M^{-T} 1[cite: 19]
        adj = spla.spsolve(M.T.tocsc(), np.ones(n, dtype=float))
        grad = (1.0 - x_A) * adj / float(n)

        p_A_elig = p_A[eligible_indices] + step_size * grad[eligible_indices]
        p_A[:] = 0.0
        p_A[eligible_indices] = project_simplex_eligible(p_A_elig, float(k))
        step_size *= 0.92

    eligible_nodes = [u for u in node_list if u not in B_set]
    ranked = sorted(
        eligible_nodes,
        key=lambda u: (-p_A[idx_map[u]], int(u))
    )
    return [int(x) for x in ranked[:k]]

# ============================== CELL 21 ==============================
def celf_lazy_greedy(pool, B, k, sim_seeds, progress_label="CELF"):
    """
    Lazy-greedy seed selection over the supplied candidate pool.

    CELF refreshes stale marginal gains when a candidate reaches the heap top.
    Because monotonicity/submodularity are not established for this signed
    competitive Maki-Thompson objective, this is an empirical heuristic rather
    than a claim of the classical CELF approximation guarantee.
    """
    pool = list(dict.fromkeys(pool))
    if len(pool) < k:
        raise ValueError(f"{progress_label}: candidate pool smaller than k.")
    cache = {}
    selected = []
    current = fitness_net_advantage(selected, B, sim_seeds, cache)
    heap = []

    print(f"{progress_label}: initial marginal evaluations for {len(pool)} candidates...")
    for idx, node in enumerate(pool, start=1):
        gain = fitness_net_advantage([node], B, sim_seeds, cache) - current
        heapq.heappush(heap, (-gain, int(node), 0))
        if idx % 50 == 0 or idx == len(pool):
            print(f"{progress_label}: initialized {idx}/{len(pool)} candidates")

    hist = []
    refreshes = 0
    while len(selected) < k:
        if not heap:
            raise RuntimeError(f"{progress_label}: CELF heap exhausted.")
        neg_gain, node, last_size = heapq.heappop(heap)
        if node in selected:
            continue
        if last_size != len(selected):
            gain = fitness_net_advantage(selected + [node], B, sim_seeds, cache) - current
            heapq.heappush(heap, (-gain, int(node), len(selected)))
            refreshes += 1
            continue

        selected.append(node)
        current = fitness_net_advantage(selected, B, sim_seeds, cache)
        hist.append({
            "step": len(selected), "node": int(node),
            "net_advantage": float(current),
            "marginal_gain_at_acceptance": float(-neg_gain),
            "heap_refreshes": int(refreshes),
            "unique_fitness_evaluations": len(cache)
        })
        print(
            f"{progress_label}: selected {len(selected)}/{k}; "
            f"net advantage={current:.2f}; refreshes={refreshes}"
        )

    if len(selected) != k or len(set(selected)) != k or set(selected) & set(B):
        raise AssertionError(f"{progress_label}: invalid selected seed set.")
    return selected, pd.DataFrame(hist), cache


def celf_sampled_local_search(initial, pool, B, sim_seeds, cache,
                              passes=1, max_swaps=300, random_seed=42,
                              progress_label="CELF + Local Search"):
    """Sampled one-for-one local search initialized from a CELF solution."""
    selected = list(initial)
    pool = list(dict.fromkeys(pool))
    B = set(B)
    rng = random.Random(random_seed)
    score = fitness_net_advantage(selected, B, sim_seeds, cache)
    history = [{
        "pass": 0, "net_advantage": float(score),
        "swap_out": None, "swap_in": None, "tested_swaps": 0,
        "unique_fitness_evaluations": len(cache)
    }]

    for p in range(1, passes + 1):
        outside = [u for u in pool if u not in set(selected) and u not in B]
        total = len(selected) * len(outside)
        if total == 0:
            break
        if total <= max_swaps:
            pairs = [(old, new) for old in selected for new in outside]
        else:
            pair_set = set()
            while len(pair_set) < max_swaps:
                pair_set.add((rng.choice(selected), rng.choice(outside)))
            pairs = sorted(pair_set)

        best_score, best_pair = score, None
        for idx, (old, new) in enumerate(pairs, start=1):
            trial = [u for u in selected if u != old] + [new]
            trial_score = fitness_net_advantage(trial, B, sim_seeds, cache)
            if trial_score > best_score + 1e-12:
                best_score, best_pair = trial_score, (old, new)
            if idx % 100 == 0 or idx == len(pairs):
                print(f"{progress_label}: pass {p}, tested {idx}/{len(pairs)} swaps")

        if best_pair is None:
            history.append({
                "pass": p, "net_advantage": float(score),
                "swap_out": None, "swap_in": None,
                "tested_swaps": len(pairs),
                "unique_fitness_evaluations": len(cache)
            })
            print(f"{progress_label}: no improving sampled swap in pass {p}.")
            break

        old, new = best_pair
        selected.remove(old)
        selected.append(new)
        score = best_score
        history.append({
            "pass": p, "net_advantage": float(score),
            "swap_out": int(old), "swap_in": int(new),
            "tested_swaps": len(pairs),
            "unique_fitness_evaluations": len(cache)
        })
        print(f"{progress_label}: pass {p}, {old} -> {new}; net advantage={score:.2f}")

    if len(selected) != len(initial) or len(set(selected)) != len(selected) or set(selected) & B:
        raise AssertionError(f"{progress_label}: invalid refined seed set.")
    return selected, pd.DataFrame(history), cache


print("CELF and CELF + sampled local-search functions ready.")

# ============================== CELL 23 ==============================
# if during experiment any set become broken like duplicate nodes or fewer than 10 nides then it is used to fix it

def repair_solution(nodes, pool, k, rng):
    pool_set = set(pool)
    result, seen = [], set()
    for u in nodes:
        if u in pool_set and u not in seen:
            result.append(u); seen.add(u)
    remaining = [u for u in pool if u not in seen]
    if len(result)<k: result.extend(rng.sample(remaining,k-len(result)))
    return result[:k]


def discrete_ssa(pool, B, k, sim_seeds, population_size=12,
                 iterations=10, mutation_rate=0.2, random_seed=42):
    rng = random.Random(random_seed)
   # cache: Creates an empty dictionary to remember scores of seed sets we already tested.
    cache = {}
    score = lambda sol: fitness_net_advantage(sol, B, sim_seeds, cache)
# Create population_size candidate seed sets, each containing k unique nodes.
    population = [rng.sample(pool,k) for _ in range(population_size)]
# Evaluate each initial candidate using mean Net Advantage over sim_seeds.
    scores = [score(s) for s in population]
#best_i: Finds the index of the highest-scoring salp in the initial population.
    best_i = int(np.argmax(scores))
#food, food_score: Saves a copy of that best seed set as food (the target the swarm chases) and its score as food_score.
    food, food_score = list(population[best_i]), float(scores[best_i])
#hist: Starts a log list recording Iteration 0's best score, average swarm score, and how many distinct seed sets exist in the population.
    hist = [{"iteration":0,"best_net_advantage":food_score,
             "mean_net_advantage":float(np.mean(scores)),
             "unique_solutions":len({tuple(sorted(s)) for s in population})}]
#c1 & exploit: Computes the SSA control parameter. Early on (iteration = 1), c1 is high ($\approx 1.7$) and exploit is low ($\approx 0.15$), encouraging broad exploration. Near the end (iteration = 10), c1 drops near 0 and exploit rises near 1.0, forcing the leader to copy nodes from food
    for iteration in range(1,iterations+1):
        c1 = 2.0*math.exp(-((4.0*iteration/iterations)**2))
        exploit = 1.0-min(c1/2.0,1.0)
    # Store population_size new candidate seed sets for this iteration
        new_pop = []
      # to create a new leader and keep track of previus leader
        leader, current = [], population[0]
      # Build the leader solution one node at a time until it contains k nodes.
        for pos in range(k):
         # if true exploit food
            if rng.random()<exploit:
                choices=[u for u in food if u not in leader]
                if choices:
                    leader.append(rng.choice(choices)); continue
          #chooing from itself
            choices=[u for u in current if u not in leader]
            if choices and rng.random()<0.5:
                leader.append(rng.choice(choices))

          # random from pool of nodes not used previosuly
            else:
                leader.append(rng.choice([u for u in pool if u not in leader]))
        #making sure it still have 10 nodes
        new_pop.append(repair_solution(leader,pool,k,rng))



       # updateing all remaining followers starting from 1
        for i in range(1,population_size):
            updated=list(population[i])
            previous=new_pop[i-1]
            # picking a random integer b/w 1 to 10 then picking that random index and updateing them from salp infront of them
            for pos in rng.sample(range(k),rng.randint(1,k)):
                used=set(updated[:pos]+updated[pos+1:])
                choices=[u for u in previous if u not in used]
                if choices: updated[pos]=rng.choice(choices)
            # Mutate selected positions using unused nodes from this variant's candidate pool.
            for pos in range(k):
                if rng.random()<mutation_rate:
                    used=set(updated[:pos]+updated[pos+1:])
                    choices=[u for u in pool if u not in used]
                    if choices: updated[pos]=rng.choice(choices)
            new_pop.append(repair_solution(updated,pool,k,rng))
       #calcualting new score
        new_scores=[score(s) for s in new_pop]
        # finding best set index
        i_best=int(np.argmax(new_scores))
        #updainf food score if its less then current best set
        if new_scores[i_best]>food_score:
            food=list(new_pop[i_best]); food_score=float(new_scores[i_best])
        # finding lowest score and replacing it with food
        i_worst=int(np.argmin(new_scores))
        new_pop[i_worst]=list(food)
        population=new_pop
        #updating
        hist.append({"iteration":iteration,"best_net_advantage":food_score,
                     "mean_net_advantage":float(np.mean(new_scores)),
                     "unique_solutions":len({tuple(sorted(s)) for s in population})})
        print(f"SSA {iteration}/{iterations}: best net advantage={food_score:.2f}, mean net advantage={np.mean(new_scores):.2f}")

    if len(food)!=k or len(set(food))!=k or set(food)&set(B):
        raise AssertionError("SSA returned invalid seed set.")
    return food,food_score,pd.DataFrame(hist),cache

# ============================== CELL 25 ==============================
# ============================================================
# SECTION 11: FOUR-OPPONENT MULTI-K EXPERIMENT RUNNER (CELF-OFF)
# ============================================================

if not (RUN_NCGE or RUN_CRIEG or RUN_SIGNED_GA or RUN_CELF or RUN_SSA):
    raise ValueError("Enable at least one optimization/literature method.")

MULTI_K_OUTPUT_DIR = OUTPUT_DIR / "multi_k_celf_off_csv_persist"
MULTI_K_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not K_VALUES or sorted(set(K_VALUES)) != K_VALUES:
    raise ValueError("K_VALUES must be a non-empty, ascending list without duplicates.")
if CONFIG["candidate_pool_ratio"] < 1:
    raise ValueError("candidate_pool_ratio must be at least 1.")
if min(K_VALUES) < 1:
    raise ValueError("All K values must be positive.")

ORIGINAL_CONFIG_K = CONFIG["k"]
SCENARIO_SPECS = [
    ("B1", "Random"),
    ("B2", "Degree-based"),
    ("B3", "Positive PageRank"),
    ("B4", "Community-concentrated"),
]
scenario_limit = len(SCENARIO_SPECS)
MAX_K = max(K_VALUES)

# Fixed nested B orderings, one per strategy.
rng_b1 = random.Random(CONFIG["master_seed"] + 810001)
B_ORDERS = {
    "B1": rng_b1.sample(list(NODES), MAX_K),
    "B2": sorted(NODES, key=lambda u: (-G.out_degree(u), int(u)))[:MAX_K],
}

# Positive-edge-only PageRank for B3.
G_POSITIVE = nx.DiGraph()
G_POSITIVE.add_nodes_from(NODES)
G_POSITIVE.add_edges_from(
    (u, v, {"weight": float(data.get("probability", 0.0))})
    for u, v, data in G.edges(data=True)
    if int(data.get("polarity", 0)) > 0
)
POSITIVE_PAGERANK_START = time.perf_counter()
if G_POSITIVE.number_of_edges() > 0:
    B3_POSITIVE_PAGERANK = nx.pagerank(
        G_POSITIVE, alpha=0.85, weight="weight", max_iter=200, tol=1e-10
    )
else:
    B3_POSITIVE_PAGERANK = {u: 1.0 / len(NODES) for u in NODES}
POSITIVE_PAGERANK_RUNTIME_SEC = time.perf_counter() - POSITIVE_PAGERANK_START
B_ORDERS["B3"] = sorted(
    NODES, key=lambda u: (-B3_POSITIVE_PAGERANK.get(u, 0.0), int(u))
)[:MAX_K]

# B4: Louvain on the undirected projection; choose the largest community.
COMMUNITY_START = time.perf_counter()
G_UNDIRECTED = nx.Graph()
G_UNDIRECTED.add_nodes_from(NODES)
G_UNDIRECTED.add_edges_from((u, v) for u, v in G.edges() if u != v)
if G_UNDIRECTED.number_of_edges() > 0:
    communities = list(
        nx.community.louvain_communities(
            G_UNDIRECTED, seed=CONFIG["master_seed"], weight=None
        )
    )
else:
    communities = [{u} for u in NODES]

largest_community = sorted(
    communities,
    key=lambda community: (-len(community), min(int(u) for u in community))
)[0]
B4_ORDER = sorted(
    largest_community, key=lambda u: (-G.out_degree(u), int(u))
)
if len(B4_ORDER) < MAX_K:
    raise ValueError(
        f"Largest Louvain community has {len(B4_ORDER)} nodes, "
        f"but max(K)={MAX_K}. B4 cannot provide nested seed sets."
    )
B_ORDERS["B4"] = B4_ORDER[:MAX_K]
COMMUNITY_DETECTION_RUNTIME_SEC = time.perf_counter() - COMMUNITY_START

# Persist one-time preprocessing runtimes immediately, before the long experiment.
preprocessing_runtime_path = MULTI_K_OUTPUT_DIR / "preprocessing_runtimes.csv"
preprocessing_runtime_df = pd.DataFrame([
    {"preprocessing": "Signed PageRank for A", "runtime_sec": float(SIGNED_PAGERANK_PREPROCESSING_RUNTIME_SEC)},
    {"preprocessing": "Positive PageRank for B3", "runtime_sec": float(POSITIVE_PAGERANK_RUNTIME_SEC)},
    {"preprocessing": "Louvain community detection for B4", "runtime_sec": float(COMMUNITY_DETECTION_RUNTIME_SEC)}
])
preprocessing_runtime_df.to_csv(preprocessing_runtime_path, index=False)
print(f"Preprocessing runtimes saved to: {preprocessing_runtime_path}")


for scenario_id, _ in SCENARIO_SPECS:
    assert len(B_ORDERS[scenario_id]) == MAX_K
    assert len(set(B_ORDERS[scenario_id])) == MAX_K

FINAL_SIMULATION_SEEDS = [
    900000000 + i for i in range(CONFIG["evaluation_mc_runs"])
]

def multi_k_checkpoint_signature(k):
    signature_config = {
        "experiment_version": "multi_k_v9_celf_off_csv_persist",
        "enabled_methods": {
            "NCGE": bool(RUN_NCGE), "CRIEG": bool(RUN_CRIEG),
            "Signed-GA": bool(RUN_SIGNED_GA), "CELF": bool(RUN_CELF),
            "Discrete SSA": bool(RUN_SSA)
        },
        "k": int(k),
        "candidate_pool_ratio": CONFIG["candidate_pool_ratio"],
        "diffusion_model": "signed_counter_influence_v1",
        "candidate_pool_scoring": [
            "weighted_positive_minus_negative_probability_v1",
            "ncge_liang2025_network_centrality_v1"
        ],
        "full_graph_literature_baselines": [
            "crieg_hong2021_crr_v1",
            "signed_ga_chakraborty2025_voter_v1"
        ],
        "optimization_objective": "mean_spread_A_minus_spread_B_v1",
        "B_scenarios": SCENARIO_SPECS,
        "B1_order": B_ORDERS["B1"][:MAX_K],
        "B2_order": B_ORDERS["B2"][:MAX_K],
        "B3_order": B_ORDERS["B3"][:MAX_K],
        "B4_order": B_ORDERS["B4"][:MAX_K],
        "candidate_pool_size": int(k) * int(CONFIG["candidate_pool_ratio"]),
        "random_candidate_fraction": CONFIG["random_candidate_fraction"],
        "optimization_mc_runs": CONFIG["optimization_mc_runs"],
        "ssa_population": CONFIG["ssa_population"],
        "evaluation_mc_runs": CONFIG["evaluation_mc_runs"],
        "ssa_iterations": CONFIG["ssa_iterations"],
        "ssa_mutation_rate": CONFIG["ssa_mutation_rate"],
        "master_seed": CONFIG["master_seed"],
        "dataset_file": DATA_FILE.name,
        "dataset_size_bytes": DATA_FILE.stat().st_size,
    }
    return hashlib.sha256(
        json.dumps(signature_config, sort_keys=True).encode()
    ).hexdigest()

def save_multi_k_checkpoint(checkpoint, checkpoint_path):
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, checkpoint_path)


multi_k_runs = []
multi_k_checkpoint_paths = {}

GREEDY_POOL_METHODS = []
if RUN_NCGE:
    GREEDY_POOL_METHODS.append(("NCGE (Liang 2025)", "NCGE", 100000))
if RUN_CELF:
    GREEDY_POOL_METHODS.append(("CELF", "Weighted", 110000))

for k in K_VALUES:
    CONFIG["k"] = int(k)
    print("\n" + "=" * 78)
    print(f"STARTING FOUR-B-SCENARIO EXPERIMENT (CELF-OFF): K = {k}")
    print("=" * 78)

    checkpoint_path = (
        MULTI_K_OUTPUT_DIR / f"checkpoint_k{k}_celf_off.json"
    )
    multi_k_checkpoint_paths[k] = checkpoint_path
    signature = multi_k_checkpoint_signature(k)

    if CONFIG["resume"] and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        if checkpoint.get("signature") != signature:
            raise ValueError(
                f"Checkpoint configuration mismatch for K={k}. "
                "Use a fresh experiment output folder."
            )
        print(f"Resuming checkpoint for K={k}.")
    else:
        checkpoint = {"signature": signature, "k": int(k), "scenarios": {}}
    save_multi_k_checkpoint(checkpoint, checkpoint_path)

    for scenario_id, scenario_name in tqdm(
        SCENARIO_SPECS, desc=f"K={k} opponent strategies"
    ):
        B = set(B_ORDERS[scenario_id][:k])
        eligible_nodes = [u for u in NODES if u not in B]
        sid = scenario_id

        if sid not in checkpoint["scenarios"]:
            # 1. Three Classic Full-Graph Baselines
            t0 = time.perf_counter()
            random_rng = random.Random(
                CONFIG["master_seed"] + 50000 + k * 100 + int(scenario_id[1:])
            )
            random_A = sorted(random_rng.sample(eligible_nodes, k))
            random_runtime = time.perf_counter() - t0

            t0 = time.perf_counter()
            degree_A = top_out_degree_seeds(eligible_nodes, k)
            degree_runtime = time.perf_counter() - t0

            t0 = time.perf_counter()
            pagerank_A = signed_pagerank_seeds(eligible_nodes, k)
            pagerank_runtime = time.perf_counter() - t0

            # 2. Two Full-Graph Literature Benchmarks (CRIEG 2021 & Signed-GA 2025)
            baseline_seed_sets = {
                "Top Degree": [int(x) for x in degree_A],
                "Signed PageRank": [int(x) for x in pagerank_A],
            }
            baseline_runtimes = {
                "Random": float(random_runtime),
                "Top Degree": float(degree_runtime),
                "Signed PageRank": float(pagerank_runtime),
            }

            if RUN_CRIEG:
                t0 = time.perf_counter()
                crieg_A = crieg_seeds(
                    B, k, num_sketches=2000,
                    random_seed=CONFIG["master_seed"] + 820000 + k * 100 + int(scenario_id[1:])
                )
                baseline_seed_sets["CRIEG (Hong 2021)"] = [int(x) for x in crieg_A]
                baseline_runtimes["CRIEG (Hong 2021)"] = float(time.perf_counter() - t0)

            if RUN_SIGNED_GA:
                t0 = time.perf_counter()
                sga_A = signed_ga_seeds(B, k, steps=30)
                baseline_seed_sets["Signed-GA (Chakraborty 2025)"] = [int(x) for x in sga_A]
                baseline_runtimes["Signed-GA (Chakraborty 2025)"] = float(time.perf_counter() - t0)

            # 3. Candidate Pools for NCGE (Liang 2025) and Discrete SSA
            pool_size_for_k = int(k) * int(CONFIG["candidate_pool_ratio"])
            weighted_pool, weighted_pool_scores = generate_weighted_candidate_pool(
                B, pool_size_for_k, CONFIG["random_candidate_fraction"],
                CONFIG["master_seed"] + 600000 + int(scenario_id[1:])
            )
            ncge_pool, ncge_pool_scores = generate_ncge_candidate_pool(
                B, pool_size_for_k
            )
            optimization_seeds = [
                CONFIG["master_seed"] * 100000
                + int(scenario_id[1:]) * 10000 + i
                for i in range(CONFIG["optimization_mc_runs"])
            ]

            checkpoint["scenarios"][sid] = {
                "scenario_id": scenario_id,
                "opponent_scenario": scenario_name,
                "experiment": int(scenario_id[1:]),
                "k": int(k),
                "seeds_B": sorted(int(x) for x in B),
                "random_seeds_A": [int(x) for x in random_A],
                "candidate_pools": {
                    "Weighted": [int(x) for x in weighted_pool],
                    "NCGE": [int(x) for x in ncge_pool],
                },
                "candidate_pool_scores": {
                    "Weighted": {
                        str(int(node)): weighted_pool_scores[node]
                        for node in weighted_pool
                    },
                    "NCGE": {
                        str(int(node)): ncge_pool_scores[node]
                        for node in ncge_pool
                    },
                },
                "optimization_seeds": [int(x) for x in optimization_seeds],
                "baseline_seed_sets": baseline_seed_sets,
                "baseline_runtimes": baseline_runtimes,
                "methods": {}
            }
            save_multi_k_checkpoint(checkpoint, checkpoint_path)

        rec = checkpoint["scenarios"][sid]
        if set(rec["seeds_B"]) != B:
            raise ValueError(
                f"Stored B seeds do not match {scenario_id} nested seeds for K={k}."
            )
        optimization_seeds = list(rec["optimization_seeds"])

        # 4. Run enabled candidate-pool methods (NCGE when enabled)
        for method_name, pool_key, _ in GREEDY_POOL_METHODS:
            if method_name not in rec["methods"]:
                pool = list(rec["candidate_pools"][pool_key])
                print(
                    f"\nRunning {method_name}: pool={len(pool)}, "
                    f"optimization MC={CONFIG['optimization_mc_runs']}"
                )
                t0 = time.perf_counter()
                selected, method_history, method_cache = celf_lazy_greedy(
                    pool, B, k, optimization_seeds,
                    progress_label=f"K={k} {scenario_id} {method_name}"
                )
                runtime = time.perf_counter() - t0
                rec["methods"][method_name] = {
                    "seeds_A": [int(x) for x in selected],
                    "optimization_fitness_net_advantage": float(
                        fitness_net_advantage(selected, B, optimization_seeds, method_cache)
                    ),
                    "runtime_sec": float(runtime),
                    "history": method_history.to_dict("records"),
                    "fitness_cache_size": int(len(method_cache)),
                    "candidate_pool_key": pool_key,
                    "candidate_pool_size": int(len(pool)),
                }
                save_multi_k_checkpoint(checkpoint, checkpoint_path)

        # 5. Run Discrete SSA on 70/30 Weighted pool
        if RUN_SSA:
            variant = "Discrete SSA"
            pool = list(rec["candidate_pools"]["Weighted"])
            if variant not in rec["methods"]:
                print(
                    f"\nRunning {variant}: pool={len(pool)}, "
                    f"population={CONFIG['ssa_population']}, "
                    f"iterations={CONFIG['ssa_iterations']}, "
                    f"optimization MC={CONFIG['optimization_mc_runs']}"
                )
                t0 = time.perf_counter()
                seeds, fitness, history, cache = discrete_ssa(
                    pool, B, k, optimization_seeds,
                    CONFIG["ssa_population"], CONFIG["ssa_iterations"],
                    CONFIG["ssa_mutation_rate"],
                    CONFIG["master_seed"] + 30000 + int(scenario_id[1:])
                )
                ssa_runtime = time.perf_counter() - t0
                rec["methods"][variant] = {
                    "seeds_A": [int(x) for x in seeds],
                    "optimization_fitness_net_advantage": float(fitness),
                    "runtime_sec": float(ssa_runtime),
                    "history": history.to_dict("records"),
                    "candidate_pool_size": len(pool),
                    "candidate_pool_key": "Weighted"
                }
                save_multi_k_checkpoint(checkpoint, checkpoint_path)

        # Validate enabled methods
        full_graph_checks = [("Random", rec["random_seeds_A"])] + list(rec["baseline_seed_sets"].items())
        for label, seed_set in full_graph_checks:
            if len(seed_set) != k or len(set(seed_set)) != k or set(seed_set) & B:
                raise AssertionError(f"Invalid {label} seeds for K={k}, scenario={scenario_id}.")

        expected_pool_methods = []
        if RUN_NCGE:
            expected_pool_methods.append("NCGE (Liang 2025)")
        if RUN_CELF:
            expected_pool_methods.append("CELF")
        if RUN_SSA:
            expected_pool_methods.append("Discrete SSA")
        for method_name in expected_pool_methods:
            method_rec = rec["methods"][method_name]
            pool_key = method_rec["candidate_pool_key"]
            pool = set(rec["candidate_pools"][pool_key])
            A = set(method_rec["seeds_A"])
            if len(A) != k or not A.issubset(pool) or A & B:
                raise AssertionError(
                    f"Invalid {method_name} seeds for K={k}, scenario={scenario_id}."
                )

        multi_k_runs.append({
            "k": int(k),
            "experiment": int(scenario_id[1:]),
            "scenario_id": scenario_id,
            "opponent_scenario": scenario_name,
            "seeds_B": rec["seeds_B"],
            "random_seeds_A": rec["random_seeds_A"],
            "candidate_pools": rec["candidate_pools"],
            "candidate_pool_scores": rec.get("candidate_pool_scores", {}),
            "baseline_seed_sets": rec["baseline_seed_sets"],
            "baseline_runtimes": rec["baseline_runtimes"],
            "methods": rec["methods"]
        })

    print(f"Finished K={k}. Checkpoint saved to: {checkpoint_path}")

CONFIG["k"] = ORIGINAL_CONFIG_K
print("\n" + "=" * 78)
print("FOUR-B-SCENARIO CELF-OFF EXPERIMENTS COMPLETED")
print("K values:", K_VALUES)
print("Opponent strategies:", SCENARIO_SPECS)
print("Scenario records:", len(multi_k_runs))
print("Output directory:", MULTI_K_OUTPUT_DIR)
print("=" * 78)

# ============================== CELL 27 ==============================
# ============================================================
# SECTION 12: VALIDATE AND EXPORT SEED SETS / RUNTIMES (CELF-OFF)
# ============================================================

completed = list(multi_k_runs)
seed_summary_rows, seed_detail_rows = [], []
selection_runtime_rows, candidate_pool_score_rows = [], []
method_history_rows = []

FULL_GRAPH_METHODS = {
    "Random",
    "Top Degree",
    "Signed PageRank",
    "CRIEG (Hong 2021)",
    "Signed-GA (Chakraborty 2025)",
}

ALL_A_METHODS = [
    "Random",
    "Top Degree",
    "Signed PageRank",
    "CRIEG (Hong 2021)",
    "NCGE (Liang 2025)",
    "Signed-GA (Chakraborty 2025)",
    "Discrete SSA",
]

for rec in completed:
    k = int(rec["k"])
    scenario_id = rec["scenario_id"]
    scenario_name = rec["opponent_scenario"]
    B = set(rec["seeds_B"])
    eligible_nodes = set(NODES) - B

    for variant in ["Weighted", "NCGE"]:
        variant_pool = rec["candidate_pools"][variant]
        variant_scores = rec.get("candidate_pool_scores", {}).get(variant, {})
        for node in variant_pool:
            detail = variant_scores.get(str(int(node)))
            if detail is not None:
                candidate_pool_score_rows.append({
                    "K": k, "scenario_id": scenario_id,
                    "opponent_scenario": scenario_name,
                    "candidate_pool_variant": variant,
                    "node_id": int(node), **detail
                })

    all_sets = [("Rumour B", scenario_name, sorted(B))]
    all_sets += [("Rumour A", "Random", rec["random_seeds_A"])]
    all_sets += [
        ("Rumour A", method, seeds)
        for method, seeds in rec["baseline_seed_sets"].items()
    ]
    all_sets += [
        ("Rumour A", method, rec["methods"][method]["seeds_A"])
        for method in ALL_A_METHODS if method not in FULL_GRAPH_METHODS
    ]

    for rumour, method, seed_nodes in all_sets:
        seeds = [int(x) for x in seed_nodes]
        if len(seeds) != k or len(set(seeds)) != k:
            raise AssertionError(f"Invalid seed set: K={k}, scenario={scenario_id}, method={method}")
        if rumour == "Rumour A" and set(seeds) & B:
            raise AssertionError(f"Rumour A/B seed overlap: K={k}, scenario={scenario_id}, method={method}")
        if rumour == "Rumour A" and method in FULL_GRAPH_METHODS:
            if not set(seeds).issubset(eligible_nodes):
                raise AssertionError(f"{method} selected outside eligible network.")
        if rumour == "Rumour A" and method in rec["methods"]:
            pool_key = rec["methods"][method]["candidate_pool_key"]
            if not set(seeds).issubset(set(rec["candidate_pools"][pool_key])):
                raise AssertionError(f"{method} selected outside candidate pool {pool_key}.")

        seed_summary_rows.append({
            "K": k, "scenario_id": scenario_id,
            "opponent_scenario": scenario_name,
            "rumour": rumour, "method": method,
            "seed_count": len(seeds),
            "seed_nodes": ", ".join(map(str, sorted(seeds)))
        })
        for position, node in enumerate(sorted(seeds), start=1):
            seed_detail_rows.append({
                "K": k, "scenario_id": scenario_id,
                "opponent_scenario": scenario_name,
                "rumour": rumour, "method": method,
                "seed_position": position, "node_id": node
            })

    for method, seconds in rec["baseline_runtimes"].items():
        selection_runtime_rows.append({
            "K": k, "scenario_id": scenario_id,
            "opponent_scenario": scenario_name,
            "method": method, "selection_runtime_sec": float(seconds)
        })
    for method, method_rec in rec["methods"].items():
        selection_runtime_rows.append({
            "K": k, "scenario_id": scenario_id,
            "opponent_scenario": scenario_name,
            "method": method, "selection_runtime_sec": float(method_rec["runtime_sec"])
        })
        for history_row in method_rec.get("history", []):
            method_history_rows.append({
                "K": k, "scenario_id": scenario_id,
                "opponent_scenario": scenario_name,
                "method": method, **history_row
            })

seed_sets_summary_df = pd.DataFrame(seed_summary_rows)
seed_sets_detail_df = pd.DataFrame(seed_detail_rows)
selection_runtime_df = pd.DataFrame(selection_runtime_rows)
candidate_pool_scores_df = pd.DataFrame(candidate_pool_score_rows)
method_history_df = pd.DataFrame(method_history_rows)

seed_sets_summary_df.to_csv(MULTI_K_OUTPUT_DIR / "multi_k_seed_sets_summary.csv", index=False)
seed_sets_detail_df.to_csv(MULTI_K_OUTPUT_DIR / "multi_k_seed_sets_detailed.csv", index=False)
selection_runtime_df.to_csv(MULTI_K_OUTPUT_DIR / "multi_k_selection_runtimes.csv", index=False)
candidate_pool_scores_df.to_csv(
    MULTI_K_OUTPUT_DIR / "candidate_pool_scores_weighted_vs_ncge.csv", index=False
)
method_history_df.to_csv(
    MULTI_K_OUTPUT_DIR / "multi_k_method_optimization_histories.csv", index=False
)

print("Seed sets, candidate-pool scores, method histories and selection runtimes exported.")
display(seed_sets_summary_df)
display(selection_runtime_df)
display(candidate_pool_scores_df.head())

# ============================== CELL 29 ==============================
# ============================================================
# SECTION 13: DISPLAY ALL SEED SETS
# ============================================================

print("FINAL SEED SETS ACROSS K AND ALL FOUR B STRATEGIES")
display(
    seed_sets_summary_df.sort_values(
        ["K", "scenario_id", "rumour", "method"]
    ).reset_index(drop=True)
)

print("\nSeed-set export files:")
print(MULTI_K_OUTPUT_DIR / "multi_k_seed_sets_summary.csv")
print(MULTI_K_OUTPUT_DIR / "multi_k_seed_sets_detailed.csv")


# ============================== CELL 32 ==============================
# ============================================================
# SECTION 14: INDEPENDENT FINAL EVALUATION (CELF-OFF)
# CSV is the durable source for evaluation data. Existing rows are reused
# so a runtime interruption does not require repeating completed evaluations.
# ============================================================

evaluation_csv_path = MULTI_K_OUTPUT_DIR / "multi_k_evaluation_raw.csv"
evaluation_tmp_path = evaluation_csv_path.with_suffix(".tmp.csv")

if evaluation_csv_path.exists():
    saved_evaluation_df = pd.read_csv(evaluation_csv_path)
    evaluation_rows = saved_evaluation_df.to_dict("records")
    completed_evaluation_keys = set(
        zip(saved_evaluation_df["K"], saved_evaluation_df["scenario_id"], saved_evaluation_df["method"])
    )
    print(f"Loaded {len(evaluation_rows)} saved evaluation rows; completed rows will be skipped.")
else:
    saved_evaluation_df = pd.DataFrame()
    evaluation_rows = []
    completed_evaluation_keys = set()

for rec in tqdm(completed, desc="Independent evaluation across B strategies"):
    k = int(rec["k"])
    scenario_id = rec["scenario_id"]
    scenario_name = rec["opponent_scenario"]
    B = set(rec["seeds_B"])

    method_sets = {
        "Random": rec["random_seeds_A"],
        **rec["baseline_seed_sets"],
    }
    selection_times = dict(rec["baseline_runtimes"])
    for method, method_rec in rec["methods"].items():
        method_sets[method] = method_rec["seeds_A"]
        selection_times[method] = method_rec["runtime_sec"]

    for method, A in method_sets.items():
        evaluation_key = (k, scenario_id, method)
        if evaluation_key in completed_evaluation_keys:
            continue
        A = set(A)
        if len(A) != k or A & B:
            raise AssertionError(
                f"Invalid evaluation seed set: K={k}, scenario={scenario_id}, method={method}"
            )
        evaluation_start = time.perf_counter()
        metrics = evaluate_seed_set(A, B, FINAL_SIMULATION_SEEDS)
        evaluation_runtime = time.perf_counter() - evaluation_start
        row = {
            "K": k, "experiment": int(scenario_id[1:]),
            "scenario_id": scenario_id, "opponent_scenario": scenario_name,
            "method": method, **metrics,
            "selection_runtime_sec": float(selection_times[method]),
            "evaluation_runtime_sec": float(evaluation_runtime),
            "final_evaluation_mc_runs": len(FINAL_SIMULATION_SEEDS)
        }
        evaluation_rows.append(row)
        completed_evaluation_keys.add(evaluation_key)

        # Save after every newly evaluated seed set using atomic replace.
        pd.DataFrame(evaluation_rows).to_csv(evaluation_tmp_path, index=False)
        os.replace(evaluation_tmp_path, evaluation_csv_path)

final_results_df = pd.DataFrame(evaluation_rows)
# Always rewrite the completed raw evaluation table, including resumed rows.
final_results_df.to_csv(evaluation_tmp_path, index=False)
os.replace(evaluation_tmp_path, evaluation_csv_path)
print(f"Independent evaluation table saved to: {evaluation_csv_path}")
print(f"Independent evaluation rows available: {len(final_results_df)}")
display(final_results_df)


# ============================== CELL 34 ==============================
# CSV-ONLY ANALYSIS SETUP
# Run this cell (and the summary/runtime/chart cells below) after a runtime reset.
# If the experiment CSV folder is elsewhere, set MULTI_K_OUTPUT_DIR to that folder.
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = (
    Path("/kaggle/working/competitive_im_results_no_celf_csv_persist")
    if Path("/kaggle/working").exists()
    else Path("./competitive_im_results_no_celf_csv_persist")
)
MULTI_K_OUTPUT_DIR = OUTPUT_DIR / "multi_k_celf_off_csv_persist"
K_VALUES = [10, 20, 30, 40, 50]
SCENARIO_ORDER = ["B1", "B2", "B3", "B4"]
SCENARIO_SPECS = [
    ("B1", "Random"),
    ("B2", "Degree-based"),
    ("B3", "Positive PageRank"),
    ("B4", "Community-concentrated"),
]
print("CSV analysis directory:", MULTI_K_OUTPUT_DIR)


# ============================== CELL 36 ==============================
# ============================================================
# SECTION 15: PERFORMANCE SUMMARY BY K AND B STRATEGY
# Summary is rebuilt from the saved raw-evaluation CSV, not runtime memory.
# ============================================================

evaluation_csv_path = MULTI_K_OUTPUT_DIR / "multi_k_evaluation_raw.csv"
if not evaluation_csv_path.exists():
    raise FileNotFoundError(f"Raw evaluation CSV not found: {evaluation_csv_path}. Run Section 14 first.")
final_results_df = pd.read_csv(evaluation_csv_path)

summary_df = (
    final_results_df
    .groupby(["K", "scenario_id", "opponent_scenario", "method"])
    .agg(
        spread_A_mean=("spread_A", "mean"),
        spread_B_mean=("spread_B", "mean"),
        blocked_B_mean=("blocked_B", "mean"),
        net_advantage_mean=("net_advantage", "mean"),
        weighted_fitness_mean=("weighted_fitness", "mean"),
        final_evaluation_mc_runs=("final_evaluation_mc_runs", "first"),
        selection_runtime_sec=("selection_runtime_sec", "mean"),
        evaluation_runtime_sec=("evaluation_runtime_sec", "mean")
    )
    .reset_index()
    .sort_values(["scenario_id", "method", "K"])
)

summary_csv_path = MULTI_K_OUTPUT_DIR / "multi_k_performance_summary.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"Performance summary saved to: {summary_csv_path}")
display(summary_df)


# ============================== CELL 38 ==============================
# ============================================================
# SECTION 16: EXECUTION TIME SUMMARY (FROM SAVED CSV)
# ============================================================

evaluation_csv_path = MULTI_K_OUTPUT_DIR / "multi_k_evaluation_raw.csv"
if not evaluation_csv_path.exists():
    raise FileNotFoundError(f"Raw evaluation CSV not found: {evaluation_csv_path}. Run Section 14 first.")
final_results_df = pd.read_csv(evaluation_csv_path)

runtime_summary = (
    final_results_df
    .groupby(["K", "scenario_id", "opponent_scenario", "method"])
    .agg(
        selection_runtime_sec=("selection_runtime_sec", "mean"),
        evaluation_runtime_sec=("evaluation_runtime_sec", "mean")
    )
    .reset_index()
)
runtime_summary["selection_plus_evaluation_sec"] = (
    runtime_summary["selection_runtime_sec"]
    + runtime_summary["evaluation_runtime_sec"]
)
runtime_summary.to_csv(
    MULTI_K_OUTPUT_DIR / "multi_k_runtime_summary.csv", index=False
)

preprocessing_runtime_path = MULTI_K_OUTPUT_DIR / "preprocessing_runtimes.csv"
if preprocessing_runtime_path.exists():
    preprocessing_runtime_df = pd.read_csv(preprocessing_runtime_path)
else:
    print("Preprocessing runtime CSV is not available yet; skipping its display/export.")
    preprocessing_runtime_df = pd.DataFrame(columns=["preprocessing", "runtime_sec"])


print("Per-K, per-scenario selection and evaluation runtimes:")
display(runtime_summary)
print("One-time preprocessing runtimes:")
display(preprocessing_runtime_df)


# ============================== CELL 40 ==============================
# ============================================================
# SECTION 17: PERFORMANCE CHARTS BY B STRATEGY (CELF-OFF; CSV SOURCE)
# ============================================================

METHOD_ORDER = [
    "Random",
    "Top Degree",
    "Signed PageRank",
    "CRIEG (Hong 2021)",
    "NCGE (Liang 2025)",
    "Signed-GA (Chakraborty 2025)",
    "Discrete SSA",
]
SCENARIO_ORDER = ["B1", "B2", "B3", "B4"]

# Fixed one-to-one colorblind-safe Okabe-Ito 8-color palette
METHOD_COLORS = {
    "Random": "#7F7F7F",                      # Neutral Gray (baseline)
    "Top Degree": "#E69F00",                  # Amber/Orange (baseline)
    "Signed PageRank": "#009E73",             # Teal Green (baseline)
    "CRIEG (Hong 2021)": "#56B4E9",           # Sky Blue (2021 literature benchmark)
    "NCGE (Liang 2025)": "#CC79A7",           # Magenta/Purple (2025 literature benchmark)
    "Signed-GA (Chakraborty 2025)": "#8C564B",# Brown/Bronze (2025 literature benchmark)
    "CELF": "#0072B2",                        # Deep Blue (proposed greedy optimizer)
    "Discrete SSA": "#D55E00",                # Vermilion Red (proposed swarm optimizer)
}
assert len(set(METHOD_COLORS.values())) == len(METHOD_COLORS)

METHOD_MARKERS = {
    "Random": "x",
    "Top Degree": "^",
    "Signed PageRank": "s",
    "CRIEG (Hong 2021)": "v",
    "NCGE (Liang 2025)": "D",
    "Signed-GA (Chakraborty 2025)": "*",
    "CELF": "o",
    "Discrete SSA": "P",
}

METHOD_LINESTYLES = {
    "Random": ":",
    "Top Degree": "--",
    "Signed PageRank": "-.",
    "CRIEG (Hong 2021)": "--",
    "NCGE (Liang 2025)": "-.",
    "Signed-GA (Chakraborty 2025)": ":",
    "CELF": "-",
    "Discrete SSA": "-",
}

def plot_metric_vs_k(metric_column, y_label, title_prefix, filename_prefix, zero_line=False):
    summary_csv_path = MULTI_K_OUTPUT_DIR / "multi_k_performance_summary.csv"
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Performance summary CSV not found: {summary_csv_path}. Run Section 15 first.")
    plot_data = pd.read_csv(summary_csv_path)
    for scenario_id in SCENARIO_ORDER:
        scenario_name = dict(SCENARIO_SPECS)[scenario_id]
        plt.figure(figsize=(12, 7))
        scenario_data = plot_data[plot_data["scenario_id"] == scenario_id]

        for method in METHOD_ORDER:
            method_data = scenario_data[
                scenario_data["method"] == method
            ].sort_values("K")
            if not method_data.empty:
                is_proposed = method == "Discrete SSA"
                plt.plot(
                    method_data["K"],
                    method_data[metric_column],
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    linestyle=METHOD_LINESTYLES[method],
                    linewidth=2.6 if is_proposed else 1.8,
                    markersize=7 if is_proposed else 5.5,
                    label=method,
                )

        if zero_line:
            plt.axhline(0, color="#444444", linestyle="--", linewidth=1.1, label="Zero reference")
        plt.xlabel("Seed budget (K)")
        plt.ylabel(y_label)
        plt.title(f"{title_prefix} — {scenario_id}: {scenario_name}")
        plt.xticks(K_VALUES)
        plt.grid(alpha=0.25)
        plt.legend(ncol=2, frameon=True)
        plt.tight_layout()
        filename = f"{filename_prefix}_{scenario_id}.png"
        plt.savefig(MULTI_K_OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

plot_metric_vs_k(
    "spread_A_mean", "Mean number of Rumour A adopters",
    "Rumour A Spread vs K", "rumour_A_spread"
)
plot_metric_vs_k(
    "spread_B_mean", "Mean number of Rumour B adopters",
    "Rumour B Spread vs K", "rumour_B_spread"
)
plot_metric_vs_k(
    "net_advantage_mean", "Mean net advantage (A spread - B spread)",
    "Competitive Net Advantage vs K", "net_advantage", True
)
plot_metric_vs_k(
    "selection_runtime_sec", "Seed-selection time (seconds)",
    "Seed Selection Runtime vs K", "selection_runtime"
)
plot_metric_vs_k(
    "evaluation_runtime_sec", "Independent evaluation time (seconds)",
    "Final Evaluation Runtime vs K", "evaluation_runtime"
)

print("Scenario-specific charts saved in:", MULTI_K_OUTPUT_DIR)
print("Unique method colors:")
for method in METHOD_ORDER:
    print(f"  {method}: {METHOD_COLORS[method]}")

# ============================== CELL 42 ==============================
# ============================================================
# SECTION 18: STATISTICAL COMPARISON NOTE
# ============================================================

# B1-B4 are four deliberately different opponent strategies, not four
# independent random replicates from one common population. Therefore, they
# must not be treated as interchangeable replicates in a Wilcoxon test.
# The notebook reports descriptive performance separately for each B strategy.
# If inferential testing is needed, retain per-Monte-Carlo-run outcomes and
# compare methods using paired simulation seeds within each fixed B strategy.
print(
    "Inferential tests are not run across B1-B4 because these are distinct "
    "opponent strategies, not independent replicate scenarios. "
    "Use paired per-run outcomes within each fixed B strategy for formal tests."
)


# ============================== CELL 44 ==============================
# ============================================================
# SECTION 19: COMPLETENESS AND SEARCH-SPACE VALIDATION (CELF-OFF)
# ============================================================

expected_methods = set(METHOD_ORDER)
expected_scenarios = {"B1", "B2", "B3", "B4"}
expected_pool_methods = {}
if RUN_NCGE:
    expected_pool_methods["NCGE (Liang 2025)"] = "NCGE"
if RUN_CELF:
    expected_pool_methods["CELF"] = "Weighted"
if RUN_SSA:
    expected_pool_methods["Discrete SSA"] = "Weighted"

assert sorted(set(final_results_df["K"])) == sorted(K_VALUES)
assert len(multi_k_runs) == len(K_VALUES) * len(expected_scenarios)
assert len(set(METHOD_COLORS.values())) == len(METHOD_COLORS)

for rec in multi_k_runs:
    k = int(rec["k"])
    scenario_id = rec["scenario_id"]
    B = set(rec["seeds_B"])
    eligible_nodes = set(NODES) - B
    assert scenario_id in expected_scenarios
    assert len(B) == k

    for method, pool_key in expected_pool_methods.items():
        pool = set(rec["candidate_pools"][pool_key])
        assert len(pool) == k * CONFIG["candidate_pool_ratio"]
        assert pool.issubset(eligible_nodes)
        A = set(rec["methods"][method]["seeds_A"])
        assert len(A) == k and A.issubset(pool) and not A.intersection(B)
        assert rec["methods"][method]["candidate_pool_key"] == pool_key
        assert rec["methods"][method]["runtime_sec"] >= 0

    weighted_scores = rec.get("candidate_pool_scores", {}).get("Weighted", {})
    assert set(weighted_scores) == {
        str(int(node)) for node in rec["candidate_pools"]["Weighted"]
    }
    for detail in weighted_scores.values():
        assert detail["positive_outgoing_score"] >= 0
        assert detail["negative_outgoing_score"] >= 0
        assert math.isclose(
            detail["net_signed_candidate_score"],
            detail["positive_outgoing_score"] - detail["negative_outgoing_score"],
            rel_tol=1e-10, abs_tol=1e-10
        )

    ncge_scores = rec.get("candidate_pool_scores", {}).get("NCGE", {})
    assert set(ncge_scores) == {
        str(int(node)) for node in rec["candidate_pools"]["NCGE"]
    }
    for detail in ncge_scores.values():
        assert detail["ncge_centrality_score"] >= 0

    baseline_sets = {"Random": rec["random_seeds_A"], **rec["baseline_seed_sets"]}
    for method, seed_nodes in baseline_sets.items():
        A = set(seed_nodes)
        assert len(A) == k and A.issubset(eligible_nodes) and not A.intersection(B)

for k in K_VALUES:
    for scenario_id in expected_scenarios:
        subset = final_results_df[
            (final_results_df["K"] == k) &
            (final_results_df["scenario_id"] == scenario_id)
        ]
        assert set(subset["method"]) == expected_methods
        assert len(subset) == len(expected_methods)

assert not final_results_df.duplicated(["K", "scenario_id", "method"]).any()
assert (final_results_df["final_evaluation_mc_runs"] == CONFIG["evaluation_mc_runs"]).all()

print("All enabled-method, four-scenario, two-pool validations passed.")
print("K values:", K_VALUES)
print("Opponent strategies:", SCENARIO_SPECS)
print("A methods:", sorted(expected_methods))
print("Evaluation rows:", len(final_results_df))
print("Final Monte Carlo runs per seed set:", len(FINAL_SIMULATION_SEEDS))
print("Results directory:", MULTI_K_OUTPUT_DIR)
print("Checkpoint files:")
for k, checkpoint_file in multi_k_checkpoint_paths.items():
    print(f"  K={k}: {checkpoint_file}")
