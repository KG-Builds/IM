
# ============================== CELL 2 ==============================
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
OUTPUT_DIR = Path("/kaggle/working/competition_blocking_ssa_variants_correct_gainA") if Path("/kaggle/working").exists() else Path("./competition_blocking_ssa_variants_correct_gainA")
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

# Only the SSA variants in this notebook are run. Other seed-selection algorithms are intentionally absent/disabled.
RUN_OTHER_ALGORITHMS = False
RUN_SSA_VARIANTS = True

print("DATA_FILE:", DATA_FILE)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("K_VALUES:", K_VALUES)

# ============================== CELL 4 ==============================
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

# ============================== CELL 6 ==============================
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

# ============================== CELL 8 ==============================
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

# ============================== CELL 9 ==============================
# Shared raw diffusion cache: each individual Monte Carlo realization is cached.
# Cache key includes sorted A seeds, sorted B seeds, and the exact simulation seed.
SIMULATION_OUTCOME_CACHE = {}
B_ONLY_BASELINE_CACHE = {}
SIMULATION_CACHE_HITS = 0
SIMULATION_CACHE_MISSES = 0

def cached_simulation(A, B, simulation_seed):
    global SIMULATION_CACHE_HITS, SIMULATION_CACHE_MISSES
    key = (tuple(sorted(set(A))), tuple(sorted(set(B))), int(simulation_seed))
    if key in SIMULATION_OUTCOME_CACHE:
        SIMULATION_CACHE_HITS += 1
        return SIMULATION_OUTCOME_CACHE[key]
    SIMULATION_CACHE_MISSES += 1
    result = competitive_maki_thompson(A, B, np.random.default_rng(int(simulation_seed)))
    SIMULATION_OUTCOME_CACHE[key] = result
    return result

def evaluate_seed_set(A, B, simulation_seeds, cache=None):
    """Aggregate cached individual simulations; cache shared across all methods/modes."""
    A, B = set(A), set(B)
    sim_seeds = tuple(map(int, simulation_seeds))
    b_key = (tuple(sorted(B)), sim_seeds)
    if b_key not in B_ONLY_BASELINE_CACHE:
        b_only = [cached_simulation([], B, s)['spread_B'] for s in sim_seeds]
        B_ONLY_BASELINE_CACHE[b_key] = float(np.mean(b_only))
    outcomes = [cached_simulation(A, B, s) for s in sim_seeds]
    mean_A = float(np.mean([r['spread_A'] for r in outcomes]))
    mean_B = float(np.mean([r['spread_B'] for r in outcomes]))
    net_adv = float(np.mean([r['spread_A'] - r['spread_B'] for r in outcomes]))
    blocked = max(0.0, B_ONLY_BASELINE_CACHE[b_key] - mean_B)
    return {'spread_A':mean_A, 'spread_B':mean_B, 'blocked_B':blocked,
            'net_advantage':net_adv, 'b_only_baseline_spread':B_ONLY_BASELINE_CACHE[b_key]}


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

# ============================== CELL 13 ==============================
# Precompute each node's signed candidate score exactly once for this graph.
SIGNED_CANDIDATE_SCORE_DETAILS = {}
for u in NODES:
    positive_score = 0.0
    negative_score = 0.0
    for v in ADJ[u]:
        p = EDGE_PROB[(u, v)]
        polarity = EDGE_POLARITY[(u, v)]
        if polarity > 0:
            positive_score += p
        elif polarity < 0:
            negative_score += p
    SIGNED_CANDIDATE_SCORE_DETAILS[u] = {
        'positive_outgoing_score': float(positive_score),
        'negative_outgoing_score': float(negative_score),
        'net_signed_candidate_score': float(positive_score-negative_score)
    }


def generate_weighted_candidate_pool(B, pool_size, random_fraction, random_seed):
    """Generate candidate pool using precomputed node scores; excludes B seeds."""
    B = set(B)
    eligible = [u for u in NODES if u not in B]
    if pool_size > len(eligible):
        raise ValueError("Not enough eligible nodes.")
    ranked = sorted(eligible, key=lambda u: (
        -SIGNED_CANDIDATE_SCORE_DETAILS[u]['net_signed_candidate_score'],
        -SIGNED_CANDIDATE_SCORE_DETAILS[u]['positive_outgoing_score'],
        SIGNED_CANDIDATE_SCORE_DETAILS[u]['negative_outgoing_score'], int(u)))
    n_random = int(round(pool_size*random_fraction))
    n_ranked = pool_size-n_random
    top_ranked = ranked[:n_ranked]
    remaining = ranked[n_ranked:]
    rng = random.Random(random_seed)
    random_nodes = rng.sample(remaining,n_random) if n_random else []
    pool = top_ranked+random_nodes
    if len(pool)!=pool_size or len(set(pool))!=pool_size or set(pool)&B:
        raise AssertionError('Invalid weighted candidate pool.')
    return pool, {u:SIGNED_CANDIDATE_SCORE_DETAILS[u] for u in eligible}

print('Precomputed signed candidate scores for',len(SIGNED_CANDIDATE_SCORE_DETAILS),'nodes.')


# ============================== CELL 15 ==============================
SCENARIO_SPECS = [("B1", "Random"), ("B2", "Degree-based"),
                   ("B3", "Positive PageRank"), ("B4", "Community-concentrated")]
MAX_K = max(K_VALUES)
rng_b1 = random.Random(CONFIG["master_seed"] + 810001)
B_ORDERS = {"B1": rng_b1.sample(list(NODES), MAX_K),
            "B2": sorted(NODES, key=lambda u: (-G.out_degree(u), int(u)))[:MAX_K]}
G_POSITIVE = nx.DiGraph(); G_POSITIVE.add_nodes_from(NODES)
G_POSITIVE.add_edges_from((u,v,{'weight':float(d.get('probability',0.0))})
    for u,v,d in G.edges(data=True) if int(d.get('polarity',0))>0)
if G_POSITIVE.number_of_edges():
    B3_POSITIVE_PAGERANK = nx.pagerank(G_POSITIVE, alpha=0.85, weight='weight', max_iter=200, tol=1e-10)
else:
    B3_POSITIVE_PAGERANK = {u:1.0/len(NODES) for u in NODES}
B_ORDERS['B3'] = sorted(NODES, key=lambda u:(-B3_POSITIVE_PAGERANK.get(u,0.0), int(u)))[:MAX_K]
G_UNDIRECTED = nx.Graph(); G_UNDIRECTED.add_nodes_from(NODES); G_UNDIRECTED.add_edges_from((u,v) for u,v in G.edges() if u!=v)
communities = list(nx.community.louvain_communities(G_UNDIRECTED, seed=CONFIG['master_seed'], weight=None)) if G_UNDIRECTED.number_of_edges() else [{u} for u in NODES]
largest_community = sorted(communities, key=lambda c:(-len(c),min(int(u) for u in c)))[0]
B4_ORDER = sorted(largest_community, key=lambda u:(-G.out_degree(u),int(u)))
if len(B4_ORDER) < MAX_K: raise ValueError(f'Largest community too small for max K={MAX_K}.')
B_ORDERS['B4'] = B4_ORDER[:MAX_K]
for sid,_ in SCENARIO_SPECS:
    assert len(B_ORDERS[sid])==MAX_K and len(set(B_ORDERS[sid]))==MAX_K
print('Opponent seed orderings prepared:', {sid:len(order) for sid,order in B_ORDERS.items()})

# ============================== CELL 17 ==============================
# Six objective functions share raw diffusion results through SIMULATION_OUTCOME_CACHE.
VARIANTS = {
    "Original SSA": {"objective": "spread_A + 1.5 * blocked_B", "type": "weighted", "alpha": 1.0, "beta": 1.5},
    "Competition-aware": {"objective": "net_advantage", "type": "net_advantage"},
    "Explicit blocking": {"objective": "blocked_B", "type": "blocking"},
    "Combined influence-blocking": {"objective": "1.0 * spread_A + 2.0 * blocked_B", "type": "weighted", "alpha": 1.0, "beta": 2.0},
    "Gain_A + blocking": {"objective": "(spread_A - spread_B) + 2.0 * blocked_B; Gain_A = spread_A - spread_B", "type": "gain_blocking", "beta": 2.0},
    "Budget-aware": {"objective": "K-specific weighted influence/blocking", "type": "budget"},
}

def score_variant(A, B, sim_seeds, variant_name, k, cache=None):
    # Optional local objective-metric cache; raw simulations are globally cached.
    key = (tuple(sorted(set(A))), tuple(sorted(set(B))), tuple(map(int,sim_seeds)))
    if cache is not None and key in cache:
        m = cache[key]
    else:
        m = evaluate_seed_set(A, B, sim_seeds)
        if cache is not None: cache[key] = m
    vtype = VARIANTS[variant_name]['type']
    if vtype == 'net_advantage': return float(m['net_advantage'])
    if vtype == 'blocking': return float(m['blocked_B'])
    if vtype == 'weighted':
        cfg=VARIANTS[variant_name]
        return float(cfg['alpha']*m['spread_A'] + cfg['beta']*m['blocked_B'])
    if vtype == 'gain_blocking':
        gain_A=float(m['spread_A']-m['spread_B'])
        return float(gain_A+VARIANTS[variant_name]['beta']*m['blocked_B'])
    if vtype == 'budget':
        if k<=20: alpha,beta=0.75,2.25
        elif k<=40: alpha,beta=1.0,1.75
        else: alpha,beta=1.25,1.50
        return float(alpha*m['spread_A']+beta*m['blocked_B'])
    raise ValueError(f'Unknown objective type: {vtype}')

print('Objectives:', {name:spec['objective'] for name,spec in VARIANTS.items()})


# ============================== CELL 19 ==============================
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


def discrete_ssa(pool, B, k, sim_seeds, variant_name, population_size=12,
                 iterations=10, mutation_rate=0.2, random_seed=42):
    rng = random.Random(random_seed)
   # cache: Creates an empty dictionary to remember scores of seed sets we already tested.
    cache = {}  # Local metric cache; raw Monte Carlo outcomes are globally cached.
    score = lambda sol: score_variant(sol, B, sim_seeds, variant_name, k, cache)
# Create population_size candidate seed sets, each containing k unique nodes.
    population = [rng.sample(pool,k) for _ in range(population_size)]
# Evaluate each initial candidate using the selected variant objective over sim_seeds.
    scores = [score(s) for s in population]
#best_i: Finds the index of the highest-scoring salp in the initial population.
    best_i = int(np.argmax(scores))
#food, food_score: Saves a copy of that best seed set as food (the target the swarm chases) and its score as food_score.
    food, food_score = list(population[best_i]), float(scores[best_i])
#hist: Starts a log list recording Iteration 0's best score, average swarm score, and how many distinct seed sets exist in the population.
    hist = [{"iteration":0,"best_objective_score":food_score,
             "mean_objective_score":float(np.mean(scores)),
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
        hist.append({"iteration":iteration,"best_objective_score":food_score,
                     "mean_objective_score":float(np.mean(new_scores)),
                     "unique_solutions":len({tuple(sorted(s)) for s in population})})
        print(f"SSA {iteration}/{iterations}: best objective score={food_score:.2f}, mean objective score={np.mean(new_scores):.2f}")

    if len(food)!=k or len(set(food))!=k or set(food)&set(B):
        raise AssertionError("SSA returned invalid seed set.")
    return food,food_score,pd.DataFrame(hist),cache

# ============================== CELL 21 ==============================
def sampled_swap_refinement(initial, pool, B, k, sim_seeds, variant_name,
                           max_swap_tests=100, random_seed=42):
    """Sample swaps and accept best improving swap per pass; bounded trial count."""
    rng=random.Random(random_seed)
    selected=list(initial); pool=list(dict.fromkeys(pool)); B=set(B)
    cache={}  # Objective cache; simulation realizations use the global shared cache.
    current=score_variant(selected,B,sim_seeds,variant_name,k,cache)
    outside=[node for node in pool if node not in set(selected) and node not in B]
    total=len(selected)*len(outside)
    if total <= max_swap_tests:
        pairs=[(old,new) for old in selected for new in outside]
    else:
        pair_set=set()
        while len(pair_set)<max_swap_tests:
            pair_set.add((rng.choice(selected),rng.choice(outside)))
        pairs=sorted(pair_set)
    best_score=current; best_pair=None
    for old,new in pairs:
        trial=[node for node in selected if node!=old]+[new]
        value=score_variant(trial,B,sim_seeds,variant_name,k,cache)
        if value>best_score+1e-12: best_score=value; best_pair=(old,new)
    if best_pair:
        old,new=best_pair; selected.remove(old); selected.append(new); current=best_score
    return sorted(selected), {'swap_tests':len(pairs),'accepted_swap':best_pair,
                            'objective_before':float(score_variant(initial,B,sim_seeds,variant_name,k,cache)),
                            'objective_after':float(current),'unique_evaluations':len(cache)}

# ============================== CELL 23 ==============================
# Set swap refinement mode inclusion. Each mode is a separate experimental track.
RUN_SWAP_REFINEMENT = True  # False runs only SSA-only mode; True includes both tracks
SWAP_MAX_TESTS = 100       # sampled one-for-one swap trials per seed set (bounded)
SWAP_RANDOM_SEED_BASE = 770000
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / 'ssa_six_variants_optimized_shared_cache_paired_swap'
EXPERIMENT_OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
CHECKPOINT_DIR = EXPERIMENT_OUTPUT_DIR / 'checkpoints'; CHECKPOINT_DIR.mkdir(exist_ok=True)
EVAL_CSV = EXPERIMENT_OUTPUT_DIR / 'evaluation_results.csv'
SUMMARY_CSV = EXPERIMENT_OUTPUT_DIR / 'summary_by_K_scenario_variant_mode.csv'
NESTED_CSV = EXPERIMENT_OUTPUT_DIR / 'nested_seed_set_checks.csv'
OPT_MC_SEEDS = [MASTER_SEED*100000 + 3000000 + i for i in range(CONFIG['optimization_mc_runs'])]
FINAL_MC_SEEDS = [900000000+i for i in range(CONFIG['evaluation_mc_runs'])]
MODE_LIST = ['SSA only'] + (['SSA + Swap refinement'] if RUN_SWAP_REFINEMENT else [])

# Budget and algorithm controls are intentionally kept explicit and unchanged from the source config.
SSA_POPULATION = CONFIG['ssa_population']
SSA_ITERATIONS = CONFIG['ssa_iterations']
SSA_MUTATION_RATE = CONFIG['ssa_mutation_rate']
OPTIMIZATION_MC_RUNS = CONFIG['optimization_mc_runs']
FINAL_EVALUATION_MC_RUNS = CONFIG['evaluation_mc_runs']

print('Variants:', list(VARIANTS))
print('Modes:', MODE_LIST)
print('SSA config:', SSA_POPULATION, SSA_ITERATIONS, SSA_MUTATION_RATE,
      'optimization MC=', OPTIMIZATION_MC_RUNS, 'final MC=', FINAL_EVALUATION_MC_RUNS)
print('Swap tests max per refinement:', SWAP_MAX_TESTS)

# ============================== CELL 25 ==============================
import itertools

# Atomic checkpoint write protects against partially written JSON.
def atomic_json_save(payload, path):
    tmp=path.with_suffix('.tmp')
    with open(tmp,'w',encoding='utf-8') as f:
        json.dump(payload,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def checkpoint_file(variant,mode,k,scenario_id):
    safe=variant.lower().replace(' ','_').replace('+','plus').replace('-','_')
    safe_mode=mode.lower().replace(' ','_').replace('+','plus')
    return CHECKPOINT_DIR/f'{safe}_{safe_mode}_k{k}_{scenario_id}.json'

def ssa_shared_checkpoint_file(variant,k,scenario_id):
    safe=variant.lower().replace(' ','_').replace('+','plus').replace('-','_')
    return CHECKPOINT_DIR/f'{safe}_shared_ssa_k{k}_{scenario_id}.json'

def load_eval_rows():
    if EVAL_CSV.exists(): return pd.read_csv(EVAL_CSV).to_dict('records')
    return []

evaluation_records=load_eval_rows()
def eval_key(row): return (int(row['K']),row['scenario_id'],row['variant'],row['mode'])
completed_eval_keys={eval_key(r) for r in evaluation_records}

for k in K_VALUES:
    pool_size=int(k*CONFIG['candidate_pool_ratio'])
    for scenario_id,scenario_name in SCENARIO_SPECS:
        B=set(B_ORDERS[scenario_id][:k])
        pool,pool_score_details=generate_weighted_candidate_pool(
            B,pool_size,CONFIG['random_candidate_fraction'],
            CONFIG['master_seed']+600000+int(scenario_id[1:]))
        opt_seeds=[MASTER_SEED*100000+int(scenario_id[1:])*10000+i
                   for i in range(CONFIG['optimization_mc_runs'])]
        for variant_name in VARIANTS:
            # One SSA optimization checkpoint shared by SSA-only and Swap branches.
            ssa_path=ssa_shared_checkpoint_file(variant_name,k,scenario_id)
            ssa_signature={
                'k':int(k),'scenario_id':scenario_id,'variant':variant_name,
                'pool_size':pool_size,'random_candidate_fraction':CONFIG['random_candidate_fraction'],
                'population':SSA_POPULATION,'iterations':SSA_ITERATIONS,
                'optimization_mc_runs':OPTIMIZATION_MC_RUNS,'mutation_rate':SSA_MUTATION_RATE,
                'master_seed':MASTER_SEED,'dataset_name':DATA_FILE.name,
                'dataset_size_bytes':DATA_FILE.stat().st_size,
                'cache_strategy':'shared_raw_simulation_cache_v1'
            }
            if ssa_path.exists():
                with open(ssa_path,encoding='utf-8') as f: ssa_saved=json.load(f)
                if ssa_saved.get('config_signature')!=ssa_signature:
                    raise ValueError(f'SSA checkpoint signature mismatch: {ssa_path}')
                ssa_seed_set=ssa_saved['ssa_seed_set']
                optimization_score=float(ssa_saved['optimization_score'])
                history=ssa_saved.get('history',[])
                ssa_runtime=float(ssa_saved.get('ssa_runtime_sec',0.0))
                print('Loaded shared SSA result:',(k,scenario_id,variant_name))
            else:
                rng_seed=MASTER_SEED+500000+int(scenario_id[1:])*1000+k*10+list(VARIANTS).index(variant_name)
                t0=time.perf_counter()
                ssa_seed_set,optimization_score,history,score_cache=discrete_ssa(
                    pool,B,k,opt_seeds,variant_name,SSA_POPULATION,SSA_ITERATIONS,
                    SSA_MUTATION_RATE,rng_seed)
                ssa_runtime=time.perf_counter()-t0
                ssa_saved={'config_signature':ssa_signature,
                    'ssa_seed_set':[int(x) for x in ssa_seed_set],
                    'optimization_score':float(optimization_score),
                    'history':history.to_dict('records'),'ssa_runtime_sec':float(ssa_runtime)}
                atomic_json_save(ssa_saved,ssa_path)

            ssa_seed_set=list(map(int,ssa_seed_set))
            # Both tracks branch from the exact same SSA solution.
            for mode in MODE_LIST:
                key=(int(k),scenario_id,variant_name,mode)
                if key in completed_eval_keys:
                    print('Skipping completed final evaluation:',key)
                    continue
                path=checkpoint_file(variant_name,mode,k,scenario_id)
                signature={**ssa_signature,'mode':mode,
                    'swap_max_tests':SWAP_MAX_TESTS if mode!='SSA only' else 0,
                    'swap_random_seed_base':SWAP_RANDOM_SEED_BASE}
                saved=None
                if path.exists():
                    with open(path,encoding='utf-8') as f: saved=json.load(f)
                    if saved.get('config_signature')!=signature:
                        raise ValueError(f'Mode checkpoint signature mismatch for {key}; use a new experiment directory.')
                    print('Loaded mode checkpoint:',key,'stage=',saved.get('stage','unknown'))
                if saved is None:
                    saved={'config_signature':signature,'stage':'ssa_ready',
                        'ssa_seed_set':ssa_seed_set,'seed_set':ssa_seed_set,
                        'optimization_score':float(optimization_score),'history':history if isinstance(history,list) else history.to_dict('records'),
                        'swap_info':None,'ssa_runtime_sec':float(ssa_runtime),'swap_runtime_sec':0.0}
                    atomic_json_save(saved,path)
                seed_set=list(map(int,saved.get('seed_set',ssa_seed_set)))
                swap_info=saved.get('swap_info')
                swap_runtime=float(saved.get('swap_runtime_sec',0.0))
                if mode=='SSA + Swap refinement' and saved.get('stage')!='swap_complete':
                    swap_t0=time.perf_counter()
                    seed_set,swap_info=sampled_swap_refinement(
                        ssa_seed_set,pool,B,k,opt_seeds,variant_name,SWAP_MAX_TESTS,
                        SWAP_RANDOM_SEED_BASE+int(scenario_id[1:])*1000+k+list(VARIANTS).index(variant_name))
                    swap_runtime=time.perf_counter()-swap_t0
                    saved.update({'stage':'swap_complete','seed_set':[int(x) for x in seed_set],
                        'swap_info':swap_info,'swap_runtime_sec':float(swap_runtime)})
                    atomic_json_save(saved,path)
                elif mode=='SSA only':
                    seed_set=list(ssa_seed_set); swap_info=None; swap_runtime=0.0
                    saved.update({'stage':'ssa_complete','seed_set':seed_set,
                        'swap_info':None,'swap_runtime_sec':0.0})
                    atomic_json_save(saved,path)
                final_eval_t0=time.perf_counter()
                final_metrics=evaluate_seed_set(seed_set,B,FINAL_MC_SEEDS)
                final_eval_runtime=time.perf_counter()-final_eval_t0
                row={'K':int(k),'scenario_id':scenario_id,'opponent_scenario':scenario_name,
                    'variant':variant_name,'mode':mode,'seed_set':json.dumps(sorted(map(int,seed_set))),
                    'optimization_objective_score':float(optimization_score),
                    'spread_A':final_metrics['spread_A'],'spread_B':final_metrics['spread_B'],
                    'blocked_B':final_metrics['blocked_B'],'net_advantage':final_metrics['net_advantage'],
                    'b_only_baseline_spread':final_metrics['b_only_baseline_spread'],
                    'ssa_runtime_sec':float(ssa_runtime),'swap_runtime_sec':float(swap_runtime),
                    'final_evaluation_runtime_sec':float(final_eval_runtime),
                    'optimization_mc_runs':len(opt_seeds),'final_evaluation_mc_runs':len(FINAL_MC_SEEDS),
                    'swap_tests':(swap_info or {}).get('swap_tests',0),
                    'swap_accepted':bool((swap_info or {}).get('accepted_swap') is not None),
                    'swap_objective_before':(swap_info or {}).get('objective_before',np.nan),
                    'swap_objective_after':(swap_info or {}).get('objective_after',np.nan)}
                evaluation_records.append(row)
                pd.DataFrame(evaluation_records).drop_duplicates(
                    subset=['K','scenario_id','variant','mode'],keep='last').to_csv(EVAL_CSV,index=False)
                completed_eval_keys.add(key)
                print('Saved evaluation row:',key,'Net Advantage=',row['net_advantage'])
print('Experiment loop finished. Evaluation CSV:',EVAL_CSV)
print('Shared simulation cache entries:',len(SIMULATION_OUTCOME_CACHE),
      'hits:',SIMULATION_CACHE_HITS,'misses:',SIMULATION_CACHE_MISSES)


# ============================== CELL 26 ==============================
print('Raw simulation cache size:', len(SIMULATION_OUTCOME_CACHE))
print('Cache hits:', SIMULATION_CACHE_HITS, 'Cache misses:', SIMULATION_CACHE_MISSES)
print('Cache hit rate:', SIMULATION_CACHE_HITS / max(1, SIMULATION_CACHE_HITS + SIMULATION_CACHE_MISSES))


# ============================== CELL 28 ==============================
if not EVAL_CSV.exists():
    raise FileNotFoundError(f'No evaluation CSV found at {EVAL_CSV}')
evaluation_df = pd.read_csv(EVAL_CSV)
summary_df = (
    evaluation_df.groupby(['K', 'scenario_id', 'opponent_scenario', 'variant', 'mode'], as_index=False)
    .agg(
        spread_A=('spread_A', 'mean'),
        spread_B=('spread_B', 'mean'),
        blocked_B=('blocked_B', 'mean'),
        net_advantage=('net_advantage', 'mean'),
        ssa_runtime_sec=('ssa_runtime_sec', 'mean'),
        swap_runtime_sec=('swap_runtime_sec', 'mean'),
        final_evaluation_runtime_sec=('final_evaluation_runtime_sec', 'mean'),
        evaluation_mc_runs=('final_evaluation_mc_runs', 'first')
    )
)
summary_df.to_csv(SUMMARY_CSV, index=False)
print('Summary CSV saved:', SUMMARY_CSV)
display(summary_df.sort_values(['mode', 'variant', 'scenario_id', 'K']))

# ============================== CELL 29 ==============================
# Plot Net Advantage from saved summary CSV (one figure per B scenario).
summary_from_csv=pd.read_csv(SUMMARY_CSV)
for mode in MODE_LIST:
    for scenario_id,scenario_name in SCENARIO_SPECS:
        subset=summary_from_csv[(summary_from_csv['mode']==mode)&(summary_from_csv['scenario_id']==scenario_id)]
        if subset.empty: continue
        plt.figure(figsize=(11,6))
        for variant in VARIANTS:
            d=subset[subset['variant']==variant].sort_values('K')
            if not d.empty: plt.plot(d['K'],d['net_advantage'],marker='o',label=variant)
        plt.xlabel('Seed budget K'); plt.ylabel('Net Advantage (Spread A - Spread B)')
        plt.title(f'Net Advantage by K | {scenario_id}: {scenario_name} | {mode}')
        plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# ============================== CELL 31 ==============================
if set(['SSA only','SSA + Swap refinement']).issubset(set(evaluation_df['mode'])):
    wide=evaluation_df.pivot_table(index=['K','scenario_id','opponent_scenario','variant'],
        columns='mode',values='net_advantage').reset_index()
    if 'SSA only' in wide and 'SSA + Swap refinement' in wide:
        wide['net_advantage_change_with_swap']=wide['SSA + Swap refinement']-wide['SSA only']
        wide.to_csv(EXPERIMENT_OUTPUT_DIR/'swap_refinement_net_advantage_comparison.csv',index=False)
        display(wide.sort_values(['variant','scenario_id','K']))
else:
    print('Both modes are not present yet; comparison will populate after both runs finish.')

# ============================== CELL 33 ==============================
def parse_seed_set(s):
    return set(map(int,json.loads(s)))
nested_rows=[]
for (mode,variant,scenario_id),group in evaluation_df.groupby(['mode','variant','scenario_id']):
    by_k={int(r.K):parse_seed_set(r.seed_set) for r in group.itertuples()}
    expected_k=[10,20,30,40,50]
    present=sorted(by_k)
    chain_ok=True; pair_checks=[]
    for k1,k2 in zip(expected_k[:-1],expected_k[1:]):
        if k1 not in by_k or k2 not in by_k:
            ok=False; subset_count=np.nan
        else:
            ok=by_k[k1].issubset(by_k[k2]); subset_count=len(by_k[k1]&by_k[k2])
        pair_checks.append(ok); chain_ok=chain_ok and ok
        nested_rows.append({'mode':mode,'variant':variant,'scenario_id':scenario_id,
            'K_from':k1,'K_to':k2,'subset_relation_holds':ok,
            'intersection_size':subset_count,'smaller_set_size':k1})
    nested_rows.append({'mode':mode,'variant':variant,'scenario_id':scenario_id,
        'K_from':'10→50','K_to':'full chain','subset_relation_holds':chain_ok,
        'intersection_size':np.nan,'smaller_set_size':np.nan})
nested_df=pd.DataFrame(nested_rows)
nested_df.to_csv(NESTED_CSV,index=False)
print('Nested-seed checks saved:',NESTED_CSV)
display(nested_df)

# ============================== CELL 35 ==============================
expected_rows=len(K_VALUES)*len(SCENARIO_SPECS)*len(VARIANTS)*len(MODE_LIST)
print(f'Rows present: {len(evaluation_df)} / expected {expected_rows}')
assert evaluation_df[['K','scenario_id','variant','mode']].duplicated().sum()==0
for r in evaluation_df.itertuples():
    A=parse_seed_set(r.seed_set); B=set(B_ORDERS[r.scenario_id][:int(r.K)])
    assert len(A)==int(r.K) and len(A)==len(set(A)) and not (A&B)
assert evaluation_df['variant'].isin(VARIANTS).all()
assert evaluation_df['mode'].isin(MODE_LIST).all()
if len(evaluation_df)==expected_rows:
    assert set(evaluation_df['K'])==set(K_VALUES)
    assert set(evaluation_df['scenario_id'])=={s for s,_ in SCENARIO_SPECS}
    assert set(evaluation_df['variant'])==set(VARIANTS)
    assert set(evaluation_df['mode'])==set(MODE_LIST)
    print('All expected experiment combinations present and validated.')
else:
    print('Experiment is partial; valid saved rows retained. Run experiment cell to resume missing combinations.')

# ============================== CELL 37 ==============================
print('Experiment output directory:',EXPERIMENT_OUTPUT_DIR)
print('Evaluation results CSV:',EVAL_CSV)
print('Summary CSV:',SUMMARY_CSV)
print('Nested-set checks CSV:',NESTED_CSV)
print('Checkpoint directory:',CHECKPOINT_DIR)
