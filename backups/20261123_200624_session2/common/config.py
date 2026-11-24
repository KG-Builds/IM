"""Central configuration for the signed competitive influence experiment.

Every value that affects results lives here and is copied into each result row
(and checkpoint signature) so that runs are reproducible and comparable.
"""
from __future__ import annotations

# --- Network ---------------------------------------------------------------
NETWORK_FILE = "synthetic_network_1000_7500.csv"
NETWORK_NODES = 1000
NETWORK_ARCS = 7500
NETWORK_SEED = 42
NETWORK_SIGN_POSITIVE_PROB = 0.5
TRUST_MAGNITUDE_RANGE = (1, 10)
PROBABILITY_MAPPING = {"1-3": 0.1, "4-7": 0.5, "8-10": 0.9}

# --- Experiment grid --------------------------------------------------------
K_VALUES = [10, 20, 30, 40, 50]
SCENARIOS = ["B1", "B2", "B3", "B4"]
SCENARIO_NAMES = {
    "B1": "Random opponent",
    "B2": "Degree-based opponent",
    "B3": "Positive PageRank opponent",
    "B4": "Community-concentrated opponent",
}
MASTER_SEED = 42

# --- Optimizer --------------------------------------------------------------
SSA_POPULATION = 12
SSA_ITERATIONS = 10
SSA_MUTATION_RATE = 0.20
OPT_MC_RUNS = 15
FINAL_MC_RUNS = 100
POOL_RATIO = 10               # candidate pool target = POOL_RATIO * K
RANDOM_POOL_FRACTION = 0.30   # 30 % random exploration, 70 % ranked
SWAP_MAX_TESTS = 100

# --- Seed protocol ----------------------------------------------------------
# All derived seeds are functions of MASTER_SEED and the configuration only;
# no wall-clock or process state enters the RNG.
SEED_PROTOCOL_VERSION = "v2-common-random-numbers"


def scenario_number(scenario_id: str) -> int:
    if scenario_id not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario {scenario_id!r}")
    return int(scenario_id[1:])


def opponent_seed(scenario_id: str) -> int:
    # Preserves the starter protocol: only B1 uses randomness, seeded once.
    return MASTER_SEED + 810001


def candidate_pool_seed(scenario_id: str) -> int:
    return MASTER_SEED + 600000 + scenario_number(scenario_id)


def optimization_mc_seeds(scenario_id: str, n_runs: int = OPT_MC_RUNS) -> list[int]:
    base = MASTER_SEED * 100000 + 3000000 + scenario_number(scenario_id) * 10000
    return [base + i for i in range(int(n_runs))]


def final_mc_seeds(n_runs: int = FINAL_MC_RUNS) -> list[int]:
    # Shared across all scenarios/configurations on purpose (common random numbers).
    return [900000000 + i for i in range(int(n_runs))]


def ssa_rng_seed(scenario_id: str, k: int, objective_index: int) -> int:
    return MASTER_SEED + 500000 + scenario_number(scenario_id) * 1000 + int(k) * 10 + int(objective_index)


def swap_rng_seed(scenario_id: str, k: int, objective_index: int) -> int:
    return 770000 + scenario_number(scenario_id) * 1000 + int(k) + int(objective_index)
