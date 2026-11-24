# Configuration — Signed Competitive Influence Maximization

This file records every constant, seed, formula and design decision used by the
experiment. Values marked **assumption** are project choices, not canonical
model parameters.

---

## 1. Paths

| Item | Path |
|---|---|
| Project root | repository root |
| Network CSV | `data/synthetic_network_1000_7500.csv` |
| Network metadata (original) | `data/network_metadata.json` |
| Network metadata v2 | `data/network_metadata_v2.json` |
| C++ engine | `cpp_engine/` |
| Results | `results/<run_id>/<scenario>/` |
| Smoke results | `results/smoke/<run_id>/` |
| Logs | `logs/<run_id>.log` |
| Analysis | `analysis/figures/<run_id>_<timestamp>/`, `analysis/tables/<run_id>_<timestamp>/` |

The network file is read-only source data. No experiment code overwrites it.

---

## 2. Network

| Parameter | Value |
|---|---|
| Nodes | 1,000 (ids 0–999) |
| Directed arcs | 7,500 (one CSV row = one directed arc) |
| Underlying unordered pairs | 7,498 |
| Reciprocal arc pairs | 2 |
| Self-loops | 0 |
| Duplicate directed arcs | 0 |
| Topology | Barabási–Albert-style undirected backbone (`m=7`, initial clique 8), randomly oriented, plus random directed fill |
| Generation seed | 42 |
| Sign | positive with probability 0.5 |
| Trust magnitude | uniform integer 1–10 |
| Rating | `polarity × strength` |
| Probability mapping | 1–3 → 0.1; 4–7 → 0.5; 8–10 → 0.9 |
| Positive / negative arcs | 3,720 / 3,780 |
| Probability classes 0.1 / 0.5 / 0.9 | 2,278 / 2,950 / 2,272 |
| Out-degree min/mean/max | 0 / 7.5 / 77 |
| In-degree min/mean/max | 1 / 7.5 / 67 |
| Weak components | 1 (all 1,000 nodes) |
| Strong components | 3 (largest 998) |
| Isolated nodes | 0 |

File SHA-256:
`659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`

Semantic edge-table SHA-256:
`461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25`

### Edge-count convention

The generator builds 7,500 undirected pairs, orients each once, and then uses a
random directed fill if orientation collisions reduce the count. Because the
pair de-duplication filter is order-sensitive, a few pairs can be added twice,
collapse after orientation, and be replaced by fill arcs. The delivered file has
exactly 7,500 unique directed arcs over 7,498 unordered pairs plus 2 reciprocal
pairs (4 reciprocal arcs). This is documented explicitly rather than
reinterpreting “7,500 edges” silently. Seed 42 reproduces the CSV byte-for-byte.

---

## 3. Diffusion model

Synchronous, discrete rounds; per-node states:

| State | Meaning |
|---|---|
| 0 | ignorant |
| 1 | A-spreader |
| 2 | B-spreader |
| 3 | A-stifler |
| 4 | B-stifler |

Rules:

1. A seeds start as A-spreaders; B seeds as B-spreaders; A/B overlap is rejected.
2. Each round, every current spreader chooses one uniformly random outgoing
   neighbour it has not contacted before (per-edge contact memory).
3. If it has no uncontacted outgoing neighbour, or the chosen target is not
   ignorant at the start of the round, it becomes a stifler of its own rumor and
   does not transmit that round.
4. Otherwise, with the edge transmission probability, it transmits; a negative
   edge flips the rumor (A→B, B→A).
5. Proposals are resolved synchronously: highest edge strength wins; ties between
   different rumors are broken uniformly at random.
6. Final `Spread_A` = count of states 1 and 3; `Spread_B` = 2 and 4;
   `ignorant` = 0; `rounds` = rounds executed.

This is a finite-contact Maki–Thompson variant. See `cpp_engine/MODEL_SPEC.md`.

---

## 4. Opponent B scenarios

All orders have length 50 and are nested across K (the K-seed set is a prefix).

| Scenario | Rule | Seed |
|---|---|---|
| B1 Random | `random.Random(master_seed + 810001).sample(nodes, 50)` | 42 + 810001 |
| B2 Degree | out-degree descending, ties ascending node id | — |
| B3 Positive PageRank | PageRank on positive-only directed subgraph, edge weight = probability, α=0.85, max_iter=200, tol=1e-10; descending, ties ascending id | — |
| B4 Community | Louvain on undirected unweighted projection (signs ignored), seed 42; largest community (ties: smallest member id); sort by out-degree descending, ties ascending id | 42 |

B4 largest community size on the delivered network: 227.

---

## 5. Candidate pool

- Target size: `10 × K`
- Polarity-aware score:
  `score(u) = Σ probability of outgoing positive edges − Σ probability of outgoing negative edges`
- Negative edges can only lower a node's score; they are never rewarded.
- Composition: `n_ranked = int(0.7 × 10K)` from the ranking; the remainder
  sampled uniformly without replacement from eligible non-ranked nodes.
- Opponent seeds excluded everywhere.
- If fewer eligible nodes exist, the pool is truncated and metadata records it;
  fewer than K eligible nodes is an error.
- Pool seed: `master_seed + 600000 + scenario_number`.
- The pool is a node-level score; it does not use target in-degree or distance
  (starter design).

For the full network, every K reached exactly 10·K with 70 ranked + 30 random
nodes and no opponent seed.

---

## 6. SSA

| Parameter | Value |
|---|---|
| Population | 12 |
| Iterations | 10 |
| Mutation rate | 0.20 |
| `c1` | `2·exp(−(4t/T)²)` |
| Exploitation probability | `1 − c1/2` |
| Representation | set of exactly K unique candidate nodes |
| Fitness | mean objective over 15 fixed optimization MC seeds |
| Elitism | best-so-far food preserved; worst individual replaced by food |
| Random seed | `master_seed + 500000 + scenario_number·1000 + K·10 + objective_index` |

Repair/dedup logic keeps only candidate-pool members and pads to exactly K.
SSA is run **once** per (scenario, K, objective); both modes use the saved set.

This is a discrete adaptation of SSA preserved from the starter. Canonical
equation-level fidelity cannot be verified without the missing reference
notebook; it is a documented researcher decision.

---

## 7. Swap refinement

- Start from the exact SSA seed set.
- One-for-one replacements only.
- Evaluate up to `max_tests = 100` pairs (all pairs when fewer).
- Apply the single best strictly improving swap; equal/worse swaps rejected
  (elitist).
- Exactly K unique candidate nodes; opponent seeds excluded.
- Random seed: `770000 + scenario_number·1000 + K + objective_index`.
- Logs tests, accepted pair, objective before/after, evaluation count.

The starter's single-acceptance policy is preserved; iterative refinement would
be a different algorithm and is not silently substituted.

---

## 8. Monte Carlo protocol

| Parameter | Value |
|---|---|
| Optimization MC runs | 15 |
| Final evaluation MC runs | 100 |
| Optimization seeds | `master_seed·100000 + 3000000 + scenario_number·10000 + i`, i=0..14 |
| Final seeds | `900000000 + i`, i=0..99 |
| Baseline | paired B-only counterfactual with the same seed list |

Common random numbers: the same final seed list is used by every scenario,
objective and mode, enabling paired comparisons. Optimization and final
evaluation use different seed lists.

### Blocked_B (provisional)

```
baseline_B(B, s) = Spread_B(A=empty, B, s)
Blocked_B        = max(0, mean_s baseline_B − mean_s Spread_B(A, B, s))
```

Per-seed unclipped differences are stored for analysis. This is the starter's
only implementation; it has no reference-notebook confirmation, so
Blocked_B-dependent objectives are labelled `provisional_blocked_b`.

---

## 9. Objectives

| ID | Name | Formula | Blocked_B | Validity |
|---|---|---|---|---|
| O1 | Original SSA | `Spread_A + 1.5·Blocked_B` | yes | provisional |
| O2 | Competition-aware | `Spread_A − Spread_B` | no | **validated** |
| O3 | Explicit blocking | `Blocked_B` | yes | provisional |
| O4 | Combined influence-blocking | `Spread_A + 2·Blocked_B` | yes | provisional |
| O5 | Gain_A + blocking | `(Spread_A − Spread_B) + 2·Blocked_B` | yes | provisional |
| O6 | Budget-aware | `α(K)·Spread_A + β(K)·Blocked_B` | yes | provisional |

Budget coefficients:

| K | α | β |
|---|---|---|
| 10, 20 | 0.75 | 2.25 |
| 30, 40 | 1.00 | 1.75 |
| 50 | 1.25 | 1.50 |

`Gain_A = Spread_A − Spread_B`. O4 and O5 are distinct.

---

## 10. Experiment matrix

- Scenarios: B1, B2, B3, B4
- K: 10, 20, 30, 40, 50
- Objectives: O1–O6
- Modes: `SSA only`, `SSA + Swap refinement`
- Per scenario: 5 × 6 × 2 = 60 configurations
- Total: 4 × 60 = 240 configurations

Smoke test: B1, K=10, O2+O3, both modes, population 4, 2 iterations, 5
optimization MC, 10 final MC, 10 swap tests (4 configurations).

---

## 11. Checkpoints, resume and signatures

- Run directory: `results/<run_id>/`
- Scenario files: `experiment_results.csv`, `checkpoints/`, `mc_runs/`,
  `ssa_history/`, `scenario_metadata.json`
- Run files: `run_metadata.json`, `manifest.json`
- CSV and JSON writes are atomic (sibling temp file + `os.replace`).
- A configuration signature includes: run kind, scenario, K, objective id/name,
  network file + semantic fingerprint, master seed, SSA parameters, MC counts,
  pool ratio/fraction/signature, opponent-order signature, swap tests,
  evaluation protocol, and optimization/final seed hashes.
- Resume skips completed rows only when the signature matches; mismatch raises
  `CheckpointMismatch` and instructs the user to use a new run ID.
- SSA is run once; if a checkpoint exists it is reused. A missing mode can be
  completed from the saved SSA seed set without rerunning SSA.
- Seed-set columns are unordered sets. `ssa_seed_set` and
  `swap_initial_seed_set` preserve the SSA return order; `final_seed_set` is
  sorted. Compare them as sets, not strings.
- `final_evaluation_runtime_sec` includes a possible metric-cache hit, so it can
  be near zero when the exact (A, B, seeds) evaluation already exists.
  `simulations_for_config`, `simulation_seconds_for_config`,
  `cache_hits_for_config` and `cache_misses_for_config` record actual work.
- `run_metadata.json` stores the absolute network/results paths used at run time.
  The experiment is reproducible from the saved seed sets and config, but
  resuming the exact run directory is tied to the current workspace path.

---

## 12. Analysis

- Read-only: never imports the engine or launches simulations.
- Validates columns, duplicates, fingerprints, net-advantage identity,
  non-negative Blocked_B, expected/missing configurations, MC seed pairing and
  MC-file paths (no path escape).
- Tables: overall, per-scenario, per-K, mode comparison, objective-vs-O2 paired
  comparison, objective ranking, runtime, variability, robustness, best
  configuration per scenario/K.
- Figures: Net Advantage vs K; Swap vs SSA paired differences; Spread_A vs
  Spread_B; Blocked_B (provisional watermark); runtime; objective ranking;
  robustness across B1–B4; configuration variability; objective×scenario
  heatmaps.
- Objective ranking always uses final Net Advantage, never optimizer fitness.
- Paired tests use MC-seed alignment; they do not treat MC replicates as
  independent optimizer runs. Holm correction is applied within each
  scenario/K/mode group for objective-vs-O2 comparisons.

---

## 13. Environment / reproducibility

- Python 3.14.10; NumPy 2.4.6; pandas 2.3.3; networkx 3.7; matplotlib 3.10.8;
  SciPy 1.16.2; pybind11 3.1.0; g++ 16.2.1; OpenMP enabled; 8 CPUs.
- Build: `cd cpp_engine && python setup.py build_ext --inplace` (no install).
- Tests: `python -m unittest discover -s tests -p 'test_*.py'` (102 tests).
- No packages were installed or upgraded by the autonomous work.
- All random seeds are recorded per row; all selected seed sets are saved.
