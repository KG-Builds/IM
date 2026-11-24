# Complete Project Context — Signed Competitive Influence Maximization

**Last updated:** 2026-11-24  
**Workspace root:** `/home/kushagragarg/im/IM_Experiment_Starter`  
**Main benchmark directory:** `IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness`

> **Handover note for another OpenCode agent.** Read this whole file before changing anything. The project is already complete and validated. Your job is to understand it, verify it, or extend it carefully — not to rerun expensive experiments or rewrite working code without a reason. All paths below are relative to the workspace root unless they are absolute.

---

## 0. How to use this document

Read in this order:

1. Sections 1–4 for the research question, model, and repository map.
2. Section 5 for the 12-method benchmark design.
3. Sections 6–7 for the 5,000-node and 1,000-node results.
4. Section 8 for the notebooks.
5. Section 9 for validation and reproduction commands.
6. Sections 10–12 for interpretation, guardrails, and common tasks.

If you only need one sentence: **this project compares seed-selection algorithms for competitive influence maximization on synthetic signed networks, using Net Advantage = Spread_A − Spread_B as the external metric, with separate 5,000-node and 1,000-node runs that must never be mixed.**

---

## 1. Executive summary

### 1.1 Research problem

A campaign (Rumor A / truth) chooses K seed nodes to compete against a fixed opponent campaign (Rumor B) on a signed directed network. The network has positive and negative trust edges. The evaluated diffusion model is a finite-contact competitive Maki–Thompson variant. The main question is:

> Which seed-selection method gives the best Net Advantage against different opponent strategies and different seed budgets?

**Net Advantage** is:

```text
Net Advantage = Spread_A - Spread_B
```

Higher is better. `Spread_A` and `Spread_B` are reported separately so a method cannot look good only by suppressing B while failing to spread A.

### 1.2 What is complete

- A **5,000-node / 37,500-edge** reverse-influence experiment with six SSA objectives and optional Swap refinement (`exp_20261124_5000_reversed`).
- A **12-method benchmark** on the same 5,000-node network, with SSA O1/O4/O6 plus notebook baselines and FIBM.
- A separate **1,000-node / 7,500-edge** version of the same 12-method benchmark, using the corrected reverse-influence model.
- Two executed research notebooks:
  - `competitive_im_benchmark_with_fibm.ipynb` — 5,000-node version.
  - `competitive_im_benchmark_1000_with_fibm.ipynb` — 1,000-node version.
- Validation, tests, figures, reports, and reuse audits.

### 1.3 Headline findings

**5,000-node 12-method benchmark**

- Highest mean Net Advantage: **SSA O1 = 47.878**.
- Close second: **SSA O6 = 47.261**.
- **CELF = 46.115**, with **6 strict wins out of 20**, tied with SSA O1 for the most wins.
- **No universal winner.** CELF wins six cells, including B4/K=50 with the highest single-cell value (92.68).
- Only **4 of 20** winner-vs-runner-up paired 95% intervals exclude zero.
- Signed-GA and CRIEG win **zero** cells on this network.

**1,000-node 12-method benchmark**

- Highest mean Net Advantage: **SSA-Notebook = 24.483**.
- But SSA-Notebook is **the same SSA algorithm as SSA O1 with a different random seed**, so this is not a genuinely different method.
- SSA O6 = 24.218; SSA O1 = 24.212; CELF = 23.638; SSA O4 = 23.416.
- **SSA O1, SSA O4, and CELF each win 4 of 20 cells.** SSA O6 and CRIEG win 3 each; SSA-Notebook wins 2.
- **CELF is the main real competitor.** CRIEG is a secondary threat on this smaller graph, especially at B2/K=40–50 and B3/K=50.
- Signed-GA is not competitive on either network: mean 12.364 and 0 wins on 1,000; mean 28.207 and 0 wins on 5,000.
- Only **4 of 20** winner-vs-runner-up paired 95% intervals exclude zero.

**FIBM**

- FIBM is a fairness-aware influence-blocking baseline, not a Net Advantage optimizer.
- 5,000-node: mean Net Advantage 18.718, mean Blocked_B 34.063.
- 1,000-node: mean Net Advantage 18.363, mean Blocked_B 19.475.
- It slightly improves blocking on average but has lower Spread_A, so its Net Advantage is lower.
- Its internal fairness values (`Pi`, `tau`, `sigma`) come from a positive-edge CIC surrogate, not from the signed Maki–Thompson evaluation.

### 1.4 Most important warning

There are **two different 5,000-node result sets** and **two different 1,000-node meanings**. Do not mix them:

| Question | Authoritative location |
|---|---|
| 1,000-node, 12 methods, reverse influence | `IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/benchmark_1000_synthetic/raw_results/` |
| 5,000-node, 12 methods, reverse influence | `IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/raw_results/` |
| 5,000-node, six SSA objectives, SSA-only + SSA+Swap | `IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/` |
| Old 1,000-node same-direction results | `IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/to_be_discarded/` — **invalid, do not use** |

---

## 2. Research model

### 2.1 Signed network

The CSV is the **trust** network. Each row is one directed trust arc:

```text
source, target, rating, probability, strength, polarity
```

- `polarity = +1` means positive trust.
- `polarity = -1` means negative trust.
- `rating = polarity * strength`.
- `strength` is an integer 1–10.
- Trust-to-transmission-probability mapping:
  - strength 1–3 → 0.1
  - strength 4–7 → 0.5
  - strength 8–10 → 0.9

### 2.2 Trust direction versus influence direction

This is the single most important modeling convention:

```text
trust:      source -> target
influence:  target -> source   (exact reverse)
```

The reversal is applied **exactly once** in `common/network.py::load_influence_network(..., reverse=True)`. No downstream component reverses it again. All graph-based operations use the influence graph:

- C++ diffusion engine
- Python reference simulator
- candidate-pool scoring
- B1–B4 opponent construction
- SSA fitness
- Swap refinement (in the separate root experiment)
- final Monte Carlo evaluation

The 5,000-node graph has **0 reciprocal trust pairs** and **0 same-direction influence edges**.

The 1,000-node graph has **2 reciprocal trust pairs**. Because both directions of each reciprocal pair exist in the trust file, the direction check reports **4 same-direction edges** even though every trust arc has been reversed exactly once. This is expected. The 1,000-node notebook asserts:

```python
assert direction["same_direction_edges"] == 2 * direction["reciprocal_trust_pairs"]
```

Do not “fix” this to `== 0`; that would be wrong for this graph.

### 2.3 Diffusion model

Finite-contact competitive Maki–Thompson variant, synchronous rounds:

- States: `0` ignorant, `1` A-spreader, `2` B-spreader, `3` A-stifler, `4` B-stifler.
- A seeds start as A-spreaders; B seeds start as B-spreaders.
- A and B seed sets must be disjoint.
- Each spreader contacts one uniformly random **uncontacted outgoing influence neighbor** per round.
- If it has no uncontacted neighbor, or the chosen target is not ignorant at the start of the round, it becomes a stifler of its own rumor.
- Otherwise it transmits with the edge probability.
- A negative edge **flips** the rumor (A→B, B→A); a positive edge preserves it.
- Proposals are resolved by highest edge strength; different-rumor ties are broken uniformly at random.
- `Spread_A` counts states 1 and 3.
- `Spread_B` counts states 2 and 4.
- `rounds` includes the terminal empty round, matching the reference semantics.

Final evaluations use the persistent C++ engine. The Python simulator is retained for model-logic tests.

### 2.4 Metrics

- **Net Advantage** = `Spread_A - Spread_B` — primary ranking metric.
- **Spread_A** — final A-side spread (states 1+3).
- **Spread_B** — final B-side spread (states 2+4).
- **Blocked_B**:

```text
baseline_B(B, s) = Spread_B(A=empty, B, s)
Blocked_B = max(0, mean_s baseline_B(B, s) - mean_s Spread_B(A, B, s))
```

The B-only baseline uses the same paired Monte Carlo seeds as the A/B run.

### 2.5 Objectives

| ID | Name | Formula |
|---|---|---|
| O1 | Original SSA | `Spread_A + 1.5 * Blocked_B` |
| O2 | Competition-aware | `Spread_A - Spread_B` |
| O3 | Explicit blocking | `Blocked_B` |
| O4 | Combined influence-blocking | `Spread_A + 2.0 * Blocked_B` |
| O5 | Gain_A + blocking | `(Spread_A - Spread_B) + 2.0 * Blocked_B` |
| O6 | Budget-aware | `alpha(K) * Spread_A + beta(K) * Blocked_B` |

O6 coefficients:

| K | alpha | beta |
|---|---|---|
| 10, 20 | 0.75 | 2.25 |
| 30, 40 | 1.00 | 1.75 |
| 50 | 1.25 | 1.50 |

The 12-method benchmark uses only **O1, O4, O6**. The root 5,000 experiment uses all six objectives.

### 2.6 Monte Carlo protocol

- **Optimization MC:** 15 common seeds per scenario.
  - Formula: `master_seed * 100000 + scenario_number * 10000 + i`, `i = 0..14`.
- **Final evaluation MC:** 100 common seeds.
  - `900000000 .. 900000099`.
- Common random numbers are used across methods, so comparisons are paired by final seed.
- The 100 final MC runs measure **evaluation uncertainty** for a fixed seed set. They do **not** measure optimizer run-to-run variability.
- There is **one optimizer run per method/configuration** unless explicitly stated otherwise.

### 2.7 Opponent scenarios B1–B4

All opponent orders have length 50 and are nested across K; the K-seed set is the prefix of length K. The same B set is used for every A method within a scenario/K cell.

| Scenario | Construction |
|---|---|
| B1 Random | `random.Random(master_seed + 810001).sample(nodes, 50)` |
| B2 Degree | influence out-degree descending, ties by ascending node id |
| B3 Positive PageRank | PageRank on the positive-only directed influence subgraph, edge weight = probability, alpha=0.85, max_iter=200, tol=1e-10; descending, ties by id |
| B4 Community | Louvain on the undirected unweighted influence projection, seed 42; largest community; sort by influence out-degree descending, ties by id |

B1–B4 are **four deliberately different opponent strategies**, not independent replicates. Do not pool them into a single significance test.

---

## 3. Repository map

### 3.1 Top level

```text
/home/kushagragarg/im/IM_Experiment_Starter/
├── complete_context.md                      # this file
├── baseline comparison research papers/     # reference PDFs
├── original datasets/                       # Slashdot / Bitcoin OTC raw data
└── IM_Synthetic_Experiment/                 # main research repository
```

### 3.2 `IM_Synthetic_Experiment/`

Key files and directories:

```text
IM_Synthetic_Experiment/
├── README.md                         # 5,000-node reverse-influence project overview
├── CONFIG.md                         # LEGACY 1,000-node same-direction config (historical)
├── BENCHMARK.md                      # LEGACY 1,000-node benchmark (historical)
├── CHANGELOG.md
├── AUDIT_REPORT.md
├── AUDIT_REPORT_ADDENDUM.md
├── REFERENCE_NOTEBOOK_AUDIT.md
├── GPT_Contexts/
│   ├── ContextForGPT.md              # historical pre-5,000 context
│   └── ContextForGPT_5000_Reversed.md # detailed 5,000 handover
├── data/
│   ├── synthetic_network_1000_7500.csv
│   ├── network_metadata_v2.json      # 1,000-node metadata/hashes
│   ├── network_metadata.json         # legacy metadata
│   └── README.md
├── common/                           # shared research code
│   ├── network.py                    # trust loading + influence reversal
│   ├── network_generator.py          # legacy 1,000 generator
│   ├── network_generator_large.py    # 5,000 generator
│   ├── opponents.py                  # B1–B4
│   ├── candidates.py                 # 10*K polarity-aware pool
│   ├── objectives.py                 # O1–O6
│   ├── diffusion.py                  # Python reference simulator
│   ├── cpp_adapter.py                # persistent C++ engine adapter
│   ├── evaluation.py                 # MC metrics / Blocked_B / caching
│   ├── ssa.py                        # discrete SSA
│   ├── swap.py                       # bounded Swap (root experiment only)
│   ├── stats.py                      # bootstrap / paired diff
│   └── ...
├── cpp_engine/                       # C++ engine + pybind11 + OpenMP
├── tests/                            # 121 root tests
├── notebooks/                        # earlier per-scenario notebooks
├── old work/                         # supplied reference notebooks (read-only)
├── experiments/
│   ├── exp_20261124_5000_reversed/   # 5,000 six-objective / Swap experiment
│   └── benchmark_ssa_baselines_fairness/  # 12-method benchmark, 5,000 + 1,000
└── ...
```

### 3.3 Benchmark experiment directory

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/
├── README.md
├── METHODS.md
├── ALGORITHM_AUDIT.md
├── PAPER_IMPLEMENTATION.md
├── FINAL_REPORT.md
├── AUDIT.md
├── config/
│   ├── benchmark_config.json                 # 5,000, 12 methods
│   └── benchmark_1000_synthetic_config.json  # 1,000, 12 methods
├── network/
│   ├── network_reference.json
│   └── network_validation.json
├── algorithms/
│   ├── core.py
│   ├── notebook_baselines.py
│   ├── fibmm.py
│   └── reuse.py
├── raw_results/                              # 5,000 12-method results
│   ├── results.csv
│   ├── ledger.csv
│   ├── manifest.json
│   ├── seed_sets.csv
│   ├── opponent_seeds.csv
│   ├── mc/                                   # 100-row per-cell tables
│   ├── checkpoints/
│   ├── preprocessing/
│   └── fibmm_diagnostics/
├── benchmark_1000_synthetic/                 # 1,000 12-method results
│   ├── README.md
│   └── raw_results/
│       ├── results.csv
│       ├── ledger.csv
│       ├── manifest.json
│       ├── mc/
│       ├── checkpoints/
│       └── fibmm_diagnostics/
├── aggregated/                               # 5,000 aggregated tables/validation
├── figures/
│   ├── notebook/                             # 5,000 notebook figures
│   └── notebook_1000/                        # 1,000 notebook figures
├── reports/
│   ├── CHECKLIST.md
│   ├── STATISTICAL_NOTES.md
│   └── ALGORITHM_BUDGETS.md
├── validation/
│   ├── tests.txt
│   ├── verify.txt
│   ├── crosscheck.txt
│   ├── notebook_validation.txt
│   ├── notebook_1000_validation.txt
│   └── benchmark_1000_validation.json
├── tests/
│   └── test_notebook_1000.py
├── run_benchmark.py                          # 5,000 CLI runner
├── run_1000_synthetic.py                     # 1,000 runner
├── analyze_benchmark.py                      # 5,000 aggregation/plots
├── notebook_analysis.py                      # 1,000 notebook analysis helper
├── competitive_im_benchmark_with_fibm.ipynb
└── competitive_im_benchmark_1000_with_fibm.ipynb
```

### 3.4 Root 5,000 experiment

```text
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/
├── README.md
├── run_experiment_5000.py
├── analyze_5000.py
├── config/
│   ├── experiment_config.json
│   └── run_20261124_5000_reversed_config.json
├── network/
│   ├── synthetic_network_5000_37500.csv
│   ├── network_metadata.json
│   └── network_validation.json
├── raw_results/run_20261124_5000_reversed/
├── aggregated/
├── final_figures/                 # exactly four Net Advantage PNGs
├── reports/FINAL_REPORT.md
├── validation/
│   ├── tests_5000.txt
│   ├── benchmark_5000.md
│   ├── benchmark_5000.json
│   ├── replay_check.json
│   └── VALIDATION_GATE.md
└── to_be_discarded/               # old same-direction results, invalid
```

---

## 4. The two 5,000-node studies (do not confuse them)

Both use the **same 5,000-node network file** and the **same reversed influence direction**, but they answer different questions.

| Property | Root experiment | 12-method benchmark |
|---|---|---|
| Directory | `experiments/exp_20261124_5000_reversed/` | `experiments/benchmark_ssa_baselines_fairness/` |
| Methods | SSA O1–O6, with and without Swap | SSA O1/O4/O6 + 8 notebook baselines + FIBM |
| Modes | SSA-only and SSA+Swap | SSA-only; no Swap |
| Cells | 4 × 5 × 6 × 2 = 240 | 4 × 5 × 12 = 240 |
| Primary result | Which SSA objective wins across B × K? | Which method wins across B × K? |
| Final report | `exp_20261124_5000_reversed/reports/FINAL_REPORT.md` | `benchmark_ssa_baselines_fairness/FINAL_REPORT.md` |
| Status | 240/240, 121 tests, 202,055 simulations | 240/240, validation PASS, 60 SSA rows reused, 180 new |

The benchmark’s SSA O1/O4/O6 rows are **reused from the root experiment** after validating network hash, direction, opponent seeds, candidate pool, SSA parameters, seed sets, and 100-row MC files. The benchmark’s 180 new rows are the notebook baselines and FIBM.

The root experiment’s **SSA+Swap** results are not part of the benchmark. The benchmark explicitly excludes Swap refinement and CELF + Local Search.

### 4.1 Root 5,000 experiment headline

- 240/240 configurations.
- 202,055 simulations.
- Wall time about 2m37s.
- 121 tests pass.
- SSA-only highest mean Net Advantage: **O1 = 47.878**.
- SSA+Swap highest mean Net Advantage: **O6 = 49.412**.
- Most strict wins SSA-only: **O1, 6/20**.
- Most strict wins SSA+Swap: **O1 and O4, 5/20 each**.
- No single SSA variant wins every B × K combination.

Use this experiment when the question is about **O1–O6** or **Swap refinement**. Use the 12-method benchmark when the question is about **CELF, CRIEG, NCGE, Signed-GA, FIBM, etc.**

---

## 5. 12-method benchmark design

### 5.1 Methods

| Method ID | Display name | Group | Selection objective / mechanism |
|---|---|---|---|
| `SSA_O1` | SSA O1 (Original) | SSA variant | `Spread_A + 1.5 * Blocked_B` |
| `SSA_O4` | SSA O4 (Combined) | SSA variant | `Spread_A + 2.0 * Blocked_B` |
| `SSA_O6` | SSA O6 (Budget-aware) | SSA variant | `alpha(K)*Spread_A + beta(K)*Blocked_B` |
| `SSA_Notebook` | SSA-Notebook | Notebook baseline | Same SSA as O1; notebook seed schedule |
| `Random` | Random | Notebook baseline | Uniform eligible-node sample |
| `TopDegree` | Top Degree | Notebook baseline | Influence out-degree ranking |
| `SignedPageRank` | Signed PageRank | Notebook baseline | Two-layer signed PageRank |
| `CRIEG` | CRIEG (2021) | Notebook baseline | Competitive reverse-reachable sketches |
| `NCGE` | NCGE (2025) | Notebook baseline | Centrality pool + lazy greedy with O1 fitness |
| `SignedGA` | Signed-GA (2025) | Notebook baseline | Signed Voter equilibrium + projected gradient |
| `CELF` | CELF (O1 fitness) | Notebook baseline | Lazy greedy on the weighted candidate pool with O1 fitness |
| `FIBMM` | **FIBM** | Fairness baseline | Paper-based maximin-fairness CRP greedy |

**Excluded:** `CELF + sampled Local Search` — it is one-for-one swap-based post-processing, and the project excludes Swap refinement. The base CELF is included. No SSA+Swap, Local Swap, or swap-based post-processing is run in this benchmark.

### 5.2 Important implementation facts

- **SSA-Notebook is not a distinct algorithm.** `run_ssa_notebook(...)` calls `run_discrete_ssa(..., "O1", ...)` with the same population, iterations, mutation rate, and candidate pool. The only difference is the random seed:
  - SSA O1: `master_seed + 500000 + scenario_number*1000 + K*10 + variant_index`
  - SSA-Notebook: `master_seed + 30000 + scenario_number`
  Treat SSA-Notebook as a **second independent SSA O1 run / sensitivity check**, not as a new method.
- **CELF and NCGE use O1 fitness**, not Net Advantage:
  - `O1 = Spread_A + 1.5 * Blocked_B`
  - External ranking is still `Net Advantage = Spread_A - Spread_B`.
- **SSA O1/O4/O6 use their own objective during selection**, then are evaluated by Net Advantage.
- **FIBM uses its own fairness objective**, not O1/O4/O6 and not Net Advantage.
- **CRIEG, NCGE, and Signed-GA are literature adaptations** to this signed competitive setting. Their original papers use different diffusion models and assumptions.

### 5.3 Candidate pool and SSA

SSA candidate pool:

- Target size `10 * K`.
- Polarity-aware score:
  ```text
  score(u) = sum(p(positive outgoing edges)) - sum(p(negative outgoing edges))
  ```
- Ranking key: `(-net_score, -positive_score, negative_score, node_id)`.
- Composition: `n_random = int(round(0.30 * pool_size))`, `n_ranked = pool_size - n_random`; random part sampled from the ranked remainder.
- Opponent seeds excluded everywhere.

SSA parameters:

- Population 12.
- Iterations 10.
- Mutation rate 0.20.
- `c1 = 2 * exp(-((4t/T)^2))`.
- Exploitation probability `1 - c1/2`.
- Leader/follower updates; best-so-far food preserved; worst individual replaced by food.
- Repair to exactly K unique candidate-pool nodes.
- 132 objective evaluations per run: 12 initial + 10 × 12.

### 5.4 FIBM adaptation

Paper: Xiao, Liu, Yang, Li, “Fairness-Aware Influence Blocking Maximization in Social Networks,” IEEE Transactions on Computational Social Systems, 12(6), 2025.

Paper objective:

```text
sigma(S) = (1 - h) * Pi(S) + h * tau(S)
```

- `Pi(S)` = overall saving ratio.
- `tau(S)` = minimum community saving ratio (maximin fairness).
- `h` controls utility/fairness trade-off.

Paper defaults used:

- `h = 0.9`
- `epsilon = 0.1`
- `l = 1`
- Louvain communities
- small-community merge threshold `0.005 * n`

Implementation:

- Selection uses a **positive-edge CIC surrogate**: only positive influence arcs are traversable; negative edges are excluded because they flip the rumor.
- Louvain on the undirected unweighted projection of **all** influence edges.
- CRP sampling with the paper’s martingale sample-size formula.
- Two-stage greedy:
  1. `k - floor(k*h)` nodes by overall marginal coverage.
  2. Remaining nodes by repeatedly choosing the worst-off community and the node covering most of its remaining CRP sets.
- CRP budget is capped for feasibility:
  - 5,000-node benchmark: **600,000** sets per configuration.
  - 1,000-node benchmark: **100,000** sets per configuration.
- Final evaluation always uses the full signed Maki–Thompson model.
- FIBM is **not** rewritten into O1/O4/O6.

### 5.5 Budgets

| Method | Selection budget | MT diffusion during selection | Final evaluation |
|---|---|---|---|
| SSA O1/O4/O6 | 132 objective evaluations | 15 MC seeds per evaluation | 100 common seeds |
| SSA-Notebook | Same SSA budget | 15 MC seeds per evaluation | 100 common seeds |
| Random | One uniform sample | None | 100 common seeds |
| Top Degree | One deterministic sort | None | 100 common seeds |
| Signed PageRank | One PageRank preprocessing + sort | None | 100 common seeds |
| CRIEG | 2,000 reverse-reachability sketches | None | 100 common seeds |
| NCGE | Centrality preprocessing + lazy greedy | 15 MC seeds per greedy evaluation | 100 common seeds |
| Signed-GA | 30 projected gradient steps | None | 100 common seeds |
| CELF | Lazy greedy on 10*K pool | 15 MC seeds per greedy evaluation | 100 common seeds |
| FIBM | Paper CRP sampling (capped) + two-stage greedy | None (CIC surrogate) | 100 common seeds |

The comparison is fair in **evaluation conditions** (same network, opponents, K values, final seeds, metric) but **not in computational budget**. Runtime and simulation counts are reported explicitly.

### 5.6 Reuse and checkpointing

- Each cell is checkpointed atomically.
- Existing valid cells are skipped on rerun.
- The ledger records `completed_valid`, `reused`, `invalidated`, `failed`, `incomplete`, and the reason.
- 5,000 benchmark: 60 SSA O1/O4/O6 rows reused from the root experiment; 180 new rows.
- 1,000 benchmark: 240 new rows; no prior 1,000 reverse-influence results existed.
- A deterministic reproduction check reruns B1/K=10/O1 and compares the seed set with the stored one.

---

## 6. 5,000-node 12-method benchmark

### 6.1 Network

| Property | Value |
|---|---|
| Nodes | 5,000 (ids 0–4999) |
| Trust arcs | 37,500 |
| Influence arcs | 37,500 (exact reverse) |
| Positive / negative trust arcs | 18,706 / 18,794 |
| Probability classes 0.1 / 0.5 / 0.9 | 11,151 / 14,911 / 11,438 |
| Reciprocal trust pairs | 0 |
| Same-direction influence edges | 0 |
| File SHA-256 | `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e` |
| Trust semantic SHA-256 | `576d00314530fc82c86d5a433e13a2ea64dbe135723350d936df21b082e2b20f` |
| Influence semantic SHA-256 | `b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9` |

Network path: `IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/network/synthetic_network_5000_37500.csv`.

### 6.2 Configuration

- `config/benchmark_config.json`
- Scenarios B1–B4.
- K = 10, 20, 30, 40, 50.
- 12 methods.
- 15 optimization MC runs; 100 final MC runs.
- FIBM CRP cap 600,000.
- SSA population 12, iterations 10, mutation 0.20.
- Candidate pool 10*K, 70/30.
- No Swap.

### 6.3 Results

All 240 cells completed. 60 SSA rows reused; 180 new. Validation PASS. Wall span about 1.60 hours (includes sequential Signed-GA and FIBM work; not pure CPU time).

| Method | Mean NetAdv | Median | Strict wins / 20 | Mean Spread_A | Mean Spread_B | Mean Blocked_B | Mean selection runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SSA O1 | 47.878 | 50.120 | 6 | 995.559 | 947.681 | 33.379 | 0.675 |
| SSA O6 | 47.261 | 51.050 | 2 | 995.419 | 948.159 | 32.901 | 0.678 |
| CELF | 46.115 | 45.015 | 6 | 994.577 | 948.462 | 32.599 | 4.346 |
| SSA-Notebook | 45.570 | 46.245 | 3 | 993.274 | 947.704 | 33.356 | 0.731 |
| SSA O4 | 43.241 | 45.720 | 3 | 991.445 | 948.203 | 32.857 | 0.675 |
| Signed-GA | 28.207 | 24.845 | 0 | 998.913 | 970.706 | 10.430 | 237.668 |
| FIBM | 18.718 | 17.150 | 0 | 965.715 | 946.998 | 34.063 | 46.103 |
| CRIEG | 14.536 | 10.685 | 0 | 958.606 | 944.069 | 36.991 | 0.679 |
| Signed PageRank | 3.934 | 6.175 | 0 | 961.004 | 957.069 | 24.036 | 0.008 |
| NCGE | 2.685 | 1.205 | 0 | 961.195 | 958.509 | 22.551 | 3.668 |
| Random | -0.150 | -0.410 | 0 | 971.389 | 971.540 | 9.887 | 0.005 |
| Top Degree | -1.746 | -1.775 | 0 | 941.311 | 943.057 | 38.003 | 0.009 |

### 6.4 Winners by scenario

- **B1 — Random opponent:** SSA O1 (2), CELF (2), SSA O4 (1).
- **B2 — Degree-based opponent:** SSA O1 (2), SSA-Notebook (2), CELF (1).
- **B3 — Positive PageRank opponent:** SSA O6 (2), SSA O1 (1), CELF (1), SSA O4 (1).
- **B4 — Community opponent:** CELF (2), SSA O1 (1), SSA-Notebook (1), SSA O4 (1).

### 6.5 Winners by K

- K=10: CELF (2), SSA O4 (1), SSA O1 (1).
- K=20: SSA O1 (3), CELF (1).
- K=30: SSA-Notebook (2), CELF (1), SSA O6 (1).
- K=40: SSA O4 (2), CELF (1), SSA O1 (1).
- K=50: SSA O1 (1), SSA-Notebook (1), SSA O6 (1), CELF (1).

### 6.6 Interpretation

- SSA O1 has the highest mean, but CELF ties it for the most strict wins.
- CELF is the strongest non-SSA baseline and the main real competitor.
- SSA-Notebook is a second SSA O1 run, not a new algorithm.
- Signed-GA has a higher mean than CRIEG but wins zero cells; its average is driven by a few high values, not consistent wins.
- CRIEG is weak on this 5,000-node graph (0 wins).
- FIBM is a fairness baseline, not a Net Advantage contender.
- Only 4/20 winner-vs-runner-up paired intervals exclude zero; most per-cell rankings are descriptive.

Full 20-row winner table: `FINAL_REPORT.md` §5.2.

---

## 7. 1,000-node 12-method benchmark

### 7.1 Why this run exists

The user requested a version of the benchmark on the synthetic **1,000-node / 7,500-edge** graph. The old 1,000-node results were produced with the **same-direction** influence model and are invalid for the corrected reverse-influence model. They must not be reused. A separate 1,000-node run was created.

### 7.2 Network

| Property | Value |
|---|---|
| Nodes | 1,000 (ids 0–999) |
| Trust arcs | 7,500 |
| Influence arcs | 7,500 (exact reverse) |
| Underlying unordered pairs | 7,498 |
| Reciprocal trust pairs | 2 |
| Reciprocal arcs | 4 |
| Same-direction edges reported by direction check | 4 (expected; both directions of reciprocal pairs) |
| Positive / negative trust arcs | 3,720 / 3,780 |
| Probability classes 0.1 / 0.5 / 0.9 | 2,278 / 2,950 / 2,272 |
| Out-degree min/mean/max | 0 / 7.5 / 77 |
| In-degree min/mean/max | 1 / 7.5 / 67 |
| Weak components | 1 |
| Strong components | 3 (largest 998) |
| Self-loops | 0 |
| Duplicate directed arcs | 0 |
| File SHA-256 | `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5` |
| Trust semantic SHA-256 | `461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25` |
| Influence semantic SHA-256 | `0c0f4e1f1d8a3abc0e2765941e94cfe233efbe32246c5b56c960f0c1ee7f7197` |

Network path: `IM_Synthetic_Experiment/data/synthetic_network_1000_7500.csv`.

Metadata: `IM_Synthetic_Experiment/data/network_metadata_v2.json`.

Generation: seed 42, Barabási–Albert-style backbone (`m=7`, initial clique 8), random directed fill, sign positive with probability 0.5, trust magnitude uniform 1–10. The generator is pinned; do not regenerate the file unless the user explicitly asks.

**Small metadata wording note:** `network_metadata_v2.json`’s prose says “7500 underlying unordered pairs,” but the verified count is **7,498** because the file contains 2 reciprocal pairs. The table above uses the verified count.

### 7.3 Configuration

- `config/benchmark_1000_synthetic_config.json`
- Scenarios B1–B4.
- K = 10, 20, 30, 40, 50.
- 12 methods.
- 15 optimization MC runs; 100 final MC runs.
- FIBM CRP cap 100,000.
- SSA population 12, iterations 10, mutation 0.20.
- Candidate pool 10*K, 70/30.
- No Swap.
- `ssa.reuse_prior_run = "__no_prior_1000_run__"` — sentinel to prevent reuse of the 5,000-node SSA rows.

### 7.4 Results

All 240 cells completed. All 240 are new computations. Manifest: 240 expected, 240 completed, 0 missing, 0 failed. Full run took about **188 seconds**.

| Method | Mean NetAdv | Median | Strict wins / 20 | Mean Spread_A | Mean Spread_B | Mean Blocked_B | Mean selection runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SSA-Notebook | 24.483 | 23.410 | 2 | 215.301 | 190.818 | 18.060 | 0.195 |
| SSA O6 | 24.218 | 25.480 | 3 | 215.352 | 191.134 | 17.744 | 0.209 |
| SSA O1 | 24.212 | 24.390 | 4 | 215.298 | 191.086 | 17.793 | 0.206 |
| CELF | 23.638 | 23.720 | 4 | 216.578 | 192.940 | 15.938 | 0.609 |
| SSA O4 | 23.416 | 20.935 | 4 | 215.488 | 192.072 | 16.807 | 0.204 |
| CRIEG | 18.870 | 19.980 | 3 | 206.703 | 187.833 | 21.046 | 0.159 |
| FIBM | 18.363 | 15.610 | 0 | 207.767 | 189.404 | 19.475 | 3.509 |
| Signed-GA | 12.364 | 12.030 | 0 | 217.501 | 205.137 | 5.105 | 2.660 |
| NCGE | 4.664 | 3.840 | 0 | 202.934 | 198.270 | 10.609 | 0.696 |
| Top Degree | -0.709 | 0.185 | 0 | 182.864 | 183.573 | 25.305 | 0.002 |
| Random | -1.582 | -0.825 | 0 | 201.919 | 203.501 | 5.539 | 0.001 |
| Signed PageRank | -6.129 | -6.310 | 0 | 189.468 | 195.596 | 13.282 | 0.001 |

### 7.5 Winners by scenario and K

| Scenario | K | Winner | Winner NetAdv | Runner-up | Margin | Paired 95% interval excludes zero? |
|---|---:|---|---:|---|---:|---|
| B1 | 10 | SSA O4 | 20.57 | SSA O6 | 1.02 | No |
| B1 | 20 | SSA-Notebook | 29.48 | CELF | 1.25 | No |
| B1 | 30 | SSA O4 | 30.34 | SSA O6 | 3.97 | No |
| B1 | 40 | CELF | 37.21 | SSA O1 | 8.59 | **Yes** |
| B1 | 50 | SSA-Notebook | 30.55 | SSA O6 | 8.00 | **Yes** |
| B2 | 10 | CELF | 17.38 | SSA O1 | 0.03 | No |
| B2 | 20 | SSA O6 | 27.94 | CELF | 4.13 | No |
| B2 | 30 | SSA O6 | 29.29 | CRIEG | 2.49 | No |
| B2 | 40 | CRIEG | 42.37 | SSA O4 | 7.41 | **Yes** |
| B2 | 50 | CRIEG | 38.70 | FIBM | 3.69 | No |
| B3 | 10 | CELF | 17.65 | SSA-Notebook | 2.24 | No |
| B3 | 20 | SSA O4 | 27.01 | SSA O1 | 1.11 | No |
| B3 | 30 | SSA O1 | 40.01 | SSA-Notebook | 9.93 | **Yes** |
| B3 | 40 | SSA O6 | 32.34 | SSA-Notebook | 3.27 | No |
| B3 | 50 | CRIEG | 32.02 | FIBM | 3.32 | No |
| B4 | 10 | SSA O1 | 22.57 | SSA-Notebook | 0.31 | No |
| B4 | 20 | SSA O1 | 32.00 | SSA O4 | 0.59 | No |
| B4 | 30 | SSA O1 | 35.26 | CELF | 2.97 | No |
| B4 | 40 | SSA O4 | 34.59 | SSA-Notebook | 1.50 | No |
| B4 | 50 | CELF | 37.66 | SSA-Notebook | 3.06 | No |

Win counts:

- SSA O1: 4
- SSA O4: 4
- CELF: 4
- SSA O6: 3
- CRIEG: 3
- SSA-Notebook: 2
- All others: 0

### 7.6 Scenario and K patterns

Mean Net Advantage by scenario:

| Method | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| SSA O1 | 19.360 | 23.190 | 24.232 | 30.068 |
| SSA O4 | 21.930 | 21.358 | 22.054 | 28.322 |
| SSA O6 | 22.680 | 24.642 | 23.808 | 25.744 |
| SSA-Notebook | 23.776 | 23.572 | 23.030 | 27.554 |
| CELF | 18.574 | 24.258 | 24.816 | 26.904 |
| CRIEG | 14.642 | 24.330 | 20.862 | 15.646 |
| FIBM | 7.386 | 23.516 | 22.374 | 20.176 |
| Signed-GA | 3.852 | 16.178 | 14.178 | 15.248 |
| NCGE | 2.968 | 3.946 | 5.318 | 6.424 |
| Top Degree | -4.468 | 0.328 | 0.050 | 1.254 |
| Random | -6.758 | -1.970 | -0.476 | 2.876 |
| Signed PageRank | -11.324 | -4.768 | -4.094 | -4.328 |

Mean Net Advantage by K:

| Method | K=10 | K=20 | K=30 | K=40 | K=50 |
|---|---:|---:|---:|---:|---:|
| SSA O1 | 14.442 | 25.170 | 27.885 | 29.478 | 24.088 |
| SSA O4 | 16.382 | 24.643 | 25.842 | 26.728 | 23.485 |
| SSA O6 | 16.370 | 25.218 | 27.685 | 27.430 | 24.390 |
| SSA-Notebook | 17.458 | 23.640 | 24.630 | 27.835 | 28.852 |
| CELF | 14.742 | 23.428 | 24.408 | 29.092 | 26.520 |
| CRIEG | 5.440 | 12.900 | 18.592 | 27.348 | 30.070 |
| FIBM | 9.565 | 13.262 | 20.922 | 21.865 | 26.200 |
| Signed-GA | 1.455 | 8.910 | 14.765 | 25.472 | 11.218 |
| NCGE | 2.785 | 3.145 | 1.285 | 8.598 | 7.508 |
| Top Degree | 0.352 | -1.233 | -1.702 | -0.283 | -0.680 |
| Random | 2.120 | -4.102 | -1.810 | -1.575 | -2.542 |
| Signed PageRank | -3.482 | -7.482 | -7.075 | -7.002 | -5.600 |

### 7.7 SSA versus baselines

Win/tie/loss counts over the 20 cells:

| SSA variant | vs Random | vs Top Degree | vs Signed PageRank | vs CRIEG | vs NCGE | vs Signed-GA | vs CELF | vs FIBM | vs SSA-Notebook |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SSA O1 | 20/0/0 | 20/0/0 | 20/0/0 | 14/0/6 | 20/0/0 | 18/0/2 | 11/0/9 | 13/0/7 | 10/0/10 |
| SSA O4 | 20/0/0 | 20/0/0 | 20/0/0 | 13/0/7 | 20/0/0 | 17/0/3 | 8/0/12 | 13/0/7 | 9/0/11 |
| SSA O6 | 20/0/0 | 20/0/0 | 20/0/0 | 15/0/5 | 20/0/0 | 17/0/3 | 11/0/9 | 14/0/6 | 9/0/11 |

Reading: wins/ties/losses for the SSA variant versus the baseline. These are descriptive paired comparisons on the same 100 evaluation seeds; they are not independent optimizer replications.

### 7.8 FIBM

- Mean Net Advantage 18.363; mean Blocked_B 19.475.
- Mean internal `Pi` about 0.708; mean `tau` about 0.690.
- Wins 0 cells on Net Advantage.
- FIBM is competitive in several B2/B3/B4 cells but is not a Net Advantage optimizer.
- Its higher Blocked_B relative to SSA (19.475 vs about 17.8) is offset by lower Spread_A.
- Internal fairness values are surrogate estimates, not signed-MT measurements.

### 7.9 Interpretation

- The top group is tight: SSA-Notebook 24.483, SSA O6 24.218, SSA O1 24.212, CELF 23.638, SSA O4 23.416.
- SSA-Notebook’s top mean is not evidence of a better algorithm; it is the same SSA O1 with a different seed.
- SSA O1, SSA O4, and CELF each win 4 cells; no method dominates.
- CELF is the main real competitor.
- CRIEG is a secondary threat on this smaller graph, with 3 wins (B2/K=40, B2/K=50, B3/K=50).
- Signed-GA is not competitive (0 wins, mean 12.364).
- FIBM is a fairness baseline, not a Net Advantage contender.
- Only 4/20 winner-vs-runner-up paired intervals exclude zero; most per-cell rankings are descriptive.

---

## 8. Notebooks

### 8.1 1,000-node notebook

**Path:**

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/competitive_im_benchmark_1000_with_fibm.ipynb
```

**What it does:**

- Loads `config/benchmark_1000_synthetic_config.json`.
- Loads and verifies `data/synthetic_network_1000_7500.csv` (file hash, trust semantic hash, influence semantic hash, node/arc counts, direction check).
- Documents the signed network, diffusion model, B1–B4 opponents, SSA O1/O4/O6, all notebook baselines, and FIBM.
- Audits the saved 1,000-node results:
  - 240 expected cells.
  - 240 present.
  - 0 missing/extra.
  - 0 signature mismatches.
  - 0 bad MC files.
  - 0 opponent-inconsistent cells.
  - Manifest 240/240.
- Loads `benchmark_1000_synthetic/raw_results/results.csv` and per-cell MC files.
- Builds winner table, algorithm summary, SSA-vs-baseline table, FIBM fairness table, runtime table.
- Produces four Net Advantage charts, one per B scenario.
- Saves figures to `figures/notebook_1000/`.

**Execution mode:**

- Default `EXECUTION_MODE = "load_existing"`.
- `ALLOW_MISSING_RUNS = False`.
- `ALLOW_FORCE_RERUN = False`.
- `SIMULATION_LAUNCHED = False`.
- No SSA, FIBM, baseline, or diffusion simulation is launched when the notebook is opened or run in default mode.
- `run_missing` requires `ALLOW_MISSING_RUNS=True`.
- `force_rerun` requires `ALLOW_FORCE_RERUN=True` and explicit method/scenario/K selections; writes to `notebook_reruns_1000/<timestamp>` and never overwrites the saved run.

**Size and validation:**

- 47 cells: 21 code, 26 markdown.
- Valid IPYNB.
- Executed end-to-end with zero errors.
- 0 unresolved TODO markers.
- Validation file: `validation/notebook_1000_validation.txt`.
- Result-integrity file: `validation/benchmark_1000_validation.json` (`ok: true`).
- There is no separate 1,000-node `FINAL_REPORT.md`; the notebook’s findings/conclusion and `benchmark_1000_synthetic/README.md` are the summary.

**Charts:**

- `figures/notebook_1000/Net_Advantage_B1.png`
- `figures/notebook_1000/Net_Advantage_B2.png`
- `figures/notebook_1000/Net_Advantage_B3.png`
- `figures/notebook_1000/Net_Advantage_B4.png`

The charts are embedded as executed notebook outputs. Opening the notebook in Jupyter shows them without rerunning. If you restart and run all, it will re-read the saved results and regenerate the charts; it will not launch simulations in `load_existing` mode.

**Chart style:**

- Two panels per scenario.
- Left panel: SSA O1/O4/O6, SSA-Notebook, FIBM.
- Right panel: Random, Top Degree, Signed PageRank, CRIEG, NCGE, Signed-GA, CELF.
- Shared y-axis within a scenario.
- 95% Monte Carlo error bars over the 100 common final seeds.
- Bottom legend, distinct colors/markers/line styles.
- 300 dpi PNG.

**Important network-check detail:**

- The 1,000-node graph has 2 reciprocal trust pairs, so the direction check reports 4 same-direction edges.
- The notebook asserts `same_direction_edges == 2 * reciprocal_trust_pairs`, not `== 0`.
- The full reversal check still requires 0 missing reversed edges and 0 extra influence edges.

### 8.2 5,000-node notebook

**Path:**

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/competitive_im_benchmark_with_fibm.ipynb
```

**What it does:**

- Same overall structure as the 1,000-node notebook.
- Uses `config/benchmark_config.json`.
- Loads the 5,000-node network from `experiments/exp_20261124_5000_reversed/network/synthetic_network_5000_37500.csv`.
- Reads `raw_results/results.csv`.
- Produces charts in `figures/notebook/`.
- Default `load_existing`; no simulations.

**Size and validation:**

- 50 cells: 23 code, 27 markdown.
- Valid IPYNB.
- Executed with zero errors.
- Validation file: `validation/notebook_validation.txt`.
- 240 rows, 12 methods, 0 missing, 0 signature mismatches, 0 bad MC files.

**Charts:**

- `figures/notebook/Net_Advantage_B1.png`
- `figures/notebook/Net_Advantage_B2.png`
- `figures/notebook/Net_Advantage_B3.png`
- `figures/notebook/Net_Advantage_B4.png`

### 8.3 Do not mix the notebooks

| Notebook | Config | Results | Figures | Network |
|---|---|---|---|---|
| 1,000 | `benchmark_1000_synthetic_config.json` | `benchmark_1000_synthetic/raw_results/` | `figures/notebook_1000/` | 1,000 / 7,500 |
| 5,000 | `benchmark_config.json` | `raw_results/` | `figures/notebook/` | 5,000 / 37,500 |

The 1,000 notebook explicitly states that the 5,000 results are not reused. The 5,000 notebook uses the 5,000 results only.

---

## 9. Validation and reproducibility

### 9.1 1,000-node validation

- `validation/notebook_1000_validation.txt`:
  - 47 cells, 21 code, 26 markdown.
  - Valid IPYNB.
  - 0 execution errors.
  - 0 unresolved TODO markers.
  - Final summary: 240 rows, 12 methods, 240 expected/present, 0 missing, 0 signature mismatches, 0 bad MC files, 0 opponent-inconsistent cells, `load_existing`, `SIMULATION_LAUNCHED=False`.
- `validation/benchmark_1000_validation.json`:
  - `ok: true`.
  - 240 rows, 12 methods.
  - File hash and influence hash match.
  - Direction check: 7,500 trust edges, 7,500 influence edges, 0 missing, 0 extra, 4 same-direction edges (2 reciprocal pairs), `all_trust_edges_reversed: true`.
  - Manifest: 240 expected, 240 completed, 0 missing, 0 failed.
  - Ledger: 240 `completed_valid`.

### 9.2 5,000-node validation

- `aggregated/validation_summary.json`:
  - `ok: true`, no issues.
  - 240 cells, 12 methods.
  - 60 reused SSA rows, 180 new.
  - Wall span about 1.60 hours.
  - Network file hash and influence hash match.
  - 15 optimization MC, 100 final MC.
- `raw_results/manifest.json`:
  - 240 expected, 240 completed, 0 missing, 0 failed.
  - Ledger: 180 `completed_valid`, 60 `reused`.
- `validation/tests.txt`: 32 tests pass (benchmark experiment).
- Root 5,000 experiment: 121 tests pass (`exp_20261124_5000_reversed/validation/tests_5000.txt`).

### 9.3 Test commands

From `IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness`:

```bash
# Benchmark experiment tests (32)
python -m unittest discover -s tests -p 'test_*.py' -v

# Root project tests (121)
cd ../..
python -m unittest discover -s tests -p 'test_*.py' -v
```

The benchmark tests include `tests/test_notebook_1000.py`, which checks:
- notebook exists and is valid;
- 1,000-node config and network references;
- required sections;
- no TODO/placeholder text;
- FIBM documentation;
- safe default execution gates;
- no Swap execution;
- all code cells compile.

### 9.4 Benchmark commands

From `IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness`:

```bash
# Plan only, no simulations
python run_benchmark.py --mode plan

# Validate network, opponent orders, prior SSA reuse, deterministic SSA check
python run_benchmark.py --mode verify

# Reduced-budget smoke test on B1/K=10
python run_benchmark.py --mode smoke

# Full 5,000 12-method run (requires explicit confirmation)
python run_benchmark.py --mode full --confirm-full

# Read-only aggregation, validation, figures, report (5,000)
python run_benchmark.py --mode analyze

# Separate 1,000-node run (skips valid 1,000 cells; never reuses 5,000 cells)
python run_1000_synthetic.py
```

Targeted rerun example:

```bash
python run_benchmark.py --mode full --confirm-full \
  --only-methods FIBMM --only-scenarios B1 --only-k 10
```

1,000 targeted rerun example:

```bash
python run_1000_synthetic.py --only-methods Random --only-scenarios B1 --only-k 10
```

The 1,000 runner:
- Uses `Synthetic1000Runner`, which overrides `_reuse_prior_cell` to return `False` so it never reuses 5,000 SSA results.
- Still skips valid existing 1,000 cells.
- Calls `_refresh_manifest` after the run so the manifest always reports the full 240-cell matrix, even after a targeted run.
- `--force` recomputes selected cells; it does not delete existing results.

### 9.5 Root 5,000 commands

From `IM_Synthetic_Experiment`:

```bash
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode plan
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode smoke
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode full --confirm-full --run-id run_20261124_5000_reversed
python experiments/exp_20261124_5000_reversed/analyze_5000.py
```

### 9.6 C++ engine

The C++ extension is already built at:

```text
IM_Synthetic_Experiment/cpp_engine/simulation_engine.cpython-314-x86_64-linux-gnu.so
```

If it needs rebuilding:

```bash
cd IM_Synthetic_Experiment/cpp_engine
python setup.py build_ext --inplace
cd ..
```

The engine uses OpenMP; the environment has 8 CPUs.

### 9.7 Environment

- Python 3.14.10
- NumPy 2.4.6
- pandas 2.3.3
- networkx 3.7
- matplotlib 3.10.8
- SciPy 1.16.2
- pybind11 3.1.0
- g++ 16.2.1
- OpenMP enabled
- 8 CPUs
- Linux

No packages were installed by the agent. No network access was used.

### 9.8 Known practical issues

1. **`BenchmarkRunner` requires an absolute `output_root`.**
   - A relative path can cause:
     ```text
     ValueError: ... is not in the subpath of ...
     ```
     in `_write_manifest`.
   - Use `Path(...).resolve()`.
   - The 1,000 runner uses `RESULTS_ROOT` under the resolved experiment directory.

2. **Partial runs can overwrite the manifest with a subset.**
   - `run_1000_synthetic.py` calls `_refresh_manifest` after each run to restore the full 240-cell expected/completed/missing fields.
   - If you write a new runner, preserve this behavior.

3. **The 1,000 graph has reciprocal pairs.**
   - 4 same-direction edges are expected.
   - Do not assert 0 same-direction edges for the 1,000 graph.

4. **`run_benchmark.py` is hardwired to the 5,000 config.**
   - Use `run_1000_synthetic.py` for the 1,000 run.
   - `analyze_benchmark.py` is also 5,000-oriented; the 1,000 notebook uses `notebook_analysis.py`.

5. **The root `CONFIG.md` and `BENCHMARK.md` are legacy.**
   - They describe the old 1,000-node same-direction setup.
   - Do not treat them as the current 1,000 reverse-influence config.

6. **There is a git repo at `IM_Synthetic_Experiment/.git` with a single commit.**
   - The workspace root itself is not a git repo.
   - Do not use destructive git commands to “clean” the project.

---

## 10. Scientific interpretation

### 10.1 What the results establish

- On both synthetic networks, **no single method dominates every opponent and K**.
- The ranking depends on the opponent scenario and seed budget.
- **CELF is the main real competitor** on both networks.
- **SSA O1** is the best average method on the 5,000-node benchmark; **CELF** ties it for the most strict wins.
- On the 1,000-node benchmark, the top group is tight; SSA O1/O4/CELF each win 4 cells.
- **SSA-Notebook is not a new algorithm**; it is a second SSA O1 run with a different seed.
- **CRIEG** is a secondary threat on the 1,000-node graph, not on the 5,000-node graph.
- **Signed-GA** is not competitive in win counts on either network, despite a higher mean than CRIEG on 5,000.
- **FIBM** is a fairness baseline, not a Net Advantage optimizer.

### 10.2 What the results do not establish

- No claim of universal superiority.
- No claim of statistical significance for most per-cell rankings; only 4/20 winner-vs-runner-up intervals exclude zero on each network.
- No estimate of optimizer run-to-run variability; there is one optimizer run per method/cell.
- No external validity on real signed networks; both networks are synthetic.
- No claim that FIBM’s internal fairness values equal signed-MT fairness.
- No claim that CRIEG/NCGE/Signed-GA results match their original papers exactly; they are adaptations.
- No claim that 1,000-node and 5,000-node results are directly comparable; they are different graphs.

### 10.3 Why CELF matters

CELF is a lazy-greedy method on the weighted candidate pool with **O1 fitness**:

```text
O1 = Spread_A + 1.5 * Blocked_B
```

It does not directly optimize Net Advantage. Even so, it is the closest competitor to SSA on both networks. On 5,000 it has 6 strict wins and a mean of 46.115; on 1,000 it has 4 strict wins and a mean of 23.638. Its competitive mean is not free: it uses many more objective evaluations and MT simulations than SSA.

### 10.4 Why SSA-Notebook is not a separate method

`run_ssa_notebook` is a thin wrapper:

```python
def run_ssa_notebook(...):
    return run_discrete_ssa(..., "O1", ...)
```

It uses the same:
- candidate pool,
- population 12,
- 10 iterations,
- mutation 0.20,
- O1 fitness,
- 15 optimization MC seeds.

The only difference is the random seed:
- SSA O1: `master_seed + 500000 + scenario_number*1000 + K*10 + variant_index`
- SSA-Notebook: `master_seed + 30000 + scenario_number`

So SSA-Notebook is useful as a **descriptive sensitivity check / replicate**, not as a distinct algorithm. Do not describe it as a new method in a paper without this caveat.

### 10.5 Why FIBM is not a Net Advantage optimizer

FIBM maximizes:

```text
sigma(S) = (1 - h) * Pi(S) + h * tau(S)
```

on a positive-edge CIC surrogate. The benchmark evaluates it on the signed Maki–Thompson model with Net Advantage. A high internal fairness value does not imply a high external Net Advantage. This is a model-adaptation limitation, not a hidden advantage.

---

## 11. Guardrails — what not to do

1. **Do not mix the 5,000 and 1,000 results.**
   - 5,000: `raw_results/`.
   - 1,000: `benchmark_1000_synthetic/raw_results/`.
   - The 1,000 notebook must never load `raw_results/results.csv`.

2. **Do not reuse the old 1,000 same-direction results.**
   - They are invalid for the reverse-influence model.
   - They are in `exp_20261124_5000_reversed/to_be_discarded/`.
   - The new 1,000 run has 240 new rows and uses reverse influence.

3. **Do not regenerate the 1,000 network.**
   - `data/synthetic_network_1000_7500.csv` is the authoritative file.
   - Its hash is checked by the notebook.
   - Regeneration is deterministic, but unnecessary and risky.

4. **Do not use Swap refinement in the 12-method benchmark.**
   - No SSA+Swap, Local Swap, or swap-based post-processing.
   - CELF + Local Search is excluded.
   - Base CELF is included.
   - The root 5,000 experiment has Swap modes; do not conflate them with the benchmark.

5. **Do not treat the 100 MC runs as optimizer replicates.**
   - They are evaluation replicates for a fixed seed set.
   - They do not estimate run-to-run optimizer variability.

6. **Do not pool B1–B4 into a single significance test.**
   - They are four different opponent strategies, not independent samples.

7. **Do not call FIBM a Net Advantage optimizer.**
   - It optimizes a fairness objective on a CIC surrogate.
   - External evaluation is signed MT.

8. **Do not describe SSA-Notebook as a new algorithm.**
   - It is SSA O1 with a different random seed.

9. **Do not claim CRIEG and Signed-GA are the only real competition.**
   - CELF is the main real competitor on both networks.
   - CRIEG is a secondary threat only on the 1,000-node graph.
   - Signed-GA is not competitive in wins on either network.

10. **Do not launch a full experiment without explicit approval.**
    - The existing runs are complete.
    - Use `plan`, `verify`, `analyze`, or the notebooks in `load_existing` mode for inspection.
    - If a rerun is needed, ask first and use targeted/force options.

11. **Do not delete or overwrite existing notebooks, results, or reference files.**
    - The project preserves old notebooks and old results for audit.
    - The 5,000 notebook and results must remain intact.

12. **Do not use a relative `output_root` with `BenchmarkRunner`.**
    - Resolve it to an absolute path.

13. **Do not edit the network file or its metadata hashes.**
    - The notebook and runner refuse to run on a hash mismatch.

14. **Do not assume `CONFIG.md` / `BENCHMARK.md` describe the current 1,000 run.**
    - They are legacy same-direction documents.

---

## 12. Common tasks

### 12.1 “Open the 1,000-node notebook and show me the charts”

Open:

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/competitive_im_benchmark_1000_with_fibm.ipynb
```

The saved outputs already contain the four charts. If Jupyter shows no outputs, run the notebook in `load_existing` mode; it will read the saved results and regenerate the figures without simulations.

### 12.2 “Run the 1,000-node notebook safely”

From the benchmark experiment directory:

```bash
jupyter notebook competitive_im_benchmark_1000_with_fibm.ipynb
```

Keep:

```python
EXECUTION_MODE = "load_existing"
ALLOW_MISSING_RUNS = False
ALLOW_FORCE_RERUN = False
```

Then “Run All”. No simulations should launch.

### 12.3 “Verify the 1,000-node results”

```bash
cd IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness
python -m unittest discover -s tests -p 'test_*.py' -v
```

Read:

```text
validation/notebook_1000_validation.txt
validation/benchmark_1000_validation.json
benchmark_1000_synthetic/raw_results/manifest.json
```

### 12.4 “Rerun a single 1,000-node cell”

```bash
python run_1000_synthetic.py --only-methods FIBMM --only-scenarios B1 --only-k 10
```

This skips valid existing cells and refreshes the manifest. Add `--force` only if the cell must be recomputed.

### 12.5 “Recompute the 1,000-node notebook figures”

Run the notebook in `load_existing` mode. It will regenerate `figures/notebook_1000/Net_Advantage_B1.png` … `B4.png`.

### 12.6 “Analyze the 5,000-node benchmark”

```bash
cd IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness
python run_benchmark.py --mode analyze
```

This is read-only with respect to raw results.

### 12.7 “Inspect the 5,000-node six-objective / Swap experiment”

Read:

```text
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/reports/FINAL_REPORT.md
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/aggregated/winner_summary.md
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/final_figures/
```

### 12.8 “Find the authoritative answer to a result question”

Use this decision table:

| Question | Answer location |
|---|---|
| 1,000-node 12-method means/wins | `benchmark_1000_synthetic/raw_results/results.csv`; notebook tables |
| 1,000-node figures | `figures/notebook_1000/` |
| 1,000-node validation | `validation/benchmark_1000_validation.json` |
| 5,000-node 12-method means/wins | `raw_results/results.csv`; `FINAL_REPORT.md` |
| 5,000-node 12-method figures | `figures/notebook/` |
| 5,000-node six-objective / Swap | `exp_20261124_5000_reversed/` |
| FIBM paper mapping | `PAPER_IMPLEMENTATION.md` |
| Algorithm audit | `ALGORITHM_AUDIT.md` |
| Methods | `METHODS.md` |
| Statistical caveats | `reports/STATISTICAL_NOTES.md` |
| Budgets | `reports/ALGORITHM_BUDGETS.md` |
| Old invalid 1,000 same-direction results | `exp_20261124_5000_reversed/to_be_discarded/` — do not use |

---

## 13. Glossary

- **A / Rumor A / truth:** the campaign being optimized; its seeds are selected by the method under test.
- **B / Rumor B / opponent:** the fixed adversarial campaign; its seeds are determined by B1–B4.
- **Trust graph:** the CSV graph, `source -> target`.
- **Influence graph:** the exact reverse, `target -> source`; all algorithms use this.
- **Net Advantage:** `Spread_A - Spread_B`; primary ranking metric.
- **Blocked_B:** reduction in B’s spread relative to a B-only baseline with paired seeds.
- **O1/O4/O6:** SSA selection objectives; not the same as Net Advantage.
- **SSA:** Salp Swarm Algorithm; discrete set-based implementation.
- **SSA-Notebook:** SSA O1 with the notebook seed schedule; not a distinct algorithm.
- **CELF:** lazy-greedy influence maximization; here uses O1 fitness.
- **NCGE:** centrality-filtered greedy; here uses O1 fitness.
- **CRIEG:** competitive reverse-reachability sketch method.
- **Signed-GA:** signed Voter equilibrium + projected gradient method.
- **FIBM / FIBMM:** fairness-aware influence blocking maximization; internal ID `FIBMM`, display name `FIBM`.
- **CRP set:** community-aware reverse protector set used by FIBM.
- **CIC:** Competitive Independent Cascade; FIBM’s selection surrogate.
- **MT:** Maki–Thompson; the evaluated finite-contact competitive diffusion model.
- **MC:** Monte Carlo.
- **CI:** confidence interval.
- **B1–B4:** four opponent strategies (Random, Degree, Positive PageRank, Community).

---

## 14. Key file inventory

### 14.1 Context and documentation

```text
complete_context.md
IM_Synthetic_Experiment/README.md
IM_Synthetic_Experiment/BENCHMARK.md                  # legacy
IM_Synthetic_Experiment/CONFIG.md                     # legacy
IM_Synthetic_Experiment/AUDIT_REPORT.md
IM_Synthetic_Experiment/AUDIT_REPORT_ADDENDUM.md
IM_Synthetic_Experiment/REFERENCE_NOTEBOOK_AUDIT.md
IM_Synthetic_Experiment/GPT_Contexts/ContextForGPT_5000_Reversed.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/README.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/METHODS.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/ALGORITHM_AUDIT.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/PAPER_IMPLEMENTATION.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/FINAL_REPORT.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/reports/CHECKLIST.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/reports/STATISTICAL_NOTES.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/reports/ALGORITHM_BUDGETS.md
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/benchmark_1000_synthetic/README.md
```

### 14.2 Configs

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/config/benchmark_config.json
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/config/benchmark_1000_synthetic_config.json
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/config/experiment_config.json
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/config/run_20261124_5000_reversed_config.json
```

### 14.3 Networks and metadata

```text
IM_Synthetic_Experiment/data/synthetic_network_1000_7500.csv
IM_Synthetic_Experiment/data/network_metadata_v2.json
IM_Synthetic_Experiment/data/network_metadata.json
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/network/synthetic_network_5000_37500.csv
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/network/network_metadata.json
IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/network/network_validation.json
```

### 14.4 Results

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/raw_results/results.csv
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/raw_results/ledger.csv
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/raw_results/manifest.json
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/raw_results/mc/
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/raw_results/checkpoints/
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/raw_results/fibmm_diagnostics/

IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/benchmark_1000_synthetic/raw_results/results.csv
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/benchmark_1000_synthetic/raw_results/ledger.csv
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/benchmark_1000_synthetic/raw_results/manifest.json
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/benchmark_1000_synthetic/raw_results/mc/
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/benchmark_1000_synthetic/raw_results/checkpoints/
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/benchmark_1000_synthetic/raw_results/fibmm_diagnostics/
```

### 14.5 Notebooks and figures

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/competitive_im_benchmark_with_fibm.ipynb
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/competitive_im_benchmark_1000_with_fibm.ipynb

IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/figures/notebook/Net_Advantage_B1.png
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/figures/notebook/Net_Advantage_B2.png
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/figures/notebook/Net_Advantage_B3.png
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/figures/notebook/Net_Advantage_B4.png

IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/figures/notebook_1000/Net_Advantage_B1.png
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/figures/notebook_1000/Net_Advantage_B2.png
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/figures/notebook_1000/Net_Advantage_B3.png
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/figures/notebook_1000/Net_Advantage_B4.png
```

### 14.6 Validation

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/validation/tests.txt
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/validation/verify.txt
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/validation/crosscheck.txt
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/validation/notebook_validation.txt
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/validation/notebook_1000_validation.txt
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/validation/benchmark_1000_validation.json
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/aggregated/validation_summary.json
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/aggregated/validation_report.md
```

### 14.7 Code

```text
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/run_benchmark.py
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/run_1000_synthetic.py
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/analyze_benchmark.py
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/notebook_analysis.py
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/algorithms/core.py
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/algorithms/notebook_baselines.py
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/algorithms/fibmm.py
IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/algorithms/reuse.py
IM_Synthetic_Experiment/common/network.py
IM_Synthetic_Experiment/common/ssa.py
IM_Synthetic_Experiment/common/objectives.py
IM_Synthetic_Experiment/common/evaluation.py
IM_Synthetic_Experiment/common/cpp_adapter.py
IM_Synthetic_Experiment/cpp_engine/
```

### 14.8 Reference papers and notebooks

```text
baseline comparison research papers/CRIEG2021.pdf
baseline comparison research papers/CIMsigned2025.pdf
baseline comparison research papers/Fairness-Aware_Influence_Blocking_Maximization_in_Social_Networks.pdf
baseline comparison research papers/MECIM2025.pdf

IM_Synthetic_Experiment/old work/competitive_im_no_celf_csv_persist.ipynb
IM_Synthetic_Experiment/old work/competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb
```

The `old work/` notebooks are read-only references. The benchmark audit (`ALGORITHM_AUDIT.md`) records exactly how their algorithms were adapted.

---

## 15. Handover checklist for a new agent

Before making changes:

- [ ] Read this file and `experiments/benchmark_ssa_baselines_fairness/README.md`.
- [ ] Read `FINAL_REPORT.md` for the 5,000-node 12-method benchmark.
- [ ] Read the 1,000-node notebook’s findings/conclusion.
- [ ] Confirm which result set the user is asking about (5,000 vs 1,000; 12-method vs six-objective; notebook vs root).
- [ ] Do not rerun simulations unless the user explicitly asks.
- [ ] Do not delete or overwrite old notebooks, results, or reference files.
- [ ] Do not mix 5,000 and 1,000 data.
- [ ] Do not reuse old same-direction 1,000 results.
- [ ] Use `load_existing` for notebook inspection.
- [ ] Use absolute `output_root` for `BenchmarkRunner`.
- [ ] Run the relevant tests after any code change.
- [ ] Update validation files if results or notebook outputs change.

If you are asked to extend the research, the safest next steps are:

1. Repeat the optimizer with independent random seeds to estimate optimizer variability.
2. Test the ranking on a second synthetic network realization.
3. Add a real signed network (Bitcoin OTC / Slashdot) with the same protocol.
4. Run a sensitivity analysis on the FIBM CRP cap.
5. Analyze why CELF wins specific cells (B1/K=30–40, B2/K=10, B3/K=20, B4/K=10, B4/K=50 on 5,000; B1/K=40, B2/K=10, B3/K=10, B4/K=50 on 1,000).

Do not treat the current ranking as universal. The honest conclusion is: **SSA and CELF are close; the winner depends on the opponent and K; most per-cell differences are within Monte Carlo uncertainty.**
