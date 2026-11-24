# ContextForGPT — Complete Project Handover

> **Update (2026-11-24):** this handover predates the 5,000-node
> **reverse-influence** rerun. The current authoritative experiment is
> `experiments/exp_20261124_5000_reversed/`; see its
> `reports/FINAL_REPORT.md` and `README.md`. The old 1,000-node,
> same-direction results have been moved to that experiment's
> `to_be_discarded/` folder and must not be used.

This document is a self-contained handover for an external assistant (ChatGPT)
that does not have access to the repository. It explains the research goal, the
conversation history, every important file and folder, the implemented logic,
the execution flow, the actual results, the validation evidence, and the
remaining researcher decisions.

Project root (in the original environment):

```
/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment
```

The project is **not a git repository**. No packages were installed, no system
configuration was changed, and no files outside the project workspace were
modified (apart from a tiny temporary timestamp file that was removed).

---

## 1. One-paragraph summary

The project studies **Competitive Influence Maximization on a signed synthetic
social network**. Rumour A is promoted by a seed set of K nodes against an
opponent Rumour B under a finite-contact competitive Maki–Thompson diffusion
model. A C++ diffusion engine (pybind11, OpenMP) performs the Monte Carlo
simulations; Python handles candidate pools, opponent strategies, Salp Swarm
Algorithm (SSA) optimization, bounded Swap refinement, checkpointing, and
analysis. The primary evaluation metric is **Net Advantage = Spread_A −
Spread_B**. Six objective functions (O1–O6) are optimized by SSA, and each SSA
solution is evaluated in two modes: SSA-only and SSA+Swap. The full design is
4 opponent scenarios × 5 K values × 6 objectives × 2 modes = **240
configurations**. A first full run completed 240/240. Later, two reference
notebooks were supplied; a line-by-line audit found three material
discrepancies (candidate-pool ranking/random-part order, optimization-seed
offset, and round-counter semantics). These were corrected, a reference-
alignment test suite was added, and the full experiment was rerun. The
authoritative run is `run_20261123_205500_reference_aligned`, also 240/240.
The main finding is that **O2 (Competition-aware)** has the highest mean final
Net Advantage, with O1 and O6 close behind; **O3 (maximise Blocked_B)** has the
highest Blocked_B but the lowest Net Advantage; and **SSA+Swap does not reliably
improve** the final metric.

---

## 2. Conversation history (what the user asked and what was done)

1. **User:** “Read `OPENCODE_OVERNIGHT_MASTER_PROMPT.md` and work according to
   the instructions in it.”
   - The master prompt asked for an autonomous audit, repair, implementation,
     validation, full experiment, analysis, and final audit, with strict
     workspace-only safety rules.
   - The workspace initially contained a starter project: a synthetic network
     CSV, a C++ diffusion engine, Python modules (`network_generator.py`,
     `experiment_core.py`, `scenario_runner.py`, `cpp_adapter.py`), and five
     six starter notebooks with malformed code cells. The reference notebooks
     named in the prompt were absent.
   - Work performed:
     - Audited the starter and wrote `AUDIT_REPORT.md`.
     - Backed up originals under `backups/`.
     - Rebuilt the Python code into focused modules.
     - Rewrote the C++ engine as a persistent graph-resident engine with
       OpenMP batch simulation.
     - Added a pure-Python reference simulator for parity tests.
     - Added a 102-test suite (later expanded to 110).
     - Rebuilt all notebooks with real line breaks and safe defaults.
     - Ran a smoke test, wrote `VALIDATION_GATE.md`, and ran the full 240-
       configuration experiment (`run_20261123_202600`).
     - Generated analysis tables/figures and wrote `RESULTS_SUMMARY.md` and
       `FINAL_AUDIT.md`.

2. **User:** “I have made a new folder — `old work`; it contains 2 older files,
   go through those.”
   - The folder contained the two missing reference notebooks:
     - `competitive_im_no_celf_csv_persist.ipynb` — four-B Net Advantage
       notebook.
     - `competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb` —
       six-objective Gain_A/Blocked_B notebook (latest reference).
   - I parsed both notebooks programmatically, extracted their code to
     `old work/_extracted/`, and compared every relevant definition.
   - Exact matches were confirmed for the diffusion model, Blocked_B, O1–O6,
     B1–B4, SSA movement/elitism, Swap policy, and final evaluation.
   - Three **material discrepancies** were found:
     1. Candidate-pool ranking and random-part order were wrong.
     2. Optimization seeds had an extra `+3,000,000`.
     3. `rounds` was off by one.
   - Blocked_B was confirmed exactly, so all six objectives were upgraded from
     `provisional_blocked_b` to `validated`.
   - Corrections were implemented, `tests/test_reference_alignment.py` was
     added, and the full experiment was rerun as
     `run_20261123_205500_reference_aligned` (240/240).
   - New reports: `REFERENCE_NOTEBOOK_AUDIT.md`,
     `VALIDATION_GATE_REFERENCE_ALIGNED.md`,
     `RESULTS_SUMMARY_REFERENCE_ALIGNED.md`,
     `FINAL_AUDIT_REFERENCE_ALIGNED.md`.
   - The old run `run_20261123_202600` was retained as a sensitivity check.

3. **User:** “Explain O1–O6 and the direction of influence vs trust score.”
   - Explained the six objective formulas and that O2 is the only one that
     directly optimizes the final Net Advantage metric.
   - Clarified that in the **synthetic network**, the trust rating and the
     influence direction are the same `source → target` arc. The reference
     Bitcoin OTC notebooks reverse the raw CSV direction (`target → source`);
     this was documented but not applied to the synthetic network.

4. **User:** “How are you executing the files so quickly?”
   - Explained the persistent C++ engine, OpenMP, exact caching, and
     checkpoint/resume. The old Python-only simulator ran at ~190 sims/s; the
     old one-shot C++ adapter at ~692 sims/s; the current persistent engine
     runs at ~10,700 sims/s serial and ~40,255 sims/s with 8 threads.

5. **User:** “Explain all folders/files; are you 100% sure the current
   implementation gives the same results just in less time?”
   - Explained the directory tree and was explicit that the current
     implementation is **not just faster**: it is a corrected, reference-
     aligned implementation. Spread counts from the C++ refactor are
     bit-identical to the backed-up original C++ starter, but the candidate
     pool, optimization seeds, and round counter changed after the reference
     audit. The current full run therefore differs from the pre-reference run
     (per-configuration correlation ≈ 0.38), and results are not directly
     comparable to the Bitcoin OTC notebooks.

6. **User:** “Create a markdown file named `ContextForGPT` explaining the entire
   work, files/folders, logic, execution flow, conversation, everything.”
   - This document is the result.

---

## 3. Research goal and experimental design

### 3.1 Research goal

Study **competitive influence maximization for Rumour A on a signed social
network against an opponent Rumour B**. The main evaluation metric is:

```
Net Advantage = Spread_A − Spread_B
```

Higher is better. The optimizer’s internal fitness is **not** the ranking
metric. Objectives are compared by their actual final Net Advantage.

### 3.2 Synthetic network

| Parameter | Value |
|---|---|
| Nodes | 1,000 (ids 0–999) |
| Directed arcs | 7,500 (one CSV row = one directed arc) |
| Underlying unordered pairs | 7,498 |
| Reciprocal arc pairs | 2 (4 reciprocal arcs) |
| Self-loops | 0 |
| Duplicate directed arcs | 0 |
| Topology | Barabási–Albert-style undirected backbone (`m=7`, initial clique 8), randomly oriented, plus random directed fill |
| Generation seed | 42 |
| Sign | positive with probability 0.5 |
| Trust magnitude | uniform integer 1–10 |
| Rating | `polarity × strength` |
| Probability mapping | strength 1–3 → 0.1; 4–7 → 0.5; 8–10 → 0.9 |
| Positive / negative arcs | 3,720 / 3,780 |
| Probability class counts 0.1 / 0.5 / 0.9 | 2,278 / 2,950 / 2,272 |
| Out-degree min/mean/max | 0 / 7.5 / 77 |
| In-degree min/mean/max | 1 / 7.5 / 67 |
| Weak components | 1 (all 1,000 nodes) |
| Strong components | 3 (largest 998) |
| Isolated nodes | 0 |

File SHA-256:

```
659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5
```

Semantic edge-table SHA-256:

```
461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25
```

**Edge-count convention.** The generator builds 7,500 undirected pairs, orients
each once, and uses a random directed fill if orientation collisions reduce the
count. Because the pair de-duplication filter is order-sensitive, a few pairs
can be added twice, collapse after orientation, and be replaced by fill arcs.
The delivered file has exactly 7,500 unique directed arcs over 7,498 unordered
pairs plus 2 reciprocal pairs. Seed 42 reproduces the CSV byte-for-byte.

**Direction convention.** In the synthetic network, the signed trust rating and
the influence direction are the same directed arc `source → target`. Positive
edges transmit the same rumour; negative edges transmit the opposite rumour.
The reference Bitcoin OTC notebooks reverse the raw CSV direction
(`target → source`); that is a dataset-specific preprocessing choice and was
**not** applied to the synthetic network, because it would be a different graph
and would invalidate the fingerprint.

### 3.3 Diffusion model

Finite-contact competitive Maki–Thompson variant, synchronous rounds.

States:

| Code | State |
|---|---|
| 0 | ignorant |
| 1 | A-spreader |
| 2 | B-spreader |
| 3 | A-stifler |
| 4 | B-stifler |

Rules:

1. A seeds start as A-spreaders; B seeds start as B-spreaders. A/B overlap is
   rejected.
2. Each round, every current spreader chooses one uniformly random outgoing
   neighbour it has not contacted before.
3. If it has no uncontacted outgoing neighbour, or the chosen target is not
   ignorant at the start of the round, it becomes a stifler of its own rumour
   and does not transmit that round.
4. Otherwise, with the edge transmission probability, it transmits. A negative
   edge flips the rumour (A→B, B→A).
5. Proposals are resolved synchronously: highest edge strength wins; ties
   between different rumours are broken uniformly at random.
6. Final `Spread_A` = count of states 1 and 3; `Spread_B` = 2 and 4;
   `ignorant` = 0.
7. `rounds` follows the reference notebook: it is the number of the terminal
   round in which no spreader remained, i.e. **executed rounds + 1**.

### 3.4 The six objectives

`Blocked_B` is defined as:

```
baseline_B(B, s) = Spread_B(A = empty, B, s)
Blocked_B        = max(0, mean_s baseline_B − mean_s Spread_B(A, B, s))
```

The baseline and the A/B run use the same Monte Carlo seeds (common random
numbers). Per-seed unclipped differences are stored for analysis. This is
exactly the supplied six-objective reference notebook’s definition.

| ID | Name | Formula |
|---|---|---|
| O1 | Original SSA | `Spread_A + 1.5·Blocked_B` |
| O2 | Competition-aware | `Spread_A − Spread_B` |
| O3 | Explicit blocking | `Blocked_B` |
| O4 | Combined influence-blocking | `Spread_A + 2·Blocked_B` |
| O5 | Gain_A + blocking | `(Spread_A − Spread_B) + 2·Blocked_B` |
| O6 | Budget-aware | `α(K)·Spread_A + β(K)·Blocked_B` |

Budget coefficients:

| K | α | β |
|---|---|---|
| 10, 20 | 0.75 | 2.25 |
| 30, 40 | 1.00 | 1.75 |
| 50 | 1.25 | 1.50 |

O4 and O5 are different and must not be conflated.

### 3.5 Opponent B scenarios

All orders have length 50 and are nested across K (the K-seed set is a prefix).

| Scenario | Rule | Seed |
|---|---|---|
| B1 Random | `random.Random(master_seed + 810001).sample(nodes, 50)` | 42 + 810001 |
| B2 Degree | out-degree descending, ties ascending node id | — |
| B3 Positive PageRank | PageRank on positive-only directed subgraph, edge weight = probability, α=0.85, max_iter=200, tol=1e-10; descending, ties ascending id | — |
| B4 Community | Louvain on undirected unweighted projection (signs ignored), seed 42; largest community (ties: smallest member id); sort by out-degree descending, ties ascending id | 42 |

B4 largest community size on the delivered network: 227.

### 3.6 Candidate pool

Reference-aligned with the six-objective notebook.

- Target size: `10 × K`.
- Polarity-aware score:
  `score(u) = Σ probability of outgoing positive edges − Σ probability of outgoing negative edges`.
- Ranking key (exact four-level tie-break):
  `(-net_score, -positive_score, negative_score, node_id)`.
- Composition:
  `n_random = int(round(pool_size × 0.30))`,
  `n_ranked = pool_size − n_random`.
  The random part is sampled uniformly without replacement from the **ranked
  remainder** `ranked[n_ranked:]`, not from node-id order.
- Opponent seeds are excluded everywhere.
- Pool seed: `master_seed + 600000 + scenario_number`.
- If fewer eligible nodes exist, the project truncates and records it; the
  reference raises. The full network never hits this branch.

### 3.7 SSA

| Parameter | Value |
|---|---|
| Population | 12 |
| Iterations | 10 |
| Mutation rate | 0.20 |
| `c1` | `2·exp(−(4t/T)²)` |
| Exploitation probability | `1 − c1/2` |
| Representation | exactly K unique candidate nodes |
| Fitness | mean objective over 15 fixed optimization MC seeds |
| Elitism | best-so-far food preserved; worst individual replaced by food |
| Random seed | `master_seed + 500000 + scenario_number·1000 + K·10 + objective_index` |

SSA is run **once** per `(scenario, K, objective)`; both modes use the saved
seed set. Repair/dedup logic keeps only candidate-pool members and pads to
exactly K.

### 3.8 Swap refinement

- Start from the exact SSA seed set.
- One-for-one replacements only.
- Evaluate up to `max_tests = 100` pairs (all pairs when fewer).
- Apply the single best strictly improving swap; equal/worse swaps rejected
  (elitist).
- Exactly K unique candidate nodes; opponent seeds excluded.
- Random seed: `770000 + scenario_number·1000 + K + objective_index`.
- Logs tests, accepted pair, objective before/after, evaluation count.

### 3.9 Modes and configuration count

- Modes: `SSA only`, `SSA + Swap refinement`.
- Per scenario: 5 K × 6 objectives × 2 modes = 60 configurations.
- Across 4 scenarios: **240 configurations**.

---

## 4. Reference notebooks and alignment

### 4.1 Files

Supplied in `old work/`:

- `competitive_im_no_celf_csv_persist.ipynb` — four-B Net Advantage notebook.
- `competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb` —
  six-objective Gain_A/Blocked_B notebook; latest reference.

Extracted plain-text copies for auditing:

- `old work/_extracted/four_b_code.py`
- `old work/_extracted/six_variants_code.py`

The notebooks were parsed and read, not executed wholesale.

### 4.2 Exact matches confirmed

- Diffusion states, contact memory, stifling, negative-edge flip,
  strength-based conflict resolution, synchronous updates, final spread counts.
- `Blocked_B = max(0, mean baseline_B − mean Spread_B)` with paired seeds.
- O1–O6 formulas and budget coefficients.
- B1–B4 rankings, tie-breaks, PageRank/Louvain settings.
- SSA movement, repair, elitism, population/iteration/mutation.
- Swap sampled one-for-one, single best improving, elitist.
- Final evaluation with 100 common seeds.

### 4.3 Material discrepancies found and fixed

1. **Candidate-pool ranking**
   - Reference: `(-net, -positive, negative, node_id)`.
   - Old code: `(-net, node_id)`.
   - Fixed in `common/candidates.py`.

2. **Candidate-pool random part**
   - Reference: sample from `ranked[n_ranked:]` (ranked remainder).
   - Old code: sample from node-id order.
   - Fixed. On the full network this changed ~30 % of each pool. Old/reference
     overlap was only 70/100 for B1/K=10 and 377/500 for B4/K=50.

3. **Optimization seeds**
   - Reference: `master_seed*100000 + scenario_number*10000 + i`.
   - Old code: an extra `+3,000,000`.
   - Fixed in `common/experiment.py`.

4. **Round counter**
   - Reference: terminal empty round included (`executed_rounds + 1`).
   - Old code: executed rounds only.
   - Fixed in C++ and Python. Spread counts are unchanged; only
     `average_rounds` changes.

5. **Blocked_B validity**
   - Reference confirms the exact paired B-only baseline.
   - Upgraded from `provisional_blocked_b` to `validated`.

6. **SSA history**
   - Added `unique_solutions` as in the reference.

### 4.4 Documented differences, not changed

- **Synthetic edge orientation**: reference reverses Bitcoin OTC edges
  (`target → source`); synthetic network keeps `source → target` as generated
  and fingerprinted. Reversing it would be a new network.
- **Zero-rated edges**: reference supports rating 0 (probability 0, polarity 0);
  synthetic network has no zero ratings.
- **Small-network pool truncation**: reference raises when `pool_size >
  eligible`; project truncates and records it. Unused on the full network.
- **Older four-B SSA seed**: the four-B notebook uses
  `master_seed + 30000 + scenario_number`; the six-objective notebook (latest
  reference) uses the per-K/per-variant formula, which is used here.

Full detail: `REFERENCE_NOTEBOOK_AUDIT.md`.

---

## 5. Directory and file inventory

### 5.1 Top-level files

| File | Purpose |
|---|---|
| `README.md` | Project overview, quick start, structure, caveats |
| `CONFIG.md` | Every constant, seed, formula, design decision |
| `CHANGELOG.md` | Version history |
| `OVERNIGHT_LOG.md` | Timestamped work log |
| `AUDIT_REPORT.md` | Original source-of-truth audit (pre-reference) |
| `AUDIT_REPORT_ADDENDUM.md` | Session-2 verification; notes reference notebooks later arrived |
| `REFERENCE_NOTEBOOK_AUDIT.md` | Line-by-line audit of the supplied notebooks |
| `VALIDATION_GATE.md` | Pre-reference validation gate (superseded) |
| `VALIDATION_GATE_REFERENCE_ALIGNED.md` | Authoritative gate |
| `BENCHMARK.md` | Throughput/speedup measurements |
| `RESULTS_SUMMARY.md` | Pre-reference results (superseded, sensitivity) |
| `RESULTS_SUMMARY_REFERENCE_ALIGNED.md` | Authoritative results summary |
| `FINAL_AUDIT.md` | Pre-reference final audit (superseded) |
| `FINAL_AUDIT_REFERENCE_ALIGNED.md` | Authoritative final audit |
| `OPENCODE_OVERNIGHT_MASTER_PROMPT.md` | The task specification |
| `ContextForGPT.md` | This document |

### 5.2 `common/` — Python research code

Active modules:

| File | Role |
|---|---|
| `utils.py` | Paths, atomic writes, hashing, deterministic seed helpers, environment info |
| `network.py` | Load/validate/fingerprint the network, metadata v2 |
| `network_generator.py` | Pinned seed-42 generator (unchanged) |
| `opponents.py` | B1–B4 strategies |
| `candidates.py` | 10·K candidate pool, four-level ranking, 70/30 split |
| `objectives.py` | O1–O6 formulas, budget coefficients, validity |
| `diffusion.py` | Pure-Python reference simulator |
| `cpp_adapter.py` | Persistent C++ engine adapter |
| `evaluation.py` | Monte Carlo metrics, Blocked_B, safe caching |
| `ssa.py` | Discrete Salp Swarm Algorithm |
| `swap.py` | Bounded one-for-one Swap refinement |
| `stats.py` | Bootstrap CIs, paired tests, Holm correction |
| `experiment.py` | Checkpointed experiment coordinator |
| `run_experiment.py` | Command-line runner |
| `analysis.py` | Read-only tables and figures |

Compatibility/historical:

| File | Role |
|---|---|
| `experiment_core.py` | Old API facade |
| `scenario_runner.py` | Old API facade |
| `config.py` | Unused alternate configuration module |
| `network_validation.py` | Unused alternate network module |
| `util.py` | Unused alternate utility module |
| `__init__.py` | Package marker |
| `__pycache__/` | Compiled Python cache |

**Important:** `config.py`, `network_validation.py`, and `util.py` are **not
imported by the active pipeline**. They appear to be an alternate/duplicate
implementation. The active code uses `utils.py` and `network.py`.

### 5.3 `cpp_engine/` — C++ diffusion engine

| File | Role |
|---|---|
| `diffusion.hpp` | Engine class and `SimResult` declaration |
| `diffusion.cpp` | Persistent graph engine, diffusion loop, OpenMP batch |
| `bindings.cpp` | pybind11 bindings |
| `setup.py` | Project-local build (`build_ext --inplace`) |
| `Makefile` | Build/check/clean targets; no package installation |
| `README.md` | Build notes |
| `MODEL_SPEC.md` | Exact model specification |
| `simulation_engine*.so` | Compiled extension |
| `build/` | Compiler artifacts |

### 5.4 `data/` — immutable source data

| File | Role |
|---|---|
| `synthetic_network_1000_7500.csv` | Shared network |
| `network_metadata.json` | Original starter metadata |
| `network_metadata_v2.json` | Fingerprints, counts, distributions, degree/connectivity |

### 5.5 `notebooks/` — six thin notebooks

| File | Role |
|---|---|
| `00_Setup_Synthetic_Network.ipynb` | Validate network/fingerprint/C++ status |
| `B1_Random_opponent.ipynb` | B1 scenario; default `RUN_MODE='none'` |
| `B2_Degree-based_opponent.ipynb` | B2 scenario |
| `B3_Positive_PageRank_opponent.ipynb` | B3 scenario |
| `B4_Community-concentrated_opponent.ipynb` | B4 scenario |
| `05_Analysis_Charts.ipynb` | Read-only analysis |

The B notebooks require `CONFIRM_FULL_RUN=True` for a full run. The analysis
notebook never launches simulations. `.ipynb_checkpoints/` holds Jupyter
autosaves.

### 5.6 `tests/` — 110 tests in 12 test modules

| File | Coverage |
|---|---|
| `test_network.py` | Counts, signs, probabilities, seed-42 reproduction, fingerprint |
| `test_diffusion_python.py` | Deterministic model cases and probability classes |
| `test_cpp_engine.py` | C++ import, determinism, serial/parallel, errors, old/new parity |
| `test_objectives.py` | Hand-calculated O1–O6, budget coefficients, validity |
| `test_candidates.py` | Pool size/composition, polarity-awareness, B exclusion, small networks |
| `test_opponents.py` | B1 exact sample, B2 degree, B3 positive-only PageRank, B4 community, nesting |
| `test_ssa_swap.py` | SSA invariants/determinism, Swap elitism/bounds |
| `test_evaluation.py` | Metrics, Blocked_B, cache keys, per-seed arrays |
| `test_experiment_resume.py` | Checkpoints, resume, signature mismatch, manifest, schema |
| `test_notebooks.py` | JSON/nbformat validity, code-cell syntax, safe defaults |
| `test_analysis.py` | Read-only ingestion, tables, figures, corruption detection |
| `test_reference_alignment.py` | Exact reference pools, seeds, objectives, SSA trajectory, Swap, rounds |
| `_helpers.py` | Path bootstrap and tiny graph builders |

### 5.7 `tools/`

| File | Role |
|---|---|
| `build_notebooks.py` | Regenerates the six notebooks |
| `benchmark.py` | Bounded throughput benchmark |

### 5.8 `results/` — experiment outputs

| Directory | Role |
|---|---|
| `run_20261123_205500_reference_aligned/` | **Authoritative** full run, 240/240 |
| `run_20261123_202600/` | Pre-reference full run, 240/240, sensitivity check |
| `smoke/` | Isolated smoke runs |

Each full run contains:

- `B1/` … `B4/` — one directory per scenario.
- `experiment_results.csv` — 60 rows per scenario, 72 columns.
- `checkpoints/` — one JSON per `(scenario, K, objective)` with the SSA seed
  set, score, runtime, history, pool, opponent set, and configuration
  signature.
- `mc_runs/` — one CSV per `(scenario, K, objective, mode)` with 100 per-seed
  rows: seed, spread_A, spread_B, ignorant, rounds, baseline_B,
  blocked_B_run, net_advantage_run.
- `ssa_history/` — one CSV per SSA run with per-iteration best/mean/worst and
  unique-solution counts.
- `scenario_metadata.json` — opponent order, K seed sets, candidate pools.
- `run_metadata.json` — config, environment, network, C++ status.
- `manifest.json` — expected/completed/missing configurations and status.

Smoke runs are under `results/smoke/` and are excluded from default analysis
run selection.

### 5.9 `analysis/` — generated tables and figures

- `tables/<run_id>_<timestamp>/` — 10 CSVs plus `summary.md`:
  `overall.csv`, `per_scenario.csv`, `per_k.csv`, `mode_comparison.csv`,
  `objective_vs_reference.csv`, `objective_ranking.csv`, `runtime.csv`,
  `variability.csv`, `robustness.csv`, `best_config.csv`, `summary.md`.
- `figures/<run_id>_<timestamp>/` — 11 PNGs:
  Net Advantage vs K (SSA-only and SSA+Swap), Swap vs SSA paired differences,
  Spread_A vs Spread_B, Blocked_B (reference-confirmed), runtime breakdown,
  objective ranking, robustness across B1–B4, variability, and two
  objective×scenario heatmaps.
- `.mplconfig/` — matplotlib cache.

The authoritative analysis output is:

```
analysis/tables/run_20261123_205500_reference_aligned_20261123_210005/
analysis/figures/run_20261123_205500_reference_aligned_20261123_210005/
```

An earlier `..._205503` directory exists from before the analysis labels were
updated; it contains the same data but the old “provisional” figure filename.
The docs point to `..._210005`.

### 5.10 `logs/`

- `run_20261123_205500_reference_aligned.log` — authoritative run log.
- `run_20261123_202600.log` — pre-reference run log.
- `full_run_reference_aligned_console.log` — full console capture.
- `full_run_console.log` — pre-reference console capture.
- Smoke logs.
- `benchmark_20261123_202024.json` — benchmark data.

### 5.11 `old work/`

- `competitive_im_no_celf_csv_persist.ipynb` — four-B reference notebook.
- `competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb` —
  six-objective reference notebook.
- `_extracted/four_b_code.py`, `_extracted/six_variants_code.py` — plain-text
  extracts for auditing.

### 5.12 `backups/`

Original files copied before modification:

- `backups/20261123_200236/` — first backup (network, notebooks, C++ sources,
  common, README).
- `backups/20261123_200624_session2/` — second backup before session-2 edits.

---

## 6. Implemented logic in detail

### 6.1 Network loading, validation, fingerprint

`common/network.py`:

- Loads the CSV, requires columns `source, target, rating, probability,
  strength, polarity`.
- Validates:
  - 1,000 nodes, contiguous 0..999.
  - 7,500 rows, unique directed arcs, no self-loops.
  - `rating == polarity × strength`, polarity ±1, strength 1..10.
  - probability exactly matches the 1–3/4–7/8–10 mapping.
- Builds a `networkx.DiGraph` with edge attributes.
- `semantic_fingerprint` sorts edges and SHA-256s their canonical text.
- `build_metadata_v2` records file SHA-256, semantic SHA-256, counts,
  distributions, degree/connectivity summary, edge-count convention.
- `regenerate_network_df` reproduces the network in memory with seed 42; a test
  verifies byte-identical output.

### 6.2 C++ engine

`cpp_engine/diffusion.hpp/.cpp`:

- `DiffusionEngine` stores a flattened CSR-style graph:
  `offsets_`, `targets_`, `probs_`, `polarity_`, `strength_`.
- Graph is built and validated once; reused read-only.
- `simulate(A, B, seed)` returns `SimResult{spread_A, spread_B, ignorant,
  rounds}`.
- `simulate_many(A, B, seeds, n_threads)` and `simulate_batch(As, Bs, seeds,
  n_threads)` use OpenMP across runs.
- Each run has its own `mt19937_64` stream and workspace; results do not depend
  on thread scheduling.
- `simulate` functional API retained for compatibility.
- `openmp_enabled()` reports build support.

### 6.3 Diffusion algorithm (exact)

Per round:

1. Snapshot `old = state`; `next = state`.
2. For each spreader `u` in ascending index:
   - Find uncontacted outgoing arcs.
   - If none: `u` becomes a stifler; no transmission.
   - Otherwise pick one uniformly at random, mark it contacted.
   - If target `v` was not ignorant at round start: `u` becomes a stifler;
     no transmission.
   - Otherwise draw `x ~ U[0,1)`; if `x < p(arc)`, transmit. Negative arc
     flips the rumour. Record proposal `(v, rumour, strength)`.
3. If no spreader existed at round start, stop. The terminal empty round is
   the returned `rounds` value.
4. For each target with proposals:
   - Keep max-strength proposals.
   - If one rumour survives, it wins.
   - If both rumours tie at max strength, choose uniformly.
5. Apply stiflers to `next`.
6. `state = next`; increment executed-round counter.
7. Safety limit 7,501 rounds.

Final counts: `Spread_A` = states 1+3; `Spread_B` = 2+4; `ignorant` = 0.

### 6.4 Evaluation and caching

`common/evaluation.py`:

- `MetricEvaluator` wraps the C++ simulator.
- Metric cache key:
  `(graph fingerprint, A tuple, B tuple, seed tuple, protocol)`.
- Baseline cache key: `(graph fingerprint, B tuple, seed tuple, protocol)`.
- `evaluate(A, B, seeds)`:
  - Runs A/B simulations.
  - Runs B-only baseline with the same seeds (cached).
  - Returns means and per-seed arrays for Spread_A, Spread_B,
    Net_Advantage, Blocked_B, baseline_B, ignorant, rounds.
- `blocked_B = max(0, baseline_mean − spread_B_mean)`.
- `blocked_B_runs = baseline_B_runs − spread_B_runs` (unclipped).
- Objective values are always recomputed from metrics with the requested
  objective id and K; they are never cached under an objective-agnostic key.
- Counters track cache hits/misses, simulations, and simulation seconds.

### 6.5 Objectives

`common/objectives.py` implements the six formulas and budget coefficients.
All six are labelled `validated` after the reference audit. O4 and O5 are
distinct.

### 6.6 Candidate pool (exact reference logic)

`common/candidates.py`:

1. Compute per-node:
   - `positive_outgoing_score`
   - `negative_outgoing_score`
   - `net_signed_candidate_score = positive − negative`
2. Eligible nodes exclude opponent B seeds.
3. Rank with:
   `(-net, -positive, negative, node_id)`.
4. `pool_size = 10·K`.
5. `n_random = int(round(pool_size × 0.30))`;
   `n_ranked = pool_size − n_random`.
6. `top_ranked = ranked[:n_ranked]`;
   `remaining = ranked[n_ranked:]`.
7. `random_nodes = random.Random(seed).sample(remaining, n_random)`.
8. `pool = top_ranked + random_nodes`.
9. Small-network fallback: fill from `remaining` and record `truncated=True`.

### 6.7 Opponents (exact reference logic)

`common/opponents.py`:

- B1: `random.Random(master_seed + 810001).sample(sorted(nodes), 50)`.
- B2: sorted by `(-out_degree, node_id)`.
- B3: PageRank on positive-only subgraph, edge weight = probability,
  α=0.85, max_iter=200, tol=1e-10; sorted by `(-pagerank, node_id)`.
- B4: Louvain on undirected unweighted projection, seed 42; largest community
  by `(-size, min_node_id)`; sorted by `(-out_degree, node_id)`.
- All orders length 50, nested across K.

### 6.8 SSA (exact reference logic)

`common/ssa.py`:

- `population = [rng.sample(pool, K) for _ in range(12)]`.
- Evaluate each; food = best.
- For each iteration:
  - `c1 = 2·exp(−(4t/T)²)`; `exploit = 1 − c1/2`.
  - Build leader position by position:
    - with probability `exploit`, take from food if not already used;
    - otherwise with 0.5 from current, else from pool.
  - Repair leader to exactly K unique pool nodes.
  - For each follower:
    - copy 1..K random positions from the previous new salp;
    - mutate each position with probability 0.20 using unused pool nodes;
    - repair.
  - Evaluate new population.
  - If new best > food score, update food.
  - Replace worst individual with food (elitism).
  - Record history: iteration, best, mean, worst, unique_solutions.
- Final invariant checks: exactly K unique, subset of pool, no B seeds.

### 6.9 Swap (exact reference logic)

`common/swap.py`:

- Start from SSA set.
- `outsiders = pool − selected − B`.
- If `K × |outsiders| ≤ 100`, use all pairs; otherwise sample 100 unique pairs
  with `random.Random(swap_seed)`.
- Evaluate all pairs; apply the single best strictly improving swap.
- Elitist; exactly K unique; no B seeds.
- Return sorted final set and info: tests, accepted pair, objective
  before/after/gain, evaluations.

### 6.10 Seed derivation (exact reference)

`common/experiment.py::derive_seeds`:

| Item | Formula |
|---|---|
| Optimization seeds | `master_seed*100000 + scenario_number*10000 + i`, i=0..14 |
| Final seeds | `900000000 + i`, i=0..99 |
| SSA random seed | `master_seed + 500000 + scenario_number*1000 + K*10 + objective_index` |
| Swap random seed | `770000 + scenario_number*1000 + K + objective_index` |
| Candidate pool seed | `master_seed + 600000 + scenario_number` |
| B1 opponent seed | `master_seed + 810001` |

Master seed = 42.

### 6.11 Checkpoints, CSV schema, resume

`common/experiment.py`:

- Run directory: `results/<run_id>/`.
- Per scenario: `experiment_results.csv`, `checkpoints/`, `mc_runs/`,
  `ssa_history/`, `scenario_metadata.json`.
- Run-level: `run_metadata.json`, `manifest.json`.
- CSV writes and JSON checkpoints are atomic (sibling temp file + `os.replace`).
- Configuration signature includes:
  - schema version 3, reference alignment tag
  - scenario, K, objective id/name
  - network file + semantic fingerprint
  - master seed, SSA parameters, MC counts
  - pool ratio/fraction/signature, ranking/random-part tags
  - opponent-order signature
  - swap max tests, evaluation protocol, round semantics
  - optimization/final seed hashes
- Resume:
  - Existing rows are skipped only if their signature hash matches.
  - A checkpoint is reused if its signature matches.
  - A missing mode can be completed from the saved SSA set.
  - Signature mismatch raises `CheckpointMismatch` and instructs a new run ID.
- CSV schema has 72 columns, including:
  - run/scenario/K/objective/mode, objective formula/validity
  - fingerprints, master seed, seed protocols, MC counts
  - SSA population/iterations/mutation/random seed
  - candidate-pool size/ranked/random/signature
  - initial/final optimizer score, final evaluation objective score
  - Spread_A, Spread_B, Net_Advantage, Gain_A, Blocked_B, baseline_B
  - ignorant nodes, average rounds
  - standard errors and bootstrap CIs
  - SSA/Swap/final/total runtimes
  - swap tests/accepted pair
  - SSA, Swap-initial, final, opponent seed sets (JSON)
  - config signature hash, checkpoint/history/MC file paths, status, timestamp.

### 6.12 Analysis

`common/analysis.py`:

- Never imports the engine and never launches simulations.
- Discovers runs by `run_metadata.json`.
- Selects a run explicitly or the latest non-smoke run.
- Validates:
  - required columns, duplicates, one run ID/fingerprint
  - `Net Advantage = Spread_A − Spread_B`
  - non-negative Blocked_B
  - expected/missing configurations
  - MC-file existence and path escape
  - SSA/Swap MC seed pairing
- Builds tables:
  - overall, per-scenario, per-K
  - Swap-vs-SSA paired differences
  - objective-vs-O2 paired differences with Holm correction
  - objective ranking by final Net Advantage
  - runtime, variability, robustness, best configuration per scenario/K
- Builds 11 figures, including the reference-confirmed Blocked_B figure.
- Writes timestamped tables/figures under `analysis/`.

---

## 7. Execution flow — which file runs when and how results travel

### 7.1 Build and test

```bash
cd cpp_engine
python setup.py build_ext --inplace
cd ..

python -m unittest discover -s tests -p 'test_*.py'
```

Build steps:

1. `setup.py` invokes pybind11/setuptools with `-O3 -fopenmp`.
2. Compiles `bindings.cpp` and `diffusion.cpp`.
3. Produces `simulation_engine.cpython-314-x86_64-linux-gnu.so`.
4. `common/cpp_adapter.py` adds `cpp_engine/` to `sys.path` before importing.

### 7.2 Full experiment CLI flow

Command:

```bash
python common/run_experiment.py --run-id run_YYYYMMDD_HHMMSS
```

Flow:

1. `common/run_experiment.py`
   - parses args, builds `ExperimentConfig`.
   - prints the plan (240 configurations).
   - creates `ExperimentRunner`.

2. `ExperimentRunner.__init__`
   - loads `common/network.py` → `df`, `G`, `nodes`.
   - computes file and semantic fingerprints.
   - validates the network.
   - creates `CppSimulator` (`common/cpp_adapter.py`) → persistent
     `DiffusionEngine`.
   - creates `MetricEvaluator` (`common/evaluation.py`).
   - sets up `logs/<run_id>.log`.

3. `ExperimentRunner.run_all`
   - writes `run_metadata.json` (config, environment, network, C++ status).
   - loops scenarios sequentially: B1 → B2 → B3 → B4.
   - after each scenario, writes `manifest.json`.
   - at the end, writes final manifest/status.

4. `ExperimentRunner.run_scenario`
   - creates `results/<run_id>/<scenario>/`.
   - reads existing `experiment_results.csv` if resuming.
   - gets the fixed opponent order from `common/opponents.py`.
   - loops K = 10,20,30,40,50.
   - for each K:
     - `B = order[:K]`.
     - `common/candidates.py` builds the 10·K pool.
     - loops objectives O1–O6.
     - `derive_seeds` computes optimization/final/SSA/Swap/pool seeds.
     - builds the configuration signature.
     - loads or creates the SSA checkpoint:
       - `common/ssa.py` runs SSA using `MetricEvaluator` for fitness.
       - writes checkpoint JSON and SSA history CSV.
     - loops modes:
       - SSA-only: final set = SSA set.
       - SSA+Swap: `common/swap.py` refines from the same SSA set.
       - `MetricEvaluator.evaluate(final_set, B, final_seeds)` produces final
         metrics and per-seed arrays.
       - writes `mc_runs/<scenario>_K<K>_<objective>_<mode>.csv`.
       - appends a row to `experiment_results.csv` (atomic write).
   - writes `scenario_metadata.json`.

5. Result travel:

```
network CSV
  → common/network.py (validate/fingerprint)
  → common/opponents.py (B order)
  → common/candidates.py (pool)
  → common/ssa.py (seed set)  ← common/evaluation.py ← common/cpp_adapter.py ← C++ engine
  → common/swap.py (refined set)
  → common/evaluation.py (final metrics + per-seed arrays)
  → common/experiment.py (CSV row, checkpoint, mc_runs, manifest)
  → common/analysis.py (read-only tables/figures)
```

### 7.3 Notebook flow

- `00_Setup_Synthetic_Network.ipynb`: load/validate network, verify
  fingerprint, check C++ import.
- `B1`–`B4`: import the shared modules, verify fingerprint, configure, and
  optionally call `ExperimentRunner` for that scenario. Default is no run.
- `05_Analysis_Charts.ipynb`: discover/select a run, call
  `analysis.run_analysis`, display tables/figures. It never simulates.

### 7.4 Resume flow

- Re-running the same `--run-id`:
  - reads existing CSV rows and checkpoints.
  - verifies configuration signatures.
  - skips completed `(scenario, K, objective, mode)` rows.
  - reuses SSA checkpoints.
  - runs only missing modes/configurations.
- If the configuration changed, the signature mismatch raises an error and
  tells the user to use a new run ID.

### 7.5 Analysis flow

```bash
python -c "import sys; sys.path.insert(0,'common'); from analysis import run_analysis; r=run_analysis(run_id='run_20261123_205500_reference_aligned'); print(r['report_path']); print(r['figure_dir'])"
```

Flow:

1. `analysis.select_run` finds the run.
2. `analysis.load_run` concatenates `B1..B4/experiment_results.csv`.
3. `analysis.validate_run` checks schema/fingerprints/duplicates/MC files.
4. `analysis.build_tables` reads `mc_runs/` for paired comparisons.
5. `analysis.build_figures` writes 11 PNGs.
6. `analysis.write_report` writes `summary.md`.
7. Outputs go to timestamped `analysis/tables/...` and `analysis/figures/...`.

---

## 8. Actual results (authoritative run)

Run: `run_20261123_205500_reference_aligned`  
Kind: full  
Started/finished: 2026-11-23 15:23:57 → 15:24:46 UTC (≈48 s wall)  
Completed: 240/240  
Actual simulations: 203,640  
Summed per-config compute: 62.4 s  
All rows `objective_validity = validated`.

### 8.1 Objective ranking by final Net Advantage

Mean across 20 configurations per objective/mode.

| Objective | Mode | Mean Net Adv | 95 % CI | Mean Spread_A | Mean Spread_B | Mean Blocked_B |
|---|---|---:|---:|---:|---:|---:|
| **O2 Competition-aware** | SSA only | **20.55** | [18.14, 23.18] | 214.97 | 194.42 | 16.67 |
| **O2 Competition-aware** | SSA+Swap | **21.97** | [19.39, 24.56] | 215.59 | 193.62 | 17.46 |
| O1 Original SSA | SSA only | 19.52 | [16.97, 22.06] | 212.49 | 192.97 | 18.12 |
| O1 Original SSA | SSA+Swap | 19.54 | [16.68, 22.29] | 213.34 | 193.80 | 17.29 |
| O6 Budget-aware | SSA only | 19.27 | [16.72, 21.85] | 211.56 | 192.29 | 18.79 |
| O6 Budget-aware | SSA+Swap | 19.84 | [17.30, 22.32] | 212.16 | 192.32 | 18.77 |
| O5 Gain_A+blocking | SSA only | 18.62 | [16.37, 21.20] | 210.47 | 191.86 | 19.23 |
| O5 Gain_A+blocking | SSA+Swap | 18.36 | [16.05, 20.82] | 210.12 | 191.76 | 19.33 |
| O4 Combined | SSA only | 17.99 | [15.36, 20.40] | 210.90 | 192.91 | 18.17 |
| O4 Combined | SSA+Swap | 17.71 | [15.17, 20.13] | 210.24 | 192.53 | 18.56 |
| O3 Explicit blocking | SSA only | 15.72 | [13.32, 18.17] | 207.59 | 191.87 | 19.22 |
| O3 Explicit blocking | SSA+Swap | 15.85 | [13.23, 18.40] | 207.33 | 191.49 | 19.60 |

### 8.2 O2 vs the other objectives

40 configuration-level paired differences (4 scenarios × 5 K × 2 modes).
Positive means the other objective beat O2.

| Comparison | Mean difference | 95 % CI | O2 better in | Holm-significant |
|---|---:|---:|---:|---:|
| O1 − O2 | −1.72 | [−3.66, +0.18] | 55.0 % | 11/40 |
| O3 − O2 | −5.47 | [−7.64, −3.22] | 72.5 % | 23/40 |
| O4 − O2 | −3.41 | [−5.48, −1.33] | 72.5 % | 23/40 |
| O5 − O2 | −2.77 | [−4.62, −0.87] | 72.5 % | 15/40 |
| O6 − O2 | −1.70 | [−3.87, +0.47] | 62.5 % | 17/40 |

Interpretation:

- O2 is not distinguishable from O1 or O6 overall.
- O2 is better on average than O3, O4, and O5.
- There is substantial configuration-level heterogeneity; no objective wins
  every cell.
- These tests are conditional on the fixed seed sets and use MC seeds as
  paired replicates; they do not estimate between-optimizer-run variance.

### 8.3 SSA vs SSA+Swap

Across 120 paired configurations:

- Mean `Swap − SSA` = **+0.27** Net Advantage, 95 % CI **[−0.24, +0.78]**.
- Median = **0.00**; 43.3 % positive, 36.7 % negative, 20.0 % zero.
- 8/120 positive-significant, 7/120 negative-significant, rest inconclusive.

By objective:

| Objective | Mean Swap−SSA | Median | Positive-sig | Negative-sig |
|---|---:|---:|---:|---:|
| O1 | +0.02 | −0.21 | 3 | 3 |
| O2 | +1.42 | +0.49 | 4 | 1 |
| O3 | +0.12 | 0.00 | 0 | 1 |
| O4 | −0.28 | 0.00 | 1 | 2 |
| O5 | −0.26 | −0.10 | 0 | 0 |
| O6 | +0.57 | +0.08 | 0 | 0 |

**Conclusion:** SSA+Swap does not reliably improve final Net Advantage. It is
elitist on the 15-seed optimization objective and can overfit it.

Swap acceptance rate: 97/120 (80.8 %).

### 8.4 Blocked_B is not a substitute for Net Advantage

| Objective | Mean Blocked_B | Mean Net Advantage |
|---|---:|---:|
| O3 (maximise Blocked_B) | **19.41** | **15.78** |
| O5 | 19.28 | 18.49 |
| O6 | 18.78 | 19.55 |
| O4 | 18.37 | 17.85 |
| O1 | 17.70 | 19.53 |
| O2 | 17.07 | **21.26** |

- Blocked_B–Net Advantage correlation: Pearson **0.46**, Spearman **0.46**.
- Maximising Blocked_B alone gives the lowest Net Advantage.
- The evaluation metric and the optimizer objective must be kept separate.

### 8.5 Optimizer fitness vs final Net Advantage

Pearson correlation between the 15-seed optimizer score and the 100-seed final
Net Advantage:

| Objective | SSA only | SSA+Swap |
|---|---:|---:|
| O2 | **0.60** | **0.87** |
| O6 | **0.59** | **0.68** |
| O1 | 0.17 | 0.24 |
| O3 | 0.37 | 0.39 |
| O4 | 0.39 | 0.29 |
| O5 | 0.37 | 0.31 |

Only O2 and O6 have a strong fitness–evaluation relationship. Never rank
objectives by internal fitness.

### 8.6 Budget trend for O2

| K | SSA only | SSA+Swap |
|---:|---:|---:|
| 10 | 16.37 | 15.00 |
| 20 | 19.40 | 23.54 |
| 30 | 22.51 | 23.90 |
| 40 | 22.38 | 25.04 |
| 50 | 22.08 | 22.35 |

Generally upward from K=10 to K=40, then flattens/declines slightly at K=50.

### 8.7 Robustness across B1–B4 (O2)

| Scenario | SSA only | SSA+Swap |
|---|---:|---:|
| B1 Random | 18.70 | 19.25 |
| B2 Degree | 18.23 | 22.59 |
| B3 Positive PageRank | 21.69 | 20.68 |
| B4 Community | **23.57** | **25.35** |

O2 is positive across all scenarios; strongest in B4, weakest in B2 under
SSA-only.

### 8.8 Best configuration per scenario/K

| Objective | Times best / 20 |
|---|---:|
| O2 | 6 |
| O6 | 4 |
| O1 | 4 |
| O5 | 3 |
| O3 | 2 |
| O4 | 1 |

Descriptive only; nearby configurations are often within MC uncertainty.

### 8.9 Runtime

- Full run wall time ≈48 s.
- Summed per-config compute 62.4 s.
- 203,640 simulations.
- SSA-only ≈0.20 s/config; SSA+Swap ≈0.32 s/config.
- Benchmark: 40,255 sims/s at 8 threads; 10,739 serial; 692 one-shot API;
  191 Python reference.
- `final_evaluation_runtime_sec` can be near zero on a metric-cache hit; use
  `simulations_for_config` / `simulation_seconds_for_config` for runtime
  comparisons.

### 8.10 Data quality

- 240 unique rows; 0 duplicates.
- 240 MC files ×100 rows; 120 checkpoints; 120 SSA histories.
- One run ID, one file fingerprint, one semantic fingerprint.
- Net Advantage identity holds to ≤3×10⁻¹⁴.
- No NaN in core metric/seed columns; all rows completed.
- SSA-only and Swap share identical SSA sets in all 120 configurations.
- Every seed set has exactly K unique nodes and no opponent seed.
- Analysis validation PASS.

### 8.11 Pre-reference vs reference-aligned sensitivity

The first run (`run_20261123_202600`) used the old candidate-pool random order,
an extra `+3,000,000` in optimization seeds, and the old round counter.

| Quantity | Pre-reference | Reference-aligned |
|---|---:|---:|
| O2 mean Net Advantage | 21.63 | 21.26 |
| O3 mean Net Advantage | 16.38 | 15.78 |
| Highest objective | O2 | O2 |
| Lowest objective | O3 | O3 |
| Blocked_B–Net correlation | 0.36 | 0.46 |
| Swap reliable? | no | no |
| Per-config old/new correlation | — | **0.38** |

Individual configurations changed materially; aggregate conclusions are stable.
The pre-reference run is retained as a sensitivity check.

---

## 9. Tests and validation

Command:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Result: **110 tests, 0 failures, 0 errors.**

Key evidence:

- Network: 1,000 nodes, 7,500 directed arcs, seed-42 byte-identical
  regeneration, fingerprint verification.
- C++: build/import, OpenMP, serial vs parallel bit-identical, error handling,
  old-vs-new spread counts bit-identical.
- Python vs C++: deterministic cases exact; 800-run distributional parity
  within 4 SE + 0.5.
- Objectives: hand-calculated O1–O6; O4 ≠ O5.
- Candidate pool: exact reference pools for B1/K=10, B2/K=30, B3/K=50,
  B4/K=50; four-level tie-break; random part from ranked remainder.
- Opponents: exact B1–B4 orders and nestedness.
- SSA: exact reference trajectory (food, score, per-iteration history).
- Swap: exact reference result (final set, accepted pair, before/after, tests).
- Rounds: exact reference semantics on deterministic tiny graphs.
- Checkpoint/resume: no duplicates, checkpoint reuse for a missing mode,
  signature mismatch rejection, atomic writes.
- Notebooks: valid JSON/nbformat, every code cell compiles, no literal `\n`,
  safe defaults.
- Analysis: validation, tables, figures, corruption detection, path-escape
  rejection.

Smoke test: `SMOKE_TEST_REFERENCE_ALIGNED_20261123_2100` — B1/K=10/O2+O3,
both modes, 4/4, resume and analysis ingestion PASS.

Full run validation: PASS, no issues.

---

## 10. Caveats and open researcher decisions

1. **Synthetic edge orientation.** The reference Bitcoin OTC notebooks reverse
   the raw CSV direction (`target → source`). The synthetic network keeps
   `source → target` as generated and fingerprinted. This is documented, not
   silently changed. Reversing it would be a new network/experiment.
2. **Between-optimizer-run variance.** One SSA run per configuration. MC
   replicates are common random numbers, not independent optimizer runs.
   Between-run variance is not estimated.
3. **External validity.** The synthetic network is denser and less reciprocal
   than typical signed social networks. Results are conditional on this design.
4. **Small-network pool truncation.** The reference raises when `pool_size >
   eligible`; the project truncates and records it. The full network never hits
   this branch.
5. **Zero-rated edges.** The reference supports rating 0 (probability 0,
   polarity 0); the synthetic network has none.
6. **Exact Python/C++ stochastic paths.** Different RNGs; not expected to match
   seed-for-seed. Deterministic cases and distributions are validated.

No code/test blockers remain.

---

## 11. Reproduction commands

```bash
# Build (project-local only; no installation)
cd cpp_engine
python setup.py build_ext --inplace
cd ..

# Tests
python -m unittest discover -s tests -p 'test_*.py'

# Network validation / fingerprint
python -c "import sys; sys.path.insert(0,'common'); from network import load_network, verify_fingerprint; df,G,nodes=load_network(); print(len(nodes), G.number_of_edges()); print(verify_fingerprint())"

# Full experiment plan
python common/run_experiment.py --print-plan

# Smoke test (separate output root)
python common/run_experiment.py --smoke

# Full experiment (checkpointed, resumable)
python common/run_experiment.py --run-id run_YYYYMMDD_HHMMSS

# Resume the same run
python common/run_experiment.py --run-id run_YYYYMMDD_HHMMSS

# Analysis (read-only)
python -c "import sys; sys.path.insert(0,'common'); from analysis import run_analysis; r=run_analysis(run_id='run_20261123_205500_reference_aligned'); print(r['report_path']); print(r['figure_dir'])"
```

Dependencies already present in the original environment: Python 3.14.10,
NumPy 2.4.6, pandas 2.3.3, networkx 3.7, matplotlib 3.10.8, SciPy 1.16.2,
pybind11 3.1.0, g++ 16.2.1, OpenMP, 8 CPUs. No packages were installed by the
agent.

---

## 12. Key paths quick reference

| Item | Path |
|---|---|
| Authoritative full run | `results/run_20261123_205500_reference_aligned/` |
| Pre-reference full run | `results/run_20261123_202600/` |
| Authoritative manifest | `results/run_20261123_205500_reference_aligned/manifest.json` |
| Authoritative analysis tables | `analysis/tables/run_20261123_205500_reference_aligned_20261123_210005/` |
| Authoritative analysis figures | `analysis/figures/run_20261123_205500_reference_aligned_20261123_210005/` |
| Authoritative result summary | `RESULTS_SUMMARY_REFERENCE_ALIGNED.md` |
| Authoritative final audit | `FINAL_AUDIT_REFERENCE_ALIGNED.md` |
| Reference notebook audit | `REFERENCE_NOTEBOOK_AUDIT.md` |
| Authoritative validation gate | `VALIDATION_GATE_REFERENCE_ALIGNED.md` |
| Configuration | `CONFIG.md` |
| Model spec | `cpp_engine/MODEL_SPEC.md` |
| Reference notebooks | `old work/*.ipynb` |
| Extracted reference code | `old work/_extracted/*.py` |
| Tests | `tests/` |
| Logs | `logs/` |
| Backups | `backups/` |

---

## 13. If you are ChatGPT reading this

Use this document as the source of truth. Key points to preserve:

1. **O2 is the best objective by mean final Net Advantage**, but O1 and O6 are
   statistically close; do not claim O2 dominates them.
2. **O3 maximises Blocked_B but has the worst Net Advantage.** Blocked_B is not
   a proxy for the evaluation metric.
3. **SSA+Swap is not reliably better than SSA-only.** The mean is slightly
   positive but the CI crosses zero; per-configuration results are mixed.
4. **Optimizer fitness is not a ranking metric.** Its correlation with final
   Net Advantage is strong only for O2 and O6.
5. **The reference notebooks are the authority** for the model, objectives,
   Blocked_B, candidate pool, B1–B4, SSA, and Swap. The project now matches
   them, with documented differences for synthetic edge orientation, zero-rated
   edges, and small-network truncation.
6. **The pre-reference run is not the authoritative result.** Use
   `run_20261123_205500_reference_aligned` and the
   `_REFERENCE_ALIGNED` reports.
7. **Do not claim 100 % numerical identity** with the original notebooks or the
   first run. The C++ refactor preserves spread counts, but the candidate pool,
   optimization seeds, and round counter were corrected after the reference
   audit.
8. **Between-optimizer-run variance and external validity are not tested.**
   State this limitation clearly.
9. The `common/config.py`, `common/network_validation.py`, and `common/util.py`
   files are unused alternate modules; the active pipeline uses `utils.py`,
   `network.py`, and the other modules listed in §5.2.
10. The synthetic network direction is `source → target` for both trust rating
    and influence. The reference Bitcoin OTC notebooks reverse their raw CSV;
    that reversal was not applied to the synthetic graph.

If continuing the work, the highest-value next steps are:

- Add between-optimizer-run replication (5–10 SSA seeds per configuration) to
  estimate optimizer variability.
- Decide whether the synthetic edge orientation should follow the reference’s
  Bitcoin OTC reversal convention (this would be a new network/experiment).
- Test external validity on real or alternative signed networks.
- Keep the pre-reference run as a sensitivity check only.
