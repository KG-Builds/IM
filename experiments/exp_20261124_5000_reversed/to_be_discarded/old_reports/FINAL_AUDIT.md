# Final Audit — Signed Competitive Influence Maximization

> **Superseded:** this is the pre-reference audit. The reference notebooks were
> supplied later and three material discrepancies were corrected. See
> `FINAL_AUDIT_REFERENCE_ALIGNED.md`, `RESULTS_SUMMARY_REFERENCE_ALIGNED.md`,
> and `REFERENCE_NOTEBOOK_AUDIT.md` for the authoritative review.

Read-only review performed after the full experiment.  
Date: 2026-11-23 (local)  
Workspace: `IM_Synthetic_Experiment/` (only location inspected)  
Full run: `run_20261123_202600`  
Status: **240 / 240 configurations completed; analysis generated; no missing real configurations.**

---

## 1. Executive status

| Item | Status |
|---|---|
| Network generated/validated with fixed seed and fingerprint | **PASS** |
| C++ extension compiles and imports | **PASS** |
| Serial vs OpenMP correctness | **PASS** (bit-identical checksums) |
| C++ refactor preserves original model/RNG stream | **PASS** (30/30 full-network runs identical to backup) |
| Python reference parity | **PASS** (deterministic exact; distributional within tolerance) |
| Six objective formulas implemented and hand-tested | **PASS** |
| Blocked_B scientific definition | **RESEARCHER DECISION REQUIRED** (provisional) |
| Candidate pool 10·K, 70/30, polarity-aware | **PASS** |
| Opponent B1–B4 strategies | **PASS** |
| SSA invariants and determinism | **PASS** |
| Swap invariants and bounded/elitist behaviour | **PASS** |
| Checkpoint/resume and atomic writes | **PASS** |
| Notebook formatting and syntax | **PASS** |
| Full experiment | **PASS** 240/240 |
| Analysis tables/figures | **PASS** (10 tables, 11 figures) |
| Reference-notebook fidelity | **NOT TESTED** (reference notebooks absent) |
| Between-optimizer-run variance / external validity | **NOT TESTED** |

---

## 2. Final project structure

```
IM_Synthetic_Experiment/
├── README.md
├── CONFIG.md
├── CHANGELOG.md
├── OVERNIGHT_LOG.md
├── AUDIT_REPORT.md
├── AUDIT_REPORT_ADDENDUM.md
├── VALIDATION_GATE.md
├── BENCHMARK.md
├── RESULTS_SUMMARY.md
├── FINAL_AUDIT.md
├── data/
│   ├── synthetic_network_1000_7500.csv
│   ├── network_metadata.json
│   └── network_metadata_v2.json
├── cpp_engine/
│   ├── diffusion.hpp / diffusion.cpp
│   ├── bindings.cpp
│   ├── setup.py / Makefile
│   ├── MODEL_SPEC.md / README.md
│   └── simulation_engine.cpython-314-x86_64-linux-gnu.so
├── common/
│   ├── utils.py / network.py / network_generator.py
│   ├── opponents.py / candidates.py / objectives.py
│   ├── diffusion.py / cpp_adapter.py / evaluation.py
│   ├── ssa.py / swap.py / stats.py
│   ├── experiment.py / run_experiment.py
│   ├── analysis.py
│   └── experiment_core.py / scenario_runner.py  (compatibility facades)
├── notebooks/  (six notebooks)
├── tests/      (12 test modules, 102 tests)
├── tools/
│   ├── build_notebooks.py
│   └── benchmark.py
├── results/
│   ├── run_20261123_202600/
│   │   ├── run_metadata.json
│   │   ├── manifest.json
│   │   ├── B1/ … B4/
│   │   │   ├── experiment_results.csv
│   │   │   ├── scenario_metadata.json
│   │   │   ├── checkpoints/ (30 JSON per scenario)
│   │   │   ├── mc_runs/ (60 CSV per scenario)
│   │   │   └── ssa_history/ (30 CSV per scenario)
│   └── smoke/
│       └── SMOKE_TEST_20261123_202100/  (separate from real results)
├── logs/
│   ├── run_20261123_202600.log
│   ├── full_run_console.log
│   └── benchmark_20261123_202024.json
└── analysis/
    ├── figures/run_20261123_202600_20261123_202328/  (11 PNG)
    └── tables/run_20261123_202600_20261123_202328/   (10 CSV + summary.md)
```

---

## 3. Implementation inventory

| File | Role | Status |
|---|---|---|
| `common/network.py` | load/validate/fingerprint/metadata v2 | PASS |
| `common/network_generator.py` | pinned seed-42 generator | PASS (unchanged, byte-identical) |
| `common/opponents.py` | B1–B4 | PASS |
| `common/candidates.py` | 10·K, 70/30, polarity-aware pool | PASS |
| `common/objectives.py` | O1–O6, budget coefficients, validity | PASS (Blocked_B provisional) |
| `common/diffusion.py` | Python reference simulator | PASS |
| `common/cpp_adapter.py` | persistent C++ adapter | PASS |
| `common/evaluation.py` | MC metrics, paired baseline, safe cache | PASS |
| `common/ssa.py` | discrete SSA | PASS |
| `common/swap.py` | bounded one-for-one Swap | PASS |
| `common/stats.py` | bootstrap CIs, paired tests, Holm | PASS |
| `common/experiment.py` | checkpointed coordinator | PASS |
| `common/run_experiment.py` | CLI runner | PASS |
| `common/analysis.py` | read-only tables/figures | PASS |
| `cpp_engine/diffusion.*` | persistent engine + OpenMP batch | PASS |
| `cpp_engine/bindings.cpp` | pybind11 API | PASS |
| `cpp_engine/MODEL_SPEC.md` | exact model specification | PASS |
| `notebooks/*.ipynb` | six thin notebooks | PASS (syntax/format) |
| `tests/*.py` | 102 tests | PASS |
| `tools/build_notebooks.py` | notebook generator | PASS |
| `tools/benchmark.py` | bounded benchmark | PASS |

---

## 4. Build, test and run instructions

```bash
# Build (project-local only; no installation)
cd cpp_engine
python setup.py build_ext --inplace
cd ..

# Tests
python -m unittest discover -s tests -p 'test_*.py'

# Network validation / fingerprint
python -c "import sys; sys.path.insert(0,'common'); from network import load_network, verify_fingerprint; df,G,nodes=load_network(); print(len(nodes), G.number_of_edges()); print(verify_fingerprint())"

# Full experiment plan (240 configurations)
python common/run_experiment.py --print-plan

# Smoke test (separate output root)
python common/run_experiment.py --smoke

# Full experiment (checkpointed, resumable)
python common/run_experiment.py --run-id run_YYYYMMDD_HHMMSS

# Resume
python common/run_experiment.py --run-id run_YYYYMMDD_HHMMSS

# Analysis (read-only)
python -c "import sys; sys.path.insert(0,'common'); from analysis import run_analysis; r=run_analysis(); print(r['report_path']); print(r['figure_dir'])"
```

---

## 5. Notebook order

1. `notebooks/00_Setup_Synthetic_Network.ipynb`
2. `notebooks/B1_Random_opponent.ipynb`
3. `notebooks/B2_Degree-based_opponent.ipynb`
4. `notebooks/B3_Positive_PageRank_opponent.ipynb`
5. `notebooks/B4_Community-concentrated_opponent.ipynb`
6. `notebooks/05_Analysis_Charts.ipynb`

The B notebooks default to `RUN_MODE='none'` and require
`CONFIRM_FULL_RUN=True` for a full scenario. The analysis notebook is read-only
and never launches simulations.

---

## 6. Configuration used for the full run

| Parameter | Value |
|---|---|
| Run ID | `run_20261123_202600` |
| Run kind | `full` |
| Scenarios | B1, B2, B3, B4 |
| K | 10, 20, 30, 40, 50 |
| Objectives | O1, O2, O3, O4, O5, O6 |
| Modes | SSA only; SSA + Swap refinement |
| Configurations | 240 |
| Master seed | 42 |
| SSA population / iterations / mutation | 12 / 10 / 0.20 |
| Optimization MC runs | 15 |
| Final evaluation MC runs | 100 |
| Candidate pool | 10·K, 70 % ranked / 30 % random |
| Swap max tests | 100 |
| Threads | 0 (OpenMP default = 8) |
| Evaluation protocol | `paired_b_only_baseline_v1` |
| Network file SHA-256 | `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5` |
| Semantic SHA-256 | `461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25` |

---

## 7. Exact result files generated

### Full run

- `results/run_20261123_202600/manifest.json` — 240/240, status `completed`
- `results/run_20261123_202600/run_metadata.json` — config, environment, network, C++ status
- `results/run_20261123_202600/B1/experiment_results.csv` — 60 rows
- `results/run_20261123_202600/B2/experiment_results.csv` — 60 rows
- `results/run_20261123_202600/B3/experiment_results.csv` — 60 rows
- `results/run_20261123_202600/B4/experiment_results.csv` — 60 rows
- 120 checkpoint JSON files
- 120 SSA history CSVs
- 240 per-seed MC CSVs (100 rows each)
- 4 `scenario_metadata.json` files

### Analysis

- `analysis/tables/run_20261123_202600_20261123_202328/`:
  `overall.csv`, `per_scenario.csv`, `per_k.csv`, `mode_comparison.csv`,
  `objective_vs_reference.csv`, `objective_ranking.csv`, `runtime.csv`,
  `variability.csv`, `robustness.csv`, `best_config.csv`, `summary.md`
- `analysis/figures/run_20261123_202600_20261123_202328/`:
  `net_advantage_vs_K_ssa_only.png`, `net_advantage_vs_K_ssa_swap.png`,
  `swap_vs_ssa_paired_differences.png`, `spread_A_vs_spread_B.png`,
  `blocked_B_provisional.png`, `runtime_breakdown.png`,
  `objective_ranking_net_advantage.png`, `robustness_across_scenarios.png`,
  `variability_configurations.png`, `objective_scenario_heatmap_ssa_only.png`,
  `objective_scenario_heatmap_ssa_swap.png`
- `RESULTS_SUMMARY.md` — narrative summary of the actual data

### Smoke (kept separate)

- `results/smoke/SMOKE_TEST_20261123_202100/` — 4/4 configurations
- `analysis/tables/SMOKE_TEST_20261123_202100_20261123_202038/`
- `analysis/figures/SMOKE_TEST_20261123_202100_20261123_202038/`

---

## 8. Validation results and test commands

```bash
python -m unittest discover -s tests -p 'test_*.py'
# Ran 102 tests in 8.697s — OK
```

Gate report: `VALIDATION_GATE.md` — all implementation gates PASS; the only
conditional item is the Blocked_B scientific definition (provisional).

Notebook execution validation: `00_Setup_Synthetic_Network.ipynb`,
`B1_Random_opponent.ipynb` (default `RUN_MODE='none'`) and
`05_Analysis_Charts.ipynb` were executed end-to-end with **0 errors**. The
analysis notebook selected the full run and produced an additional timestamped
output set (`analysis/.../run_20261123_202600_20261123_202728/`); the primary
output set referenced throughout this audit is
`analysis/.../run_20261123_202600_20261123_202328/`.

Full-run analysis validation: **PASS**, no issues:
- 240 unique rows; no duplicates
- one run ID, one file fingerprint, one semantic fingerprint
- `Net Advantage = Spread_A − Spread_B` to ≤3×10⁻¹⁴
- no NaN in core metric/seed columns
- all rows `completed`
- 240 MC files, 100 rows each
- 120 checkpoints, 120 SSA histories
- SSA-only and Swap rows share identical SSA seed sets in all 120 configurations
- every seed set has exactly K unique nodes; no opponent-seed overlap
- no temporary files left behind
- console log contains no errors/exceptions

---

## 9. Research correctness findings

### 9.1 Source of truth
- No reference notebooks named in the master prompt exist in the workspace.
- Definitions were audited against the starter code and the master prompt.
- Original files were backed up; the network and generator were not overwritten.

### 9.2 Network
- 1,000 nodes, 7,500 unique directed arcs, no self-loops, no duplicate arcs.
- 7,498 underlying unordered pairs + 2 reciprocal pairs.
- 3,720 positive / 3,780 negative arcs; probability classes 2,278 / 2,950 / 2,272.
- Seed-42 regeneration is byte-identical; file and semantic fingerprints verified.

### 9.3 Diffusion model
- States, contact memory, stifling, negative-edge rumor flip, strength-based
  conflict resolution, synchronous updates and termination are implemented as
  specified in `cpp_engine/MODEL_SPEC.md`.
- Deterministic tiny-graph tests cover no edges, positive/negative edges,
  probability classes, A/B reachability, conflict, stifling, fully informed and
  termination.
- C++ and Python reference agree exactly on deterministic cases and within 4 SE
  on distributional comparisons.

### 9.4 Objectives
- O1–O6 formulas and budget coefficients are hand-tested.
- O4 ≠ O5.
- O2 is fully validated. O1/O3/O4/O5/O6 are `provisional_blocked_b`.

### 9.5 Blocked_B
- Implemented as the explicit paired B-only baseline with common random seeds.
- Unit-tested; correlation with Net Advantage is only 0.36.
- Maximising Blocked_B alone (O3) gives the lowest Net Advantage.
- Scientific definition remains **RESEARCHER DECISION REQUIRED**.

### 9.6 Candidate pool / opponents / SSA / Swap
- Pool is 10·K with 70/30 composition and no opponent seeds; negative edges
  reduce candidate scores.
- B1–B4 ranking/tie/community logic verified; orders nested across K.
- SSA returns exactly K unique pool nodes, preserves best-so-far, is deterministic
  under the same seed, and shares its set across modes.
- Swap is one-for-one, bounded, elitist, preserves K and excludes opponent seeds.

### 9.7 Statistics
- Final evaluation uses 100 common random seeds across all configurations.
- Paired comparisons are by MC seed within the same scenario/K/objective.
- MC replicates are **not** treated as independent optimizer runs.
- Between-optimizer-run variance is not estimated (one optimizer run per
  configuration).

---

## 10. C++/Python parity findings

| Check | Result |
|---|---|
| Refactored persistent engine vs backed-up original engine, 30 full-network runs | **Bit-identical** (same RNG stream preserved) |
| Functional API vs persistent engine, 30 random small-graph runs | **Exact** |
| Serial vs 2/4/8-thread batch | **Bit-identical checksums** |
| Batch order independence | **PASS** |
| Python vs C++ deterministic tiny graphs | **Exact** |
| Python vs C++ distributional comparison (800 runs, random graph) | **Within 4 SE + 0.5** for spread_A, spread_B, ignorant, rounds |
| Python vs C++ exact stochastic path | **NOT TESTED / not expected** (different RNGs) |

The Python reference is a readable model check, not a seed-for-seed replica of
the C++ RNG.

---

## 11. Benchmark / speedup

From `BENCHMARK.md` (2,000 simulations per thread setting, full network):

| Threads | Sims/s | Speedup vs serial |
|---:|---:|---:|
| 1 | 10,739 | 1.00× |
| 2 | 18,620 | 1.73× |
| 4 | 29,239 | 2.72× |
| 8 | 40,255 | 3.75× |

- One-shot API (rebuilds graph per call): 692 sims/s.
- Persistent 1-thread: 10,739 sims/s → **15.5×** faster than one-shot.
- Python reference: 191 sims/s.
- Full-design upper-bound estimate: 443,900 simulations; measured run executed
  202,800 simulations (cache reuse) in ≈48 s wall time.
- `final_evaluation_runtime_sec` can be near zero on a metric-cache hit; use
  `simulations_for_config` / `simulation_seconds_for_config` for runtime
  comparisons.

---

## 12. Full experiment completion status

| Scenario | Completed | Expected |
|---|---:|---:|
| B1 | 60 | 60 |
| B2 | 60 | 60 |
| B3 | 60 | 60 |
| B4 | 60 | 60 |
| **Total** | **240** | **240** |

- Status: `completed`
- Missing configurations: **none**
- Wall time: 2026-11-23 14:52:17 → 14:53:05 UTC (≈48 s)
- Summed per-config compute: 60.3 s
- Actual simulations: 202,800
- C++/OpenMP: available and enabled
- No competing jobs, no concurrent full runs

---

## 13. Result summary (actual data only)

Detailed tables and CIs are in `RESULTS_SUMMARY.md`. Key points:

1. **O2 (Competition-aware) has the highest mean final Net Advantage** in both
   modes: 21.62 (SSA only) and 21.64 (SSA+Swap), across 20 configurations each.
2. **O1 is statistically indistinguishable from O2 overall** (mean difference
   −1.04, 95 % CI [−2.69, +0.57]); O2 is better in 55 % of configurations.
3. **O2 is better on average than O3/O4/O5/O6**; the 95 % CIs of the mean
   per-config differences exclude zero. There is substantial configuration-level
   heterogeneity, including cells where O6 or O3 beat O2.
4. **O3 (maximise Blocked_B) has the highest mean Blocked_B but the lowest mean
   Net Advantage.** Blocked_B–Net Advantage correlation is only 0.36.
5. **SSA+Swap does not reliably improve final Net Advantage.** Mean difference
   +0.51, 95 % CI [+0.13, +0.89], but median 0.00, 45 % positive, only 5/120
   positive-significant and 2/120 negative-significant comparisons. The positive
   mean is driven by a few large gains; Swap can overfit the 15-seed optimization
   objective.
6. **Optimizer fitness is not a valid ranking metric.** Its correlation with
   final Net Advantage is strong only for O2 (0.71–0.80) and near zero for
   O1/O3.
7. **O2 improves with K overall** (15.28 at K=10 to 25.48 at K=50, SSA only),
   with a dip at K=30.
8. **O2 is positive across all four opponent scenarios**; B3 is easiest for
   SSA-only (23.70), B4 hardest (19.26).
9. **Runtime is modest:** ≈0.19 s per SSA-only configuration and ≈0.30 s per
   SSA+Swap configuration; the full 240-configuration run took ≈48 s wall time.

These are results for this synthetic network and design. They are not claims
about other networks or real social systems.

---

## 14. Missing configurations and why

- **Real experiment:** none. All 240 planned configurations completed.
- **Smoke test:** separate by design (`results/smoke/`), 4/4 completed.
- **Blocked_B-dependent objectives:** all 200 configurations completed, but
  labelled `provisional_blocked_b`; they are not missing, they are exploratory.
- **Reference-notebook comparison:** not possible because no reference notebooks
  exist in the workspace.
- **Between-optimizer-run replication:** not run (one optimizer run per
  configuration by design). This is a limitation, not a missing configuration.

---

## 15. Classification summary

### PASS
- Network generation/validation/fingerprint/seed-42 reproduction
- C++ build/import/OpenMP
- Serial vs parallel correctness
- Old vs new C++ engine bit-identical regression
- Deterministic diffusion tests and Python/C++ distributional parity
- Objective formulas and hand calculations
- Candidate pool composition/polarity/B-exclusion
- Opponent B1–B4 logic/nesting
- SSA/Swap invariants, determinism, elitism
- Checkpoint/resume/atomic writes/signature rejection
- Notebook JSON/syntax/real line breaks
- Smoke test 4/4, resume, analysis ingestion
- Full experiment 240/240
- Analysis validation and outputs
- Benchmark

### RESEARCHER DECISION REQUIRED
- Blocked_B scientific definition (provisional paired B-only baseline)
- SSA/Swap fidelity to the missing intended reference implementation
- Diffusion fidelity to the missing intended reference notebook
- Whether the 7,500 count should mean directed arcs (current) or undirected
  links; the current convention is documented and met exactly

### NOT TESTED
- Reference-notebook parity (notebooks absent)
- Between-optimizer-run variance / multiple optimizer seeds
- External validity on real signed networks
- Exact Python/C++ RNG-path equality (different RNGs; not expected)
- Formal convergence of SSA (population 12, 10 iterations is the starter design)

### FAIL
- None in the implemented and tested scope.

---

## 16. Limitations and next steps

1. **Obtain the missing reference notebooks** and compare model, SSA, candidate,
   Blocked_B and Swap definitions line by line. This is the highest-value next
   step for scientific validity.
2. **Resolve Blocked_B** with the researcher. If a different baseline or
   definition is intended, rerun the five dependent objectives; O2 results stand.
3. **Add between-optimizer-run replication** (e.g., 5–10 SSA seeds per
   configuration) to estimate optimizer variability and enable proper
   between-run significance tests. This is the main statistical limitation.
4. **Consider iterative Swap** as a separate, clearly labelled mode; the current
   single-best-swap policy is preserved from the starter.
5. **Test external validity** on real signed networks or alternative synthetic
   topologies; the current graph is dense and only weakly reciprocal.
6. **Consider larger SSA budgets** or convergence diagnostics; the starter uses
   population 12 and 10 iterations.
7. **Do not present O1/O3/O4/O5/O6 as validated** until Blocked_B is resolved.

---

## 17. Final statement

The implementation, validation and full 240-configuration experiment are
complete for the scope that can be verified inside this workspace. The primary
validated result is that the Competition-aware objective (O2) achieves the
highest mean final Net Advantage, with O1 close behind, while explicit
Blocked_B maximisation (O3) performs worst on Net Advantage. SSA+Swap shows only
a small, heavy-tailed average gain and is not a reliable per-configuration
improvement. The Blocked_B-dependent objectives and reference-notebook fidelity
remain researcher decisions, and the report does not claim otherwise.
