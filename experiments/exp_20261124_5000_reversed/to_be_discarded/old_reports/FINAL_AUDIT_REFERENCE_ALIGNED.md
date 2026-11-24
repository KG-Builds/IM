# Final Audit — Reference-Aligned Implementation and Experiment

Read-only review performed after the reference notebooks were supplied, the
implementation was corrected, and the full experiment was rerun.

Date: 2026-11-23 (local)  
Workspace: `IM_Synthetic_Experiment/`  
Authoritative run: `run_20261123_205500_reference_aligned`  
Status: **240 / 240 configurations completed; all six objectives validated against the supplied reference notebooks; no missing real configurations.**

Supersedes the pre-reference `FINAL_AUDIT.md`.

---

## 1. Executive status

| Item | Status |
|---|---|
| Reference notebooks available and audited | **PASS** (`old work/`, `REFERENCE_NOTEBOOK_AUDIT.md`) |
| Network generated/validated with fixed seed and fingerprint | **PASS** |
| C++ extension compiles and imports | **PASS** |
| Serial vs OpenMP correctness | **PASS** (bit-identical checksums) |
| C++ refactor preserves spread counts | **PASS** (bit-identical to backup) |
| Round counter matches reference | **PASS** (terminal empty round included) |
| Python reference parity | **PASS** (deterministic exact; distributional within tolerance) |
| Six objective formulas | **PASS** (exact reference match) |
| Blocked_B definition | **PASS** (reference-confirmed paired B-only baseline) |
| Candidate pool 10·K, 70/30, four-level ranking, ranked-remainder random part | **PASS** (exact reference match) |
| Opponent B1–B4 strategies | **PASS** (exact reference match) |
| SSA invariants, trajectory and seed schedule | **PASS** (exact reference match) |
| Swap invariants and result | **PASS** (exact reference match) |
| Checkpoint/resume and atomic writes | **PASS** |
| Notebook formatting and execution | **PASS** |
| Full experiment | **PASS** 240/240 |
| Analysis tables/figures | **PASS** (10 tables, 11 figures) |
| Between-optimizer-run variance | **NOT TESTED** (one optimizer run per configuration) |
| External validity | **NOT TESTED** (synthetic network only) |
| Synthetic edge orientation | **RESEARCHER DECISION** (reference reverses Bitcoin OTC edges; synthetic keeps `source → target`) |
| Small-network pool truncation | **DOCUMENTED DIFFERENCE** (reference raises; project truncates; unused on full network) |

---

## 2. Project structure (current)

```
IM_Synthetic_Experiment/
├── README.md / CONFIG.md / CHANGELOG.md / OVERNIGHT_LOG.md
├── AUDIT_REPORT.md / AUDIT_REPORT_ADDENDUM.md
├── REFERENCE_NOTEBOOK_AUDIT.md
├── VALIDATION_GATE.md / VALIDATION_GATE_REFERENCE_ALIGNED.md
├── BENCHMARK.md
├── RESULTS_SUMMARY.md / RESULTS_SUMMARY_REFERENCE_ALIGNED.md
├── FINAL_AUDIT.md / FINAL_AUDIT_REFERENCE_ALIGNED.md
├── old work/
│   ├── competitive_im_no_celf_csv_persist.ipynb
│   ├── competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb
│   └── _extracted/{four_b_code.py, six_variants_code.py}
├── data/ (immutable network + metadata v2)
├── cpp_engine/ (persistent engine, OpenMP, MODEL_SPEC.md)
├── common/ (modular implementation + compatibility facades)
├── notebooks/ (six thin notebooks)
├── tests/ (110 tests, 12 test modules)
├── tools/ (notebook generator, benchmark)
├── results/
│   ├── run_20261123_205500_reference_aligned/  # authoritative
│   ├── run_20261123_202600/                     # pre-reference sensitivity
│   └── smoke/ (isolated smoke runs)
├── logs/
└── analysis/{figures,tables}/...
```

---

## 3. Implementation inventory

| File | Role | Status |
|---|---|---|
| `common/network.py` | load/validate/fingerprint/metadata v2 | PASS |
| `common/network_generator.py` | pinned seed-42 generator | PASS (unchanged) |
| `common/opponents.py` | B1–B4 | PASS |
| `common/candidates.py` | four-level ranking, ranked-remainder 70/30 pool | PASS |
| `common/objectives.py` | O1–O6, budget coefficients, all validated | PASS |
| `common/diffusion.py` | Python reference simulator (reference rounds) | PASS |
| `common/cpp_adapter.py` | persistent C++ adapter | PASS |
| `common/evaluation.py` | MC metrics, paired baseline, safe cache | PASS |
| `common/ssa.py` | discrete SSA (exact reference trajectory) | PASS |
| `common/swap.py` | bounded one-for-one Swap (exact reference result) | PASS |
| `common/stats.py` | bootstrap CIs, paired tests, Holm | PASS |
| `common/experiment.py` | checkpointed coordinator, reference-aligned seeds | PASS |
| `common/run_experiment.py` | CLI runner | PASS |
| `common/analysis.py` | read-only tables/figures | PASS |
| `cpp_engine/diffusion.*` | persistent engine + OpenMP; reference rounds | PASS |
| `cpp_engine/bindings.cpp` | pybind11 API | PASS |
| `cpp_engine/MODEL_SPEC.md` | exact model specification | PASS |
| `notebooks/*.ipynb` | six thin notebooks | PASS |
| `tests/*.py` | 110 tests including reference alignment | PASS |
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

The B notebooks default to `RUN_MODE='none'`; full runs require
`CONFIRM_FULL_RUN=True`. The analysis notebook is read-only.

---

## 6. Configuration used for the authoritative run

| Parameter | Value |
|---|---|
| Run ID | `run_20261123_205500_reference_aligned` |
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
| Optimization seeds | `42*100000 + scenario*10000 + i` |
| Final seeds | `900000000 + i` |
| Candidate pool | 10·K; `n_random=int(round(size*0.30))`; four-level ranking; ranked-remainder random part |
| Swap max tests | 100 |
| Threads | 0 (OpenMP default = 8) |
| Evaluation protocol | `paired_b_only_baseline_v1` |
| Network file SHA-256 | `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5` |
| Semantic SHA-256 | `461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25` |

---

## 7. Exact result files generated

### Authoritative run

- `results/run_20261123_205500_reference_aligned/manifest.json` — 240/240, `completed`
- `results/run_20261123_205500_reference_aligned/run_metadata.json`
- `results/run_20261123_205500_reference_aligned/B1..B4/experiment_results.csv` — 60 rows each
- 120 checkpoint JSON files
- 120 SSA history CSVs
- 240 per-seed MC CSVs (100 rows each)
- 4 `scenario_metadata.json` files

### Analysis

- `analysis/tables/run_20261123_205500_reference_aligned_20261123_210005/`:
  `overall.csv`, `per_scenario.csv`, `per_k.csv`, `mode_comparison.csv`,
  `objective_vs_reference.csv`, `objective_ranking.csv`, `runtime.csv`,
  `variability.csv`, `robustness.csv`, `best_config.csv`, `summary.md`
- `analysis/figures/run_20261123_205500_reference_aligned_20261123_210005/`:
  `net_advantage_vs_K_ssa_only.png`, `net_advantage_vs_K_ssa_swap.png`,
  `swap_vs_ssa_paired_differences.png`, `spread_A_vs_spread_B.png`,
  `blocked_B_reference_confirmed.png`, `runtime_breakdown.png`,
  `objective_ranking_net_advantage.png`, `robustness_across_scenarios.png`,
  `variability_configurations.png`, `objective_scenario_heatmap_ssa_only.png`,
  `objective_scenario_heatmap_ssa_swap.png`
- `RESULTS_SUMMARY_REFERENCE_ALIGNED.md` — narrative summary of actual data

### Smoke (isolated)

- `results/smoke/SMOKE_TEST_REFERENCE_ALIGNED_20261123_2100/` — 4/4
- Analysis tables/figures for that smoke run under `analysis/`.

### Pre-reference sensitivity

- `results/run_20261123_202600/` — 240/240, retained for comparison
- `analysis/tables/run_20261123_202600_20261123_202328/`
- `analysis/figures/run_20261123_202600_20261123_202328/`

---

## 8. Validation results and test commands

```bash
python -m unittest discover -s tests -p 'test_*.py'
# Ran 110 tests in 9.0s — OK
```

Reference-alignment tests (`tests/test_reference_alignment.py`) check exact
candidate pools, seeds, objectives, SSA trajectory, Swap result and round
semantics against the supplied notebooks.

Reference-aligned smoke: `SMOKE_TEST_REFERENCE_ALIGNED_20261123_2100` — 4/4,
resume, analysis ingestion all PASS.

Authoritative run validation: **PASS**, no issues:
- 240 unique rows; no duplicates
- one run ID, one file fingerprint, one semantic fingerprint
- `Net Advantage = Spread_A − Spread_B` to ≤3×10⁻¹⁴
- no NaN in core metric/seed columns
- all rows `completed` and `objective_validity = validated`
- 240 MC files, 100 rows each
- 120 checkpoints, 120 SSA histories
- SSA-only and Swap rows share identical SSA seed sets in all 120 configurations
- every seed set has exactly K unique nodes; no opponent-seed overlap
- no temporary files left behind
- console log contains no errors/exceptions

---

## 9. Reference-notebook findings and corrections

Full detail: `REFERENCE_NOTEBOOK_AUDIT.md`.

### Exact matches
- Diffusion model states, contact memory, stifling, negative-edge flip,
  strength-based conflict resolution, synchronous updates, final spread counts.
- Blocked_B paired B-only baseline.
- O1–O6 formulas and budget coefficients.
- B1–B4 rankings, tie-breaks, Louvain/PageRank settings.
- SSA movement, repair, elitism, population/iteration/mutation.
- Swap sampled one-for-one, single best improving, elitist.
- Final evaluation with 100 common random seeds.

### Corrections applied
1. **Candidate-pool ranking**: exact four-level key
   `(-net, -positive, negative, id)` (was `(-net, id)`).
2. **Random-pool source**: sampled from `ranked[n_ranked:]` (was node-id order).
3. **Pool rounding**: `n_random=int(round(size*fraction))`, `n_ranked=size-n_random`.
4. **Optimization seeds**: removed an erroneous `+3,000,000`; now
   `master_seed*100000 + scenario*10000 + i`.
5. **Round counter**: now includes the terminal empty round
   (`executed_rounds + 1`), matching the reference.
6. **Blocked_B validity**: upgraded to `validated`; formula was already correct.
7. **SSA history**: added `unique_solutions`.

### Documented differences (not changed)
- **Synthetic edge orientation**: reference reverses Bitcoin OTC edges
  (`target → source`); synthetic network keeps `source → target` as generated
  and fingerprinted. Reversing would be a new network.
- **Zero-rated edges**: reference supports rating 0 (probability 0, polarity 0);
  synthetic network has no zero ratings.
- **Small-network pool truncation**: reference raises when `pool_size > eligible`;
  project truncates and records it. Unused on the full network.
- **SSA seed in older four-B notebook**: four-B uses
  `master_seed + 30000 + scenario`; the six-objective notebook (latest reference)
  uses the per-K/per-variant formula, which is used here.

---

## 10. C++/Python parity findings

| Check | Result |
|---|---|
| Refactored engine vs backed-up original, 30 full-network runs | **Spread counts bit-identical**; `rounds` now old+1 by design |
| Functional API vs persistent engine | **Exact** |
| Serial vs 2/4/8-thread batch | **Bit-identical checksums** |
| Batch order independence | **PASS** |
| Python vs C++ deterministic tiny graphs | **Exact**, including reference round semantics |
| Python vs C++ distributional comparison (800 runs) | **Within 4 SE + 0.5** for spread_A, spread_B, ignorant, rounds |
| Exact Python/C++ stochastic path | **NOT TESTED / not expected** (different RNGs) |

---

## 11. Benchmark / speedup

From `BENCHMARK.md` (full network, 2,000 simulations per thread setting):

| Threads | Sims/s | Speedup vs serial |
|---:|---:|---:|
| 1 | 10,739 | 1.00× |
| 2 | 18,620 | 1.73× |
| 4 | 29,239 | 2.72× |
| 8 | 40,255 | 3.75× |

- One-shot API: 692 sims/s; persistent 1-thread: 10,739 sims/s → **15.5×**.
- Python reference: 191 sims/s.
- Full-design upper bound: 443,900 simulations; measured run executed 203,640
  simulations (cache reuse) in ≈48 s wall time.
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
- Missing real configurations: **none**
- Wall time: 2026-11-23 15:23:57 → 15:24:46 UTC (≈48 s)
- Summed per-config compute: 62.4 s
- Actual simulations: 203,640
- C++/OpenMP: available and enabled
- No competing jobs, no concurrent full runs

---

## 13. Result summary (actual data only)

Detailed tables and CIs: `RESULTS_SUMMARY_REFERENCE_ALIGNED.md`.

1. **O2 (Competition-aware) has the highest mean final Net Advantage**:
   20.55 (SSA only) and 21.97 (SSA+Swap).
2. **O1 and O6 are statistically indistinguishable from O2 overall**
   (O1−O2 mean −1.72, 95 % CI [−3.66, +0.18]; O6−O2 mean −1.70, 95 % CI
   [−3.87, +0.47]).
3. **O2 is better on average than O3, O4 and O5**; their mean per-config
   differences have CIs excluding zero. Configuration-level heterogeneity is
   substantial; no objective dominates every cell.
4. **O3 (maximise Blocked_B) has the highest mean Blocked_B (19.41) but the
   lowest mean Net Advantage (15.78).** Blocked_B–Net Advantage correlation is
   0.46.
5. **SSA+Swap is not a reliable improvement**: mean difference +0.27, 95 % CI
   [−0.24, +0.78], median 0.00, 43 % positive, 8/120 positive-significant and
   7/120 negative-significant.
6. **Optimizer fitness is not a valid ranking metric**: correlation with final
   Net Advantage is strong for O2 (0.60–0.87) and O6 (0.59–0.68), weak for
   O1/O3/O4/O5.
7. **O2 is positive across all scenarios**; strongest in B4 (23.57 SSA-only,
   25.35 Swap), weakest in B2 SSA-only (18.23).
8. **Runtime is modest**: ≈0.20 s per SSA-only configuration, ≈0.32 s per
   SSA+Swap configuration; full 240-configuration run ≈48 s wall time.

These are results for this synthetic network and design, not claims about other
networks or real social systems.

---

## 14. Missing configurations and why

- **Real experiment:** none; all 240 completed.
- **Smoke:** separate under `results/smoke/`; 4/4.
- **Reference comparison:** completed; see `REFERENCE_NOTEBOOK_AUDIT.md`.
- **Between-optimizer-run replication:** not run (one optimizer run per
  configuration by design). This is a limitation, not a missing configuration.
- **Alternative synthetic edge orientation:** not run; documented researcher
  decision.

---

## 15. Classification summary

### PASS
- Reference-notebook audit and exact alignment tests
- Network generation/validation/fingerprint/seed-42 reproduction
- C++ build/import/OpenMP
- Serial vs parallel correctness
- Spread-count parity with the original engine
- Deterministic diffusion and Python/C++ distributional parity
- Reference round semantics
- Objective formulas and Blocked_B
- Candidate pool composition/ranking/random part
- Opponent B1–B4
- SSA trajectory and seed schedule
- Swap result and invariants
- Checkpoint/resume/atomic writes/signature rejection
- Notebook JSON/syntax/execution
- Smoke test and analysis ingestion
- Full experiment 240/240
- Analysis validation and outputs
- Benchmark

### RESEARCHER DECISION REQUIRED
- Synthetic edge orientation: reference reverses Bitcoin OTC edges; synthetic
  network uses `source → target` as generated/fingerprinted.
- Whether between-optimizer-run replication is required for publication.
- Whether external validity on real/alternative signed networks is needed.

### NOT TESTED
- Between-optimizer-run variance / multiple optimizer seeds
- External validity on real signed networks
- Exact Python/C++ RNG-path equality (different RNGs; not expected)
- Formal convergence of SSA (population 12, 10 iterations is the reference design)

### FAIL
- None in the implemented and tested scope.

---

## 16. Limitations and next steps

1. **Add between-optimizer-run replication** (e.g., 5–10 SSA seeds per
   configuration) to estimate optimizer variability and enable proper
   between-run significance tests. This is the main statistical limitation.
2. **Decide the synthetic edge orientation** relative to the reference's
   Bitcoin OTC convention. The current network is documented and unchanged.
3. **Test external validity** on real signed networks or alternative synthetic
   topologies; the current graph is dense and only weakly reciprocal.
4. **Consider iterative Swap** as a separate, clearly labelled mode; the current
   single-best-swap policy matches the reference.
5. **Consider larger SSA budgets** or convergence diagnostics; the reference
   uses population 12 and 10 iterations.
6. **Keep the pre-reference run** as a sensitivity check; do not mix it with the
   authoritative results.

---

## 17. Final statement

The implementation now matches the supplied reference notebooks on the diffusion
model, Blocked_B, all six objectives, the candidate pool, B1–B4, SSA, Swap and
the seed schedule. The corrected full experiment completed 240/240
configurations. The primary validated result is that the Competition-aware
objective (O2) has the highest mean final Net Advantage, with O1 and O6 close
behind and not statistically distinguishable; explicit Blocked_B maximisation
(O3) performs worst on Net Advantage. SSA+Swap does not reliably improve the
final metric. The remaining limitations are between-optimizer-run variance,
external validity, and the documented synthetic-orientation choice.
