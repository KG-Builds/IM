# Validation Gate — Reference-Aligned Implementation

Generated: 2026-11-23 (local)
Workspace: `IM_Synthetic_Experiment/`
Decision: **FULL EXPERIMENT ALLOWED AND COMPLETED. All six objectives are
validated against the supplied reference notebooks.**

Supersedes the pre-reference `VALIDATION_GATE.md`.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.14.10 |
| NumPy / pandas / networkx / matplotlib / SciPy | 2.4.6 / 2.3.3 / 3.7 / 3.10.8 / 1.16.2 |
| pybind11 | 3.1.0 |
| Compiler | g++ 16.2.1 |
| Python headers | `/usr/include/python3.14/Python.h` present |
| OpenMP | compiled and available |
| CPU count | 8 |
| C++ extension | imports |

No packages were installed, upgraded, or removed. No system configuration was changed.

---

## 2. Reference notebooks and alignment

Supplied in `old work/`:

- `competitive_im_no_celf_csv_persist.ipynb` — four-B Net Advantage notebook
- `competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb` — six-objective Gain_A/Blocked_B notebook; latest reference

Full comparison: `REFERENCE_NOTEBOOK_AUDIT.md`.

### Corrections made after the reference audit

| Discrepancy | Correction |
|---|---|
| Candidate-pool ranking used only `(-net, id)` | Now exact four-level `(-net, -positive, negative, id)` |
| Random pool part sampled from node-id order | Now sampled from `ranked[n_ranked:]` (ranked remainder) |
| Pool rounding `int(size*(1-fraction))` | Now `n_random=int(round(size*fraction))`, `n_ranked=size-n_random` |
| Optimization seeds had an extra `+3,000,000` | Now `master_seed*100000 + scenario*10000 + i` |
| `rounds` excluded the terminal empty round | Now `executed_rounds + 1` (reference semantics) |
| Blocked_B labelled provisional | Now `validated`; reference confirms the exact paired baseline |
| SSA history lacked `unique_solutions` | Added |

Spread counts, objectives, B1–B4, SSA movement, Swap and final evaluation were
already exact matches.

---

## 3. Gate checklist

| Gate | Status | Evidence |
|---|---|---|
| Network tests | **PASS** | `test_network.py` (9 tests); seed-42 byte-identical |
| C++ compile/import | **PASS** | `python setup.py build_ext --inplace`; import + OpenMP |
| Diffusion tests | **PASS** | `test_diffusion_python.py` + `test_cpp_engine.py` |
| Python/C++ parity | **PASS** | Deterministic exact; 800-run distributional within 4 SE; old/new spread counts bit-identical |
| Reference alignment | **PASS** | `test_reference_alignment.py` (8 tests): pools, seeds, objectives, SSA trajectory, Swap, rounds |
| Objective tests | **PASS** | `test_objectives.py` (7 tests); hand-calculated O1–O6; all validated |
| Candidate-pool tests | **PASS** | `test_candidates.py` (7 tests) + exact full-network reference pools |
| Opponent tests | **PASS** | `test_opponents.py` (10 tests) |
| SSA tests | **PASS** | `test_ssa_swap.py` (10 tests) + exact reference trajectory |
| Swap tests | **PASS** | `test_ssa_swap.py` + exact reference result |
| Checkpoint/resume | **PASS** | `test_experiment_resume.py` (8 tests) |
| Analysis ingestion | **PASS** | `test_analysis.py` (6 tests) |
| Notebook tests | **PASS** | `test_notebooks.py` (6 tests) |
| Smoke test | **PASS** | `SMOKE_TEST_REFERENCE_ALIGNED_20261123_2100`: 4/4, resume, analysis |
| Workload estimate | **PASS** | `BENCHMARK.md`; full run 203,640 simulations, ≈48 s wall |

### Test command and result

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

**Result: 110 tests, 0 failures, 0 errors.**

---

## 4. Smoke test (reference-aligned)

- Run: `results/smoke/SMOKE_TEST_REFERENCE_ALIGNED_20261123_2100/`
- B1, K=10, O2+O3, both modes, population 4, 2 iterations, 5 optimization MC,
  10 final MC, 10 swap tests.
- 4/4 configurations completed.
- Re-running the same run ID skips all 4 with no duplicates.
- Analysis ingests the smoke run: 10 tables + 11 figures.

---

## 5. Full experiment

- Run: `results/run_20261123_205500_reference_aligned/`
- 240/240 configurations completed (B1–B4, 60 each).
- Wall time ≈48 s (15:23:57 → 15:24:46 UTC).
- 203,640 simulations; 240 MC files ×100 rows; 120 checkpoints; 120 SSA histories.
- Validation: **PASS**, no issues.
- All rows `objective_validity = validated`.
- No missing real configurations.

---

## 6. Scientific decisions remaining

| Item | Status |
|---|---|
| Blocked_B definition | **RESOLVED** (reference-confirmed paired B-only baseline) |
| Six objective formulas | **RESOLVED** (exact match) |
| Candidate pool / ranking / random part | **RESOLVED** (exact match) |
| SSA / Swap logic and seeds | **RESOLVED** (exact match to six-objective reference) |
| Diffusion model | **RESOLVED** (exact logic; rounds semantics fixed) |
| Synthetic edge orientation | **RESEARCHER DECISION** — reference reverses Bitcoin OTC edges; synthetic network keeps `source → target` |
| Small-network pool truncation | **DOCUMENTED DIFFERENCE** — reference raises, project truncates; unused on the full network |
| Between-optimizer-run variance | **NOT TESTED** — one SSA run per configuration |
| External validity | **NOT TESTED** — synthetic network only |

---

## 7. Decision

The reference-aligned implementation passes all gates. The full experiment has
been run and validated. The authoritative results are in
`RESULTS_SUMMARY_REFERENCE_ALIGNED.md` and
`FINAL_AUDIT_REFERENCE_ALIGNED.md`.
