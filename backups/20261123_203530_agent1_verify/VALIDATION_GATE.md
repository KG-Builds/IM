# Validation Gate — Signed Competitive Influence Maximization

Generated: 2026-11-23 (local)
Workspace: `IM_Synthetic_Experiment/`
Decision: **FULL EXPERIMENT ALLOWED, with O2 as the validated primary objective and
O1/O3/O4/O5/O6 explicitly labelled `provisional_blocked_b`.**

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.14.10 |
| NumPy / pandas / networkx / matplotlib / SciPy | 2.4.6 / 2.3.3 / 3.7 / 3.10.8 / 1.16.2 |
| pybind11 | 3.1.0 |
| Compiler | g++ 16.2.1 |
| Python headers | `/usr/include/python3.14/Python.h` present |
| OpenMP | compiled and available (`simulation_engine.openmp_enabled() == True`) |
| CPU count | 8 |
| C++ extension | `cpp_engine/simulation_engine.cpython-314-x86_64-linux-gnu.so` imports |

No packages were installed, upgraded, or removed. No system configuration was changed.

---

## 2. Gate checklist

| Gate | Status | Evidence |
|---|---|---|
| Network tests pass | **PASS** | `tests/test_network.py` (9 tests): 1000 nodes, 7500 directed arcs, no self-loops, no duplicate arcs, probability mapping, seed-42 byte-identical regeneration, fingerprint |
| C++ compilation/import passes | **PASS** | `cd cpp_engine && python setup.py build_ext --inplace`; import + `openmp_enabled()` OK |
| Diffusion tests pass | **PASS** | `tests/test_diffusion_python.py` (13 tests) + `tests/test_cpp_engine.py` (14 tests) |
| Python/C++ model comparison acceptable | **PASS** | Deterministic cases exact; 800-run distributional parity on a random graph within 4 SE; refactored engine is bit-identical to the backed-up original engine on 30 full-network runs |
| Objective tests pass | **PASS** | `tests/test_objectives.py` (7 tests), hand-calculated values for O1–O6; O4 ≠ O5 |
| Candidate-pool tests pass | **PASS** | `tests/test_candidates.py` (7 tests): 10·K size, 70/30 composition, polarity-aware scoring, B exclusion, small-network truncation |
| Opponent strategy tests pass | **PASS** | `tests/test_opponents.py` (10 tests): B1 exact sample, B2 degree/tie, B3 positive-only PageRank, B4 largest community, nestedness |
| SSA tests pass | **PASS** | `tests/test_ssa_swap.py` (10 tests): exactly K unique, pool/B restrictions, determinism, elitism, K=pool edge case |
| Swap tests pass | **PASS** | `tests/test_ssa_swap.py`: one-for-one, bounded tests, elitist acceptance, K preserved, B excluded |
| Checkpoint/resume tests pass | **PASS** | `tests/test_experiment_resume.py` (8 tests): atomic CSV, resume without duplicates, checkpoint reuse for a missing mode, signature mismatch rejection, manifest |
| Analysis ingestion tests pass | **PASS** | `tests/test_analysis.py` (6 tests): validation, tables, figures, discovery, corruption detection, path-escape rejection |
| Notebook tests pass | **PASS** | `tests/test_notebooks.py` (6 tests): valid JSON/nbformat, every code cell compiles, no literal `\n`, safe defaults |
| No unresolved ambiguity affects the chosen experiment | **CONDITIONAL PASS** | O2 is independent of Blocked_B. The five Blocked_B-dependent objectives are run but labelled provisional and reported separately. See §4. |
| Workload estimate based on bounded benchmark | **PASS** | `BENCHMARK.md`: 40,255 sims/s at 8 threads; full design ≈ 443,900 simulations ≈ 11 s pure simulation (upper bound; Python overhead extra) |

### Test command

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

**Result: 102 tests, 0 failures, 0 errors.** (`Ran 102 tests in 8.697s — OK`)

---

## 3. Smoke test (separate from real results)

Run ID: `SMOKE_TEST_20261123_202100`
Location: `results/smoke/SMOKE_TEST_20261123_202100/`
Configuration: B1 only, K=10, O2 + O3, both modes, SSA population 4,
2 iterations, 5 optimization MC runs, 10 final MC runs, 10 swap tests.

| Check | Result |
|---|---|
| 4/4 configurations completed | **PASS** |
| CSV schema contains all 72 required columns | **PASS** |
| SSA-only and SSA+Swap share the same saved SSA seed set | **PASS** |
| Per-seed MC files written for every configuration | **PASS** |
| Checkpoints written and signature-checked | **PASS** |
| Re-running the same run ID skips all 4 configurations, no duplicates | **PASS** |
| Analysis notebook/module ingests the smoke run and produces 10 tables + 11 figures | **PASS** |
| Smoke results kept separate under `results/smoke/` | **PASS** |

Smoke output example: O2 SSA-only and Swap both Net Advantage 14.5; O3 SSA-only
12.1, Swap 13.3 (one accepted swap). These are pipeline checks, **not research
results**.

---

## 4. Scientific ambiguities and decisions

### 4.1 Reference notebooks are absent

The master prompt refers to original/reference notebooks, but none exist in the
workspace. Every definition has been audited against the starter code and the
master prompt. Items that cannot be verified against a reference notebook are
documented as researcher decisions, not presented as canonically verified.

### 4.2 Blocked_B (provisional)

The starter's only implementation is the paired B-only baseline:

```
baseline_B(B, s) = Spread_B(A=empty, B, s)
Blocked_B        = max(0, mean_s baseline_B - mean_s Spread_B(A, B, s))
```

The same Monte Carlo seeds are used for the baseline and the A/B run. This is
explicit, reproducible, and unit-tested, but the master prompt forbids assuming
it is the intended research definition without source support. Therefore:

- **O2 (Competition-aware) = Spread_A − Spread_B is fully validated.**
- O1/O3/O4/O5/O6 are implemented exactly as specified and run, but every row is
  labelled `objective_validity = provisional_blocked_b`.
- Analysis reports them separately and never claims a validated six-objective result.

This is the conservative choice that still allows the primary Net Advantage
comparison (O2) and the SSA-vs-Swap study to proceed.

### 4.3 SSA and Swap fidelity

The discrete set-based SSA and the single-best-swap policy are preserved exactly
from the starter. They are not replaced with another optimizer. Canonical SSA
equation-level fidelity cannot be verified without the missing reference
notebook; this is documented as a researcher decision.

### 4.4 Diffusion model fidelity

The C++ engine is the only executable authority in the workspace. Its model is
specified in `cpp_engine/MODEL_SPEC.md`; a Python reference implementation and
deterministic/distributional tests validate the logic. The refactored engine is
bit-identical to the original engine on the same seeds. Fidelity to the missing
intended reference notebook cannot be independently verified.

---

## 5. Full-run decision

The full 4-scenario × 5-K × 6-objective × 2-mode matrix (240 configurations) is
allowed with these conditions:

1. O2 results are the validated primary comparison.
2. All Blocked_B-dependent objectives are labelled provisional in the CSV,
   tables, figures, and final report.
3. SSA is run once per (scenario, K, objective) and shared by both modes.
4. Scenarios run sequentially; checkpoints and resume are enabled.
5. No result is fabricated or inferred; missing configurations are reported.

The full run uses a single master run ID and the shared network fingerprint
`659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`.
