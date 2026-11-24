# Changelog

All notable changes to this project are recorded here. The original starter
files were backed up under `backups/20261123_200236/` and
`backups/20261123_200624_session2/` before modification.

## [0.2.1] — 2026-11-23 — Full experiment and analysis

### Added

- Full 240-configuration experiment: `results/run_20261123_202600/`
  (4 scenarios × 5 K × 6 objectives × 2 modes), 240/240 completed in ≈48 s.
- Analysis outputs: 10 tables + 11 figures under
  `analysis/tables/run_20261123_202600_20261123_202328/` and
  `analysis/figures/run_20261123_202600_20261123_202328/`.
- `RESULTS_SUMMARY.md` and `FINAL_AUDIT.md` based on actual results.
- Notebook execution validation: setup, default-no-run B1, and read-only
  analysis notebooks ran end-to-end with 0 errors.

### Results (actual data)

- O2 (Competition-aware) has the highest mean final Net Advantage in both modes
  (21.62 SSA-only, 21.64 SSA+Swap).
- O1 is statistically indistinguishable from O2 overall; O2 is better on average
  than O3/O4/O5/O6.
- O3 maximises Blocked_B but has the lowest Net Advantage; Blocked_B–Net
  Advantage correlation is 0.36.
- SSA+Swap shows a small heavy-tailed average gain (+0.51, 95 % CI [0.13, 0.89])
  but is not a reliable per-configuration improvement (5/120 positive-significant,
  2/120 negative-significant).
- Optimizer fitness correlates strongly with final Net Advantage only for O2
  (0.71–0.80); near zero for O1/O3.

### Notes

- No code changes were needed after the validation gate; the full run used the
  validated implementation.
- The five Blocked_B-dependent objectives remain labelled
  `provisional_blocked_b`; O2 is the only validated objective.

## [0.2.0] — 2026-11-23 — Overnight repair, validation and experiment build

### Added

- `common/` modular architecture:
  - `utils.py` (paths, atomic writes, hashing, deterministic seed derivation,
    environment info)
  - `network.py` (load/validate/fingerprint/metadata v2)
  - `opponents.py` (B1–B4, exact starter ranking/tie rules)
  - `candidates.py` (polarity-aware 70/30 pool, truncation metadata)
  - `objectives.py` (O1–O6, budget coefficients, validity labels)
  - `diffusion.py` (pure-Python reference simulator)
  - `cpp_adapter.py` (persistent C++ engine adapter, import-order fix)
  - `evaluation.py` (paired B-only baseline, safe metric cache, per-seed arrays)
  - `ssa.py` (discrete SSA preserved from starter, invariant checks)
  - `swap.py` (bounded one-for-one elitist Swap, evaluation count fix)
  - `stats.py` (bootstrap CIs, paired tests, Holm–Bonferroni)
  - `experiment.py` (checkpointed coordinator, 72-column schema, manifest)
  - `run_experiment.py` (CLI, `--smoke`, `--print-plan`, resume)
  - `analysis.py` (read-only validation, tables, figures)
  - compatibility facades `experiment_core.py` and `scenario_runner.py`
- `cpp_engine`:
  - persistent `DiffusionEngine` (graph built once, read-only, thread-safe)
  - OpenMP batch execution with deterministic per-run RNG streams
  - functional API retained for compatibility
  - `MODEL_SPEC.md`, updated `README.md`
- `tests/` — 102 stdlib `unittest` tests covering every validation gate.
- `tools/build_notebooks.py` and `tools/benchmark.py`.
- `data/network_metadata_v2.json` — file/semantic SHA-256, counts,
  distributions, degree/connectivity summary, edge-count convention.
- Six thin notebooks with real line breaks, fingerprint checks, safe defaults
  (`RUN_MODE='none'`, `CONFIRM_FULL_RUN=False`) and a read-only analysis notebook.
- `VALIDATION_GATE.md`, `BENCHMARK.md`, `AUDIT_REPORT_ADDENDUM.md`, `CONFIG.md`,
  `README.md`, `OVERNIGHT_LOG.md`.
- Results/checkpoint infrastructure under `results/<run_id>/<scenario>/`.

### Fixed

- **D1** `cpp_adapter.py` added `cpp_engine/` to `sys.path` *after* importing the
  extension. Now it is added before the import.
- **D2** All notebook code cells contained literal `\n` sequences. Notebooks
  were rebuilt with genuine line breaks; every code cell is compile-checked.
- **D3** `scenario_runner.score_cache` keyed objective scores only by
  `(A,B,seeds)`, so different objectives could receive a stale score. The new
  metric cache is objective-independent and keyed by graph fingerprint,
  A/B seed sets, seed list and protocol; objective values are always recomputed.
- **D4** `metric_cache_eval` referenced a missing `blocked_B` key. Replaced with
  an explicit paired-baseline implementation and unit tests.
- **D5** No network fingerprint. Added file + semantic SHA-256 and metadata v2.
- **D6** Resume keys/signatures were incomplete and CSV writes were not atomic.
  Added full configuration signatures, fingerprint checks, atomic writes,
  run-id directories and manifest.
- **D7** CSV lacked required fields. New 72-column schema includes seed sets,
  seed protocols, pool composition, SSA settings, initial/final scores,
  uncertainty, runtimes, swap details, fingerprints and file paths.
- **D8** The adapter re-serialised the whole graph per simulation (~692 sims/s).
  A persistent engine raises this to ~10,700 sims/s serial and ~40,300 sims/s
  with 8 OpenMP threads.
- **D9** Runtime columns included cache hits. Added per-configuration simulation
  counts/time and cache hit/miss counters.
- **D10** Removed the dead/incorrect helper behaviour; compatibility facade now
  returns a correct paired-baseline metric dictionary.
- **D12** Analysis notebook was not read-only, had no validation/run selection,
  and wrote to a non-existent directory. New analysis module validates, selects
  runs explicitly, and writes unique timestamped tables/figures.
- **D13** B notebooks executed 60 configurations immediately. They now default
  to no run, require explicit confirmation for a full run, and the CLI is
  available for robust execution.
- **D14** README/Makefile installation instructions. README now documents manual
  dependency installation only; the Makefile performs a project-local build only.

### Preserved

- `common/network_generator.py` is unchanged; seed-42 regeneration is
  byte-identical and pinned by a regression test.
- The original network CSV is unchanged.
- Original notebooks, C++ sources and the prebuilt `.so` are backed up.
- The starter's SSA/Swap/objective formulas and opponent logic are preserved.
- No files were deleted.

### Known limitations

- Reference notebooks named in the master prompt are absent from the workspace.
- `Blocked_B` is implemented as the starter's explicit paired B-only protocol but
  is scientifically provisional; only O2 is fully validated.
- SSA/Swap fidelity to the intended reference cannot be independently verified.
- Between-optimizer-run variance is not estimated (one optimizer run per
  configuration).
