# Changelog

All notable changes to this project are recorded here. The original starter
files were backed up under `backups/20261123_200236/` and
`backups/20261123_200624_session2/` before modification.

## [0.4.0] — 2026-11-24 — Reverse influence direction, 5,000-node rerun, winner analysis

### Changed

- **Influence direction reversed.** The trust CSV is `source → target`; the
  influence graph is now the exact reverse `target → source`, applied once in
  `common/network.load_influence_network`. All downstream components use the
  influence graph as-is: C++ diffusion, Python reference, candidate pool, B1–B4,
  SSA, Swap and final evaluation.
- **New network:** 5,000 nodes / 37,500 directed trust arcs, seed 42, generated
  once and reused. Influence graph has 37,500 reversed arcs.
- **Dedicated experiment directory:**
  `experiments/exp_20261124_5000_reversed/` with config, network, raw results,
  aggregated tables, final figures, reports, logs, checkpoints, validation and
  `to_be_discarded/`.
- **Old results isolated:** all pre-correction 1,000-node same-direction
  results, analysis, logs and derived reports moved (not deleted) to
  `experiments/exp_20261124_5000_reversed/to_be_discarded/`, with a move
  manifest. They are excluded from all new calculations.

### Added

- `common/network_generator_large.py` — memory-safe 5,000-node generator.
- `common/network.py::load_influence_network`, `direction_check`,
  `influence_semantic_fingerprint`, `build_influence_metadata_v2`.
- `tests/test_influence_direction.py` — deterministic direction/network tests.
- `experiments/.../run_experiment_5000.py` — dedicated runner.
- `experiments/.../analyze_5000.py` — aggregated tables, 20-comparison winner
  analysis, exactly four final figures, final report.
- Direction/count fields in the result schema, checkpoint signature, run
  metadata and analysis validation.

### Results (actual data)

- Full run `run_20261124_5000_reversed`: **240/240 configurations**,
  202,055 simulations, ≈2m37s wall.
- **No single SSA variant consistently wins every B scenario and every K value.**
- Most strict wins in SSA-only: **O1** (6/20).
- Most strict wins in SSA+Swap: **O1 and O4** (5/20 each).
- Best mean Net Advantage: **O1** in SSA-only (47.878), **O6** in SSA+Swap
  (49.412).
- Four final figures only, one per B scenario.
- Test suite: **121 tests, 0 failures**.

## [0.3.0] — 2026-11-23 — Reference notebooks supplied; audit, correction and rerun

### Added

- `REFERENCE_NOTEBOOK_AUDIT.md` — line-by-line comparison against
  `old work/competitive_im_no_celf_csv_persist.ipynb` and
  `old work/competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb`.
- `tests/test_reference_alignment.py` — direct checks of candidate pools, seed
  derivation, objectives, SSA, Swap and round semantics against the notebooks.
- `VALIDATION_GATE_REFERENCE_ALIGNED.md`,
  `RESULTS_SUMMARY_REFERENCE_ALIGNED.md`,
  `FINAL_AUDIT_REFERENCE_ALIGNED.md`.
- New authoritative run: `run_20261123_205500_reference_aligned` (240/240).

### Fixed (reference discrepancies)

- **Candidate-pool ranking** now uses the exact four-level key
  `(-net, -positive, negative, node_id)` instead of `(-net, node_id)`.
- **Candidate-pool random part** now samples from `ranked[n_ranked:]` (ranked
  remainder), not node-id order. This changed ~30 % of each pool.
- **Pool composition** now uses `n_random = int(round(size*fraction))`,
  `n_ranked = size - n_random`.
- **Optimization seeds** now use
  `master_seed*100000 + scenario_number*10000 + i` (removed the erroneous
  `+3,000,000`).
- **Round counter** now follows the reference: terminal empty round included
  (`executed_rounds + 1`). Spread counts are unchanged.
- **Blocked_B validity** upgraded to `validated`; the reference confirms the
  paired B-only baseline exactly.
- SSA history now records `unique_solutions` as in the reference.

### Results (reference-aligned, actual data)

- O2 highest mean final Net Advantage: 20.55 (SSA only), 21.97 (SSA+Swap).
- O1 and O6 are statistically indistinguishable from O2 overall; O2 is better on
  average than O3/O4/O5.
- O3 maximises Blocked_B (19.41) but has the lowest Net Advantage (15.78);
  Blocked_B–Net Advantage correlation is 0.46.
- SSA+Swap mean difference +0.27 (95 % CI [−0.24, +0.78]); 8/120
  positive-significant, 7/120 negative-significant — not a reliable improvement.
- Optimizer fitness correlates strongly with final Net Advantage for O2
  (0.60–0.87) and O6 (0.59–0.68), weakly for O1/O3/O4/O5.

### Notes

- The pre-reference run `run_20261123_202600` is retained as a sensitivity check;
  per-configuration old/new correlation is 0.38, aggregate conclusions stable.
- Test suite now has **110 tests**.
- The synthetic edge orientation and small-network pool truncation are
  documented researcher choices, not silently changed.

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
