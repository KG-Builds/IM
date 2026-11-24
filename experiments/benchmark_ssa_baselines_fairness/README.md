# Benchmark — SSA O1/O4/O6 vs. Notebook Baselines and FIBMM

This directory is a self-contained experiment inside
`IM_Synthetic_Experiment/`.  It compares three SSA objective variants in
SSA-only mode against every valid seed-selection algorithm from
`old work/competitive_im_no_celf_csv_persist.ipynb`, plus the fairness-aware
FIBMM baseline from Xiao et al. (IEEE TCSS 2025).

**Primary metric:** Net Advantage = Spread_A − Spread_B.  
**Network (5,000-node benchmark):** the exact 5,000-node, 37,500-arc
reversed-influence synthetic graph used by `exp_20261124_5000_reversed`.  
**Separate 1,000-node run:** the 1,000-node, 7,500-arc synthetic graph is
documented below; its results are never mixed with the 5,000-node benchmark.  
**Swap refinement:** excluded by instruction.

## Headline result

All 240 cells completed and validation passed.  The highest mean Net Advantage is
**SSA O1** (47.878), followed by **SSA O6** (47.261), **CELF** (46.115),
**SSA-Notebook** (45.570) and **SSA O4** (43.241).  SSA O1 and CELF each have
**6 strict wins out of 20**; the SSA-vs-CELF contest is 10W/10L for O1 and O6,
and 9W/11L for O4.  There is **no universal winner**: CELF wins six cells,
including B4/K=50 with the highest single-cell Net Advantage (92.68).
FIBMM is a defensible fairness baseline but has a lower mean Net Advantage
(18.718); it slightly improves Blocked_B while substantially reducing Spread_A.
See `FINAL_REPORT.md` for the full analysis.


## Research notebooks

**1,000-node synthetic notebook (current request):**

`competitive_im_benchmark_1000_with_fibm.ipynb`

It loads `data/synthetic_network_1000_7500.csv` (1,000 nodes, 7,500 arcs,
reversed influence direction), reads the separate `benchmark_1000_synthetic/`
run, documents the algorithms and FIBM adaptation, and writes scenario-specific
figures under `figures/notebook_1000/`.  The default mode is `load_existing`.

**1,000-node headline result:** SSA-Notebook has the highest mean Net Advantage
(**24.483**), followed by SSA O6 (**24.218**), SSA O1 (**24.212**), CELF
(**23.638**) and SSA O4 (**23.416**).  No method dominates: SSA O1, SSA O4 and
CELF each win 4 of 20 cells, SSA O6 wins 3 and SSA-Notebook wins 2.  FIBM is a
fairness baseline with mean Net Advantage **18.363** and slightly higher mean
Blocked_B than the SSA variants.

**5,000-node benchmark notebook (reference):**

`competitive_im_benchmark_with_fibm.ipynb`

This is the earlier 5,000-node interface.  Its results are **not** used by the
1,000-node notebook, because they belong to a different graph.

## Quick start

Run from this directory:

```bash
# 1. Inspect the plan (no simulations)
python run_benchmark.py --mode plan

# 2. Validate network, opponent orders, prior SSA reuse and a deterministic
#    SSA reproduction check (no full experiment)
python run_benchmark.py --mode verify

# 3. Reduced-budget smoke test for every algorithm on B1/K=10
python run_benchmark.py --mode smoke

# 4. Full run (checkpointed, resumable; requires explicit confirmation)
python run_benchmark.py --mode full --confirm-full

# 5. Read-only aggregation, validation, figures and final report
python run_benchmark.py --mode analyze

# Separate 1,000-node run (skips valid 1,000-node cells; never reuses 5,000-node cells)
python run_1000_synthetic.py
```

The full run can be targeted for debugging without changing the configuration:

```bash
python run_benchmark.py --mode full --confirm-full \
  --only-methods FIBMM --only-scenarios B1 --only-k 10
```

Existing valid cells are skipped on rerun.  Use `--force-recompute` only when
a dependency has changed; the ledger records the reason.

## Directory layout

```
benchmark_ssa_baselines_fairness/
├── README.md
├── AUDIT.md                    # Phase-1 network/experiment audit
├── METHODS.md                  # full methodology
├── ALGORITHM_AUDIT.md          # notebook algorithm audit
├── PAPER_IMPLEMENTATION.md     # FIBMM paper-to-code mapping
├── FINAL_REPORT.md             # generated final report
├── config/
│   └── benchmark_config.json
├── network/
│   ├── network_reference.json  # identity, hashes, direction proof
│   └── network_validation.json
├── algorithms/
│   ├── core.py
│   ├── notebook_baselines.py
│   ├── fibmm.py
│   └── reuse.py
├── raw_results/
│   ├── results.csv             # one row per method × scenario × K
│   ├── ledger.csv              # reuse/completion/failure ledger
│   ├── manifest.json
│   ├── seed_sets.csv
│   ├── opponent_seeds.csv
│   ├── mc/                     # 100-row per-run tables
│   ├── checkpoints/            # per-cell seed sets + metadata
│   ├── preprocessing/          # cached PageRank/centrality/communities
│   └── fibmm_diagnostics/      # internal Pi/tau/sigma and sample plans
├── aggregated/
│   ├── benchmark_results.csv
│   ├── winners_by_scenario_k.csv
│   ├── algorithm_summary.csv
│   ├── performance_by_scenario.csv
│   ├── performance_by_k.csv
│   ├── ssa_vs_baselines.csv
│   ├── fibmm_fairness.csv
│   ├── runtime_summary.csv
│   ├── reuse_ledger_summary.csv
│   ├── validation_summary.json
│   └── validation_report.md
├── figures/
│   ├── Net_Advantage_B1_Random.png
│   ├── Net_Advantage_B2_Degree.png
│   ├── Net_Advantage_B3_Positive_PageRank.png
│   ├── Net_Advantage_B4_Community.png
│   ├── Net_Advantage_SSA_and_Fairness_All_Scenarios.png
│   ├── Mean_Net_Advantage_by_Algorithm.png
│   └── Selection_Runtime_by_Algorithm.png
├── reports/
│   ├── FINAL_REPORT.md
│   ├── STATISTICAL_NOTES.md
│   ├── ALGORITHM_BUDGETS.md
│   └── CHECKLIST.md
├── logs/
├── smoke/                      # isolated smoke outputs, never used in results
├── validation/                 # verify/test outputs
└── tests/
```

## Method list

**SSA variants:** `SSA_O1`, `SSA_O4`, `SSA_O6` (SSA-only).

**Notebook algorithms:** `SSA_Notebook` (O1 fitness, notebook seed schedule),
`Random`, `TopDegree`, `SignedPageRank`, `CRIEG`, `NCGE`, `SignedGA`, `CELF`.

**Fairness baseline:** `FIBMM`.

**Excluded:** `CELF + sampled Local Search` (one-for-one swap-based
post-processing, prohibited by the no-swap instruction).  The base CELF is
included.

## Experimental settings

- Opponent scenarios: B1 Random, B2 Degree, B3 Positive PageRank, B4 Community.
- K = 10, 20, 30, 40, 50.
- Optimization MC: 15 common seeds.
- Final evaluation MC: 100 common seeds (`900000000..900000099`).
- SSA: population 12, iterations 10, mutation 0.20.
- Candidate pool: 10·K, 70% ranked / 30% random from the ranked remainder.
- FIBMM: h=0.9, ε=0.1, l=1, Louvain communities, CRP cap 600,000.
- Primary metric: Net Advantage = Spread_A − Spread_B.

## Reuse

The SSA-only O1/O4/O6 results are reused from
`experiments/exp_20261124_5000_reversed/raw_results/run_20261124_5000_reversed`
after validating the network hash, direction, opponent seeds, candidate-pool
signature, SSA parameters, seed sets and 100-row MC files.  A deterministic
reproduction check reruns B1/K=10/O1 and compares the seed set.  No old
same-direction result is reused.

## Reproducibility and safety

- All writes stay inside this experiment directory.
- The network is read-only; it is referenced, not regenerated.
- The source notebook and paper are not modified.
- No packages are installed, no network access is used, no git operations are
  performed, and no files are deleted.
- CSV/JSON writes are atomic (temporary sibling file + `os.replace`).
- The full run is checkpointed per method × scenario × K and is resumable.
- Analysis is read-only with respect to raw results.
