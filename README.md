# Signed Competitive Influence Maximization — 5,000-Node Reverse-Influence Experiment

Research code for **Competitive Influence Maximization on a signed network**.
Rumour A is promoted by selected seed nodes against an opponent Rumour B under a
finite-contact competitive Maki–Thompson diffusion model. The main evaluation
metric is:

**Net Advantage = Spread_A − Spread_B**

Higher is better. The optimizer's internal fitness is **not** the ranking metric.

---

## Current authoritative experiment

**All old results from the 1,000-node, same-direction simulation have been moved
to `experiments/exp_20261124_5000_reversed/to_be_discarded/` and are excluded
from every calculation, table, figure and conclusion.**

| Item | Value |
|---|---|
| Experiment directory | `experiments/exp_20261124_5000_reversed/` |
| Run ID | `run_20261124_5000_reversed` |
| Network | 5,000 nodes, 37,500 directed trust arcs |
| Influence direction | **Reverse of trust direction** (`trust source → target`, influence `target → source`) |
| Configurations | 240 / 240 completed |
| Simulations | 202,055 |
| Wall time | ≈2m37s |
| Tests | 121 passing |
| Final figures | Exactly four Net Advantage PNGs in `final_figures/` |
| Final report | `experiments/exp_20261124_5000_reversed/reports/FINAL_REPORT.md` |

### Headline result

**No single SSA variant consistently wins every B scenario and every K value.**

- Most strict wins in SSA-only: **O1** (6/20).
- Most strict wins in SSA+Swap: **O1 and O4** (5/20 each).
- Highest mean Net Advantage in SSA-only: **O1** (47.878).
- Highest mean Net Advantage in SSA+Swap: **O6** (49.412).
- Best average is not the same as consistent per-comparison winning.

Full 20-comparison tables and win/tie/loss counts:
`experiments/exp_20261124_5000_reversed/aggregated/winner_summary.md`.

---

## Project layout

```
.
├── README.md
├── CONFIG.md                      # legacy 1,000-node configuration (see experiment config)
├── CHANGELOG.md
├── OVERNIGHT_LOG.md
├── AUDIT_REPORT.md                # original source-of-truth audit
├── AUDIT_REPORT_ADDENDUM.md
├── REFERENCE_NOTEBOOK_AUDIT.md    # audit of the supplied reference notebooks
├── BENCHMARK.md                   # legacy 1,000-node benchmark
├── old work/                      # supplied reference notebooks + extracted code
├── GPT_Contexts/                  # handover documents for ChatGPT
│   ├── ContextForGPT.md           # historical pre-5,000 context
│   └── ContextForGPT_5000_Reversed.md  # current authoritative context
├── data/                          # legacy 1,000-node raw network + metadata
├── common/                        # Python research code
│   ├── network.py                 # trust loading + explicit influence reversal
│   ├── network_generator.py       # legacy 1,000-node generator (unchanged)
│   ├── network_generator_large.py # 5,000-node generator
│   ├── opponents.py               # B1–B4 on the influence graph
│   ├── candidates.py              # 10·K polarity-aware candidate pool
│   ├── objectives.py              # O1–O6
│   ├── diffusion.py               # Python reference simulator
│   ├── cpp_adapter.py             # persistent C++ engine adapter
│   ├── evaluation.py              # MC metrics, Blocked_B, safe caching
│   ├── ssa.py / swap.py           # SSA and bounded Swap
│   ├── experiment.py              # checkpointed coordinator
│   ├── analysis.py                # read-only tables/figures
│   └── run_experiment.py          # CLI runner
├── cpp_engine/                    # C++ engine, pybind11, OpenMP
├── tests/                         # 121 tests
├── tools/
│   ├── build_notebooks.py
│   ├── benchmark.py
│   └── move_old_results.py
├── experiments/
│   └── exp_20261124_5000_reversed/
│       ├── config/
│       ├── network/
│       ├── raw_results/
│       ├── aggregated/
│       ├── final_figures/
│       ├── reports/
│       ├── logs/
│       ├── checkpoints/
│       ├── validation/
│       └── to_be_discarded/
├── results/                       # legacy results moved out; now a pointer README
├── analysis/                      # legacy analysis moved out; cache + pointer README
└── logs/                          # legacy run logs moved out; benchmark + pointer README
```

---

## Trust direction vs influence direction

The network CSV is the **trust** network:

```
source → target, rating, probability, strength, polarity
```

The **influence** graph is the exact reverse:

```
target → source
```

The reversal is applied exactly once in
`common/network.py::load_influence_network`. All downstream components use that
influence graph as-is:

- C++ diffusion engine
- Python reference simulator
- candidate-pool scoring
- B1 random (node sampling is direction-independent)
- B2 degree (influence out-degree)
- B3 positive PageRank (positive influence edges)
- B4 community (undirected influence projection; degree ranking uses influence out-degree)
- SSA fitness
- Swap refinement
- final Monte Carlo evaluation

Direction validation for the 5,000-node network:

- 37,500 trust edges
- 37,500 influence edges
- 0 missing reversed edges
- 0 extra influence edges
- 0 same-direction edges
- 0 reciprocal trust pairs

---

## Network

| Parameter | Value |
|---|---|
| Nodes | 5,000 (ids 0–4999) |
| Trust arcs | 37,500 (one CSV row = one directed trust arc) |
| Influence arcs | 37,500 (exact reverse) |
| Topology | Barabási–Albert backbone (initial clique 8, m=7), random fill to 37,500 undirected pairs, random orientation |
| Generation seed | 42 |
| Sign | positive with probability 0.5 |
| Trust magnitude | uniform integer 1–10 |
| Probability mapping | 1–3 → 0.1; 4–7 → 0.5; 8–10 → 0.9 |
| Positive / negative trust arcs | 18,706 / 18,794 |
| Probability classes 0.1 / 0.5 / 0.9 | 11,151 / 14,911 / 11,438 |
| Reciprocal trust pairs | 0 |
| Trust file SHA-256 | `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e` |
| Trust semantic SHA-256 | `576d00314530fc82c86d5a433e13a2ea64dbe135723350d936df21b082e2b20f` |
| Influence semantic SHA-256 | `b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9` |

Seed-42 regeneration is deterministic and identical. The legacy 1,000-node
network remains in `data/` as raw historical data and is not used by the current
experiment.

---

## Diffusion model

Finite-contact competitive Maki–Thompson variant, synchronous rounds:

- states: 0 ignorant, 1 A-spreader, 2 B-spreader, 3 A-stifler, 4 B-stifler
- each spreader contacts one uniformly random uncontacted outgoing influence neighbour per round
- no uncontacted neighbour, or a non-ignorant target → spreader becomes a stifler
- otherwise transmits with the edge probability; negative edge flips A↔B
- proposals resolved by highest edge strength; different-rumour ties random
- `Spread_A` counts states 1 and 3; `Spread_B` counts 2 and 4
- `rounds` includes the terminal empty round (reference semantics)

Full specification: `cpp_engine/MODEL_SPEC.md`.

---

## Objectives

| ID | Name | Formula |
|---|---|---|
| O1 | Original SSA | `Spread_A + 1.5·Blocked_B` |
| O2 | Competition-aware | `Spread_A − Spread_B` |
| O3 | Explicit blocking | `Blocked_B` |
| O4 | Combined influence-blocking | `Spread_A + 2·Blocked_B` |
| O5 | Gain_A + blocking | `(Spread_A − Spread_B) + 2·Blocked_B` |
| O6 | Budget-aware | `α(K)·Spread_A + β(K)·Blocked_B` |

`Blocked_B = max(0, mean_s baseline_B − mean_s Spread_B)` with paired Monte
Carlo seeds. All six objectives are reference-validated.

Budget coefficients: (0.75, 2.25) for K≤20; (1.00, 1.75) for K≤40;
(1.25, 1.50) for K=50.

---

## Experiment design

- Opponent scenarios B1, B2, B3, B4.
- K = 10, 20, 30, 40, 50.
- Six objectives O1–O6.
- Two modes: SSA-only and SSA + Swap refinement.
- 4 × 5 × 6 × 2 = 240 configurations.
- SSA: population 12, iterations 10, mutation 0.20.
- Optimization MC: 15 seeds; final evaluation MC: 100 seeds.
- Candidate pool: 10·K, 70% ranked / 30% random from the ranked remainder.
- Swap: up to 100 sampled one-for-one tests, single best improving swap.
- SSA is run once per `(scenario, K, objective)`; both modes share its seed set.

---

## Quick start

```bash
# Build the C++ extension (project-local; no installation)
cd cpp_engine
python setup.py build_ext --inplace
cd ..

# Tests
python -m unittest discover -s tests -p 'test_*.py'

# Plan the 5,000-node experiment
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode plan

# Smoke test (separate output)
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode smoke

# Full run (checkpointed, resumable)
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode full --confirm-full --run-id run_20261124_5000_reversed

# Analysis: aggregated tables, winner analysis, four final figures, report
python experiments/exp_20261124_5000_reversed/analyze_5000.py
```

No packages were installed by the agent. If a fresh environment lacks
dependencies, install them manually in your own environment:

```bash
python -m pip install numpy pandas networkx matplotlib scipy pybind11 setuptools jupyter
```

---

## Final figures (only these four)

- `experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B1_Random.png`
- `experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B2_Degree.png`
- `experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B3_Positive_PageRank.png`
- `experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B4_Community.png`

Each is a two-panel line chart (SSA-only and SSA+Swap), with six objective lines,
95% Monte Carlo CIs, consistent colors/markers, a shared legend, and a clear
title. No other charts are final deliverables.

---

## Validation

- **121 tests, 0 failures** (`validation/tests_5000.txt`).
- Deterministic trust→influence reversal tests, signed propagation, multiple
  edges, candidate scoring, B2 degree, B3 PageRank.
- C++/Python consistency: deterministic tiny graphs exact; large-network
  invariants checked; distributional parity checked on small random graphs.
- Network: 5,000 nodes, 37,500 trust/influence arcs, no self-loops, no duplicate
  arcs, seed-42 regeneration identical.
- Smoke: `SMOKE_5000_reversed_20261124` — 4/4, resume and analysis ingestion.
- Full run: 240/240, validation PASS, 240 MC files ×100 rows, 120 checkpoints,
  one run/fingerprint/direction/count, no seed-set violations.
- Old results: excluded; move manifest at
  `experiments/exp_20261124_5000_reversed/to_be_discarded/MOVE_MANIFEST.csv`.

---

## Scientific caveats

1. **One optimizer run per configuration.** The winner analysis is an empirical
   comparison of one SSA run per B × K × objective; it does not estimate
   between-optimizer-run variance.
2. **Monte Carlo uncertainty only.** Figure error bars reflect 100 MC evaluation
   seeds, not optimizer stochasticity.
3. **Strict empirical winner.** A win is based on observed mean Net Advantage;
   it is not a statistical-significance claim. Ties are reported explicitly.
4. **Synthetic network.** Results are conditional on this 5,000-node synthetic
   design and the reversed influence convention.
5. **No external validity claim** on real signed networks.

---

## Historical material

- `old work/` contains the supplied reference notebooks and extracted code.
- `REFERENCE_NOTEBOOK_AUDIT.md` documents the line-by-line alignment and the
  three corrections made after the notebooks were supplied.
- `experiments/exp_20261124_5000_reversed/to_be_discarded/` contains the old
  1,000-node, same-direction results, analysis, logs and derived reports. They
  are retained for audit only and must not be used in new research conclusions.
