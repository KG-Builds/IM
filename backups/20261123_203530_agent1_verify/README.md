# Signed Competitive Influence Maximization — Synthetic Network Experiment

Research code for **Competitive Influence Maximization on a signed network**.
Rumor A is promoted by selected seed nodes against an opponent Rumor B under a
competitive Maki–Thompson diffusion model. The main evaluation metric is:

**Net Advantage = Spread_A − Spread_B**

Higher is better. The optimizer's internal fitness is **not** the ranking metric.

---

## Project layout

```
.
├── README.md
├── CONFIG.md                  # every constant, seed and design decision
├── CHANGELOG.md
├── OVERNIGHT_LOG.md           # autonomous work log
├── AUDIT_REPORT.md            # original source-of-truth audit
├── AUDIT_REPORT_ADDENDUM.md   # session-2 verification/corrections
├── VALIDATION_GATE.md         # test/benchmark/smoke gate before the full run
├── BENCHMARK.md               # bounded performance benchmark
├── FINAL_AUDIT.md             # final read-only review (after the run)
├── data/
│   ├── synthetic_network_1000_7500.csv   # immutable shared network
│   ├── network_metadata.json             # original starter metadata
│   └── network_metadata_v2.json          # fingerprint, counts, distributions
├── cpp_engine/
│   ├── diffusion.hpp / diffusion.cpp     # persistent C++ engine
│   ├── bindings.cpp                      # pybind11 bindings
│   ├── setup.py / Makefile               # project-local build only
│   ├── MODEL_SPEC.md                     # precise model specification
│   └── README.md
├── common/
│   ├── utils.py            # paths, atomic writes, hashing, seeds
│   ├── network.py          # load/validate/fingerprint/metadata
│   ├── network_generator.py# pinned seed-42 generator (unchanged)
│   ├── opponents.py        # B1–B4 strategies
│   ├── candidates.py       # polarity-aware 70/30 candidate pool
│   ├── objectives.py       # O1–O6 formulas and validity labels
│   ├── diffusion.py        # pure-Python reference simulator
│   ├── cpp_adapter.py      # persistent C++ adapter
│   ├── evaluation.py       # Monte Carlo metrics + safe caching
│   ├── ssa.py              # discrete SSA
│   ├── swap.py             # bounded one-for-one Swap
│   ├── stats.py            # bootstrap CIs, paired tests, Holm
│   ├── experiment.py       # checkpointed experiment coordinator
│   ├── run_experiment.py   # CLI runner
│   ├── analysis.py         # read-only tables/figures
│   ├── scenario_runner.py  # deprecated compatibility facade
│   └── experiment_core.py  # deprecated compatibility facade
├── notebooks/
│   ├── 00_Setup_Synthetic_Network.ipynb
│   ├── B1_Random_opponent.ipynb
│   ├── B2_Degree-based_opponent.ipynb
│   ├── B3_Positive_PageRank_opponent.ipynb
│   ├── B4_Community-concentrated_opponent.ipynb
│   └── 05_Analysis_Charts.ipynb
├── tests/                  # 102 stdlib unittest tests
├── tools/
│   ├── build_notebooks.py
│   └── benchmark.py
├── results/<run_id>/<scenario>/...
├── logs/
└── analysis/{figures,tables}/<run_id>_<timestamp>/
```

---

## Network

- 1,000 nodes, ids 0–999
- 7,500 **unique directed arcs** (one CSV row = one directed arc)
- Barabási–Albert-style undirected backbone, randomly oriented, plus a random
  directed fill; the delivered file has 7,498 underlying unordered pairs and
  2 reciprocal arc pairs, for 7,500 arcs total
- Positive/negative signs (`P(positive)=0.5`), trust magnitude `U{1..10}`
- Probability mapping: magnitude 1–3 → 0.1; 4–7 → 0.5; 8–10 → 0.9
- Generation seed 42; seed-42 regeneration is byte-identical
- File SHA-256: `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`
- Semantic edge-table SHA-256: `461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25`

The network is **never regenerated or overwritten** by experiment code. Details
and the edge-count convention are in `data/network_metadata_v2.json` and
`CONFIG.md`.

---

## Diffusion model

Finite-contact competitive Maki–Thompson variant, synchronous rounds:

- states: 0 ignorant, 1 A-spreader, 2 B-spreader, 3 A-stifler, 4 B-stifler
- each spreader contacts one uniformly random uncontacted outgoing neighbour per round
- no uncontacted neighbour, or a non-ignorant target → spreader becomes a stifler
- otherwise transmits with the edge probability; negative edge flips A↔B
- proposals resolved by highest edge strength; different-rumor ties random
- `Spread_A` counts states 1 and 3; `Spread_B` counts 2 and 4

Full specification: `cpp_engine/MODEL_SPEC.md`.

---

## Objectives

| ID | Name | Formula | Validity |
|---|---|---|---|
| O1 | Original SSA | `Spread_A + 1.5·Blocked_B` | provisional |
| O2 | Competition-aware | `Spread_A − Spread_B` | **validated** |
| O3 | Explicit blocking | `Blocked_B` | provisional |
| O4 | Combined influence-blocking | `Spread_A + 2·Blocked_B` | provisional |
| O5 | Gain_A + blocking | `(Spread_A − Spread_B) + 2·Blocked_B` | provisional |
| O6 | Budget-aware | `α(K)·Spread_A + β(K)·Blocked_B` | provisional |

`α,β` = (0.75, 2.25) for K≤20; (1.00, 1.75) for K≤40; (1.25, 1.50) for K=50.
O4 and O5 are different. See `CONFIG.md` and `VALIDATION_GATE.md` for the
Blocked_B protocol and why five objectives are labelled provisional.

---

## Opponent scenarios

- **B1 Random**: uniform random sample, fixed order, nested across K
- **B2 Degree**: out-degree descending, ties by node id
- **B3 Positive PageRank**: PageRank on the positive-only directed subgraph
- **B4 Community**: largest Louvain community, ranked by out-degree

K = 10, 20, 30, 40, 50. Candidate pool = 10·K, 70 % polarity-aware ranked and
30 % random, with opponent seeds excluded.

---

## Quick start

Dependencies are expected to be pre-installed. **Do not install packages as part
of this project**; if something is missing, install it manually in your own
environment:

```bash
# Only if your environment is missing these (do not run inside the autonomous agent):
python -m pip install numpy pandas networkx matplotlib scipy pybind11 setuptools jupyter
```

Build the C++ extension (project-local, no installation):

```bash
cd cpp_engine
python setup.py build_ext --inplace
cd ..
```

Run the tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Validate the shared network:

```bash
python -c "import sys; sys.path.insert(0,'common'); from network import load_network, verify_fingerprint; df,G,nodes=load_network(); print(len(nodes), G.number_of_edges()); print(verify_fingerprint())"
```

Show the full experiment plan (240 configurations):

```bash
python common/run_experiment.py --print-plan
```

Run a smoke test (separate from real results):

```bash
python common/run_experiment.py --smoke
```

Run the full experiment (checkpointed, resumable, sequential scenarios):

```bash
python common/run_experiment.py --run-id run_YYYYMMDD_HHMMSS
```

Resume the same run after an interruption:

```bash
python common/run_experiment.py --run-id run_YYYYMMDD_HHMMSS
```

Run one scenario only:

```bash
python common/run_experiment.py --scenarios B1 --run-id run_YYYYMMDD_HHMMSS
```

Run analysis (read-only; never launches simulations):

```bash
python -c "import sys; sys.path.insert(0,'common'); from analysis import run_analysis; r=run_analysis(); print(r['report_path']); print(r['figure_dir'])"
```

Or open `notebooks/05_Analysis_Charts.ipynb`.

---

## Full run results (2026-11-23)

The full 240-configuration experiment completed:

- Run: `results/run_20261123_202600/` (240/240 configurations, 48 s wall)
- Manifest: `results/run_20261123_202600/manifest.json`
- Tables: `analysis/tables/run_20261123_202600_20261123_202328/`
- Figures: `analysis/figures/run_20261123_202600_20261123_202328/`
- Narrative summary: `RESULTS_SUMMARY.md`
- Final audit: `FINAL_AUDIT.md`

Headline findings (from actual data; O2 is the only validated objective):

- **O2 (Competition-aware)** has the highest mean final Net Advantage in both
  modes: 21.62 (SSA only) and 21.64 (SSA+Swap) across 20 configurations each.
- **O1 (Original SSA)** is statistically indistinguishable from O2 overall
  (mean difference −1.04, 95 % CI [−2.69, +0.57]).
- **O2 is better on average than O3/O4/O5/O6**; their mean per-config
  differences have CIs excluding zero.
- **O3 (maximise Blocked_B)** has the highest mean Blocked_B (19.26) but the
  lowest mean Net Advantage (16.38). Blocked_B–Net Advantage correlation is only
  0.36.
- **SSA+Swap is not a reliable per-configuration improvement**: mean difference
  +0.51, 95 % CI [+0.13, +0.89], median 0.00, 45 % positive, 5/120
  positive-significant and 2/120 negative-significant.
- **Optimizer fitness is not a valid ranking metric**: its correlation with final
  Net Advantage is strong only for O2 (0.71–0.80) and near zero for O1/O3.

The smoke test (`results/smoke/SMOKE_TEST_20261123_202100/`) is kept separate
from the real results.

---

## Notebook order

1. `00_Setup_Synthetic_Network.ipynb` — validate network/fingerprint/C++ status
2. `B1_Random_opponent.ipynb`
3. `B2_Degree-based_opponent.ipynb`
4. `B3_Positive_PageRank_opponent.ipynb`
5. `B4_Community-concentrated_opponent.ipynb`
6. `05_Analysis_Charts.ipynb`

The B notebooks default to `RUN_MODE='none'` and require
`CONFIRM_FULL_RUN=True` for a full scenario. The CLI is recommended for
overnight runs.

---

## Output schema

Each scenario writes `experiment_results.csv` with 72 columns, including:

- run/scenario/K/objective/mode, objective formula and validity
- network file + semantic fingerprint
- master seed, optimization/final MC counts, seed protocols
- SSA population/iterations/mutation/random seed
- candidate-pool size, ranked/random counts, pool signature
- initial/final optimizer score and final evaluation objective score
- Spread_A, Spread_B, Net_Advantage, Gain_A, Blocked_B, baseline_B
- ignorant nodes, average rounds
- MC standard errors and bootstrap CIs
- SSA/Swap/final-evaluation/total runtimes
- swap tests/accepted swap
- SSA, Swap-initial, final and opponent seed sets (JSON)
- config signature hash, checkpoint/history/MC file paths, status, timestamp

Per-seed raw outcomes are stored in `mc_runs/`; SSA histories in `ssa_history/`;
per-configuration checkpoints in `checkpoints/`.

---

## Reproducibility and fairness

- One shared network for all scenarios; fingerprint verified before each run.
- SSA run once per (scenario, K, objective); both modes use the saved seed set.
- SSA and Swap RNG seeds are derived deterministically from the master seed.
- Optimization uses 15 fixed MC seeds; final evaluation uses 100 fixed MC seeds
  shared across all configurations (common random numbers), enabling paired
  comparisons.
- Checkpoints and CSVs are written atomically; resume never duplicates completed
  configurations and rejects signature mismatches.
- Smoke tests live under `results/smoke/` and are never mixed with real results.

---

## Scientific caveats

1. **No reference notebooks exist in the workspace.** Definitions are audited
   against the starter code and master prompt, not an external canonical notebook.
2. **Blocked_B is provisional.** Only O2 is fully validated. The other five
   objectives are exploratory and labelled as such.
3. **SSA/Swap fidelity** is preserved from the starter; canonical equation-level
   fidelity cannot be independently verified without the missing reference.
4. **Statistical unit.** Monte Carlo replicates are common random numbers, not
   independent optimizer runs. Between-optimizer-run variance is not estimated.
   Do not treat MC replicates as replication of the optimizer.
5. **External validity.** The synthetic network is denser and less reciprocal
   than typical signed social networks. Results are conditional on this design.

See `VALIDATION_GATE.md`, `FINAL_AUDIT.md`, and `AUDIT_REPORT_ADDENDUM.md` for
the full evidence trail.
