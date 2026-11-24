# ContextForGPT — 5,000-Node Reverse-Influence Rerun

This document is a self-contained handover for an external assistant (ChatGPT)
that does not have access to the repository. It explains the **new
developments after the earlier reference-aligned work**:

- the influence direction was corrected to be the exact reverse of the trust
  direction;
- the network was increased to **5,000 nodes / 37,500 edges**;
- all old 1,000-node same-direction results were isolated in `to_be_discarded/`;
- the full experiment was rerun;
- exactly four final Net Advantage figures were produced;
- a 20-comparison winner analysis was performed to test whether one SSA variant
  wins every B × K combination.

The earlier handover is retained as
`GPT_Contexts/ContextForGPT.md`. It describes the pre-5,000-node work and is
historical.

---

## 1. One-paragraph summary

The new authoritative experiment uses a 5,000-node signed synthetic network with
37,500 directed **trust** arcs (`source → target`). The **influence** graph is
the exact reverse (`target → source`), applied once at load time and used by the
C++ diffusion engine, Python reference, candidate pool, B1–B4 opponent
strategies, SSA, Swap and final evaluation. The full 4 scenarios × 5 K × 6
objectives × 2 modes = 240 configurations completed successfully
(`run_20261124_5000_reversed`), with 202,055 simulations and 121 passing tests.
The main winner result is negative: **no single SSA variant consistently wins
every B scenario and every K value**. The closest alternative by strict wins is
**O1**; the highest average Net Advantage is **O1 in SSA-only** and **O6 in
SSA+Swap**, but best average is not consistent winning. Exactly four final
figures were produced, one per B scenario, each with SSA-only and SSA+Swap
panels.

---

## 2. What changed relative to the earlier context

| Area | Earlier state | Current state |
|---|---|---|
| Network size | 1,000 nodes / 7,500 arcs | **5,000 nodes / 37,500 arcs** |
| Influence direction | Same as trust (`source → target`) | **Exact reverse (`target → source`)** |
| Old results | Retained alongside corrected results | **Moved to `to_be_discarded/` with manifest; excluded** |
| Result schema | Schema 3 | **Schema 4**, includes influence direction, influence fingerprint, node/edge counts |
| Tests | 110 | **121** (direction + 5,000-node tests added) |
| Experiment runner | `common/run_experiment.py` generic | Dedicated `experiments/exp_20261124_5000_reversed/run_experiment_5000.py` |
| Analysis | Generic 11-chart analysis | Dedicated `analyze_5000.py`: aggregated tables, 20-comparison winner analysis, exactly four final figures, final report |
| Final figures | 11 charts | **Exactly four Net Advantage PNGs**, one per B scenario |

---

## 3. New conversation request

The user asked for:

1. **Correct influence direction**: trust edge `A → B` must become influence
   edge `B → A`. Influence must not propagate in the trust direction.
2. **Do not delete old results**: move old/incorrect results to a clearly named
   `to_be_discarded/` folder with a manifest.
3. **New network**: 5,000 nodes, 37,500 edges, generated once and validated.
4. **Revalidate everything** after the direction and size changes.
5. **Full rerun** with the corrected implementation.
6. **Only four final charts**: one Net Advantage line chart per B scenario.
7. **Find the SSA variant that always wins** across all 20 B × K comparisons.
8. Be statistically honest: no fabrication, no missing runs, no old results.

This document records the result of that work.

---

## 4. Current authoritative paths

| Item | Path |
|---|---|
| Experiment directory | `experiments/exp_20261124_5000_reversed/` |
| Full run | `experiments/exp_20261124_5000_reversed/raw_results/run_20261124_5000_reversed/` |
| Configuration | `experiments/exp_20261124_5000_reversed/config/experiment_config.json` |
| Network | `experiments/exp_20261124_5000_reversed/network/synthetic_network_5000_37500.csv` |
| Network metadata | `experiments/exp_20261124_5000_reversed/network/network_metadata.json` |
| Network validation | `experiments/exp_20261124_5000_reversed/network/network_validation.json` |
| Aggregated tables | `experiments/exp_20261124_5000_reversed/aggregated/` |
| Winner analysis | `experiments/exp_20261124_5000_reversed/aggregated/winner_summary.md` |
| Final figures | `experiments/exp_20261124_5000_reversed/final_figures/` |
| Final report | `experiments/exp_20261124_5000_reversed/reports/FINAL_REPORT.md` |
| Validation gate | `experiments/exp_20261124_5000_reversed/validation/VALIDATION_GATE.md` |
| Validation summary | `experiments/exp_20261124_5000_reversed/validation/validation_summary.md` |
| Replay check | `experiments/exp_20261124_5000_reversed/validation/replay_check.json` |
| Test output | `experiments/exp_20261124_5000_reversed/validation/tests_5000.txt` |
| Old results | `experiments/exp_20261124_5000_reversed/to_be_discarded/` |
| Move manifest | `experiments/exp_20261124_5000_reversed/to_be_discarded/MOVE_MANIFEST.csv` |
| Earlier context | `GPT_Contexts/ContextForGPT.md` |

---

## 5. Network

### 5.1 Trust vs influence direction

The network CSV is the **trust** network:

```
source → target, rating, probability, strength, polarity
```

The **influence** graph is the exact reverse:

```
target → source
```

The reversal is applied exactly once in
`common/network.py::load_influence_network`. No downstream component reverses it
again. All graph-based operations use the influence graph:

- C++ diffusion engine
- Python reference simulator
- candidate-pool polarity scores
- B1 random sampling (direction-independent)
- B2 degree (influence out-degree)
- B3 positive PageRank (positive influence edges)
- B4 Louvain community (undirected influence projection; degree ranking uses
  influence out-degree)
- SSA fitness
- Swap refinement
- final Monte Carlo evaluation

### 5.2 Network statistics

| Parameter | Value |
|---|---|
| Nodes | 5,000 (ids 0–4999) |
| Trust arcs | 37,500 |
| Influence arcs | 37,500 |
| Directed | yes |
| Underlying unordered pairs | 37,500 |
| Reciprocal trust pairs | 0 |
| Self-loops | 0 |
| Duplicate directed arcs | 0 |
| Positive / negative trust arcs | 18,706 / 18,794 |
| Probability classes 0.1 / 0.5 / 0.9 | 11,151 / 14,911 / 11,438 |
| Trust magnitude | uniform integer 1–10 |
| Generation seed | 42 |
| Topology | Barabási–Albert backbone (initial clique 8, m=7), random fill to 37,500 undirected pairs, random orientation |

Direction check:

- all 37,500 trust edges reversed: **true**
- missing reversed edges: **0**
- extra influence edges: **0**
- same-direction edges: **0**
- reciprocal trust pairs: **0**

Hashes:

```
Trust file SHA-256:
d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e

Trust semantic SHA-256:
576d00314530fc82c86d5a433e13a2ea64dbe135723350d936df21b082e2b20f

Influence semantic SHA-256:
b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9
```

Signs and trust magnitudes are randomly assigned from the fixed seed; seed-42
regeneration is identical.

---

## 6. Code and logic changes

### 6.1 Network generation

`common/network_generator_large.py`:

- `nx.barabasi_albert_graph(5000, 7, seed=42, initial_graph=complete_graph(8))`
- random undirected fill to exactly 37,500 pairs
- each pair oriented uniformly at random
- sign positive with probability 0.5
- trust magnitude uniform integer 1–10
- probability mapping 1–3 → 0.1, 4–7 → 0.5, 8–10 → 0.9
- hard checks: exact edge count, no duplicates, no self-loops, contiguous nodes,
  `rating = polarity × strength`

The legacy `common/network_generator.py` remains unchanged for the old
1,000-node network.

### 6.2 Influence loading and validation

`common/network.py` now provides:

- `load_influence_network(path, validate=True, n_nodes=..., n_edges=..., reverse=True)`
- `direction_check(df, G_influence)`
- `influence_semantic_fingerprint(df)`
- `build_influence_metadata_v2(...)`

The reversal is single-pass and verified edge-by-edge.

### 6.3 Experiment schema and runner

`common/experiment.py`:

- `ExperimentConfig` now includes `influence_direction`, `expected_nodes`,
  `expected_edges`.
- `ExperimentRunner` loads the influence graph when direction is `"reverse"`.
- cache key includes file fingerprint + direction + influence fingerprint.
- result schema is version 4 and includes:
  - `influence_direction`
  - `influence_semantic_fingerprint`
  - `network_nodes`
  - `network_edges`
- checkpoint signature includes direction and counts, so old and new
  configurations cannot be mixed.
- `validate_results` checks one direction/count/fingerprint per run.

Dedicated runner:
`experiments/exp_20261124_5000_reversed/run_experiment_5000.py`

### 6.4 Tests

`tests/test_influence_direction.py` adds deterministic tests:

- trust `A → B` becomes influence `B → A`
- positive propagation only in the reverse direction
- negative propagation/flip only in the reverse direction
- multiple-edge reverse chain
- candidate pool uses influence outgoing edges
- B2 degree uses influence out-degree
- B3 PageRank uses influence direction
- 5,000-node counts/direction validation
- seed-42 regeneration deterministic
- C++/Python consistency and invariants on the large graph

All tests use persistent `tests_tmp/` directories because deletion was
disallowed.

### 6.5 Old result handling

`tools/move_old_results.py` moved 30 old items (results, analysis, logs,
derived reports) into
`experiments/exp_20261124_5000_reversed/to_be_discarded/` with
`MOVE_MANIFEST.csv`. Nothing was deleted. Old results are excluded from all new
calculations.

---

## 7. Execution flow

### 7.1 Build and test

```bash
cd cpp_engine
python setup.py build_ext --inplace
cd ..

python -m unittest discover -s tests -p 'test_*.py'
```

### 7.2 Dedicated experiment runner

```bash
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode plan
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode smoke
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode full --confirm-full --run-id run_20261124_5000_reversed
```

Flow:

1. Load trust CSV, validate, build influence graph (reverse once).
2. Build persistent C++ engine.
3. Loop scenarios B1–B4.
4. Loop K = 10,20,30,40,50.
5. Build B seed set from influence graph.
6. Build 10·K candidate pool from influence graph.
7. Derive optimization/final/SSA/Swap/pool seeds.
8. Run SSA once per `(scenario, K, objective)`, save checkpoint.
9. Evaluate SSA-only and SSA+Swap from the same SSA set.
10. Write per-seed MC files and CSV row atomically.
11. Write scenario metadata and manifest.

### 7.3 Analysis

```bash
python experiments/exp_20261124_5000_reversed/analyze_5000.py
```

This reads only the new run and produces:

- aggregated tables in `aggregated/`
- winner tables and counts
- exactly four final figures in `final_figures/`
- `reports/FINAL_REPORT.md`
- `validation/validation_summary.md`

---

## 8. Experiment design

- Scenarios: B1 Random, B2 Degree, B3 Positive PageRank, B4 Community.
- K = 10, 20, 30, 40, 50.
- Objectives O1–O6.
- Modes: SSA-only and SSA + Swap refinement.
- Configurations: 4 × 5 × 6 × 2 = **240**.
- SSA: population 12, iterations 10, mutation 0.20.
- Optimization MC: 15 seeds.
- Final evaluation MC: 100 seeds.
- Candidate pool: 10·K, 70 % ranked / 30 % random from ranked remainder.
- Swap: up to 100 sampled one-for-one tests, single best improving swap.
- SSA is run once per `(scenario, K, objective)`; both modes share its seed set.

---

## 9. Results

Run: `run_20261124_5000_reversed`  
Completed: 240/240  
Simulations: 202,055  
Wall time: ≈2m37s  
Summed per-config compute: 225.6 s  
All rows `objective_validity = validated`.

### 9.1 SSA-only winner per B × K

| B | K | Winner | Winner Net Adv |
|---|---:|---|---:|
| B1 | 10 | O4 | 19.08 |
| B1 | 20 | O1 | 53.00 |
| B1 | 30 | O1 | 55.46 |
| B1 | 40 | O2 | 74.24 |
| B1 | 50 | O1 | 81.54 |
| B2 | 10 | O5 | 22.37 |
| B2 | 20 | O1 | 48.93 |
| B2 | 30 | O3 | 56.53 |
| B2 | 40 | O1 | 68.92 |
| B2 | 50 | O5 | 67.85 |
| B3 | 10 | O1 | 39.08 |
| B3 | 20 | O3 | 46.74 |
| B3 | 30 | O2 | 68.85 |
| B3 | 40 | O4 | 60.91 |
| B3 | 50 | O5 | 76.54 |
| B4 | 10 | O3 | 29.33 |
| B4 | 20 | O3 | 56.19 |
| B4 | 30 | O5 | 68.30 |
| B4 | 40 | O3 | 66.63 |
| B4 | 50 | O2 | 77.55 |

No ties in the SSA-only table.

### 9.2 SSA + Swap winner per B × K

| B | K | Winner | Winner Net Adv |
|---|---:|---|---:|
| B1 | 10 | O4 | 19.08 |
| B1 | 20 | O1 | 61.75 |
| B1 | 30 | O1 | 52.26 |
| B1 | 40 | O2 | 77.87 |
| B1 | 50 | O4 | 85.30 |
| B2 | 10 | O4 | 32.70 |
| B2 | 20 | O2 | 45.43 |
| B2 | 30 | O3 | 56.53 |
| B2 | 40 | O1 | 67.24 |
| B2 | 50 | O4 | 67.54 |
| B3 | 10 | O1 | 33.50 |
| B3 | 20 | O4 | 46.82 |
| B3 | 30 | O5 | 61.27 |
| B3 | 40 | O6 | 64.16 |
| B3 | 50 | O6 | 74.82 |
| B4 | 10 | O3 | 28.05 |
| B4 | 20 | O3 | 57.82 |
| B4 | 30 | O2 | 64.46 |
| B4 | 40 | O3 | 66.63 |
| B4 | 50 | O1 | 85.85 |

No ties in the SSA+Swap table.

### 9.3 Win / tie / loss counts

**SSA only**

| Variant | Strict wins | Ties | Losses |
|---|---:|---:|---:|
| O1 | 6 | 0 | 14 |
| O3 | 5 | 0 | 15 |
| O5 | 4 | 0 | 16 |
| O2 | 3 | 0 | 17 |
| O4 | 2 | 0 | 18 |
| O6 | 0 | 0 | 20 |

**SSA + Swap**

| Variant | Strict wins | Ties | Losses |
|---|---:|---:|---:|
| O1 | 5 | 0 | 15 |
| O4 | 5 | 0 | 15 |
| O3 | 4 | 0 | 16 |
| O2 | 3 | 0 | 17 |
| O6 | 2 | 0 | 18 |
| O5 | 1 | 0 | 19 |

**Combined 12 candidates:** the only ties are O4 (both modes) at B1/K=10, and
O3 (both modes) at B2/K=30 and B4/K=40. Full table:
`aggregated/winner_table_Combined.csv`.

### 9.4 Is there an always-winner?

**No single SSA variant consistently wins every B scenario and every K value.**

No variant has 20 strict wins with zero ties and zero losses in either mode or
in the combined 12-candidate comparison.

### 9.5 Best average vs consistent winner

| Mode | Best mean Net Advantage | Value |
|---|---|---:|
| SSA only | O1 | 47.878 |
| SSA + Swap | O6 | 49.412 |

- Most strict wins in SSA-only: **O1** (6/20).
- Most strict wins in SSA+Swap: **O1 and O4** (5/20 each).
- Most strict wins combined: **O1 | SSA only** (4/20).

O6 has the best SSA+Swap average but only 2/20 strict wins; O1 has the most
consistent strict wins. Best average is not the same as consistent winning.

---

## 10. Final figures

Exactly four PNGs, one per B scenario:

- `experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B1_Random.png`
- `experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B2_Degree.png`
- `experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B3_Positive_PageRank.png`
- `experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B4_Community.png`

Each has:

- two panels: SSA-only (left) and SSA+Swap (right)
- Net Advantage vs K for O1–O6
- 95% Monte Carlo CIs over 100 evaluation seeds
- consistent colors, markers, line styles and shared legend
- clear titles and high-resolution output (300 dpi)

All four were inspected visually and programmatically; no clipping, legend
overlap or unreadable labels.

---

## 11. Validation

- **121 tests, 0 failures** (`validation/tests_5000.txt`).
- Deterministic direction tests: trust `A → B` → influence `B → A`; positive,
  negative, multiple-edge, candidate pool, B2 degree, B3 PageRank.
- C++/Python: deterministic tiny graphs exact; large-network invariants checked;
  distributional parity checked on small random graphs.
- **Replay check:** 5/5 sampled configurations reproduced their stored per-seed
  MC arrays exactly; one SSA checkpoint reproduced its seed set and objective
  score exactly (`validation/replay_check.json`).
- Network: 5,000 nodes, 37,500 trust/influence arcs, no self-loops, no duplicate
  arcs, seed-42 regeneration identical.
- Smoke: `SMOKE_5000_reversed_20261124` — 4/4, resume and analysis ingestion.
- Full run: 240/240, validation PASS, 240 MC files ×100 rows, 120 checkpoints,
  one run/fingerprint/direction/count, valid paired seed sets.
- Old results: moved to `to_be_discarded/` with manifest; not used anywhere.

---

## 12. Old results handling

The following were moved, not deleted, into
`experiments/exp_20261124_5000_reversed/to_be_discarded/`:

- old full runs `run_20261123_202600` and
  `run_20261123_205500_reference_aligned`
- old smoke runs
- old analysis tables and figures
- old run/console/smoke logs
- old `RESULTS_SUMMARY*`, `FINAL_AUDIT*`, `VALIDATION_GATE*`

The manifest records original path, new path, reason and confirmation status.
No old result is used in any new table, figure, comparison or conclusion.

---

## 13. Limitations

1. **One optimizer run per configuration.** The winner analysis is an empirical
   comparison of one SSA run per B × K × objective; it does not estimate
   between-optimizer-run variance.
2. **Monte Carlo uncertainty only.** Figure CIs reflect 100 MC evaluation seeds,
   not optimizer stochasticity.
3. **Strict empirical winner.** Wins are based on observed mean Net Advantage,
   not statistical significance. Ties are reported explicitly.
4. **Synthetic network.** Results are conditional on this 5,000-node design and
   the reversed influence convention.
5. **No external validity claim** on real signed networks.
6. For stronger claims, repeat SSA optimization with 5–10 independent optimizer
   seeds per configuration and analyse between-run variance.

---

## 14. Recommendation

**No SSA variant consistently wins every opponent scenario and every K value.**

- If priority is **consistent per-comparison performance**, **O1** is the
  closest alternative: 6/20 strict wins in SSA-only, and tied with O4 at 5/20 in
  SSA+Swap. It still loses 14–15 comparisons; do not call it universal.
- If priority is **highest average Net Advantage**, use **O1** for SSA-only and
  **O6** for SSA+Swap — but O6 is not consistent (2/20 strict wins in SSA+Swap,
  0/20 in SSA-only).
- **O1 is the safest single choice** for the most frequent empirical winner, but
  no variant dominates every scenario and budget in this experiment.

---

## 15. What ChatGPT should preserve

1. The authoritative experiment is `run_20261124_5000_reversed` under
   `experiments/exp_20261124_5000_reversed/`.
2. Trust direction is `source → target`; influence direction is the exact
   reverse `target → source`, applied once.
3. The network has 5,000 nodes and 37,500 trust/influence arcs; seed 42.
4. All old 1,000-node same-direction results are in `to_be_discarded/` and must
   not be used.
5. No single SSA variant always wins. O1 has the most strict wins; O6 has the
   best SSA+Swap average but is inconsistent.
6. Exactly four final figures exist; do not generate or use other charts as
   final presentation figures.
7. The winner analysis uses one optimizer run per configuration; do not claim
   statistical superiority or universal dominance.
8. The full report is
   `experiments/exp_20261124_5000_reversed/reports/FINAL_REPORT.md`.

---

## 16. Earlier context

The previous handover, `GPT_Contexts/ContextForGPT.md`, describes the earlier
1,000-node same-direction work, the reference-notebook audit, and the three
corrections made after the notebooks were supplied. It is historical; the
current document supersedes its results and direction convention.
