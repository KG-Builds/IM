# Final Report — 5,000-Node Reverse-Influence Experiment

Run: `run_20261124_5000_reversed`  
Generated: 2026-11-23T20:26:39+00:00  
Wall time: 2026-11-23T20:13:14+00:00 → 2026-11-23T20:15:51+00:00  
Completed: 240/240 configurations  
Validation: **PASS**

---

## A. Changes made

1. **Influence direction corrected.** The trust CSV records `source → target`; the influence graph is now the exact reverse, `target → source`. The reversal is applied once in `common/network.load_influence_network`; all downstream components (diffusion, C++ engine, Python reference, candidate pool, B1–B4, SSA, Swap) use that influence graph as-is. No double reversal.
2. **New network.** 5,000 nodes and 37,500 directed trust arcs generated once with seed 42; the influence graph has 37,500 reversed arcs.
3. **Old results isolated.** All pre-correction results, analysis tables/figures, logs and derived reports were moved (not deleted) to `experiments/exp_20261124_5000_reversed/to_be_discarded/`, with a move manifest.
4. **Tests added.** Deterministic trust→influence reversal tests, signed propagation tests, candidate/degree/PageRank direction tests, 5,000-node network validation, seed-42 regeneration, and C++/Python consistency checks.
5. **Dedicated experiment directory** with config, network, raw results, aggregated tables, final figures, reports, logs, checkpoints, validation, and `to_be_discarded/`.

## B. Network

- Nodes: **5000**
- Trust arcs (CSV): **37500**
- Influence arcs (reversed): **37500**
- Directed: **True**
- Trust direction: `source → target`
- Influence direction: `target → source` (exact reverse)
- Direction check: all trust edges reversed = **True**; missing = 0; extra = 0; same-direction = 0
- Reciprocal trust pairs: 0
- Positive / negative trust arcs: 18706 / 18794
- Trust magnitude distribution: {'1': 3728, '2': 3689, '3': 3734, '4': 3679, '5': 3753, '6': 3681, '7': 3798, '8': 3818, '9': 3848, '10': 3772}
- Probability classes 0.1 / 0.5 / 0.9: {'0.1': 11151, '0.5': 14911, '0.9': 11438}
- Trust file SHA-256: `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e`
- Trust semantic SHA-256: `576d00314530fc82c86d5a433e13a2ea64dbe135723350d936df21b082e2b20f`
- Influence semantic SHA-256: `b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9`
- Generation seed: 42

## C. Experiment

- Configurations: **240 / 240** (4 scenarios × 5 K × 6 objectives × 2 modes)
- Simulations executed: **202,055**
- Optimization Monte Carlo runs: 15 per evaluation
- Final evaluation Monte Carlo runs: 100 per configuration
- SSA: population 12, iterations 10, mutation 0.20
- Swap: up to 100 sampled one-for-one tests, single best improving swap
- Summed per-configuration compute time: 225.6 s
- Wall time: 2026-11-23T20:13:14+00:00 → 2026-11-23T20:15:51+00:00
- Failed/skipped runs: **none**

## D. Results

### D.1 Winner per B × K — SSA only

| scenario   |   K | winner   |   winner_net_advantage | is_tie   | runner_up   |   margin |
|:-----------|----:|:---------|-----------------------:|:---------|:------------|---------:|
| B1         |  10 | O4       |                 19.080 | False    | O5          |    5.050 |
| B1         |  20 | O1       |                 53.000 | False    | O6          |    2.820 |
| B1         |  30 | O1       |                 55.460 | False    | O6          |    4.710 |
| B1         |  40 | O2       |                 74.240 | False    | O3          |    8.560 |
| B1         |  50 | O1       |                 81.540 | False    | O5          |    0.610 |
| B2         |  10 | O5       |                 22.370 | False    | O2          |    0.970 |
| B2         |  20 | O1       |                 48.930 | False    | O2          |   10.570 |
| B2         |  30 | O3       |                 56.530 | False    | O6          |    5.180 |
| B2         |  40 | O1       |                 68.920 | False    | O6          |    2.470 |
| B2         |  50 | O5       |                 67.850 | False    | O4          |    0.310 |
| B3         |  10 | O1       |                 39.080 | False    | O5          |    0.280 |
| B3         |  20 | O3       |                 46.740 | False    | O4          |    5.130 |
| B3         |  30 | O2       |                 68.850 | False    | O6          |    5.710 |
| B3         |  40 | O4       |                 60.910 | False    | O6          |    4.250 |
| B3         |  50 | O5       |                 76.540 | False    | O6          |    3.530 |
| B4         |  10 | O3       |                 29.330 | False    | O5          |    1.030 |
| B4         |  20 | O3       |                 56.190 | False    | O1          |    9.960 |
| B4         |  30 | O5       |                 68.300 | False    | O2          |    3.840 |
| B4         |  40 | O3       |                 66.630 | False    | O5          |    2.660 |
| B4         |  50 | O2       |                 77.550 | False    | O1          |    2.830 |

### D.2 Winner per B × K — SSA + Swap refinement

| scenario   |   K | winner   |   winner_net_advantage | is_tie   | runner_up   |   margin |
|:-----------|----:|:---------|-----------------------:|:---------|:------------|---------:|
| B1         |  10 | O4       |                 19.080 | False    | O2          |    4.590 |
| B1         |  20 | O1       |                 61.750 | False    | O6          |   12.830 |
| B1         |  30 | O1       |                 52.260 | False    | O6          |    1.510 |
| B1         |  40 | O2       |                 77.870 | False    | O6          |    4.180 |
| B1         |  50 | O4       |                 85.300 | False    | O6          |    1.910 |
| B2         |  10 | O4       |                 32.700 | False    | O1          |    5.710 |
| B2         |  20 | O2       |                 45.430 | False    | O5          |    4.470 |
| B2         |  30 | O3       |                 56.530 | False    | O4          |    2.550 |
| B2         |  40 | O1       |                 67.240 | False    | O6          |    0.790 |
| B2         |  50 | O4       |                 67.540 | False    | O5          |    0.160 |
| B3         |  10 | O1       |                 33.500 | False    | O5          |    2.560 |
| B3         |  20 | O4       |                 46.820 | False    | O3          |    0.400 |
| B3         |  30 | O5       |                 61.270 | False    | O6          |    1.750 |
| B3         |  40 | O6       |                 64.160 | False    | O4          |    3.250 |
| B3         |  50 | O6       |                 74.820 | False    | O1          |    8.370 |
| B4         |  10 | O3       |                 28.050 | False    | O5          |    1.690 |
| B4         |  20 | O3       |                 57.820 | False    | O4          |    0.070 |
| B4         |  30 | O2       |                 64.460 | False    | O5          |    9.350 |
| B4         |  40 | O3       |                 66.630 | False    | O2          |    4.290 |
| B4         |  50 | O1       |                 85.850 | False    | O6          |   15.430 |

Full per-variant values for every B × K comparison are in `aggregated/winner_table_SSA_only.csv`, `aggregated/winner_table_SSA_plus_Swap_refinement.csv`, and `aggregated/winner_table_Combined.csv`.

### D.3 Win / tie / loss counts

**SSA only**

| candidate   |   strict_wins |   ties |   losses | always_wins   |
|:------------|--------------:|-------:|---------:|:--------------|
| O1          |             6 |      0 |       14 | False         |
| O3          |             5 |      0 |       15 | False         |
| O5          |             4 |      0 |       16 | False         |
| O2          |             3 |      0 |       17 | False         |
| O4          |             2 |      0 |       18 | False         |
| O6          |             0 |      0 |       20 | False         |

**SSA + Swap refinement**

| candidate   |   strict_wins |   ties |   losses | always_wins   |
|:------------|--------------:|-------:|---------:|:--------------|
| O1          |             5 |      0 |       15 | False         |
| O4          |             5 |      0 |       15 | False         |
| O3          |             4 |      0 |       16 | False         |
| O2          |             3 |      0 |       17 | False         |
| O6          |             2 |      0 |       18 | False         |
| O5          |             1 |      0 |       19 | False         |

Note: **O1, O4** are tied for the most strict wins in SSA+Swap (5/20 each).

**Combined 12 candidates (6 objectives × 2 modes)**

| candidate                  |   strict_wins |   ties |   losses | always_wins   |
|:---------------------------|--------------:|-------:|---------:|:--------------|
| O1 | SSA only              |             4 |      0 |       16 | False         |
| O4 | SSA + Swap refinement |             3 |      1 |       16 | False         |
| O5 | SSA only              |             3 |      0 |       17 | False         |
| O1 | SSA + Swap refinement |             2 |      0 |       18 | False         |
| O3 | SSA only              |             1 |      2 |       17 | False         |
| O3 | SSA + Swap refinement |             1 |      2 |       17 | False         |
| O2 | SSA only              |             1 |      0 |       19 | False         |
| O2 | SSA + Swap refinement |             1 |      0 |       19 | False         |
| O6 | SSA + Swap refinement |             1 |      0 |       19 | False         |
| O4 | SSA only              |             0 |      1 |       19 | False         |
| O6 | SSA only              |             0 |      0 |       20 | False         |
| O5 | SSA + Swap refinement |             0 |      0 |       20 | False         |

### D.4 Is there an always-winner?

**No single SSA variant consistently wins every B scenario and every K value.**

No objective has 20 strict wins with zero ties and zero losses in either mode or in the combined 12-candidate comparison.

### D.5 Best average is not the same as consistent winner

Best mean Net Advantage across the 20 configurations:

| mode                  | objective_id   |   mean_net_advantage |   rank_in_mode |
|:----------------------|:---------------|---------------------:|---------------:|
| SSA + Swap refinement | O6             |               49.412 |              1 |
| SSA + Swap refinement | O1             |               48.643 |              2 |
| SSA + Swap refinement | O2             |               47.409 |              3 |
| SSA + Swap refinement | O4             |               47.099 |              4 |
| SSA + Swap refinement | O5             |               45.681 |              5 |
| SSA + Swap refinement | O3             |               43.257 |              6 |
| SSA only              | O1             |               47.878 |              1 |
| SSA only              | O5             |               47.633 |              2 |
| SSA only              | O6             |               47.261 |              3 |
| SSA only              | O2             |               47.211 |              4 |
| SSA only              | O3             |               44.480 |              5 |
| SSA only              | O4             |               43.241 |              6 |

- Best average in SSA only: **O1** (47.878).
- Best average in SSA+Swap: **O6** (49.412).
- Most strict wins in SSA only: **O1** (6/20).
- Most strict wins in SSA+Swap: **O1, O4** (5/20 each).
- Most strict wins combined: **O1 | SSA only** (4/20).

The highest average objective is not the most consistent winner. A high mean can come from a few large wins while losing many individual B × K comparisons.

## E. Figures

Only the four required final Net Advantage figures were produced:

- `/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B1_Random.png`
- `/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B2_Degree.png`
- `/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B3_Positive_PageRank.png`
- `/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/final_figures/Net_Advantage_B4_Community.png`

Each figure has two panels (SSA-only and SSA+Swap), six objective lines, 95% Monte Carlo CIs, consistent colors/markers, and a shared legend.

## F. Validation

- Test suite: **121 tests, 0 failures**.
- Deterministic direction tests: trust `A → B` becomes influence `B → A`; positive/negative propagation, multiple edges, candidate scoring, B2 degree and B3 PageRank all verified on the influence graph.
- C++ and Python reference consistency: deterministic tiny graphs exact; large-network invariants checked; distributional parity checked on small random graphs.
- Replay check: 5/5 sampled configurations reproduced their stored per-seed MC arrays exactly; one SSA checkpoint reproduced its seed set and objective score exactly (`validation/replay_check.json`).
- Network validation: 5,000 nodes, 37,500 trust arcs, 37,500 influence arcs, no self-loops, no duplicate arcs, seed-42 regeneration identical.
- Smoke test: `SMOKE_5000_reversed_20261124` — 4/4, resume and analysis ingestion pass.
- Full run validation: PASS; 240 unique rows; 240 MC files ×100 rows; 120 checkpoints; one run ID/fingerprint/direction/count; no seed-set violations.
- Old results: excluded from all calculations; moved to `to_be_discarded/` with manifest.

## G. Limitations

- **One optimizer run per configuration.** The winner analysis is an empirical comparison of one SSA run per B × K × objective; it does not estimate between-optimizer-run variance.
- **Recommended for stronger claims:** repeat SSA optimization with 5–10 independent optimizer seeds per configuration and analyse between-run variance. This was not performed here.
- **Monte Carlo uncertainty only.** The CIs in the figures reflect 100 MC evaluation seeds, not optimizer stochasticity.
- **Strict empirical winner.** A win is based on the observed mean Net Advantage; it is not a statistical-significance claim. Ties are reported only for exact/near-exact equality.
- **Synthetic network.** Results are conditional on this 5,000-node synthetic design and the reversed influence convention.
- **No external validity claim** on real signed networks.

## H. Recommendation

**No SSA variant consistently wins every opponent scenario and every K value.**

If your priority is the most consistent empirical performance, **O1 | SSA only** is the closest alternative in the combined comparison (4 strict wins, 0 ties, 16 losses out of 20). In SSA-only, **O1** has the most strict wins (6/20); in SSA+Swap, **O1, O4** are tied for the most (5/20 each).

If your priority is the highest average Net Advantage, **O1** is best in SSA-only and **O6** is best in SSA+Swap, but neither is a consistent per-comparison winner.

Do not describe any variant as universally superior based on this single optimizer run.
