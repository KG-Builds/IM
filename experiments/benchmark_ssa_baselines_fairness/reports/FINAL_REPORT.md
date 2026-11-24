# Final Report — SSA O1/O4/O6 vs. Notebook Baselines and FIBMM

**Experiment:** `benchmark_ssa_baselines_fairness`  
**Network:** 5,000 nodes, 37,500 directed trust arcs, influence direction reversed (target → source), identical to `exp_20261124_5000_reversed`.  
**Primary metric:** Net Advantage = Spread_A − Spread_B (higher is better).  
**Secondary metrics:** Spread_A, Spread_B, Blocked_B, runtime.  
**Monte Carlo:** 15 optimization seeds; 100 common final-evaluation seeds.  
**Swap refinement:** excluded by instruction; no SSA+Swap or local-swap variant was run.

---

## 1. Experiment overview

This experiment compares three SSA objective variants (O1, O4, O6) in SSA-only mode against every valid seed-selection algorithm implemented in `old work/competitive_im_no_celf_csv_persist.ipynb`, plus the FIBMM fairness-aware influence-blocking baseline from Xiao et al. (IEEE TCSS 2025).  All methods are evaluated on the same current 5,000-node synthetic network, with the same signed edges, trust values, trust-to-probability mapping, competitive Maki–Thompson diffusion, opponent seeds, K values and 100 final Monte Carlo seeds.  The benchmark ranks methods by Net Advantage, not by Spread_A alone or by a blocking metric alone.

The notebook audit and the paper-to-code mapping are in `ALGORITHM_AUDIT.md` and `PAPER_IMPLEMENTATION.md`.  The methodology is in `METHODS.md`.

## 2. Network

| Property | Value |
|---|---|
| Nodes | 5,000 |
| Trust arcs | 37,500 |
| Influence arcs | 37,500 (exact reverse of trust) |
| File SHA-256 | `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e` |
| Influence semantic SHA-256 | `b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9` |
| Direction check | 0 missing reversed edges, 0 extra, 0 same-direction |

The network is not regenerated or duplicated.  Every result row records the same file hash and influence hash, and the validation section checks that no cell used a different graph or direction.  The machine-readable identity proof is `network/network_reference.json`; the direction/count validation is `network/network_validation.json`.

## 3. Algorithms

**Group A — SSA variants:** SSA O1, SSA O4, SSA O6.  SSA-only; no Swap refinement.

| ID | Formula | K-dependent coefficients |
|---|---|---|
| O1 | `Spread_A + 1.5·Blocked_B` | none |
| O4 | `Spread_A + 2.0·Blocked_B` | none |
| O6 | `α(K)·Spread_A + β(K)·Blocked_B` | K≤20: (0.75, 2.25); K≤40: (1.00, 1.75); K=50: (1.25, 1.50) |

**Group B — notebook algorithms:** Random, Top Degree, Signed PageRank, CRIEG (Hong 2021), NCGE (Liang 2025), Signed-GA (Chakraborty 2025), CELF (O1 fitness), and SSA-Notebook (the exact notebook SSA with O1 fitness and the notebook seed schedule).  CELF + sampled Local Search is excluded because it is a one-for-one swap-based post-processing step and the master task excludes Swap-based post-processing.

**Group C — fairness-aware baseline:** FIBMM (Xiao et al. 2025), with the paper's two-stage maximin-fairness greedy on a positive-edge CIC surrogate.  Final evaluation always uses the full signed Maki–Thompson model.

## 4. Experimental settings

| Setting | Value |
|---|---|
| Opponent scenarios | B1 Random, B2 Degree, B3 Positive PageRank, B4 Community |
| Seed budgets | K = 10, 20, 30, 40, 50 |
| Optimization MC | 15 common seeds per objective evaluation |
| Final evaluation MC | 100 common seeds, `900000000..900000099` |
| SSA | population 12, iterations 10, mutation 0.20 |
| SSA candidate pool | 10·K, 70% ranked / 30% random from ranked remainder |
| CELF/NCGE fitness | O1 weighted fitness (`Spread_A + 1.5·Blocked_B`), 15 MC seeds |
| FIBMM | h = 0.9, ε = 0.1, l = 1, Louvain communities, CRP cap 600,000 |
| Final metric | Net Advantage = Spread_A − Spread_B |

Opponent seed sets are nested across K and identical for every method within each scenario/K cell; this is checked programmatically.  The exact B sets are in `raw_results/opponent_seeds.csv`; the per-cell consistency signatures are in `aggregated/opponent_consistency.csv`.

Reproduction commands (from the experiment directory): `python run_benchmark.py --mode verify`, `python run_benchmark.py --mode full --confirm-full`, `python run_benchmark.py --mode analyze`.  The full run is checkpointed and skips valid existing cells.

## 5. Results

### 5.1 Algorithm-level summary (20 scenario × K cells)

| method_name            |   mean_net_advantage |   median_net_advantage |   strict_wins |   ties |   losses |   mean_rank |   mean_spread_A |   mean_spread_B |   mean_blocked_B |
|:-----------------------|---------------------:|-----------------------:|--------------:|-------:|---------:|------------:|----------------:|----------------:|-----------------:|
| SSA O1 (Original)      |               47.878 |                 50.120 |             6 |      0 |       14 |       3.150 |         995.559 |         947.681 |           33.379 |
| SSA O6 (Budget-aware)  |               47.261 |                 51.050 |             2 |      0 |       18 |       3.050 |         995.419 |         948.159 |           32.901 |
| CELF (O1 fitness)      |               46.115 |                 45.015 |             6 |      0 |       14 |       2.850 |         994.577 |         948.462 |           32.599 |
| SSA-Notebook (O1)      |               45.570 |                 46.245 |             3 |      0 |       17 |       3.250 |         993.274 |         947.704 |           33.356 |
| SSA O4 (Combined)      |               43.241 |                 45.720 |             3 |      0 |       17 |       3.200 |         991.445 |         948.203 |           32.857 |
| Signed-GA (2025)       |               28.207 |                 24.845 |             0 |      0 |       20 |       5.800 |         998.913 |         970.706 |           10.430 |
| FIBMM (fairness-aware) |               18.718 |                 17.150 |             0 |      0 |       20 |       7.650 |         965.715 |         946.998 |           34.063 |
| CRIEG (2021)           |               14.536 |                 10.685 |             0 |      0 |       20 |       8.200 |         958.606 |         944.069 |           36.991 |
| Signed PageRank        |                3.934 |                  6.175 |             0 |      0 |       20 |       9.700 |         961.004 |         957.069 |           24.036 |
| NCGE (2025)            |                2.685 |                  1.205 |             0 |      0 |       20 |      10.050 |         961.195 |         958.509 |           22.551 |
| Random                 |               -0.150 |                 -0.410 |             0 |      0 |       20 |      10.300 |         971.389 |         971.540 |            9.887 |
| Top Degree             |               -1.746 |                 -1.775 |             0 |      0 |       20 |      10.750 |         941.311 |         943.057 |           38.003 |

The average is descriptive.  The four opponent scenarios are deliberately different, not independent replicates, so they are not pooled into a significance test.

### 5.2 Winners by scenario and K

| scenario   |   K | winner       |   winner_net_advantage |   winner_spread_A |   winner_spread_B |   winner_blocked_B | runner_up    |   margin | difference_meaningful_95   |
|:-----------|----:|:-------------|-----------------------:|------------------:|------------------:|-------------------:|:-------------|---------:|:---------------------------|
| B1         |  10 | SSA_O4       |                 19.080 |          1000.470 |           981.390 |             19.240 | CELF         |    4.970 | False                      |
| B1         |  20 | SSA_O1       |                 53.000 |          1021.310 |           968.310 |             25.890 | SSA_O6       |    2.820 | False                      |
| B1         |  30 | CELF         |                 57.200 |          1020.540 |           963.340 |             43.580 | SSA_O1       |    1.740 | False                      |
| B1         |  40 | CELF         |                 64.740 |          1025.220 |           960.480 |             42.620 | SSA_Notebook |    1.280 | False                      |
| B1         |  50 | SSA_O1       |                 81.540 |          1047.350 |           965.810 |             46.110 | SSA_O6       |    3.460 | False                      |
| B2         |  10 | CELF         |                 22.680 |           982.680 |           960.000 |             21.130 | SSA_O4       |    1.680 | False                      |
| B2         |  20 | SSA_O1       |                 48.930 |           985.110 |           936.180 |             37.180 | SSA_O6       |   10.850 | True                       |
| B2         |  30 | SSA_Notebook |                 59.100 |           986.130 |           927.030 |             36.540 | SSA_O6       |    7.750 | False                      |
| B2         |  40 | SSA_O1       |                 68.920 |           997.540 |           928.620 |             36.430 | SSA_O6       |    2.470 | False                      |
| B2         |  50 | SSA_Notebook |                 76.700 |           981.950 |           905.250 |             55.730 | CELF         |    7.940 | False                      |
| B3         |  10 | SSA_O1       |                 39.080 |           989.540 |           950.460 |             20.750 | CELF         |   15.000 | True                       |
| B3         |  20 | CELF         |                 46.700 |           987.720 |           941.020 |             27.060 | SSA_O4       |    5.090 | False                      |
| B3         |  30 | SSA_O6       |                 63.140 |           994.160 |           931.020 |             36.090 | CELF         |    8.500 | False                      |
| B3         |  40 | SSA_O4       |                 60.910 |           986.570 |           925.660 |             36.520 | SSA_O6       |    4.250 | False                      |
| B3         |  50 | SSA_O6       |                 73.010 |          1001.020 |           928.010 |             34.740 | SSA_Notebook |    1.580 | False                      |
| B4         |  10 | CELF         |                 29.170 |           991.740 |           962.570 |             18.290 | SSA_Notebook |    0.500 | False                      |
| B4         |  20 | SSA_O1       |                 46.230 |           997.830 |           951.600 |             23.940 | SSA_O4       |    3.110 | False                      |
| B4         |  30 | SSA_Notebook |                 67.350 |          1002.210 |           934.860 |             57.450 | SSA_O4       |   15.300 | True                       |
| B4         |  40 | SSA_O4       |                 55.680 |          1003.450 |           947.770 |             39.730 | SSA_O6       |    3.810 | False                      |
| B4         |  50 | CELF         |                 92.680 |          1018.680 |           926.000 |             66.810 | SSA_Notebook |   13.980 | True                       |

All 20 cells have a unique winner (no ties; 0 tied cells).  Only 4 of 20 winner-vs-runner-up paired 95% intervals exclude zero; the remaining margins are within Monte Carlo uncertainty, so most per-cell rankings are descriptive rather than decisive.  The highest mean Net Advantage is **SSA O1 (Original)** (47.878); the most strict wins are **SSA O1 (Original), CELF (O1 fitness)** (6/20 each).  These are different criteria and need not identify the same method.

### 5.3 Winners by opponent scenario

- **B1 — Random opponent:** SSA_O1 (2), CELF (2), SSA_O4 (1)
- **B2 — Degree-based opponent:** SSA_O1 (2), SSA_Notebook (2), CELF (1)
- **B3 — Positive PageRank-based opponent:** SSA_O6 (2), SSA_O1 (1), CELF (1), SSA_O4 (1)
- **B4 — Community-concentrated opponent:** CELF (2), SSA_O1 (1), SSA_Notebook (1), SSA_O4 (1)

### 5.4 Winners by K

- **K = 10:** CELF (2), SSA_O4 (1), SSA_O1 (1)
- **K = 20:** SSA_O1 (3), CELF (1)
- **K = 30:** SSA_Notebook (2), CELF (1), SSA_O6 (1)
- **K = 40:** SSA_O4 (2), CELF (1), SSA_O1 (1)
- **K = 50:** SSA_O1 (1), SSA_Notebook (1), SSA_O6 (1), CELF (1)

### 5.5 Mean Net Advantage by opponent scenario

| method_name            |     B1 |     B2 |     B3 |     B4 |
|:-----------------------|-------:|-------:|-------:|-------:|
| SSA O1 (Original)      | 50.170 | 47.214 | 48.104 | 46.022 |
| SSA O4 (Combined)      | 42.252 | 43.026 | 41.962 | 45.726 |
| SSA O6 (Budget-aware)  | 48.516 | 46.750 | 47.704 | 46.072 |
| SSA-Notebook (O1)      | 44.384 | 44.222 | 44.358 | 49.316 |
| FIBMM (fairness-aware) | 15.292 | 18.914 | 23.452 | 17.214 |
| Random                 | -1.634 |  0.480 |  0.868 | -0.316 |
| Top Degree             | -1.244 | -4.760 | -0.806 | -0.176 |
| Signed PageRank        |  6.360 |  0.300 |  6.988 |  2.090 |
| CRIEG (2021)           | 18.824 | 17.336 | 18.926 |  3.058 |
| NCGE (2025)            |  2.482 |  8.828 |  5.592 | -6.160 |
| Signed-GA (2025)       | 23.566 | 31.338 | 32.318 | 25.606 |
| CELF (O1 fitness)      | 48.156 | 42.938 | 43.226 | 50.140 |

SSA O1 has its highest scenario mean in B1 (50.17), SSA O6 in B1/B3 (48.52/47.70), and CELF in B4 (50.14).  No SSA variant has the highest mean in every scenario; CELF is the best method by mean in B4, and SSA-Notebook is second in B4 (49.32).

### 5.6 Mean Net Advantage by K

| method_name            |      10 |     20 |     30 |      40 |     50 |
|:-----------------------|--------:|-------:|-------:|--------:|-------:|
| SSA O1 (Original)      |  18.755 | 47.312 | 47.407 |  56.085 | 69.827 |
| SSA O4 (Combined)      |  20.203 | 36.445 | 43.010 |  56.197 | 60.353 |
| SSA O6 (Budget-aware)  |  16.233 | 39.508 | 52.867 |  57.538 | 70.157 |
| SSA-Notebook (O1)      |  16.930 | 33.810 | 55.038 |  48.785 | 73.287 |
| FIBMM (fairness-aware) |   5.038 |  7.432 | 21.312 |  30.248 | 29.560 |
| Random                 |   1.860 |  4.900 |  1.498 | -10.140 |  1.130 |
| Top Degree             |  -1.445 | -0.312 | -1.265 |  -5.248 | -0.462 |
| Signed PageRank        | -10.045 |  7.950 | 10.307 |   5.647 |  5.812 |
| CRIEG (2021)           |   5.853 |  8.328 | 14.170 |  19.977 | 24.352 |
| NCGE (2025)            |  -2.615 |  1.185 |  6.803 |   4.375 |  3.680 |
| Signed-GA (2025)       |  13.208 | 19.822 | 28.327 |  35.645 | 44.032 |
| CELF (O1 fitness)      |  22.510 | 39.565 | 49.285 |  48.128 | 71.088 |

At K=10, CELF and SSA O4 lead; at K=20, SSA O1 leads; at K=30, SSA-Notebook/CELF/SSA O6 trade places; at K=40, SSA O4/SSA O6 lead; at K=50, SSA O1/SSA-Notebook/SSA O6/CELF are all close.  SSA generally improves with K, while CELF peaks at K=50 (71.09).  The K-dependent reversals are a substantive part of the result, not noise to be averaged away.

### 5.7 Computational cost

| method_name            |   mean_selection_runtime_sec |   total_selection_runtime_sec |   mean_optimization_simulations |
|:-----------------------|-----------------------------:|------------------------------:|--------------------------------:|
| Signed-GA (2025)       |                      237.668 |                      4753.357 |                           0.000 |
| FIBMM (fairness-aware) |                       46.103 |                       922.064 |                           0.000 |
| CELF (O1 fitness)      |                        4.346 |                        86.924 |                        8662.500 |
| NCGE (2025)            |                        3.668 |                        73.352 |                        9516.750 |
| SSA-Notebook (O1)      |                        0.731 |                        14.622 |                        1904.250 |
| CRIEG (2021)           |                        0.679 |                        13.586 |                           0.000 |
| SSA O6 (Budget-aware)  |                        0.678 |                        13.562 |                         100.000 |
| SSA O4 (Combined)      |                        0.675 |                        13.502 |                         100.000 |
| SSA O1 (Original)      |                        0.675 |                        13.494 |                         200.000 |
| Top Degree             |                        0.009 |                         0.187 |                           0.000 |
| Signed PageRank        |                        0.008 |                         0.158 |                           0.000 |
| Random                 |                        0.005 |                         0.093 |                           0.000 |

One-time preprocessing (Signed PageRank, NCGE centrality, Louvain
communities, opponent orders) is recorded separately in
`raw_results/preprocessing_runtimes.json` and
`aggregated/preprocessing_runtimes.csv`.  Selection runtimes are per
configuration and exclude those one-time costs.  SSA O1/O4/O6 all use the
same 132 objective evaluations; the simulation-count column reflects cache
hits in the reused prior run, not a different SSA search budget.

### 5.8 Robustness

- **K sensitivity:** mean Net Advantage ranges from -1.746 to 47.878 across methods; the per-K table shows that the ranking is not constant across K.
- **Opponent sensitivity:** the per-scenario table and winner table show different winners in different B scenarios; B1–B4 are not replicates.
- **Evaluation uncertainty:** the 100-seed 95% CIs are shown in the figures and winner table; the winner-vs-runner-up paired interval is marked when it excludes zero.
- **Optimizer variability:** only one optimizer run is available per method/cell.  The SSA-Notebook baseline is a second SSA run for O1 with a different seed schedule; its comparison with SSA O1 is a descriptive sensitivity check, not a variance estimate.
- **Computational cost:** selection time ranges from sub-second heuristics to hundreds of seconds for SignedGA and FIBMM; see the runtime table and chart.

Cells whose winner is not one of the three retained SSA objectives (O1/O4/O6):

| scenario   |   K | winner       |   winner_net_advantage | runner_up    |   margin |
|:-----------|----:|:-------------|-----------------------:|:-------------|---------:|
| B1         |  30 | CELF         |                 57.200 | SSA_O1       |    1.740 |
| B1         |  40 | CELF         |                 64.740 | SSA_Notebook |    1.280 |
| B2         |  10 | CELF         |                 22.680 | SSA_O4       |    1.680 |
| B2         |  30 | SSA_Notebook |                 59.100 | SSA_O6       |    7.750 |
| B2         |  50 | SSA_Notebook |                 76.700 | CELF         |    7.940 |
| B3         |  20 | CELF         |                 46.700 | SSA_O4       |    5.090 |
| B4         |  10 | CELF         |                 29.170 | SSA_Notebook |    0.500 |
| B4         |  30 | SSA_Notebook |                 67.350 | SSA_O4       |   15.300 |
| B4         |  50 | CELF         |                 92.680 | SSA_Notebook |   13.980 |

**CELF is the strongest non-SSA baseline in this design.**  Its mean Net Advantage is 46.115, it has 6 strict wins (tied with SSA O1), and it costs about 4.35 s per configuration versus 0.68 s for SSA.  It wins B1 K=30/40, B2 K=10, B3 K=20, B4 K=10 and B4 K=50.  The SSA-vs-CELF comparison is therefore a genuine 10W/10L (O1 and O6) or 9W/11L (O4) contest, not a clean SSA sweep.  CELF's competitive mean is not free: it uses about 633 objective evaluations and 8,663 MT simulations per cell on average, versus 132 evaluations and roughly 100–200 simulations for SSA, and costs about 4.35 s versus 0.68 s per cell.  This is the main reason the report does not declare a universal SSA winner, and also why SSA remains attractive on cost-performance.

## 6. Winners

**Overall highest mean Net Advantage:** SSA O1 (Original) (47.878); second is SSA O6 (Budget-aware) (47.261).  
**Most consistent strict winner:** SSA O1 (Original), CELF (O1 fitness) (6 strict wins each; ties are counted separately, and there were no ties in this matrix).  SSA O1 has the higher mean of the two.

The two statements answer different questions: the first is an average over a heterogeneous 20-cell design; the second counts per-cell first places.  Neither is a claim of universal dominance, and the CELF tie in strict wins is substantive: CELF is a much cheaper baseline with a comparable average.

## 7. O1, O4, O6 vs. baselines

| ssa_method   | baseline_method   |   wins |   ties |   losses |   mean_net_advantage_difference |   median_net_advantage_difference |
|:-------------|:------------------|-------:|-------:|---------:|--------------------------------:|----------------------------------:|
| SSA_O1       | Random            |     20 |      0 |        0 |                          48.028 |                            46.490 |
| SSA_O1       | TopDegree         |     20 |      0 |        0 |                          49.624 |                            51.010 |
| SSA_O1       | SignedPageRank    |     20 |      0 |        0 |                          43.943 |                            43.190 |
| SSA_O1       | CRIEG             |     19 |      0 |        1 |                          33.342 |                            34.255 |
| SSA_O1       | NCGE              |     20 |      0 |        0 |                          45.192 |                            47.460 |
| SSA_O1       | SignedGA          |     17 |      0 |        3 |                          19.670 |                            21.380 |
| SSA_O1       | CELF              |     10 |      0 |       10 |                           1.762 |                             0.625 |
| SSA_O1       | FIBMM             |     19 |      0 |        1 |                          29.159 |                            27.650 |
| SSA_O1       | SSA_Notebook      |     11 |      0 |        9 |                           2.308 |                             3.770 |
| SSA_O4       | Random            |     20 |      0 |        0 |                          43.392 |                            45.000 |
| SSA_O4       | TopDegree         |     20 |      0 |        0 |                          44.988 |                            45.280 |
| SSA_O4       | SignedPageRank    |     20 |      0 |        0 |                          39.307 |                            35.945 |
| SSA_O4       | CRIEG             |     20 |      0 |        0 |                          28.706 |                            27.755 |
| SSA_O4       | NCGE              |     20 |      0 |        0 |                          40.556 |                            43.030 |
| SSA_O4       | SignedGA          |     18 |      0 |        2 |                          15.034 |                            16.595 |
| SSA_O4       | CELF              |      9 |      0 |       11 |                          -2.874 |                            -1.450 |
| SSA_O4       | FIBMM             |     20 |      0 |        0 |                          24.523 |                            25.165 |
| SSA_O4       | SSA_Notebook      |     10 |      0 |       10 |                          -2.329 |                            -0.265 |
| SSA_O6       | Random            |     20 |      0 |        0 |                          47.411 |                            47.440 |
| SSA_O6       | TopDegree         |     20 |      0 |        0 |                          49.007 |                            52.810 |
| SSA_O6       | SignedPageRank    |     20 |      0 |        0 |                          43.326 |                            41.695 |
| SSA_O6       | CRIEG             |     20 |      0 |        0 |                          32.724 |                            34.140 |
| SSA_O6       | NCGE              |     20 |      0 |        0 |                          44.575 |                            42.925 |
| SSA_O6       | SignedGA          |     19 |      0 |        1 |                          19.054 |                            20.985 |
| SSA_O6       | CELF              |     10 |      0 |       10 |                           1.145 |                             0.480 |
| SSA_O6       | FIBMM             |     20 |      0 |        0 |                          28.543 |                            29.380 |
| SSA_O6       | SSA_Notebook      |     10 |      0 |       10 |                           1.691 |                             0.535 |

Reading the table: `wins` means the SSA variant has higher mean Net Advantage in that many of the 20 scenario × K cells.  These are descriptive paired comparisons on the same 100 evaluation seeds; they are not independent optimizer replications.

- **SSA_O1 vs baselines (20 cells each):** Random: 20W/0T/0L; TopDegree: 20W/0T/0L; SignedPageRank: 20W/0T/0L; CRIEG: 19W/0T/1L; NCGE: 20W/0T/0L; SignedGA: 17W/0T/3L; CELF: 10W/0T/10L; FIBMM: 19W/0T/1L; SSA_Notebook: 11W/0T/9L.
- **SSA_O4 vs baselines (20 cells each):** Random: 20W/0T/0L; TopDegree: 20W/0T/0L; SignedPageRank: 20W/0T/0L; CRIEG: 20W/0T/0L; NCGE: 20W/0T/0L; SignedGA: 18W/0T/2L; CELF: 9W/0T/11L; FIBMM: 20W/0T/0L; SSA_Notebook: 10W/0T/10L.
- **SSA_O6 vs baselines (20 cells each):** Random: 20W/0T/0L; TopDegree: 20W/0T/0L; SignedPageRank: 20W/0T/0L; CRIEG: 20W/0T/0L; NCGE: 20W/0T/0L; SignedGA: 19W/0T/1L; CELF: 10W/0T/10L; FIBMM: 20W/0T/0L; SSA_Notebook: 10W/0T/10L.

A 20W/0T/0L row is a clean sweep in this design; a mixed row means the answer depends on the opponent and/or K.  The table does not support a claim that any SSA variant beats every baseline in every cell.

## 8. Fairness-aware baseline

| scenario   |   K |   fibmm_net_advantage |   fibmm_spread_A |   fibmm_spread_B |   fibmm_blocked_B |   fibmm_rank_net_advantage |   pi_total |   tau |   sigma |
|:-----------|----:|----------------------:|-----------------:|-----------------:|------------------:|---------------------------:|-----------:|------:|--------:|
| B1         |  10 |                 6.150 |          981.770 |          975.620 |            25.010 |                          6 |      0.971 | 0.963 |   0.963 |
| B1         |  20 |                10.790 |          986.260 |          975.470 |            18.730 |                          9 |      0.972 | 0.967 |   0.967 |
| B1         |  30 |                 6.600 |          976.070 |          969.470 |            37.450 |                         10 |      0.956 | 0.945 |   0.946 |
| B1         |  40 |                36.070 |          995.260 |          959.190 |            43.910 |                          6 |      0.956 | 0.945 |   0.946 |
| B1         |  50 |                16.850 |          987.430 |          970.580 |            41.340 |                          8 |      0.958 | 0.944 |   0.945 |
| B2         |  10 |                -0.080 |          965.860 |          965.940 |            15.190 |                          9 |      0.518 | 0.495 |   0.498 |
| B2         |  20 |                 2.350 |          956.390 |          954.040 |            19.320 |                          9 |      0.478 | 0.451 |   0.454 |
| B2         |  30 |                19.020 |          959.470 |          940.450 |            23.120 |                          7 |      0.479 | 0.450 |   0.453 |
| B2         |  40 |                30.320 |          958.310 |          927.990 |            37.060 |                          7 |      0.454 | 0.438 |   0.439 |
| B2         |  50 |                42.960 |          961.160 |          918.200 |            42.780 |                          7 |      0.454 | 0.434 |   0.436 |
| B3         |  10 |                 8.050 |          965.930 |          957.880 |            13.330 |                          8 |      0.607 | 0.575 |   0.578 |
| B3         |  20 |                 8.730 |          956.520 |          947.790 |            20.290 |                          8 |      0.490 | 0.461 |   0.463 |
| B3         |  30 |                33.330 |          958.130 |          924.800 |            42.310 |                          7 |      0.495 | 0.469 |   0.472 |
| B3         |  40 |                26.170 |          956.860 |          930.690 |            31.490 |                          8 |      0.489 | 0.465 |   0.467 |
| B3         |  50 |                40.980 |          951.240 |          910.260 |            52.490 |                          7 |      0.494 | 0.471 |   0.474 |
| B4         |  10 |                 6.030 |          966.870 |          960.840 |            20.020 |                          9 |      0.757 | 0.717 |   0.721 |
| B4         |  20 |                 7.860 |          962.210 |          954.350 |            21.190 |                          9 |      0.773 | 0.735 |   0.739 |
| B4         |  30 |                26.300 |          959.670 |          933.370 |            58.940 |                          6 |      0.780 | 0.731 |   0.736 |
| B4         |  40 |                28.430 |          957.990 |          929.560 |            57.940 |                          6 |      0.777 | 0.729 |   0.734 |
| B4         |  50 |                17.450 |          950.910 |          933.460 |            59.350 |                          7 |      0.783 | 0.735 |   0.740 |

FIBMM vs the SSA variants (descriptive, 20 cells):

| ssa_method   | fibmm_method   |   net_advantage_wins |   net_advantage_ties |   net_advantage_losses |   mean_net_advantage_difference |   median_net_advantage_difference |   mean_blocked_B_difference |   median_blocked_B_difference |
|:-------------|:---------------|---------------------:|---------------------:|-----------------------:|--------------------------------:|----------------------------------:|----------------------------:|------------------------------:|
| SSA_O1       | FIBMM          |                   19 |                    0 |                      1 |                          29.159 |                            27.650 |                      -0.684 |                         0.505 |
| SSA_O4       | FIBMM          |                   20 |                    0 |                      0 |                          24.523 |                            25.165 |                      -1.206 |                         0.380 |
| SSA_O6       | FIBMM          |                   20 |                    0 |                      0 |                          28.543 |                            29.380 |                      -1.161 |                        -0.980 |

FIBMM's internal objective is `σ(S) = (1−h)·Π(S) + h·τ(S)`, where Π is the overall saving ratio and τ is the minimum community saving ratio.  It is not O1, O4 or O6, and it was not rewritten into one of them.  The external table above shows that a method can have a high internal fairness value and a modest competitive Net Advantage; maximizing fairness does not necessarily maximize Net Advantage.  The internal values are estimates from the positive-edge CIC surrogate, not from the signed Maki–Thompson evaluation.

Correlations across the 20 FIBMM cells (descriptive only):

| variable            |   fibmm_net_advantage |   fibmm_blocked_B |   pi_total |    tau |   sigma |
|:--------------------|----------------------:|------------------:|-----------:|-------:|--------:|
| fibmm_net_advantage |                 1.000 |             0.706 |     -0.246 | -0.242 |  -0.242 |
| fibmm_blocked_B     |                 0.706 |             1.000 |      0.150 |  0.128 |   0.130 |
| pi_total            |                -0.246 |             0.150 |      1.000 |  0.998 |   0.998 |
| tau                 |                -0.242 |             0.128 |      0.998 |  1.000 |   1.000 |
| sigma               |                -0.242 |             0.130 |      0.998 |  1.000 |   1.000 |

FIBMM mean Net Advantage is 18.718; mean Blocked_B is 34.063; mean internal Π is 0.682; mean τ is 0.656. For context, the SSA means are O1 47.878, O4 43.241, and O6 47.261. FIBMM's mean Blocked_B (34.063) is slightly higher than SSA O1's (33.379), so it does improve blocking on average; its Spread_A is much lower, which is why its Net Advantage is lower.  The internal Pi/tau values are slightly negatively correlated with external Net Advantage (−0.25/−0.24) and slightly positively correlated with Blocked_B (0.15/0.13), which is consistent with a fairness-vs-competitive-advantage trade-off in this design.  CRIEG (36.99) and Top Degree (38.00) have even higher mean Blocked_B than FIBMM, but much lower Spread_A, so a blocking-only ranking would mis-rank them; this is exactly why the benchmark uses Net Advantage as the primary metric.  The fairness-aware method is therefore a legitimate baseline for the paper's own objective, but it is not a Net-Advantage optimizer and should not be expected to win the competitive ranking.

## 9. Figures

- `figures/Net_Advantage_B1_Random.png`
- `figures/Net_Advantage_B2_Degree.png`
- `figures/Net_Advantage_B3_Positive_PageRank.png`
- `figures/Net_Advantage_B4_Community.png`
- `figures/Net_Advantage_SSA_and_Fairness_All_Scenarios.png`
- `figures/Mean_Net_Advantage_by_Algorithm.png`
- `figures/Selection_Runtime_by_Algorithm.png`

The four primary figures are scenario-specific two-panel Net Advantage line charts.  The left panel contains the SSA variants, SSA-Notebook and FIBMM; the right panel contains the traditional and literature baselines.  All panels share the same y-axis within a scenario.  Error bars are 95% Monte Carlo CIs over the 100 common evaluation seeds; they do not represent optimizer variability.  The supplementary focus figure and the two summary charts are descriptive.

## 10. Validation

**Status:** PASS  
Cells: 240; methods: 12; reused SSA rows: 60; new rows: 180.  
Wall span between first and last result timestamp: 1.60 hours (includes the sequential SignedGA and FIBMM work; not a pure CPU-time measure).

The code fingerprint (SHA-256 of the runner, analysis, algorithms and config) is in `aggregated/code_fingerprint.json` and is embedded in the validation summary.

Checks passed: complete 4 × 5 × 12 matrix; unique keys; one network hash and one influence direction; opponent seed sets identical within every scenario/K cell; every A set has exactly K unique nodes and is disjoint from B; Net Advantage identity; non-negative Blocked_B; 100 MC rows per cell with the common seed list; 60/60 prior SSA-only rows reused after metadata and MC validation.

## 11. Computation reuse

| status          | source          | reason                                 |   count |
|:----------------|:----------------|:---------------------------------------|--------:|
| completed_valid | new_computation | new computation                        |     171 |
| completed_valid | new_computation | existing result has matching signature |       9 |
| reused          | prior_ssa_only  | validated prior SSA-only result        |      57 |
| reused          | prior_ssa_only  | existing result has matching signature |       3 |

Reused prior SSA cells: 60.  Newly computed cells: 180.  Skipped because a valid result already existed: 12.  Invalidated: 0.  Failed: 0.

The authoritative SSA-only O1/O4/O6 seed sets and their 100-run final evaluations were reused from `exp_20261124_5000_reversed` after validating the network hash, direction, opponent seeds, candidate-pool signature, SSA parameters, seed sets and MC files.  A deterministic reproduction check reran B1/K=10/O1 and matched the stored seed set exactly.  No old same-direction result was reused.

## 12. Limitations

- One optimizer run per method/configuration.  The 100 Monte Carlo runs measure evaluation uncertainty, not between-optimizer-run variability.
- B1–B4 are distinct opponent strategies, not independent replicates; no pooled significance test across scenarios is claimed.
- The network is synthetic; results are conditional on this 5,000-node design and the reversed influence convention.
- FIBMM's selection uses a positive-edge CIC surrogate because the paper's model is unsigned CIC, while the external evaluation is the signed finite-contact Maki–Thompson model.  The CRP sample budget is capped for feasibility; the exact martingale sample sizes are recorded.
- CELF and NCGE use the notebook's O1 weighted fitness, not Net Advantage; this is faithful to the notebook but means their internal objective differs from the ranking metric.
- CRIEG, NCGE and Signed-GA are adaptations of literature methods to this signed, competitive setting; their original papers assume different diffusion models.
- CELF + sampled Local Search is excluded by the no-swap instruction, so the notebook's strongest greedy-refinement variant is not represented.

## 13. Research recommendation

Prioritize **SSA O1 (Original)** for the next research stage if the goal is the best average Net Advantage across the current B1–B4 design.  Keep **SSA O6** as the close second (mean 47.261) and **CELF** as the primary cheap baseline (mean 46.115, 6 strict wins, ~4.35 s per cell).  Do not treat O4 as an improvement over O1 on this evidence: its mean is 43.241, it trails O1 by about 4.64, and it loses 9 of 20 cells to CELF.  The result does not support declaring a universal winner: the per-cell table shows scenario- and K-dependent reversals, and CELF wins six cells outright.  The next stage should (i) repeat the SSA optimizer with independent random seeds to estimate optimizer variability, (ii) analyze why CELF wins B1 K=30/40, B2 K=10, B3 K=20, B4 K=10 and B4 K=50 (its greedy marginal-gain search may exploit the O1 fitness better in those cells), and (iii) test whether the O1/O4/O6/CELF ranking is stable under a second network realization.  FIBMM is a defensible fairness baseline but should not replace the Net Advantage objective unless fairness is an explicit deployment constraint.

