# Results Summary — Full Experiment

> **Superseded:** this is the pre-reference summary. The authoritative,
> reference-aligned results are in `RESULTS_SUMMARY_REFERENCE_ALIGNED.md`
> (run `run_20261123_205500_reference_aligned`). This file is retained as a
> sensitivity check.

Run: `run_20261123_202600`  
Kind: `full`  
Started/finished: 2026-11-23 14:52:17 → 14:53:05 UTC (≈48 s wall)  
Completed: **240 / 240 configurations** (4 scenarios × 5 K × 6 objectives × 2 modes)  
Network fingerprint: `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`  
Semantic fingerprint: `461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25`  
Validation: **PASS** (no duplicate rows, all fingerprints consistent, all 240 MC files present, 100 final MC runs per configuration)

All numbers below are computed from the stored CSVs. **O2 is the only fully
validated objective.** O1/O3/O4/O5/O6 use the provisional paired B-only
`Blocked_B` protocol and are exploratory.

---

## 1. Primary metric: final Net Advantage

Mean final Net Advantage across the 20 configurations per objective/mode
(5 K × 4 scenarios). 95 % intervals are percentile bootstrap intervals across
configurations, not a claim about independent optimizer runs.

| Objective | Mode | Mean Net Adv | 95 % CI | Mean Spread_A | Mean Spread_B | Mean Blocked_B |
|---|---|---:|---:|---:|---:|---:|
| **O2 Competition-aware** | SSA only | **21.62** | [19.39, 23.83] | 214.92 | 193.29 | 17.80 |
| **O2 Competition-aware** | SSA+Swap | **21.64** | [19.80, 23.46] | 214.60 | 192.97 | 18.12 |
| O1 Original SSA | SSA only | 19.73 | [17.95, 21.49] | 213.34 | 193.61 | 17.48 |
| O1 Original SSA | SSA+Swap | 21.45 | [20.01, 22.85] | 214.26 | 192.81 | 18.28 |
| O6 Budget-aware | SSA only | 18.99 | [15.97, 22.05] | 212.30 | 193.31 | 17.78 |
| O6 Budget-aware | SSA+Swap | 19.45 | [16.43, 22.53] | 212.60 | 193.15 | 17.94 |
| O4 Combined | SSA only | 18.86 | [16.53, 21.21] | 210.62 | 191.76 | 19.33 |
| O4 Combined | SSA+Swap | 19.08 | [16.52, 21.68] | 211.72 | 192.64 | 18.44 |
| O5 Gain_A+blocking | SSA only | 17.13 | [14.47, 19.71] | 210.39 | 193.27 | 17.82 |
| O5 Gain_A+blocking | SSA+Swap | 17.82 | [15.50, 20.05] | 210.21 | 192.38 | 18.70 |
| O3 Explicit blocking | SSA only | 16.42 | [13.42, 19.44] | 208.25 | 191.83 | 19.25 |
| O3 Explicit blocking | SSA+Swap | 16.34 | [13.32, 19.40] | 208.17 | 191.83 | 19.26 |

**O2 has the highest point estimate in both modes.** O1 is close, especially with
Swap. O3 (maximising Blocked_B alone) has the lowest Net Advantage despite the
highest Blocked_B.

---

## 2. O2 vs the other objectives (paired by MC seed within configuration)

Each row summarises 40 configuration-level paired differences (4 scenarios ×
5 K × 2 modes). Positive means the other objective beat O2. The CI is a
bootstrap CI across the 40 configuration-level mean differences; Holm counts
are per scenario/K/mode groups (5 comparisons each, 40 groups = 200 tests).

| Comparison | Mean difference | 95 % CI | O2 better in | Holm-significant differences |
|---|---:|---:|---:|---:|
| O1 − O2 | −1.04 | [−2.69, +0.57] | 55.0 % | 14 / 40 |
| O3 − O2 | −5.25 | [−7.54, −3.07] | 80.0 % | 20 / 40 |
| O4 − O2 | −2.66 | [−4.57, −0.79] | 62.5 % | 19 / 40 |
| O5 − O2 | −4.16 | [−5.69, −2.55] | 80.0 % | 18 / 40 |
| O6 − O2 | −2.41 | [−4.54, −0.36] | 72.5 % | 14 / 40 |

- O2 is **not clearly better than O1** overall (CI crosses zero; O2 better in
  55 % of configurations). The two are statistically indistinguishable in this
  design.
- O2 is better than O3, O4, O5 and O6 on average; the CIs exclude zero.
- There is substantial configuration-level heterogeneity: e.g. B4/K=40 strongly
  favours O6 (+16.2 vs O2, Holm-adjusted p≈8×10⁻⁹), while B1/K=20 strongly
  favours O2 over O3 (−22.8, p≈2×10⁻¹⁴). No objective dominates everywhere.
- These tests are **conditional on the fixed seed sets** and use MC seeds as
  paired replicates. They do not estimate between-optimizer-run variance.

---

## 3. SSA vs SSA+Swap

Swap is elitist on the 15-seed optimization objective, not on the 100-seed final
evaluation. Across all 120 paired configurations:

- Mean `Swap − SSA` = **+0.51** Net Advantage, 95 % CI **[+0.13, +0.89]**.
- Median = **0.00**; IQR = [−0.37, +1.58]; 45.0 % positive, 28.3 % negative,
  26.7 % exactly zero.
- Only **5 / 120** configurations have a paired 95 % CI strictly above zero, and
  **2 / 120** strictly below zero. The rest are inconclusive.
- The positive mean is driven by a few large gains: B4/K=10/O1 +8.80,
  B3/K=20/O1 +7.74, B1/K=10/O5 +5.85, B4/K=40/O2 +5.32. There are also losses:
  B3/K=20/O2 −6.07, B4/K=30/O6 −4.73.

**Interpretation:** bounded Swap refinement produces a small average positive
shift but is **not a reliable per-configuration improvement** on the final
metric. It can overfit the 15-seed optimization objective. Do not claim that
SSA+Swap dominates SSA.

Swap acceptance rates (accepted at least one improving swap):

| Objective | Accepted / 20 |
|---|---:|
| O1 | 17 |
| O2 | 17 |
| O3 | 11 |
| O4 | 13 |
| O5 | 14 |
| O6 | 16 |
| **Total** | **88 / 120 (73.3 %)** |

---

## 4. Blocked_B is not a substitute for Net Advantage

| Objective | Mean Blocked_B | Mean Net Advantage |
|---|---:|---:|
| O3 (maximise Blocked_B) | **19.26** | **16.38** |
| O4 | 18.88 | 18.97 |
| O1 | 17.88 | 20.59 |
| O2 | 17.96 | **21.63** |
| O6 | 17.86 | 19.22 |
| O5 | 18.26 | 17.48 |

- Correlation `Blocked_B` vs `Net Advantage` across all 240 rows: Pearson
  **0.361**, Spearman **0.337** (positive but weak).
- Maximising Blocked_B alone (O3) yields the **lowest** Net Advantage.
- This is exactly why the evaluation metric and the optimizer objective must be
  kept separate. The result is consistent with the provisional status of
  Blocked_B: it is a useful diagnostic, not a validated proxy.

---

## 5. Optimizer fitness vs final Net Advantage

Pearson correlation between the internal 15-seed optimizer score and the final
100-seed Net Advantage, per objective/mode (20 configurations each):

| Objective | SSA only | SSA+Swap |
|---|---:|---:|
| O2 Competition-aware | **0.71** | **0.80** |
| O1 Original SSA | 0.05 | −0.14 |
| O3 Explicit blocking | −0.01 | 0.04 |
| O4 Combined | 0.29 | 0.35 |
| O5 Gain_A+blocking | 0.44 | 0.54 |
| O6 Budget-aware | 0.35 | 0.39 |

Only O2's internal fitness is a strong predictor of final Net Advantage. For
O1 and O3, internal fitness is essentially uncorrelated with the final metric.
**Never rank objectives by optimizer fitness.**

---

## 6. Budget (K) trend for O2

Mean Net Advantage by K (4 scenarios per K):

| K | SSA only | SSA+Swap |
|---:|---:|---:|
| 10 | 15.28 | 16.12 |
| 20 | 23.44 | 23.10 |
| 30 | 20.91 | 21.35 |
| 40 | 23.01 | 22.90 |
| 50 | 25.48 | 24.73 |

The trend is upward from K=10 to K=50, with a dip at K=30. Larger seed budgets
generally help, but the relationship is not monotone in this sample.

---

## 7. Robustness across B1–B4 (O2)

Mean O2 Net Advantage by scenario (5 K per scenario):

| Scenario | SSA only | SSA+Swap |
|---|---:|---:|
| B1 Random | 21.99 | 21.37 |
| B2 Degree | 21.55 | 22.42 |
| B3 Positive PageRank | **23.70** | 21.70 |
| B4 Community | 19.26 | 21.06 |

O2 remains positive across all scenarios. B3 is easiest for SSA-only; B4 is
hardest for SSA-only. The across-scenario range is 4.44 (SSA-only) and 1.36
(SSA+Swap).

---

## 8. Best configuration per scenario/K

Ranked by final Net Advantage, ignoring statistical uncertainty:

| Objective | Times best / 20 |
|---|---:|
| O2 | 7 |
| O1 | 4 |
| O6 | 4 |
| O3 | 2 |
| O4 | 2 |
| O5 | 1 |

O2 is the most frequent winner, but it does not win every cell. This table is
descriptive; differences between nearby configurations are often within MC
uncertainty.

---

## 9. Runtime and throughput

- Full run wall time: **≈48 s** (14:52:17 → 14:53:05 UTC), sequential scenarios,
  8 OpenMP threads.
- Summed per-configuration compute time: **60.3 s** (includes SSA, Swap and final
  evaluation; parallel work means this can exceed wall time).
- Actual simulations executed: **202,800** (metric caching reduced the
  upper-bound estimate of 443,900).
- Mean per configuration: SSA-only ≈ **0.19 s**, SSA+Swap ≈ **0.30 s**.
- Benchmark (`BENCHMARK.md`): persistent engine **40,255 sims/s** at 8 threads
  (3.75× serial); one-shot API 692 sims/s; Python reference 191 sims/s.

Caveat: `final_evaluation_runtime_sec` can be near zero when the exact
(A, B, seed list) evaluation was already cached. `simulations_for_config`,
`simulation_seconds_for_config`, `cache_hits_for_config` and
`cache_misses_for_config` record actual work and should be used for runtime
comparisons.

---

## 10. Data quality

- 240 unique `(scenario, K, objective_id, mode)` rows; 0 duplicates.
- 240 per-seed MC files, 100 rows each; 120 SSA checkpoints; 120 SSA histories.
- All rows share one run ID, one file fingerprint and one semantic fingerprint.
- `Net Advantage = Spread_A − Spread_B` holds to ≤3×10⁻¹⁴.
- No NaN in core metric/seed columns; all 240 rows marked `completed`.
- SSA-only and Swap rows share the identical saved SSA seed set in every one of
  the 120 configurations; Swap-initial equals the SSA set in every Swap row.
- Every seed set has exactly K unique nodes, contains no opponent seed, and
  every opponent seed set has exactly K nodes.
- Analysis validation: **PASS**, no issues.

---

## 11. What is validated vs exploratory

| Claim | Status |
|---|---|
| O2 is the best objective by mean final Net Advantage | Supported descriptively; O2 vs O1 is not distinguishable |
| O2 beats O3/O4/O5/O6 on average | Supported by paired per-config comparisons with bootstrap CIs |
| SSA+Swap improves final Net Advantage reliably | **Not supported**; small average shift, mostly inconclusive, heavy-tailed |
| Blocked_B is a validated objective component | **No**; provisional protocol only |
| O1/O3/O4/O5/O6 are validated objectives | **No**; exploratory |
| Optimizer fitness can rank objectives | **No**; correlation is weak for O1/O3 |
| Results generalize beyond this synthetic network | **Not tested**; external validity limited |

---

## 12. Files

- Results: `results/run_20261123_202600/`
- Manifest: `results/run_20261123_202600/manifest.json`
- Run metadata: `results/run_20261123_202600/run_metadata.json`
- Analysis tables: `analysis/tables/run_20261123_202600_20261123_202328/`
- Analysis figures: `analysis/figures/run_20261123_202600_20261123_202328/`
- Narrative report: `analysis/tables/run_20261123_202600_20261123_202328/summary.md`
- Validation gate: `VALIDATION_GATE.md`
- Benchmark: `BENCHMARK.md`
- Final audit: `FINAL_AUDIT.md`
