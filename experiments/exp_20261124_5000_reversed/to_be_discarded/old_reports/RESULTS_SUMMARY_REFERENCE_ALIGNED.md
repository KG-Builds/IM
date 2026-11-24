# Results Summary — Reference-Aligned Full Experiment

Run: `run_20261123_205500_reference_aligned`  
Kind: `full`  
Started/finished: 2026-11-23 15:23:57 → 15:24:46 UTC (≈48 s wall)  
Completed: **240 / 240 configurations** (4 scenarios × 5 K × 6 objectives × 2 modes)  
Network fingerprint: `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`  
Semantic fingerprint: `461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25`  
Validation: **PASS** — no duplicate rows, all fingerprints consistent, 240 MC files present, 100 final MC runs each, all six objectives reference-validated.

All numbers are computed from the stored CSVs. This run supersedes the
pre-reference `run_20261123_202600`; the latter is retained as a sensitivity
check.

---

## 1. Objective ranking by final Net Advantage

Mean across 20 configurations per objective/mode (5 K × 4 scenarios). 95 %
intervals are percentile bootstrap intervals across configurations.

| Objective | Mode | Mean Net Adv | 95 % CI | Mean Spread_A | Mean Spread_B | Mean Blocked_B |
|---|---|---:|---:|---:|---:|---:|
| **O2 Competition-aware** | SSA only | **20.55** | [18.14, 23.18] | 214.97 | 194.42 | 16.67 |
| **O2 Competition-aware** | SSA+Swap | **21.97** | [19.39, 24.56] | 215.59 | 193.62 | 17.46 |
| O1 Original SSA | SSA only | 19.52 | [16.97, 22.06] | 212.49 | 192.97 | 18.12 |
| O1 Original SSA | SSA+Swap | 19.54 | [16.68, 22.29] | 213.34 | 193.80 | 17.29 |
| O6 Budget-aware | SSA only | 19.27 | [16.72, 21.85] | 211.56 | 192.29 | 18.79 |
| O6 Budget-aware | SSA+Swap | 19.84 | [17.30, 22.32] | 212.16 | 192.32 | 18.77 |
| O5 Gain_A+blocking | SSA only | 18.62 | [16.37, 21.20] | 210.47 | 191.86 | 19.23 |
| O5 Gain_A+blocking | SSA+Swap | 18.36 | [16.05, 20.82] | 210.12 | 191.76 | 19.33 |
| O4 Combined | SSA only | 17.99 | [15.36, 20.40] | 210.90 | 192.91 | 18.17 |
| O4 Combined | SSA+Swap | 17.71 | [15.17, 20.13] | 210.24 | 192.53 | 18.56 |
| O3 Explicit blocking | SSA only | 15.72 | [13.32, 18.17] | 207.59 | 191.87 | 19.22 |
| O3 Explicit blocking | SSA+Swap | 15.85 | [13.23, 18.40] | 207.33 | 191.49 | 19.60 |

**O2 has the highest point estimate in both modes.** O1 and O6 are close; O3
(maximising Blocked_B alone) is lowest.

---

## 2. O2 vs the other objectives (paired by MC seed within configuration)

Each row summarises 40 configuration-level paired differences (4 scenarios ×
5 K × 2 modes). Positive means the other objective beat O2. CI is a bootstrap CI
across the 40 configuration-level mean differences; Holm counts are per
scenario/K/mode groups (5 comparisons each, 40 groups = 200 tests).

| Comparison | Mean difference | 95 % CI | O2 better in | Holm-significant |
|---|---:|---:|---:|---:|
| O1 − O2 | −1.72 | [−3.66, +0.18] | 55.0 % | 11 / 40 |
| O3 − O2 | −5.47 | [−7.64, −3.22] | 72.5 % | 23 / 40 |
| O4 − O2 | −3.41 | [−5.48, −1.33] | 72.5 % | 23 / 40 |
| O5 − O2 | −2.77 | [−4.62, −0.87] | 72.5 % | 15 / 40 |
| O6 − O2 | −1.70 | [−3.87, +0.47] | 62.5 % | 17 / 40 |

- **O2 is not distinguishable from O1 or O6 overall** (both CIs cross zero).
- **O2 is better on average than O3, O4 and O5** (CIs exclude zero).
- There is substantial configuration-level heterogeneity; no objective dominates
  every cell.
- Tests are conditional on the fixed seed sets and use MC seeds as paired
  replicates; they do not estimate between-optimizer-run variance.

---

## 3. SSA vs SSA+Swap

Across all 120 paired configurations:

- Mean `Swap − SSA` = **+0.27** Net Advantage, 95 % CI **[−0.24, +0.78]**.
- Median = **0.00**; 43.3 % positive, 36.7 % negative, 20.0 % exactly zero.
- **8 / 120** configurations have a paired 95 % CI strictly above zero;
  **7 / 120** strictly below zero; the rest are inconclusive.

By objective:

| Objective | Mean Swap−SSA | Median | Positive-sig | Negative-sig |
|---|---:|---:|---:|---:|
| O1 | +0.02 | −0.21 | 3 | 3 |
| O2 | +1.42 | +0.49 | 4 | 1 |
| O3 | +0.12 | 0.00 | 0 | 1 |
| O4 | −0.28 | 0.00 | 1 | 2 |
| O5 | −0.26 | −0.10 | 0 | 0 |
| O6 | +0.57 | +0.08 | 0 | 0 |

**Interpretation:** bounded Swap refinement does **not** reliably improve the
final Net Advantage. It is elitist on the 15-seed optimization objective and can
overfit it. The small positive mean is not statistically distinguishable from
zero overall. Do not claim SSA+Swap dominates SSA.

Swap acceptance rate: **97 / 120 (80.8 %)**.

---

## 4. Blocked_B is not a substitute for Net Advantage

| Objective | Mean Blocked_B | Mean Net Advantage |
|---|---:|---:|
| O3 (maximise Blocked_B) | **19.41** | **15.78** |
| O5 | 19.28 | 18.49 |
| O6 | 18.78 | 19.55 |
| O4 | 18.37 | 17.85 |
| O1 | 17.70 | 19.53 |
| O2 | 17.07 | **21.26** |

- Correlation `Blocked_B` vs `Net Advantage` across all 240 rows: Pearson
  **0.46**, Spearman **0.46** (moderate but far from a substitute).
- Maximising Blocked_B alone (O3) yields the **lowest** Net Advantage.
- The evaluation metric and the optimizer objective must be kept separate.

---

## 5. Optimizer fitness vs final Net Advantage

Pearson correlation between the internal 15-seed optimizer score and the final
100-seed Net Advantage, per objective/mode (20 configurations each):

| Objective | SSA only | SSA+Swap |
|---|---:|---:|
| O2 Competition-aware | **0.60** | **0.87** |
| O6 Budget-aware | **0.59** | **0.68** |
| O1 Original SSA | 0.17 | 0.24 |
| O3 Explicit blocking | 0.37 | 0.39 |
| O4 Combined | 0.39 | 0.29 |
| O5 Gain_A+blocking | 0.37 | 0.31 |

Only O2 and O6 have a strong fitness–evaluation relationship. **Never rank
objectives by optimizer fitness.**

---

## 6. Budget (K) trend for O2

Mean Net Advantage by K (4 scenarios per K):

| K | SSA only | SSA+Swap |
|---:|---:|---:|
| 10 | 16.37 | 15.00 |
| 20 | 19.40 | 23.54 |
| 30 | 22.51 | 23.90 |
| 40 | 22.38 | 25.04 |
| 50 | 22.08 | 22.35 |

The trend is generally upward from K=10 to K=40, then flattens/declines slightly
at K=50. The relationship is not strictly monotone in this sample.

---

## 7. Robustness across B1–B4 (O2)

Mean O2 Net Advantage by scenario (5 K per scenario):

| Scenario | SSA only | SSA+Swap |
|---|---:|---:|
| B1 Random | 18.70 | 19.25 |
| B2 Degree | 18.23 | 22.59 |
| B3 Positive PageRank | 21.69 | 20.68 |
| B4 Community | **23.57** | **25.35** |

O2 is positive across all scenarios; strongest in B4, weakest in B2 under
SSA-only. The across-scenario range is 5.34 (SSA-only) and 6.10 (SSA+Swap).

---

## 8. Best configuration per scenario/K

Ranked by final Net Advantage, ignoring statistical uncertainty:

| Objective | Times best / 20 |
|---|---:|
| O2 | 6 |
| O6 | 4 |
| O1 | 4 |
| O5 | 3 |
| O3 | 2 |
| O4 | 1 |

O2 is the most frequent winner, but it does not win every cell. Differences
between nearby configurations are often within MC uncertainty.

---

## 9. Runtime and throughput

- Full run wall time: **≈48 s** (15:23:57 → 15:24:46 UTC), sequential scenarios,
  8 OpenMP threads.
- Summed per-configuration compute: **62.4 s**.
- Actual simulations: **203,640** (metric caching).
- Mean per configuration: SSA-only ≈ **0.20 s**, SSA+Swap ≈ **0.32 s**.
- Benchmark (`BENCHMARK.md`): persistent engine **40,255 sims/s** at 8 threads
  (3.75× serial); one-shot API 692 sims/s; Python reference 191 sims/s.

Caveat: `final_evaluation_runtime_sec` can be near zero on a metric-cache hit;
use `simulations_for_config` / `simulation_seconds_for_config` for runtime
comparisons.

---

## 10. Data quality

- 240 unique `(scenario, K, objective_id, mode)` rows; 0 duplicates.
- 240 per-seed MC files, 100 rows each; 120 SSA checkpoints; 120 SSA histories.
- All rows share one run ID, one file fingerprint and one semantic fingerprint.
- `Net Advantage = Spread_A − Spread_B` to ≤3×10⁻¹⁴.
- No NaN in core metric/seed columns; all 240 rows marked `completed`.
- SSA-only and Swap rows share the identical saved SSA seed set in all 120
  configurations; Swap-initial equals the SSA set in every Swap row.
- Every seed set has exactly K unique nodes and no opponent seed.
- Analysis validation: **PASS**, no issues.
- All rows `objective_validity = validated`.

---

## 11. Pre-reference vs reference-aligned sensitivity

The first full run (`run_20261123_202600`) used the old candidate-pool random
order, an extra +3,000,000 in optimization seeds, and the old round counter.
After correction:

| Quantity | Pre-reference | Reference-aligned |
|---|---:|---:|
| O2 mean Net Advantage | 21.63 | 21.26 |
| O3 mean Net Advantage | 16.38 | 15.78 |
| Highest objective | O2 | O2 |
| Lowest objective | O3 | O3 |
| Blocked_B–Net correlation | 0.36 | 0.46 |
| Swap reliable? | no | no |
| Per-config old/new correlation | — | **0.38** |

Individual configurations changed materially, but the aggregate conclusions
(O2 best, O3 worst, Blocked_B not a proxy, Swap not reliable) are stable. The
pre-reference run is retained as a sensitivity check.

---

## 12. Validated vs remaining limitations

| Claim | Status |
|---|---|
| Six objective formulas | **Validated against the supplied reference notebook** |
| Blocked_B protocol | **Validated against the reference** |
| Candidate pool/ranking/random part | **Validated against the reference** |
| SSA/Swap logic and seeds | **Validated against the six-objective reference** |
| Diffusion model | **Validated logic; rounds semantics aligned** |
| O2 highest mean Net Advantage | Descriptive; O1/O6 not distinguishable |
| O2 better than O3/O4/O5 | Supported by paired per-config CIs |
| SSA+Swap reliably improves | **Not supported** |
| Between-optimizer-run variance | **Not estimated** (one run per configuration) |
| External validity | **Not tested** (synthetic network only) |
| Synthetic edge orientation | **Researcher decision** (reference reverses Bitcoin OTC edges) |

---

## 13. Files

- Results: `results/run_20261123_205500_reference_aligned/`
- Manifest: `results/run_20261123_205500_reference_aligned/manifest.json`
- Run metadata: `results/run_20261123_205500_reference_aligned/run_metadata.json`
- Tables: `analysis/tables/run_20261123_205500_reference_aligned_20261123_210005/`
- Figures: `analysis/figures/run_20261123_205500_reference_aligned_20261123_210005/`
- Narrative report: `analysis/tables/run_20261123_205500_reference_aligned_20261123_210005/summary.md`
- Reference audit: `REFERENCE_NOTEBOOK_AUDIT.md`
- Validation gate: `VALIDATION_GATE_REFERENCE_ALIGNED.md`
- Final audit: `FINAL_AUDIT_REFERENCE_ALIGNED.md`
