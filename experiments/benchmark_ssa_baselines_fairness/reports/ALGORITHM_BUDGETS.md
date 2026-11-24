# Algorithm Budgets and Fairness of Comparison

Different seed-selection algorithms have inherently different computational
requirements.  This experiment records the actual budget rather than forcing a
single artificial budget on every method.

| Method | Selection budget | Diffusion evaluations during selection | Final evaluation |
|---|---|---|---|
| SSA O1/O4/O6 | 12 initial + 10 × 12 = 132 objective evaluations | 15 optimization MC seeds per evaluation (paired A/B and B-only baseline) | 100 common MC seeds |
| SSA-Notebook | Same SSA budget, notebook seed schedule | 15 optimization MC seeds per evaluation | 100 common MC seeds |
| Random | One uniform sample | None | 100 common MC seeds |
| Top Degree | One deterministic sort | None | 100 common MC seeds |
| Signed PageRank | One two-layer PageRank preprocessing + sort | None | 100 common MC seeds |
| CRIEG | 2000 reverse-reachability sketches | None (sketch sampling only) | 100 common MC seeds |
| NCGE | One centrality preprocessing + lazy greedy | 15 optimization MC seeds per greedy evaluation | 100 common MC seeds |
| Signed-GA | 30 projected gradient steps, 30 sparse linear systems (plus adjoint solves) | None | 100 common MC seeds |
| CELF | Lazy greedy on 10·K pool | 15 optimization MC seeds per greedy evaluation | 100 common MC seeds |
| FIBMM | Paper's CRP sampling (capped at 600,000 sets/configuration) + two-stage greedy | None (CIC surrogate CRP sampling, not MT diffusion) | 100 common MC seeds |

## What is fair

- Every method uses the same network, B seeds, K values, diffusion model and
  100 final evaluation seeds.
- SSA, SSA-Notebook, NCGE and CELF use the same 15 optimization MC seeds per
  scenario; this is the notebook's protocol.
- FIBMM uses its own paper-defined sampling budget; it is not forced into the
  15-seed MT optimization loop, because that would no longer be FIBMM.
- Random, Top Degree, Signed PageRank, CRIEG and Signed-GA use their own
  deterministic or internal-sampling logic and no MT optimization MC.
- No method receives an easier opponent or a different final evaluation seed.

## What is not equal

- The computational budgets are not equal, and cannot be equal without
  changing the algorithms.  Runtime and simulation counts are reported so the
  reader can judge the cost-performance trade-off.
- CELF/NCGE use O1 fitness, not Net Advantage.  This is faithful to the
  notebook code, but it means their internal objective differs from the
  external ranking metric.
- FIBMM's internal objective is maximin fairness on a CIC surrogate, not Net
  Advantage on the signed MT model.
- CRIEG, NCGE and Signed-GA are literature adaptations; their original papers
  assume different diffusion models.

The comparison is therefore fair in **evaluation conditions** (same network,
opponents, seeds and metric) but not in **computational budget**.  This is the
standard and honest way to compare algorithms with different designs.
