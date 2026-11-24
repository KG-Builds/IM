# Expected Runtime

Observed on the 8-CPU environment used for this experiment:

| Component | Observed cost | Notes |
|---|---|---|
| Network load + direction validation | ~0.5 s | Once per process |
| Opponent orders | ~1.6 s | Cached |
| Signed PageRank | ~0.4 s | Cached |
| NCGE centrality | ~25 s | Cached; exact closeness + 200-pivot betweenness |
| FIBMM communities | ~1.3 s | Cached |
| SSA O1/O4/O6 | ~0.6–0.7 s per cell | Reused; not recomputed |
| SSA-Notebook | ~0.7 s per cell | New |
| Random / Top Degree / Signed PageRank | <0.1 s per cell | New |
| CRIEG | ~1.7 s per cell | 2000 sketches |
| NCGE / CELF | ~1–6 s per cell | Lazy greedy, 15 MC |
| Signed-GA | mean 237.7 s per cell (range ~199–292 s) | 30 sparse direct solves × 2 (forward + adjoint); fill-in depends on B/K |
| FIBMM | mean 46.1 s per cell (range ~10–146 s) | CRP sampling + coverage updates; capped at 600k sets |
| Final evaluation | ~0.05 s per cell | 100 seeds, C++ engine |

The full matrix is 240 cells.  The dominant costs are Signed-GA and FIBMM
(20 cells each).  The exact wall time depends on the machine and on whether
preprocessing caches are already populated.  The run is checkpointed per cell;
restarting it skips valid completed cells.
