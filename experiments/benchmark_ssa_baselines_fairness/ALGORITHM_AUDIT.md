# Algorithm Audit — `competitive_im_no_celf_csv_persist.ipynb`

Source inspected (read-only):
`old work/competitive_im_no_celf_csv_persist.ipynb`

The notebook is a Bitcoin-OTC experiment. It builds a signed directed influence
graph, runs a competitive signed Maki–Thompson simulator, and compares seed
selection methods under four fixed Rumour B strategies and K = 10, 20, 30, 40,
50. This audit records every seed-selection algorithm actually implemented in
the notebook, including algorithms whose run toggle was off.

## 1. Audit table

| Algorithm | Implemented in notebook | Original objective | Seed-selection mechanism | Required adaptation to the current experiment | Status |
|---|---|---|---|---|---|
| **Random** | Yes, cell 25 | None (uniform baseline) | `random.Random(master_seed + 50000 + K*100 + scenario_number).sample(eligible, K)` | Use the current influence graph and the same eligible set (all nodes except B). Preserve the seed formula. | Included |
| **Top Degree** | Yes, cell 16 | None (degree heuristic) | Rank eligible nodes by directed out-degree descending, ties by node id; take K. | Use influence out-degree on the current reversed influence graph. | Included |
| **Signed PageRank** | Yes, cell 16 | None (signed centrality heuristic) | Two-layer signed random walk: positive edges preserve polarity, negative edges flip; transition weight = absolute original rating; teleport only to positive layer; score = positive-layer mass − negative-layer mass. | Apply to the current influence graph with the stored `rating` attribute. One-time preprocessing. | Included |
| **CRIEG** (Hong et al., IEEE TCYB 2021) | Yes, cell 19 | Competitive influence blocking via reverse-reachability sketches | 2000 Competitive Reverse Reachable sketches on the positive subgraph; nodes receive weight 1 if they reach a target strictly before B, 0.5 if tied; greedy coverage selection over all eligible nodes. | Use the current positive influence subgraph and the same B seeds. The notebook's dict-of-dicts implementation was re-expressed with equivalent lists for speed; RNG draws and greedy logic are unchanged. | Included |
| **NCGE** (Liang et al., ESWA 2025) | Yes, cell 14 + cell 21 | Network-centrality-filtered greedy influence maximization | Centrality score `D_v = 0.5·degree_norm + 0.3·closeness_norm + 0.2·betweenness_norm`; take top 10·K eligible nodes; then lazy-greedy marginal-gain selection. | Use the current directed influence graph. The greedy fitness is the notebook's O1 weighted fitness (see note below), with 15 optimization MC seeds. | Included |
| **Signed-GA** (Chakraborty et al., 2025) | Yes, cell 19 | Signed Voter equilibrium / gradient ascent | Build signed Laplacian `L = diag(in_abs) − W_signed`; solve `(L + diag(p_A+p_B))x_A = p_A + 1^T W^-`; gradient ascent on the eligible simplex with 30 steps and projection; take top-K allocated nodes. | Use the current influence graph and B seeds. Sparse LU uses the `MMD_AT_PLUS_A` ordering (same direct solve, less fill-in). | Included |
| **CELF** | Yes, cell 21 | Lazy-greedy marginal gain; notebook fitness is O1 (`Spread_A + 1.5·Blocked_B`) | Lazy greedy with a max-heap over the 70/30 weighted candidate pool; stale gains are refreshed when a candidate reaches the top. | Use the current weighted candidate pool, current evaluator, and 15 optimization MC seeds. | Included |
| **CELF + sampled Local Search** | Yes, cell 21 | CELF objective plus one-for-one sampled swaps | Starts from CELF, tests up to 300 sampled one-for-one replacements, accepts the best improving swap. | Not adapted. It is a swap-based post-processing step. The master task explicitly excludes Swap refinement, Local Swap and Swap-based post-processing. | **Excluded by instruction** |
| **Discrete SSA** | Yes, cell 23 | Notebook fitness is O1 (`Spread_A + 1.5·Blocked_B`), despite markdown calling it Net Advantage | Population 12, 10 iterations, mutation 0.20; `c1 = 2·exp(−(4t/T)²)`, exploit `1−c1/2`; leader/follower updates; best-so-far food; repair to K unique pool nodes. | The notebook's SSA is algorithmically the same as SSA O1. This experiment includes **SSA O1/O4/O6** with the current reference seed schedule, and additionally **SSA-Notebook** with the notebook's original `master_seed + 30000 + scenario_number` schedule as a replication/sensitivity baseline. | Included (as O1/O4/O6 + SSA-Notebook) |
| **Full Greedy + Local Search** | No (removed in cell 17) | — | The notebook states that full Greedy and Greedy + Local Search were removed. | Not counted as an implemented algorithm. | Not applicable |

Helper functions (`repair_solution`, `fitness_net_advantage`,
`evaluate_seed_set`, `competitive_maki_thompson`, candidate-pool builders,
`compute_signed_pagerank`, etc.) are not counted as separate algorithms.

## 2. Important notebook discrepancies found during the audit

1. **CELF and SSA fitness.** The notebook markdown says the optimizers maximize
   mean Net Advantage, but the actual code's `fitness_net_advantage` returns
   `weighted_fitness = 1.0·Spread_A + 1.5·Blocked_B` (O1). The experiment
   preserves the code, not the markdown. CELF and NCGE therefore use O1
   fitness; SSA-Notebook also uses O1. The external ranking metric is still Net
   Advantage.
2. **SSA seed schedule.** The four-B notebook uses
   `master_seed + 30000 + scenario_number` for its single SSA method. The
   six-objective notebook uses
   `master_seed + 500000 + scenario_number·1000 + K·10 + variant_index`. The
   current experiment uses the latter for O1/O4/O6 and the former for the
   explicit SSA-Notebook baseline.
3. **CELF was disabled in the notebook run** (`RUN_CELF = False`), but its
   implementation is complete. It is included here because the task asks for
   every implemented seed-selection algorithm, not only the toggled-on ones.
4. **NCGE's centrality computation** uses exact closeness centrality for
   graphs with at most 10,000 nodes and sampled betweenness with 200 pivots.
   That branch is used on the 5,000-node network.
5. **CRIEG's original implementation** stores per-node/per-sketch gains in
   nested dictionaries. The same weights are stored in lists here; this is a
   data-structure optimization, not a change to the sketch construction or
   greedy rule.
6. **Signed-GA's original implementation** calls `scipy.sparse.linalg.spsolve`
   with the default COLAMD ordering. On the 5,000-node graph the fill-in is
   large; this experiment uses `splu(..., permc_spec="MMD_AT_PLUS_A")`, which
   solves the same linear system exactly with a different fill-reducing
   ordering.

## 3. SSA objectives retained in this experiment

Only O1, O4 and O6 are retained. All are run in **SSA-only** mode. Swap
refinement is not run, not reported, and not used as a post-processing step.

| ID | Name | Formula | Parameters / K-dependence | Candidate pool | Fitness evaluation |
|---|---|---|---|---|---|
| O1 | Original SSA | `Spread_A + 1.5·Blocked_B` | α=1.0, β=1.5 | 10·K, 70/30 polarity-aware | mean over 15 optimization MC seeds |
| O4 | Combined influence-blocking | `Spread_A + 2.0·Blocked_B` | α=1.0, β=2.0 | 10·K, 70/30 polarity-aware | mean over 15 optimization MC seeds |
| O6 | Budget-aware | `α(K)·Spread_A + β(K)·Blocked_B` | K≤20: (0.75, 2.25); K≤40: (1.00, 1.75); K=50: (1.25, 1.50) | 10·K, 70/30 polarity-aware | mean over 15 optimization MC seeds |

`Blocked_B = max(0, mean_s baseline_B(B,s) − mean_s Spread_B(A,B,s))`, with the
same paired Monte Carlo seeds for the B-only baseline and the A/B run.

## 4. Why CELF + Local Search is excluded

The master task says:

> Do not include SSA + Swap, Local Swap, Swap-based post-processing, Swap
> refinement as a separate mode, or any previously implemented Swap variant.

The notebook's `celf_sampled_local_search` is exactly a one-for-one swap-based
post-processing step initialized from CELF. It is therefore excluded by
instruction, not because it is slow or uncompetitive. The base CELF algorithm
is included. No CELF+Local Search result is generated or reported in this
experiment.

## 5. Fair-comparison notes

- All methods use the same network, B seeds, K values, diffusion model and 100
  final evaluation seeds.
- CELF, NCGE and SSA use the same 15 optimization MC seeds per scenario; this
  is the notebook's optimization protocol.
- FIBMM uses its own paper-defined CRP sampling budget; it does not use the
  15-seed MT optimization loop. This is an inherent algorithmic difference and
  is recorded in the runtime/budget tables.
- CRIEG, Signed-GA, Signed PageRank, Top Degree and Random do not use MT
  optimization MC; they are deterministic or use their own internal sampling.
- No method receives a different B seed set, a different final evaluation
  seed set, or a different network.
