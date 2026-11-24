# Methods — SSA O1/O4/O6 vs. Notebook Baselines and FIBMM

## 1. Scope

This experiment compares:

- **Group A:** SSA O1, SSA O4, SSA O6, in SSA-only mode.
- **Group B:** every valid seed-selection algorithm implemented in
  `old work/competitive_im_no_celf_csv_persist.ipynb`: Random, Top Degree,
  Signed PageRank, CRIEG, NCGE, Signed-GA, CELF, and SSA-Notebook.
- **Group C:** FIBMM, the fairness-aware influence-blocking method of Xiao et
  al. (IEEE TCSS 2025).

The primary metric is **Net Advantage = Spread_A − Spread_B**. Spread_A,
Spread_B, Blocked_B and runtime are secondary metrics. Methods are not ranked
by Spread_A alone or by Blocked_B alone.

Swap refinement is excluded. No SSA+Swap, Local Swap, or swap-based
post-processing is run. CELF + sampled Local Search is excluded for the same
reason; the base CELF algorithm is included.

## 2. Network and influence direction

The network is the exact file used by the authoritative
`exp_20261124_5000_reversed` experiment:

```
experiments/exp_20261124_5000_reversed/network/synthetic_network_5000_37500.csv
```

Properties verified at run time:

- 5,000 nodes, ids 0–4999.
- 37,500 directed trust arcs, one CSV row per arc.
- 37,500 influence arcs, each the exact reverse of a trust arc.
- 0 missing reversed edges, 0 extra influence edges, 0 same-direction edges,
  0 reciprocal trust pairs.
- File SHA-256: `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e`.
- Trust semantic SHA-256: `576d00314530fc82c86d5a433e13a2ea64dbe135723350d936df21b082e2b20f`.
- Influence semantic SHA-256: `b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9`.

The CSV is the trust network (`source → target`). The influence graph reverses
every edge once (`target → source`) and preserves probability, polarity,
strength and rating. All algorithms, the diffusion engine, PageRank, degree
ranking, communities, candidate pools and FIBMM operate on the influence graph.
The network is not regenerated or duplicated.

## 3. Diffusion model

Finite-contact competitive Maki–Thompson variant, synchronous rounds:

- states: 0 ignorant, 1 A-spreader, 2 B-spreader, 3 A-stifler, 4 B-stifler;
- A seeds start as A-spreaders, B seeds as B-spreaders; A/B overlap rejected;
- each spreader contacts one uniformly random uncontacted outgoing influence
  neighbour per round;
- if it has no uncontacted neighbour, or the target is not ignorant at the
  start of the round, it becomes a stifler of its own rumor;
- otherwise it transmits with the edge probability; a negative edge flips
  A↔B; a positive edge preserves the rumor;
- proposals are resolved by highest edge strength; different-rumor ties are
  broken uniformly at random;
- `Spread_A` counts states 1 and 3; `Spread_B` counts 2 and 4;
- `rounds` includes the terminal empty round (reference semantics).

The C++ engine is used for all final evaluations. The Python reference
implementation remains available for model-logic tests.

## 4. Trust-to-probability mapping

| Trust strength | Probability |
|---|---|
| 1–3 | 0.10 |
| 4–7 | 0.50 |
| 8–10 | 0.90 |

Sign is stored separately as polarity `+1`/`−1`; negative edges transmit with
the same probability but flip the rumor. The synthetic network has no
zero-rated edges.

## 5. Opponent scenarios

All opponent orders have length 50 and are nested across K; the K-seed set is
the prefix of length K. They are identical for every A method within a
scenario/K cell.

| Scenario | Construction |
|---|---|
| B1 Random | `random.Random(master_seed + 810001).sample(nodes, 50)` |
| B2 Degree | influence out-degree descending, ties by ascending node id |
| B3 Positive PageRank | PageRank on the positive-only directed influence subgraph, edge weight = probability, α=0.85, max_iter=200, tol=1e-10; descending, ties by id |
| B4 Community | Louvain on the undirected unweighted influence projection, seed 42; largest community; sort by influence out-degree descending, ties by id |

## 6. Candidate pools and SSA

### 6.1 SSA candidate pool

Target size `10·K`. Polarity-aware score:

```
score(u) = Σ p(positive outgoing edges) − Σ p(negative outgoing edges)
```

Ranking key: `(-net_score, -positive_score, negative_score, node_id)`.
Composition: `n_random = int(round(0.30·pool_size))`, `n_ranked =
pool_size − n_random`; the random part is sampled without replacement from the
ranked remainder. Opponent seeds are excluded.

### 6.2 SSA

Population 12, iterations 10, mutation 0.20,
`c1 = 2·exp(−(4t/T)²)`, exploitation probability `1 − c1/2`; leader/follower
updates; best-so-far food preserved; worst individual replaced by food;
solutions repaired to exactly K unique pool nodes.

SSA is run once per (scenario, K, objective). The seed sets for O1/O4/O6 are
reused from the authoritative prior SSA-only run after validation; SSA-Notebook
is run new with the notebook's original seed schedule.

### 6.3 SSA objectives

| ID | Formula |
|---|---|
| O1 | `Spread_A + 1.5·Blocked_B` |
| O4 | `Spread_A + 2.0·Blocked_B` |
| O6 | `α(K)·Spread_A + β(K)·Blocked_B` |

O6 coefficients: K=10,20 → (0.75, 2.25); K=30,40 → (1.00, 1.75);
K=50 → (1.25, 1.50).

`Blocked_B = max(0, mean_s baseline_B(B,s) − mean_s Spread_B(A,B,s))`, with the
same paired Monte Carlo seeds for the B-only baseline and the A/B run.

## 7. Notebook baselines

Detailed audit: `ALGORITHM_AUDIT.md`.

- **Random:** uniform sample of eligible nodes, notebook seed formula.
- **Top Degree:** top influence out-degree, ties by id.
- **Signed PageRank:** two-layer signed random walk, transition weight
  `abs(rating)`, teleport to positive layer only, score = positive minus
  negative.
- **CRIEG:** 2000 competitive reverse-reachable sketches on the positive
  subgraph, weights 1.0 (strictly before B) / 0.5 (tie), greedy coverage.
- **NCGE:** centrality score
  `0.5·degree_norm + 0.3·closeness_norm + 0.2·betweenness_norm`; top `10·K`
  eligible nodes; lazy-greedy marginal gain with O1 fitness.
- **Signed-GA:** signed Voter equilibrium linear system plus 30 projected
  gradient steps; top-K allocated nodes.
- **CELF:** lazy-greedy over the weighted candidate pool with O1 fitness.
- **SSA-Notebook:** the notebook's SSA with O1 fitness and
  `master_seed + 30000 + scenario_number` seed schedule.

## 8. FIBMM

Detailed mapping: `PAPER_IMPLEMENTATION.md`.

FIBMM uses the paper's maximin fairness objective
`σ(S) = (1−h)·Π(S) + h·τ(S)` and its two-stage greedy selection. The
selection surrogate is a positive-edge CIC model; the external evaluation is
the full signed Maki–Thompson model. Defaults: h=0.9, ε=0.1, l=1, Louvain
communities, small communities (<0.5% of nodes) merged, CRP total cap 600,000
per configuration. The internal fairness metrics and sample-size plan are
saved per cell.

## 9. Monte Carlo and random seeds

- **Optimization MC:** 15 common seeds per scenario:
  `master_seed·100000 + scenario_number·10000 + i`, i=0..14.
- **Final evaluation MC:** 100 common seeds:
  `900000000 + i`, i=0..99.
- **SSA O1/O4/O6:** reused prior run; same optimization/final seeds.
- **SSA-Notebook:** random seed `master_seed + 30000 + scenario_number`.
- **Random:** `master_seed + 50000 + K·100 + scenario_number`.
- **CRIEG:** `master_seed + 820000 + K·100 + scenario_number`.
- **FIBMM CRP:** `master_seed + 900000 + scenario_number·1000 + K`.
- **Signed PageRank / NCGE / Signed-GA:** deterministic; NCGE betweenness uses
  seed 42.

The 15 optimization MC runs are **not** 15 independent SSA runs. They estimate
fitness during search. The 100 final MC runs measure the selected seed set.
There is one optimizer run per method/configuration; between-optimizer-run
variability is not estimated. The SSA-Notebook baseline is an additional
independent SSA run for O1 and provides a descriptive sensitivity check.

## 10. Evaluation protocol

For each method, scenario and K:

1. Select exactly K unique eligible A seeds, disjoint from B.
2. Evaluate the A seed set against the fixed B set on 100 common final seeds
   with the C++ engine.
3. Compute the paired B-only baseline on the same seeds.
4. Report means and 95% bootstrap CIs for Net Advantage, Spread_A, Spread_B
   and Blocked_B.
5. Save the per-run MC table and a checkpoint containing the seed set,
   selection metadata and configuration signature.

The primary ranking is by mean Net Advantage. The component spreads are
reported to explain the result. Paired comparisons between the winner and
runner-up use the same 100 seeds; they are conditional on the two fixed seed
sets and do not represent optimizer variability.

## 11. Computation reuse and checkpointing

- The prior SSA-only O1/O4/O6 rows are reused only after validating: network
  file hash, influence semantic hash, direction, node/edge counts, master
  seed, SSA parameters, optimization/final MC counts, candidate-pool signature,
  opponent seed set, A seed set, and the 100-row MC file.
- A deterministic reproduction check reruns B1/K=10/O1 and compares the seed
  set with the stored one.
- Each new cell is checkpointed atomically. Results and the ledger are written
  atomically. Existing valid cells are skipped on rerun.
- The ledger records `completed_valid`, `reused`, `invalidated`, `failed`,
  `incomplete` and the reason.
- A chart-style change does not trigger simulations. Analysis is read-only.

## 12. Validation

- Network direction and identity checks.
- Opponent nested-order checks.
- A seed-set size, uniqueness and A/B disjointness checks.
- Candidate-pool membership checks for pool-based methods.
- Net Advantage identity and non-negative Blocked_B checks.
- 100-row common-seed MC file checks.
- 60/60 prior SSA reuse validation.
- Deterministic reproduction of one SSA configuration.
- Figure text-clipping checks.
- All validation outputs are in `aggregated/validation_summary.json` and
  `aggregated/validation_report.md`.

## 13. Limitations

- One optimizer run per method/configuration; no between-run variance.
- B1–B4 are distinct opponent strategies, not independent replicates.
- Synthetic network; conditional conclusions.
- FIBMM's selection surrogate is unsigned CIC, while evaluation is signed MT.
- CELF and NCGE use O1 fitness, not Net Advantage.
- Literature baselines are adaptations; their original papers use different
  diffusion models and assumptions.
- CELF + Local Search is excluded by the no-swap instruction.
