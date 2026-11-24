# Reference Notebook Audit and Alignment

> **Update (2026-11-24):** the earlier synthetic experiment kept
> `source → target`. The new authoritative 5,000-node experiment now applies
> the reference convention: trust `source → target`, influence
> `target → source`. See `experiments/exp_20261124_5000_reversed/reports/FINAL_REPORT.md`.
> The audit below remains the line-by-line comparison of the algorithms.

Date: 2026-11-23  
Reference files supplied in `old work/`:

| File | Role |
|---|---|
| `competitive_im_no_celf_csv_persist.ipynb` | Original four-B Net Advantage notebook |
| `competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb` | Six-objective Gain_A/Blocked_B notebook; latest reference for this experiment |

The notebooks target the Bitcoin OTC signed network. Their algorithms and
definitions are the authority for this synthetic-network implementation. This
audit compares the project code against the notebook code cell by cell and
records every correction.

---

## 1. Verified exact matches

The following definitions match the six-objective reference notebook exactly:

### 1.1 Diffusion model
Reference cells 8 and 9 (`competitive_maki_thompson`, `evaluate_seed_set`):

- states `I`, `A_S`, `B_S`, `A_T`, `B_T`
- A/B seed overlap rejected; unknown seeds rejected
- each spreader contacts each outgoing neighbour at most once
- one uniformly random uncontacted neighbour per round
- no available neighbour or non-ignorant target → stifler, no transmission
- transmission with edge probability; negative edge flips A↔B
- proposals resolved by highest edge strength; different-rumour ties random
- synchronous updates; final `Spread_A` counts `A_S` + `A_T`, likewise B
- `Blocked_B = max(0, mean baseline_B − mean Spread_B)` with the same MC seeds

### 1.2 Objectives
Reference cell 17 (`VARIANTS`, `score_variant`):

| ID | Formula | Match |
|---|---|---|
| O1 | `Spread_A + 1.5·Blocked_B` | exact |
| O2 | `Spread_A − Spread_B` | exact |
| O3 | `Blocked_B` | exact |
| O4 | `Spread_A + 2·Blocked_B` | exact |
| O5 | `(Spread_A − Spread_B) + 2·Blocked_B` | exact |
| O6 | budget coefficients 0.75/2.25 (K≤20), 1.00/1.75 (K≤40), 1.25/1.50 (K=50) | exact |

### 1.3 Opponent scenarios
Reference cell 15 (`B_ORDERS`):

- B1: `random.Random(master_seed + 810001).sample(nodes, 50)`
- B2: out-degree descending, ties by node id
- B3: PageRank on positive-only directed subgraph, weight = probability,
  α=0.85, max_iter=200, tol=1e-10
- B4: Louvain on undirected unweighted projection, seed 42; largest community
  (ties: smallest member id); sort by out-degree descending, ties by id

All match exactly, including the 50-node nested order.

### 1.4 SSA
Reference cell 19 (`discrete_ssa`):

- population 12, iterations 10, mutation 0.20
- `c1 = 2·exp(−(4t/T)²)`, exploitation probability `1 − c1/2`
- leader construction from food / current / pool
- follower update from the previous salp, then mutation
- repair to exactly K unique pool nodes
- best-so-far food preserved; worst replaced by food
- random seed `master_seed + 500000 + scenario·1000 + K·10 + variant_index`

The project SSA matches this logic exactly. A direct test compares the full
trajectory (food, score, per-iteration best/mean/unique count) against a local
re-implementation of the reference and passes.

### 1.5 Swap
Reference cell 21 (`sampled_swap_refinement`):

- start from the exact SSA set
- up to 100 sampled one-for-one pairs
- apply the single best strictly improving swap
- elitist; exactly K unique; opponent seeds excluded
- random seed `770000 + scenario·1000 + K + variant_index`

The project Swap matches exactly; a direct test compares the final set, accepted
pair, objective before/after, and test count.

### 1.6 Seed schedule
Reference cell 25 (experiment loop):

```
optimization seeds = master_seed*100000 + scenario_number*10000 + i
final seeds        = 900000000 + i
SSA seed           = master_seed + 500000 + scenario_number*1000 + K*10 + variant_index
Swap seed          = 770000 + scenario_number*1000 + K + variant_index
candidate pool     = master_seed + 600000 + scenario_number
```

The four-B notebook (lines 1195–1199) independently confirms the optimization
seed formula. The project now matches this exactly.

---

## 2. Discrepancies found and corrected

### D-R1 — Candidate-pool ranking (material)
**Reference** (cell 13; four-B cell 14):

```
(-net_signed_candidate_score,
 -positive_outgoing_score,
 negative_outgoing_score,
 node_id)
```

**Project before:** only `(-net_score, node_id)`. Ties in net score were broken
by id, not by positive/negative components.

**Impact:** on the full network, old vs reference pool set overlap was only
70/100 (B1, K=10) and 377/500 (B4, K=50). The old 30 % exploration component
also differed because of D-R2.

**Fix:** `common/candidates.py` now implements the exact four-level key.

### D-R2 — Random-part source (material)
**Reference:** `remaining = ranked[n_ranked:]`; random nodes are sampled from
this **ranked remainder**.

**Project before:** `remaining = [u for u in eligible if u not in ranked_part]`,
i.e. node-id order. `random.Random.sample` is order-sensitive, so this selected
different exploration nodes.

**Fix:** `common/candidates.py` now samples from `ranked[n_ranked:]` exactly.

### D-R3 — Pool composition rounding
**Reference:** `n_random = int(round(pool_size * random_fraction))`;
`n_ranked = pool_size − n_random`.

**Project before:** `n_ranked = int(pool_size * (1 − random_fraction))`.

For 10·K pools the result is the same (70/30), but the formulas differ for other
sizes. **Fix:** reference formula used.

### D-R4 — Optimization seed offset (material)
**Reference:** `master_seed*100000 + scenario_number*10000 + i`.

**Project before:** `master_seed*100000 + 3000000 + scenario_number*10000 + i`
(an extra +3,000,000).

**Impact:** SSA optimized on a different 15-seed sample in every configuration.

**Fix:** `common/experiment.py::derive_seeds` now uses the reference formula.
The new checkpoint signature (`schema_version 3`) records the alignment.

### D-R5 — Round counter semantics
**Reference:** returns `round_no`, the number of the terminal round in which no
spreader remained, i.e. **executed rounds + 1**. (The reference's
`for ... else` guard raises only if the `edges+1` safety bound is exhausted.)

**Project before:** returned executed rounds only (one lower).

**Impact:** only the `average_rounds` column; spread counts and all objectives
are unchanged.

**Fix:** C++ and Python now return `executed_rounds + 1`. The old-vs-new engine
regression test still checks that `Spread_A`, `Spread_B`, and `ignorant` are
bit-identical, and now asserts `new_rounds = old_rounds + 1`.

### D-R6 — Blocked_B validity status
**Reference:** exactly `max(0, baseline_B_mean − Spread_B_mean)` with paired
seeds. The project implementation already matched this.

**Fix:** the `provisional_blocked_b` label was removed. All six objectives are
now `validated` against the supplied reference. No numerical change was needed
to the Blocked_B formula itself.

### D-R7 — SSA history diagnostic
**Reference:** records `unique_solutions` per iteration.

**Fix:** added to the project history (alongside `worst_objective_score`). No
scientific effect.

### D-R8 — Objective cache scope
**Reference:** `score_variant` uses a local per-call cache keyed by
`(A, B, seeds)`; raw simulations use a global cache.

**Project:** a global metric cache keyed by graph fingerprint, A, B, seeds, and
protocol; objective values are recomputed from objective-independent metrics.
This is safe (metrics do not depend on the objective) and avoids the starter's
cross-variant cache bug. **No change required.**

### D-R9 — Small-network pool behaviour
**Reference:** raises if `pool_size > len(eligible)`.

**Project:** truncates the pool and records `truncated=True` in metadata, as a
small-network safety choice. On the 1,000-node network this branch is never
used. **Documented difference, not changed.**

### D-R10 — Graph orientation
**Reference:** builds `G.add_edge(target, source)` for Bitcoin OTC.

**Project:** the synthetic network uses `source → target`, exactly as generated
and fingerprinted. Reversing the synthetic edges would be a different network.
**Documented as a data-preprocessing difference; not changed.** This is a
researcher decision if the synthetic orientation is to be revisited.

### D-R11 — Zero-rated edges
**Reference:** rating 0 → probability 0, polarity 0, no transmission.

**Project:** synthetic ratings are magnitudes 1–10 with polarity ±1; there are
no zero ratings. **Not applicable; documented.**

### D-R12 — SSA seed in the older four-B notebook
The four-B notebook uses `master_seed + 30000 + scenario_number` for its single
SSA method. The six-objective notebook (the latest reference for this
six-variant experiment) uses the per-K/per-variant formula in §1.6. The project
uses the six-objective formula; the older formula is documented, not merged.

---

## 3. Reference-alignment tests

`tests/test_reference_alignment.py` (8 tests) re-implements the reference logic
locally and checks:

- exact full-network candidate pool for B1/K=10, B2/K=30, B3/K=50, B4/K=50
- exact four-level tie-break
- exact random-part source and composition
- exact optimization/final/SSA/Swap/pool seeds
- all six objective formulas
- exact SSA trajectory (food, score, per-iteration history)
- exact Swap result (final set, accepted pair, before/after, tests)
- exact diffusion rounds/spreads on deterministic tiny graphs

Test suite: **110 tests, 0 failures**.

---

## 4. Rerun and sensitivity

The corrected implementation was rerun as:

- Run: `run_20261123_205500_reference_aligned`
- 240/240 configurations completed
- Wall time ≈48 s; 203,640 simulations
- Validation: PASS

### Old (pre-reference) vs new (reference-aligned)

Matching 240 configurations:

- per-configuration Net Advantage correlation: **0.381**
- mean Net Advantage: old 19.04 → new 18.75 (difference −0.30)
- old O2 mean 21.63 → new O2 mean 21.26

Individual configurations changed materially because the candidate pool and
optimization seeds changed. Aggregate conclusions are broadly stable:

| Finding | Pre-reference | Reference-aligned |
|---|---|---|
| Highest mean Net Advantage | O2 (21.63) | O2 (21.26) |
| Lowest mean Net Advantage | O3 (16.38) | O3 (15.78) |
| O3 highest Blocked_B | yes (19.26) | yes (19.41) |
| Blocked_B–Net correlation | 0.36 | 0.46 |
| Swap reliably improves Net Advantage | no | no |
| Optimizer fitness strong predictor only for O2 | yes | yes (O2 0.60–0.87) |

The pre-reference run is retained under `results/run_20261123_202600/` as a
sensitivity check. The reference-aligned run is authoritative.

---

## 5. Remaining researcher decisions

1. **Synthetic edge orientation** — the reference reverses Bitcoin OTC edges;
   the synthetic network uses `source → target`. This is documented and
   preserved. Reversing it would be a new network/experiment.
2. **Small-network pool truncation** — the project truncates where the reference
   raises; the full network never hits this branch.
3. **Between-optimizer-run variance** — still not estimated; one SSA run per
   configuration.
4. **External validity** — results remain conditional on the synthetic design.

All other reference-alignment items are resolved and tested.
