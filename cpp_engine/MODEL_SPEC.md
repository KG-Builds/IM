# Model Specification — Signed Competitive Diffusion

Status: implemented and validated against the supplied reference notebooks in
`old work/` (six-objective and four-B notebooks).
Sources of truth: the reference notebooks, `cpp_engine/diffusion.cpp`
(production engine) and `common/diffusion.py` (readable Python reference). If
this document and the code ever disagree, the code wins; if the code and the
reference notebooks disagree, `REFERENCE_NOTEBOOK_AUDIT.md` records the
resolution.

---

## 1. Network representation

* Nodes are contiguous integers `0..999`. The engine also accepts arbitrary
  integer ids; it maps them to dense indices internally.
* One CSV row is one **directed arc** `(u -> v)`. On the delivered network:
  1,000 nodes, 7,500 unique directed arcs, 7,498 underlying unordered pairs
  and 2 reciprocal arc pairs (4 arcs). There are no self-loops and no isolated
  nodes. See `data/network_metadata_v2.json`.
* Each arc carries `probability`, `polarity` (`+1` / `-1`) and `strength`
  (`1..10`). `rating` is stored in the CSV but is not used by the dynamics.
* For the C++ engine the graph is flattened once into CSR-style arrays
  (`offsets_`, `targets_`, `probs_`, `polarity_`, `strength_`) with neighbours
  sorted by node id. Construction validates: unique node ids, per-node array
  lengths, target range, `p ∈ [0,1]` and finite, polarity `±1`, strength
  `1..10`.
* Transmission probability is a project assumption, bucketed by strength:

  | strength | probability |
  |---|---|
  | 1–3 | 0.10 |
  | 4–7 | 0.50 |
  | 8–10 | 0.90 |

## 2. States

| code | state |
|---|---|
| 0 | ignorant |
| 1 | A-spreader |
| 2 | B-spreader |
| 3 | A-stifler |
| 4 | B-stifler |

`Spread_A` counts states `{1, 3}`; `Spread_B` counts `{2, 4}`; `ignorant`
counts state `0`. Stiflers therefore remain part of their side's final spread.

## 3. Initialisation

* A seeds start as A-spreaders (state 1); B seeds start as B-spreaders
  (state 2).
* Unknown seed ids and A/B overlap are rejected.
* Every directed arc has exactly one "contacted" bit, initially clear. A
  contact, once consumed, is never available again.

## 4. Round semantics (synchronous, finite-contact)

Each round executes the following against a snapshot `old = state`; all
writes go to a separate `next` buffer initially equal to `state`.

1. **Contact phase**, in ascending node index over all `u` with
   `old[u] ∈ {1,2}`:
   * Count `available` = outgoing arcs of `u` not yet contacted.
   * If `available == 0`: `u` becomes a stifler of its own rumor; no
     transmission; no RNG draw.
   * Otherwise draw one uniform index in `[0, available-1]`, mark that arc
     contacted (one draw; the arc is consumed even if nothing is transmitted),
     and let `v` be its target.
   * If `old[v] != 0` (target not ignorant at round start): `u` becomes a
     stifler; no transmission.
   * Otherwise draw `x ~ U[0,1)`; if `x < p(arc)` the rumor is transmitted.
     The transmitted rumor is `u`'s own rumor, **flipped** on a negative arc
     (`A -> B`, `B -> A`). A positive arc transmits the rumor unchanged.
     Record a proposal `(v, rumor, strength(arc))`.
2. If no spreader existed at round start, stop **before** resolution. This
   terminal empty round is the value returned in `rounds` (reference-notebook
   semantics).
3. **Resolution phase**, in ascending target index over targets with at least
   one proposal:
   * Keep proposals with the maximum edge strength. If the surviving
     proposals agree on the rumor, that rumor wins. If both rumors tie at the
     maximum strength, choose uniformly between them (one RNG draw).
   * Write the winner into `next[v]`.
4. **Stifling**: every spreader that stifled this round moves to state 3 (if
   it was an A-spreader) or 4 (if B). This cannot collide with a proposal,
   because proposals only target nodes that were ignorant at round start.
5. `state = next`; increment the executed-round counter. On termination due to
   no spreaders, return the terminal round number = executed rounds + 1. A
   safety limit of 7,501 rounds raises an error rather than looping forever.

Termination is guaranteed because each spreader has finitely many outgoing
arcs and each arc is consumed at most once.

### Consequences

* Negative arcs transmit with the same probability as positive arcs; polarity
  changes *whose* rumor arrives, not whether it arrives.
* Strength only breaks contention; it does not scale transmission.
* There is no reinfection, no recovery back to ignorant, and no reactivation
  of stiflers.
* Contacts are per directed arc, so `u` may contact `v` and later `v` may
  contact `u` along the reciprocal arc, if one exists.

## 5. Randomness and determinism

* Each simulation uses its own `std::mt19937_64` stream seeded with the
  supplied 64-bit seed. Results depend only on `(graph, seeds_A, seeds_B,
  seed)`, never on thread scheduling.
* C++ draw order per round: contact draws in ascending spreader order
  (one edge choice, then at most one transmission draw), then one resolution
  draw per target that received at least one proposal (a genuine coin flip
  whenever both rumors tie at maximum strength).
* `simulate_many` / `simulate_batch` parallelise across runs with OpenMP,
  giving each thread a private workspace. Serial and OpenMP outputs are
  bit-identical (verified by the test suite).
* `DiffusionEngine` builds the CSR graph once and reuses it, so repeated
  Monte Carlo runs do not pay reconstruction cost.
* The Python reference uses `random.Random`, a different generator. It is
  used for model-logic parity on deterministic cases and distributional
  parity elsewhere; cross-language seed-for-seed equality is not claimed.

## 6. Monte Carlo protocol

* Optimisation evaluation: 15 common random seeds per configuration,
  `master_seed*100000 + scenario_number*10000 + i`.
* Final evaluation: 100 common random seeds
  (`900000000 .. 900000099`), shared by the SSA-only and SSA+Swap modes of the
  same `(scenario, K, objective)`.
* Metrics are means over those runs; the per-run values are stored in
  `mc_runs/*.csv`.

## 7. Metrics and objectives

Let `Spread_A`, `Spread_B` be final means, and define

```
Net_Advantage = Spread_A - Spread_B
baseline_B(B, s) = Spread_B(A = empty, B, s)
Blocked_B = max(0, mean_s baseline_B - mean_s Spread_B(A, B, s))
```

The baseline and the A/B run use the same Monte Carlo seeds (common random
numbers). This is exactly the supplied six-objective reference notebook's
`evaluate_seed_set` definition; all objectives are **validated**.

| id | name | score | validity |
|---|---|---|---|
| O1 | Original SSA | `Spread_A + 1.5·Blocked_B` | validated |
| O2 | Competition-aware | `Spread_A - Spread_B` | validated |
| O3 | Explicit blocking | `Blocked_B` | validated |
| O4 | Combined influence-blocking | `Spread_A + 2·Blocked_B` | validated |
| O5 | Gain_A + blocking | `(Spread_A - Spread_B) + 2·Blocked_B` | validated |
| O6 | Budget-aware | `α(K)·Spread_A + β(K)·Blocked_B` | validated |

O4 and O5 are distinct and must not be conflated.

Budget coefficients:

| K | α | β |
|---|---|---|
| 10 | 0.75 | 2.25 |
| 20 | 0.75 | 2.25 |
| 30 | 1.00 | 1.75 |
| 40 | 1.00 | 1.75 |
| 50 | 1.25 | 1.50 |

## 8. Optimiser and refinement

* Candidate pool: target `10·K`. Polarity-aware score
  `Σ p(positive out-arcs) − Σ p(negative out-arcs)`. Ranking key is the exact
  reference four-level key `(-net, -positive, negative, id)`. Composition:
  `n_random = int(round(size*0.30))`, `n_ranked = size - n_random`; the random
  part is sampled without replacement from `ranked[n_ranked:]` (the ranked
  remainder). Opponent B seeds are always excluded.
* Discrete SSA: population 12, 10 iterations, mutation 0.20,
  `c1 = 2·exp(-(4t/T)²)`, exploit probability `1 − c1/2`, leader/follower
  updates, best-so-far elitism (worst individual replaced by the food
  source). Every solution is repaired to exactly K unique pool nodes.
* Swap refinement: sample at most 100 one-for-one replacement pairs, evaluate
  all of them, and apply the single best strictly improving swap. It is
  elitist, preserves exactly K unique non-B nodes, and records
  `objective_before/after/gain`.
* SSA is run once per `(scenario, K, objective)`; the stored seed set is
  shared by the SSA-only and SSA+Swap modes, which makes the mode comparison
  paired.

## 9. Known limitations

* The model is a finite-contact Maki–Thompson variant, not the canonical
  infinite-contact process; `rounds` is the terminal round number (executed
  rounds + 1), matching the reference notebook.
* The 2 reciprocal arc pairs are a generator artefact of the delivered file
  and are treated as legitimate directed arcs.
* The reference notebooks reverse Bitcoin OTC edges (`target -> source`); the
  synthetic network keeps `source -> target` as generated and fingerprinted.
  This is a documented data-preprocessing difference, not a model difference.
* Zero-rated reference edges (probability 0, polarity 0) are not present in the
  synthetic network; the C++ engine requires polarity ±1.
* Between-optimizer-run variance is not estimated (one SSA run per
  configuration).
