# Phase 1 Audit — Current Experiment and Network

This audit was performed before implementing the benchmark.  It records what
was verified from the actual project files, not from prior reports.

## 1. Latest synthetic network

| Item | Verified value |
|---|---|
| Authoritative experiment | `experiments/exp_20261124_5000_reversed/` |
| Network file | `experiments/exp_20261124_5000_reversed/network/synthetic_network_5000_37500.csv` |
| Nodes | 5,000 (ids 0–4999) |
| Trust arcs (CSV rows) | 37,500 |
| Influence arcs | 37,500 |
| Influence direction | exact reverse of trust (`target → source`) |
| Trust file SHA-256 | `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e` |
| Trust semantic SHA-256 | `576d00314530fc82c86d5a433e13a2ea64dbe135723350d936df21b082e2b20f` |
| Influence semantic SHA-256 | `b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9` |
| Direction check | 37,500 trust edges; 37,500 expected reversed; 0 missing; 0 extra; 0 same-direction; 0 reciprocal trust pairs |
| Positive / negative arcs | 18,706 / 18,794 |
| Probability classes | 0.1: 11,151; 0.5: 14,911; 0.9: 11,438 |
| Topology | Barabási–Albert backbone (initial clique 8, m=7) plus random fill; random orientation |
| Generation seed | 42 |
| Sign | positive with probability 0.5 |
| Trust magnitude | uniform integer 1–10 |

The network was not regenerated.  The benchmark's `load_verified_network`
checks the file hash, trust/influence semantic hashes, node count, arc count
and the full reversal relation before any result is produced.

## 2. Signed-edge representation

The CSV columns are `source, target, rating, probability, strength, polarity`.
`rating = polarity * strength`; `polarity ∈ {−1,+1}`; `strength ∈ 1..10`.
The influence graph stores the same attributes on the reversed arc.

## 3. Trust-to-probability mapping

| Strength | Probability |
|---|---|
| 1–3 | 0.10 |
| 4–7 | 0.50 |
| 8–10 | 0.90 |

The mapping is a project assumption.  Negative edges use the same probability
tiers but flip the rumor.

## 4. Diffusion model

The current model is the finite-contact competitive Maki–Thompson variant in
`cpp_engine/MODEL_SPEC.md`:

- states 0 ignorant, 1 A-spreader, 2 B-spreader, 3 A-stifler, 4 B-stifler;
- synchronous rounds; each spreader contacts one uniformly random uncontacted
  outgoing influence neighbour;
- no uncontacted neighbour or non-ignorant target → stifler;
- otherwise transmit with edge probability; negative edge flips A↔B;
- proposals resolved by highest edge strength; different-rumor ties random;
- `Spread_A` counts 1+3; `Spread_B` counts 2+4;
- terminal empty round included in `rounds`.

## 5. Rumor A and B seeds

- A seeds are selected by each method, exactly K unique nodes, disjoint from B.
- B seeds are the fixed nested opponent orders B1–B4, identical for every A
  method within a scenario/K cell.
- The B orders are length 50; the K-set is the prefix.
- B1 random seed: `master_seed + 810001`.
- B2 degree: influence out-degree descending, ties by id.
- B3 positive PageRank: positive-only directed subgraph, edge weight =
  probability, α=0.85, max_iter=200, tol=1e-10.
- B4 community: Louvain on the undirected unweighted influence projection,
  seed 42; largest community; out-degree ranking.

## 6. Current SSA

`common/ssa.py` implements the discrete set-based SSA:

- population 12, iterations 10, mutation 0.20;
- `c1 = 2·exp(−(4t/T)²)`, exploitation `1 − c1/2`;
- leader/follower updates; best-so-far food; worst replaced by food;
- repair to exactly K unique candidate-pool nodes;
- candidate pool 10·K, 70/30 polarity-aware ranking;
- fitness mean over 15 optimization MC seeds.

## 7. Objectives

| ID | Formula |
|---|---|
| O1 | `Spread_A + 1.5·Blocked_B` |
| O4 | `Spread_A + 2.0·Blocked_B` |
| O6 | `α(K)·Spread_A + β(K)·Blocked_B`; K≤20 (0.75,2.25), K≤40 (1.00,1.75), K=50 (1.25,1.50) |

`Blocked_B = max(0, mean_s baseline_B(B,s) − mean_s Spread_B(A,B,s))`, paired
seeds.

## 8. Opponent strategies B1–B4

Retained exactly: B1 Random, B2 Degree, B3 Positive PageRank, B4
Community-concentrated.  See `common/opponents.py` and `METHODS.md`.

## 9. Configuration and random seeds

The authoritative prior config:

- master seed 42;
- scenarios B1–B4; K 10–50;
- optimization MC 15; final MC 100;
- SSA population 12, iterations 10, mutation 0.20;
- pool ratio 10, random fraction 0.30;
- influence direction reverse.

These values were read from
`experiments/exp_20261124_5000_reversed/config/experiment_config.json` and the
prior result rows, not assumed.

## 10. Existing artifacts and reuse decision

The prior run contains SSA-only and SSA+Swap results for O1–O6.  This
experiment:

- **reuses** SSA-only O1/O4/O6 seed sets and 100-run final evaluations after
  validating network, direction, opponent seeds, candidate-pool signature, SSA
  parameters, seed sets and MC files;
- **does not reuse** SSA+Swap results (Swap is excluded);
- **does not reuse** any old 1,000-node or same-direction result;
- **does not modify** the prior experiment.

A deterministic SSA reproduction check (B1/K=10/O1) matched the prior seed set
exactly.

## 11. What changed for the new benchmark

- New experiment directory; prior files untouched.
- New algorithms: Random, Top Degree, Signed PageRank, CRIEG, NCGE, Signed-GA,
  CELF, SSA-Notebook, FIBMM.
- Same network, same diffusion, same opponents, same K values, same 100 final
  seeds.
- Swap refinement removed from the comparison.
- Primary ranking metric: Net Advantage, not optimizer fitness.
