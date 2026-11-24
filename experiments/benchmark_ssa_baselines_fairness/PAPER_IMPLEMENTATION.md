# Paper-to-Implementation Mapping — FIBMM

Paper:
**Yu Xiao, Ding Liu, Lan Yang, Zhiwu Li, “Fairness-Aware Influence Blocking
Maximization in Social Networks,” IEEE Transactions on Computational Social
Systems, 12(6), 2025.**

Source inspected (read-only):
`baseline comparison research papers/Fairness-Aware_Influence_Blocking_Maximization_in_Social_Networks.pdf`

## 1. What the paper actually does

### 1.1 Diffusion model

The paper uses the **Competitive Independent Cascade (CIC)** model on an
unsigned directed graph `G = (V,E,p)`. There is a negative rumor cascade with
seed set `S_r` and a positive truth cascade with seed set `S`. A node is
inactive, negatively active, or positively active. A newly active node gets one
attempt on each out-neighbor with probability `p_uv`; if both cascades activate
a node simultaneously, the positive cascade wins the tie. A node is
**protected** (saved) by `S` if it would be negatively active without `S` but is
positively active with `S`.

### 1.2 Fairness definition

For a community `C_i`, let `f_{C_i}(∅)` be the expected number of infected
nodes without truth seeds and `f_{C_i}(S)` the expected number with truth
seeds. The community saving ratio is

```
Π_{C_i}(S) = (f_{C_i}(∅) - f_{C_i}(S)) / f_{C_i}(∅)
```

and the overall saving ratio is `Π(S) = f*(S) / f(∅)`, where
`f*(S) = f(∅) - f(S)`. Maximin fairness is

```
τ(S) = min_{C_i ∈ C} Π_{C_i}(S)
```

The FIBM objective is

```
σ(S) = (1 - h) · Π(S) + h · τ(S)
```

with `h ∈ [0,1]` controlling the trade-off. The paper's default experiments use
`h = 0.9`, `ε = 0.1`, `l = 1`, and `n_r = 50` rumor seeds.

### 1.3 CRP sets and sampling

A **Community-Aware Reverse Protector (CRP) set** for a node `v ∈ C_i \ S_r`
in a realization `g` is the set of nodes `u ∈ V \ S_r` with
`d_g(u,v) ≤ d_g(S_r,v)`. If `v` is not reachable from any rumor seed,
`RC_i(v) = ∅`. A random CRP set samples a realization and a random `v` from
`C_i \ (S ∪ S_r)`, then performs a reverse BFS from `v`, stopping at the rumor
seeds.

The paper estimates the saving ratios by the fraction of CRP sets covered by
`S`:

```
F(RC_i,S) = (# covered CRP sets in community i) / θ_i
F(R,S)    = (# covered CRP sets overall) / θ
```

and chooses the per-community sample size `θ_i` by a martingale/Chernoff
argument. The sample-size formula uses a two-hop lower bound `LB` on the number
of saved nodes (Algorithm 2) and

```
α = sqrt(l·ln(n_i) + ln 2)
β = sqrt((1 - 1/e)·ln(n_i/k_i) + l·ln(n_i) + ln 2)
θ_i = 2·n_i·((1 - 1/e)·α + β)^2 / (LB_i·ε^2)
```

where `n_i` is the number of eligible nodes in community `i`, `k_i` is the
community's share of the budget, and `LB_i` is the community's share of the
two-hop lower bound.

### 1.4 Two-stage node selection (Algorithm 1)

1. Generate `θ_i` random CRP sets for each community.
2. **Stage 1 (utility):** select `k - floor(k·h)` nodes greedily by marginal
   gain in the estimated overall saving ratio `Π(S)`.
3. **Stage 2 (fairness):** repeatedly identify the community `C*` with the
   smallest estimated saving ratio, and add the node that covers the largest
   number of that community's remaining CRP sets. Repeat until `|S| = k`.

The paper proves FIBM is NP-hard and that FIBMM is a sampling-based
approximation algorithm.

## 2. Paper-to-code mapping

| Paper component | Paper definition | Our implementation | Adaptation / deviation |
|---|---|---|---|
| Diffusion for selection | Unsigned CIC | `algorithms/fibmm.py` builds a **positive-edge CIC surrogate**: only positive influence arcs with `p > 0` are traversable; each edge is live independently with its stored probability; the positive cascade wins ties | The evaluated model is the signed finite-contact Maki–Thompson model. The paper's CIC cannot represent negative-edge rumor flipping, so the surrogate excludes negative edges. This is the minimum defensible adaptation for the selection step. |
| Diffusion for evaluation | CIC | Full signed Maki–Thompson via the project C++ engine, 100 common MC seeds | External evaluation is deliberately different from the paper's model; this is required by the benchmark. |
| Communities | Louvain; merge communities smaller than `0.005·n` | Louvain on the undirected unweighted projection of **all** influence edges (signs ignored), seed 42; communities smaller than `round(0.005·5000)=25` merged into one | Uses all edges, matching the project's B4 community convention. The merge threshold is the paper's rule. |
| Rumor seeds `S_r` | 50 random rumor nodes | The benchmark's B1–B4 opponent seed set for the current scenario and K | The paper fixes `n_r = 50`; the benchmark varies K = 10–50, so `S_r` has size K. This is required to compare at the same budgets. |
| Truth budget `k` | `k` truth seeds | The benchmark's K | Same. |
| Fairness `Π`, `τ`, `σ` | Maximin saving ratio | Same definitions estimated from CRP coverage; reported in `fibmm_fairness.csv` and diagnostics | Internal values are surrogate estimates, not signed-MT measurements. |
| CRP set | Reverse BFS until rumor seed; `∅` if unreachable | `_sample_crp` performs reverse BFS on the positive-edge CIC surrogate, records nodes with distance ≤ the nearest rumor seed, returns `[]` if none | Exact paper semantics on the surrogate. |
| Sample size `θ_i` | Martingale formula with two-hop `LB` | `two_hop_lower_bound` implements Algorithm 2; `sample_size_plan` implements the α/β/θ formula | The total sample count is capped at 600,000 per configuration for feasibility; the raw and capped values are recorded. |
| Two-hop lower bound | Algorithm 2 | `two_hop_lower_bound`: Q1/Q2, κ1, κ2, κ1′, `LB = inf1 - inf2` (floored at 1) | The paper's OCR is ambiguous about the summation/min in Algorithm 2; we use the natural sum-based lower bound. This is documented. |
| Stage 1 count | `k - k·h` | `k - floor(k·h)` (avoids floating-point `floor(0.9999)=0` at K=10, h=0.9) | Equivalent intended integer count. |
| Stage 2 community choice | `argmin_C Π_C(S)` | Same, excluding communities with zero non-empty CRP sets (already safe) | If all communities are safe, the fallback uses overall coverage. |
| Stage 2 node choice | `argmax_v F(RC_C*, {v})` | Same; ties by smallest node id | Deterministic tie-break. |
| Coverage definition | `V1` covered by `V2` if they intersect | `RC ∩ S ≠ ∅` | The paper text contains an apparent typo (`V1 ∩ V2 = ∅`); the algorithm and standard coverage semantics require non-empty intersection. |
| Parameters | `h=0.9`, `ε=0.1`, `l=1` | Same | No tuning on the benchmark. |
| Evaluation metrics | `Π`, `τ`, `σ`, POF | External Net Advantage, Spread_A, Spread_B, Blocked_B; internal `Π`, `τ`, `σ` also reported | The benchmark's primary metric is Net Advantage, as required. FIBMM is **not** rewritten into O1/O4/O6. |

## 3. What was modified and why

1. **Positive-edge CIC surrogate.** The paper assumes unsigned edges. Our
   network is signed, and negative edges flip the rumor in the evaluated
   model. A truth campaign cannot reliably use a negative edge to spread the
   truth, so the FIBMM selection surrogate uses only positive edges. This is
   the minimum change that lets the paper's reverse-reachability machinery
   operate without pretending the signed model is unsigned.
2. **Sample-size cap.** The exact martingale formula would require roughly
   0.35–2.0 million CRP sets per configuration on this network (depending on
   the scenario and K). Storing and processing that many sets is not practical
   in Python. The experiment caps
   the total at 600,000 per configuration, allocated proportionally to the
   paper's `θ_i`, with a minimum of 50 per included community. The raw
   `θ_i` values, scale factor and realized counts are saved for every cell.
3. **Community projection.** The paper's networks are unsigned; our Louvain
   projection ignores signs and uses all influence edges, matching the
   project's B4 convention. This is a documented choice, not a claim that
   signed communities are equivalent.
4. **Rumor-seed count.** The paper uses `n_r = 50`; the benchmark uses the
   scenario's K seeds. This is necessary for a fair K-matched comparison.
5. **External evaluation.** The paper evaluates `Π`, `τ`, `σ` under CIC. This
   benchmark evaluates every method with the same signed Maki–Thompson
   simulator and Net Advantage. FIBMM's internal fairness values are therefore
   reported separately and are not used to rank methods.

## 4. What could not be preserved

- The paper's exact CIC diffusion cannot be imposed on the benchmark without
  changing the evaluated model for all methods. The selection surrogate is a
  CIC approximation; the evaluation is the project's signed MT model.
- The paper's theoretical `(1 - 1/e - ε)` style guarantee is not claimed here.
  The sample budget is capped, and the surrogate differs from the evaluated
  diffusion.
- External community-level saved-node counts under the signed MT model are not
  produced by the aggregate C++ engine. The internal CRP-based `Π`, `τ`, `σ`
  are reported as the method's own fairness estimates; external Net Advantage
  and Blocked_B are reported for every method.
- The paper's POF (price of fairness) requires a fairness-blind IBM solution
  under the same CIC model. This benchmark instead compares FIBMM directly to
  the SSA variants and notebook baselines under the common external metric.

## 5. Is the adaptation a fair baseline?

Yes, with the limitations above stated explicitly:

- It retains the paper's problem definition, maximin fairness objective,
  CRP-set construction, two-stage greedy selection, and default `h`, `ε`, `l`.
- It does not rewrite FIBMM into O1/O4/O6 or tune it to the external metric.
- It uses its own paper-defined sampling budget, not the SSA optimization
  budget.
- Every method, including FIBMM, is evaluated by the same external evaluator,
  network, opponent seeds, K values and 100 final MC seeds.
- The main scientific caveat is that FIBMM's internal surrogate is unsigned
  CIC while the external model is signed MT; this is a model-adaptation
  limitation, not a hidden advantage for FIBMM.

## 6. Code locations

| Component | File |
|---|---|
| Community detection | `algorithms/fibmm.py::detect_communities` |
| Two-hop lower bound | `algorithms/fibmm.py::two_hop_lower_bound` |
| Sample-size plan | `algorithms/fibmm.py::sample_size_plan` |
| CRP sampling | `algorithms/fibmm.py::_sample_crp`, `build_crp_sets` |
| Two-stage selection | `algorithms/fibmm.py::select_fibmm` |
| Runner integration | `run_benchmark.py::BenchmarkRunner._select` |
| Diagnostics | `raw_results/fibmm_diagnostics/FIBMM_<scenario>_K<K>.json` |
| Aggregated fairness table | `aggregated/fibmm_fairness.csv` |
