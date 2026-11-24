# Statistical Notes

## Unit of uncertainty

The 100 final Monte Carlo runs are **evaluation replicates**, not independent
optimizer runs.  A 95% CI on a mean Net Advantage describes how precisely that
fixed seed set's mean is estimated under the 100 common diffusion seeds.  It
does not describe how much the seed set would change if SSA were rerun with a
different optimizer seed.

## Paired comparisons

All methods use the same 100 final seeds.  Therefore, method comparisons can be
paired by replicate: for a given seed `s`, the difference
`NetAdv_A(s) − NetAdv_B(s)` is computed, and the 100 differences are
bootstrapped.  The winner table reports this paired interval for the winner vs
runner-up.  This is conditional on the two fixed seed sets and the one
optimizer run per method.

## B1–B4 are not replicates

B1 (random), B2 (degree), B3 (positive PageRank) and B4 (community) are four
deliberately different opponent strategies.  They are not independent samples
from a common opponent population.  The analysis therefore reports results
separately by scenario and does not pool B1–B4 into a single significance test.
Averaging over the 20 scenario × K cells is descriptive.

## Multiple comparisons

The winner table reports one paired comparison per cell (winner vs runner-up).
No family-wise adjustment is applied to those descriptive intervals.  The
`ssa_vs_baselines.csv` table reports win/tie/loss counts and mean differences,
not p-values for 27 simultaneous hypotheses.

## Optimizer variability

There is one optimizer run per method/configuration.  The explicit
SSA-Notebook baseline is an additional SSA run for O1 with the notebook's seed
schedule; its comparison with SSA O1 is a descriptive sensitivity check, not a
formal variance estimate.  A future experiment should repeat each optimizer
with independent random seeds.

## FIBMM

FIBMM's internal `Π`, `τ` and `σ` are estimates from the positive-edge CIC
surrogate.  They are not measured under the signed Maki–Thompson model.  The
external Net Advantage and Blocked_B are measured under the common signed MT
protocol.  Do not interpret a high internal `σ` as a high external Net
Advantage.
