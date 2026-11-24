# Analysis report — SMOKE_5000_reversed_20261124

- Run directory: `experiments/exp_20261124_5000_reversed/validation/smoke_results/SMOKE_5000_reversed_20261124`
- Run kind: `SMOKE_TEST_5000_REVERSED`
- Generated: 2026-11-23T20:13:08+00:00
- Rows: 4
- Network fingerprint: `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e`
- Influence direction: `reverse`
- Nodes / edges: 5000 / 37500
- Validation: PASS

## Validation issues

- None.

## Data provenance

- Config: `{"expected_edges": 37500, "expected_nodes": 5000, "final_mc_runs": 10, "influence_direction": "reverse", "k_values": [10], "master_seed": 42, "modes": ["SSA only", "SSA + Swap refinement"], "n_threads": 0, "network_path": "/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/network/synthetic_network_5000_37500.csv", "objective_ids": ["O2", "O3"], "optimization_mc_runs": 5, "pool_ratio": 10, "random_pool_fraction": 0.3, "results_root": "/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/experiments/exp_20261124_5000_reversed/validation/smoke_results", "resume": true, "run_id": "SMOKE_5000_reversed_20261124", "run_kind": "SMOKE_TEST_5000_REVERSED", "scenarios": ["B1"], "ssa_iterations": 2, "ssa_mutation_rate": 0.2, "ssa_population": 4, "swap_max_tests": 10}`
- Evaluation protocol: `paired_b_only_baseline_v1`

## Caveats

- All six objective formulas and the Blocked_B protocol are confirmed against the supplied six-objective reference notebook.
- Blocked_B uses the reference-confirmed paired B-only baseline with common random seeds.
- Net Advantage is the ranking metric; optimizer fitness is not used for ranking.
- Monte Carlo replicates are common random numbers, not independent optimizer runs.

## Overall summary

| objective_id   | objective_name    | objective_validity   | mode                  |   n_configs |   net_advantage_mean |   net_advantage_std |   net_advantage_se |   net_advantage_min |   net_advantage_max |   spread_A_mean |   spread_B_mean |   blocked_B_mean |   ignorant_mean |   rounds_mean |   total_runtime_mean |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   net_advantage_se_mean |
|:---------------|:------------------|:---------------------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------:|----------------:|----------------:|-----------------:|----------------:|--------------:|---------------------:|-------------------:|--------------------:|--------------------------:|------------------------:|
| O2             | Competition-aware | validated            | SSA + Swap refinement |           1 |             -13.5000 |                 nan |                nan |            -13.5000 |            -13.5000 |        980.3000 |        993.8000 |           0.0000 |       3025.9000 |       35.4000 |               0.0922 |             0.0486 |              0.0435 |                    0.0000 |                 11.3717 |
| O2             | Competition-aware | validated            | SSA only              |           1 |             -13.5000 |                 nan |                nan |            -13.5000 |            -13.5000 |        980.3000 |        993.8000 |           0.0000 |       3025.9000 |       35.4000 |               0.0593 |             0.0486 |              0.0000 |                    0.0106 |                 11.3717 |
| O3             | Explicit blocking | validated            | SSA + Swap refinement |           1 |              13.6000 |                 nan |                nan |             13.6000 |             13.6000 |        985.0000 |        971.4000 |          20.4000 |       3043.6000 |       36.0000 |               0.0784 |             0.0393 |              0.0391 |                    0.0000 |                  8.7244 |
| O3             | Explicit blocking | validated            | SSA only              |           1 |              13.6000 |                 nan |                nan |             13.6000 |             13.6000 |        985.0000 |        971.4000 |          20.4000 |       3043.6000 |       36.0000 |               0.0454 |             0.0393 |              0.0000 |                    0.0060 |                  8.7244 |

## Objective ranking (final Net Advantage)

| objective_id   | objective_name    | objective_validity   | mode                  |   n_configs |   net_advantage_mean |   net_advantage_std |   net_advantage_se |   net_advantage_min |   net_advantage_max |   spread_A_mean |   spread_B_mean |   blocked_B_mean |   ignorant_mean |   rounds_mean |   total_runtime_mean |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   net_advantage_se_mean |   rank_in_mode |
|:---------------|:------------------|:---------------------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------:|----------------:|----------------:|-----------------:|----------------:|--------------:|---------------------:|-------------------:|--------------------:|--------------------------:|------------------------:|---------------:|
| O3             | Explicit blocking | validated            | SSA + Swap refinement |           1 |              13.6000 |                 nan |                nan |             13.6000 |             13.6000 |        985.0000 |        971.4000 |          20.4000 |       3043.6000 |       36.0000 |               0.0784 |             0.0393 |              0.0391 |                    0.0000 |                  8.7244 |              1 |
| O2             | Competition-aware | validated            | SSA + Swap refinement |           1 |             -13.5000 |                 nan |                nan |            -13.5000 |            -13.5000 |        980.3000 |        993.8000 |           0.0000 |       3025.9000 |       35.4000 |               0.0922 |             0.0486 |              0.0435 |                    0.0000 |                 11.3717 |              2 |
| O3             | Explicit blocking | validated            | SSA only              |           1 |              13.6000 |                 nan |                nan |             13.6000 |             13.6000 |        985.0000 |        971.4000 |          20.4000 |       3043.6000 |       36.0000 |               0.0454 |             0.0393 |              0.0000 |                    0.0060 |                  8.7244 |              1 |
| O2             | Competition-aware | validated            | SSA only              |           1 |             -13.5000 |                 nan |                nan |            -13.5000 |            -13.5000 |        980.3000 |        993.8000 |           0.0000 |       3025.9000 |       35.4000 |               0.0593 |             0.0486 |              0.0000 |                    0.0106 |                 11.3717 |              2 |

## Swap vs SSA paired comparison

| scenario   |   K | objective_id   | objective_name    | objective_validity   |   n_pairs |   swap_minus_ssa |   ci_low |   ci_high |   p_value |   fraction_swap_better |   cohens_dz |   ssa_net_advantage |   swap_net_advantage |
|:-----------|----:|:---------------|:------------------|:---------------------|----------:|-----------------:|---------:|----------:|----------:|-----------------------:|------------:|--------------------:|---------------------:|
| B1         |  10 | O2             | Competition-aware | validated            |        10 |           0.0000 |   0.0000 |    0.0000 |       nan |                 0.0000 |         nan |            -13.5000 |             -13.5000 |
| B1         |  10 | O3             | Explicit blocking | validated            |        10 |           0.0000 |   0.0000 |    0.0000 |       nan |                 0.0000 |         nan |             13.6000 |              13.6000 |

## Best configuration per scenario/K

| scenario   |   K | objective_id   | objective_name    | mode     | objective_validity   |   net_advantage |   spread_A |   spread_B |   blocked_B |
|:-----------|----:|:---------------|:------------------|:---------|:---------------------|----------------:|-----------:|-----------:|------------:|
| B1         |  10 | O3             | Explicit blocking | SSA only | validated            |         13.6000 |   985.0000 |   971.4000 |     20.4000 |

## Robustness across scenarios

| objective_id   | objective_name    | mode                  |       B1 |   across_scenario_mean |   across_scenario_std |   across_scenario_range |
|:---------------|:------------------|:----------------------|---------:|-----------------------:|----------------------:|------------------------:|
| O2             | Competition-aware | SSA + Swap refinement | -13.5000 |               -13.5000 |                   nan |                  0.0000 |
| O2             | Competition-aware | SSA only              | -13.5000 |               -13.5000 |                   nan |                  0.0000 |
| O3             | Explicit blocking | SSA + Swap refinement |  13.6000 |                13.6000 |                   nan |                  0.0000 |
| O3             | Explicit blocking | SSA only              |  13.6000 |                13.6000 |                   nan |                  0.0000 |

## Runtime

| scenario   | mode                  |   n_configs |   total_runtime_mean |   total_runtime_std |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   simulations_mean |   simulation_seconds_mean |
|:-----------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------------:|-------------------:|--------------------------:|
| B1         | SSA + Swap refinement |           2 |               0.0853 |              0.0098 |             0.0440 |              0.0413 |                    0.0000 |            50.0000 |                    0.0394 |
| B1         | SSA only              |           2 |               0.0523 |              0.0098 |             0.0440 |              0.0000 |                    0.0083 |            15.0000 |                    0.0081 |
