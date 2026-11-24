# Analysis report — SMOKE_TEST_REFERENCE_ALIGNED_20261123_2100

- Run directory: `results/smoke/SMOKE_TEST_REFERENCE_ALIGNED_20261123_2100`
- Run kind: `SMOKE_TEST`
- Generated: 2026-11-23T15:27:42+00:00
- Rows: 4
- Network fingerprint: `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`
- Validation: PASS

## Validation issues

- None.

## Data provenance

- Config: `{"final_mc_runs": 10, "k_values": [10], "master_seed": 42, "modes": ["SSA only", "SSA + Swap refinement"], "n_threads": 0, "network_path": "/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/data/synthetic_network_1000_7500.csv", "objective_ids": ["O2", "O3"], "optimization_mc_runs": 5, "pool_ratio": 10, "random_pool_fraction": 0.3, "results_root": "/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/results/smoke", "resume": true, "run_id": "SMOKE_TEST_REFERENCE_ALIGNED_20261123_2100", "run_kind": "SMOKE_TEST", "scenarios": ["B1"], "ssa_iterations": 2, "ssa_mutation_rate": 0.2, "ssa_population": 4, "swap_max_tests": 10}`
- Evaluation protocol: `paired_b_only_baseline_v1`

## Caveats

- All six objective formulas and the Blocked_B protocol are confirmed against the supplied six-objective reference notebook.
- Blocked_B uses the reference-confirmed paired B-only baseline with common random seeds.
- Net Advantage is the ranking metric; optimizer fitness is not used for ranking.
- Monte Carlo replicates are common random numbers, not independent optimizer runs.

## Overall summary

| objective_id   | objective_name    | objective_validity   | mode                  |   n_configs |   net_advantage_mean |   net_advantage_std |   net_advantage_se |   net_advantage_min |   net_advantage_max |   spread_A_mean |   spread_B_mean |   blocked_B_mean |   ignorant_mean |   rounds_mean |   total_runtime_mean |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   net_advantage_se_mean |
|:---------------|:------------------|:---------------------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------:|----------------:|----------------:|-----------------:|----------------:|--------------:|---------------------:|-------------------:|--------------------:|--------------------------:|------------------------:|
| O2             | Competition-aware | validated            | SSA + Swap refinement |           1 |               6.1000 |                 nan |                nan |              6.1000 |              6.1000 |        208.0000 |        201.9000 |           6.7000 |        590.1000 |       24.6000 |               0.0255 |             0.0131 |              0.0116 |                    0.0009 |                  5.2227 |
| O2             | Competition-aware | validated            | SSA only              |           1 |              11.9000 |                 nan |                nan |             11.9000 |             11.9000 |        209.1000 |        197.2000 |          11.4000 |        593.7000 |       25.3000 |               0.0178 |             0.0131 |              0.0000 |                    0.0047 |                  7.1328 |
| O3             | Explicit blocking | validated            | SSA + Swap refinement |           1 |              -5.0000 |                 nan |                nan |             -5.0000 |             -5.0000 |        206.2000 |        211.2000 |           0.0000 |        582.6000 |       25.8000 |               0.0188 |             0.0090 |              0.0090 |                    0.0009 |                  4.4994 |
| O3             | Explicit blocking | validated            | SSA only              |           1 |              -0.8000 |                 nan |                nan |             -0.8000 |             -0.8000 |        204.7000 |        205.5000 |           3.1000 |        589.8000 |       22.8000 |               0.0101 |             0.0090 |              0.0000 |                    0.0011 |                  3.3658 |

## Objective ranking (final Net Advantage)

| objective_id   | objective_name    | objective_validity   | mode                  |   n_configs |   net_advantage_mean |   net_advantage_std |   net_advantage_se |   net_advantage_min |   net_advantage_max |   spread_A_mean |   spread_B_mean |   blocked_B_mean |   ignorant_mean |   rounds_mean |   total_runtime_mean |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   net_advantage_se_mean |   rank_in_mode |
|:---------------|:------------------|:---------------------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------:|----------------:|----------------:|-----------------:|----------------:|--------------:|---------------------:|-------------------:|--------------------:|--------------------------:|------------------------:|---------------:|
| O2             | Competition-aware | validated            | SSA + Swap refinement |           1 |               6.1000 |                 nan |                nan |              6.1000 |              6.1000 |        208.0000 |        201.9000 |           6.7000 |        590.1000 |       24.6000 |               0.0255 |             0.0131 |              0.0116 |                    0.0009 |                  5.2227 |              1 |
| O3             | Explicit blocking | validated            | SSA + Swap refinement |           1 |              -5.0000 |                 nan |                nan |             -5.0000 |             -5.0000 |        206.2000 |        211.2000 |           0.0000 |        582.6000 |       25.8000 |               0.0188 |             0.0090 |              0.0090 |                    0.0009 |                  4.4994 |              2 |
| O2             | Competition-aware | validated            | SSA only              |           1 |              11.9000 |                 nan |                nan |             11.9000 |             11.9000 |        209.1000 |        197.2000 |          11.4000 |        593.7000 |       25.3000 |               0.0178 |             0.0131 |              0.0000 |                    0.0047 |                  7.1328 |              1 |
| O3             | Explicit blocking | validated            | SSA only              |           1 |              -0.8000 |                 nan |                nan |             -0.8000 |             -0.8000 |        204.7000 |        205.5000 |           3.1000 |        589.8000 |       22.8000 |               0.0101 |             0.0090 |              0.0000 |                    0.0011 |                  3.3658 |              2 |

## Swap vs SSA paired comparison

| scenario   |   K | objective_id   | objective_name    | objective_validity   |   n_pairs |   swap_minus_ssa |   ci_low |   ci_high |   p_value |   fraction_swap_better |   cohens_dz |   ssa_net_advantage |   swap_net_advantage |
|:-----------|----:|:---------------|:------------------|:---------------------|----------:|-----------------:|---------:|----------:|----------:|-----------------------:|------------:|--------------------:|---------------------:|
| B1         |  10 | O2             | Competition-aware | validated            |        10 |          -5.8000 | -18.7875 |    6.9100 |    0.4475 |                 0.4000 |     -0.2512 |             11.9000 |               6.1000 |
| B1         |  10 | O3             | Explicit blocking | validated            |        10 |          -4.2000 | -12.4575 |    4.0150 |    0.4236 |                 0.4000 |     -0.2651 |             -0.8000 |              -5.0000 |

## Best configuration per scenario/K

| scenario   |   K | objective_id   | objective_name    | mode     | objective_validity   |   net_advantage |   spread_A |   spread_B |   blocked_B |
|:-----------|----:|:---------------|:------------------|:---------|:---------------------|----------------:|-----------:|-----------:|------------:|
| B1         |  10 | O2             | Competition-aware | SSA only | validated            |         11.9000 |   209.1000 |   197.2000 |     11.4000 |

## Robustness across scenarios

| objective_id   | objective_name    | mode                  |      B1 |   across_scenario_mean |   across_scenario_std |   across_scenario_range |
|:---------------|:------------------|:----------------------|--------:|-----------------------:|----------------------:|------------------------:|
| O2             | Competition-aware | SSA + Swap refinement |  6.1000 |                 6.1000 |                   nan |                  0.0000 |
| O2             | Competition-aware | SSA only              | 11.9000 |                11.9000 |                   nan |                  0.0000 |
| O3             | Explicit blocking | SSA + Swap refinement | -5.0000 |                -5.0000 |                   nan |                  0.0000 |
| O3             | Explicit blocking | SSA only              | -0.8000 |                -0.8000 |                   nan |                  0.0000 |

## Runtime

| scenario   | mode                  |   n_configs |   total_runtime_mean |   total_runtime_std |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   simulations_mean |   simulation_seconds_mean |
|:-----------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------------:|-------------------:|--------------------------:|
| B1         | SSA + Swap refinement |           2 |               0.0222 |              0.0048 |             0.0110 |              0.0103 |                    0.0009 |            60.0000 |                    0.0097 |
| B1         | SSA only              |           2 |               0.0139 |              0.0055 |             0.0110 |              0.0000 |                    0.0029 |            15.0000 |                    0.0027 |
