# Analysis report — SMOKE_TEST_dev_20261123_2015

- Run directory: `results/smoke/SMOKE_TEST_dev_20261123_2015`
- Run kind: `SMOKE_TEST`
- Generated: 2026-11-23T14:48:19+00:00
- Rows: 4
- Network fingerprint: `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`
- Validation: PASS

## Validation issues

- None.

## Data provenance

- Config: `{"final_mc_runs": 10, "k_values": [10], "master_seed": 42, "modes": ["SSA only", "SSA + Swap refinement"], "n_threads": 0, "network_path": "/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/data/synthetic_network_1000_7500.csv", "objective_ids": ["O2", "O3"], "optimization_mc_runs": 5, "pool_ratio": 10, "random_pool_fraction": 0.3, "results_root": "/home/kushagragarg/im/IM_Synthetic_Experiment_Starter/IM_Synthetic_Experiment/results/smoke", "resume": true, "run_id": "SMOKE_TEST_dev_20261123_2015", "run_kind": "SMOKE_TEST", "scenarios": ["B1"], "ssa_iterations": 2, "ssa_mutation_rate": 0.2, "ssa_population": 4, "swap_max_tests": 10}`
- Evaluation protocol: `paired_b_only_baseline_v1`

## Caveats

- Objective O2 (Competition-aware) is the only fully validated objective.
- O1/O3/O4/O5/O6 depend on the provisional paired B-only Blocked_B protocol.
- Net Advantage is the ranking metric; optimizer fitness is not used for ranking.
- Monte Carlo replicates are common random numbers, not independent optimizer runs.

## Overall summary

| objective_id   | objective_name    | objective_validity    | mode                  |   n_configs |   net_advantage_mean |   net_advantage_std |   net_advantage_se |   net_advantage_min |   net_advantage_max |   spread_A_mean |   spread_B_mean |   blocked_B_mean |   ignorant_mean |   rounds_mean |   total_runtime_mean |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   net_advantage_se_mean |
|:---------------|:------------------|:----------------------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------:|----------------:|----------------:|-----------------:|----------------:|--------------:|---------------------:|-------------------:|--------------------:|--------------------------:|------------------------:|
| O2             | Competition-aware | validated             | SSA + Swap refinement |           1 |              14.5000 |                 nan |                nan |             14.5000 |             14.5000 |        207.1000 |        192.6000 |          16.0000 |        600.3000 |       23.8000 |               0.0167 |             0.0086 |              0.0081 |                    0.0000 |                  5.1430 |
| O2             | Competition-aware | validated             | SSA only              |           1 |              14.5000 |                 nan |                nan |             14.5000 |             14.5000 |        207.1000 |        192.6000 |          16.0000 |        600.3000 |       23.8000 |               0.0105 |             0.0086 |              0.0000 |                    0.0019 |                  5.1430 |
| O3             | Explicit blocking | provisional_blocked_b | SSA + Swap refinement |           1 |              13.3000 |                 nan |                nan |             13.3000 |             13.3000 |        214.1000 |        200.8000 |           7.8000 |        585.1000 |       23.6000 |               0.0171 |             0.0081 |              0.0080 |                    0.0010 |                  4.5486 |
| O3             | Explicit blocking | provisional_blocked_b | SSA only              |           1 |              12.1000 |                 nan |                nan |             12.1000 |             12.1000 |        219.1000 |        207.0000 |           1.6000 |        573.9000 |       23.2000 |               0.0091 |             0.0081 |              0.0000 |                    0.0010 |                  6.1544 |

## Objective ranking (final Net Advantage)

| objective_id   | objective_name    | objective_validity    | mode                  |   n_configs |   net_advantage_mean |   net_advantage_std |   net_advantage_se |   net_advantage_min |   net_advantage_max |   spread_A_mean |   spread_B_mean |   blocked_B_mean |   ignorant_mean |   rounds_mean |   total_runtime_mean |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   net_advantage_se_mean |   rank_in_mode |
|:---------------|:------------------|:----------------------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------:|----------------:|----------------:|-----------------:|----------------:|--------------:|---------------------:|-------------------:|--------------------:|--------------------------:|------------------------:|---------------:|
| O2             | Competition-aware | validated             | SSA + Swap refinement |           1 |              14.5000 |                 nan |                nan |             14.5000 |             14.5000 |        207.1000 |        192.6000 |          16.0000 |        600.3000 |       23.8000 |               0.0167 |             0.0086 |              0.0081 |                    0.0000 |                  5.1430 |              1 |
| O3             | Explicit blocking | provisional_blocked_b | SSA + Swap refinement |           1 |              13.3000 |                 nan |                nan |             13.3000 |             13.3000 |        214.1000 |        200.8000 |           7.8000 |        585.1000 |       23.6000 |               0.0171 |             0.0081 |              0.0080 |                    0.0010 |                  4.5486 |              2 |
| O2             | Competition-aware | validated             | SSA only              |           1 |              14.5000 |                 nan |                nan |             14.5000 |             14.5000 |        207.1000 |        192.6000 |          16.0000 |        600.3000 |       23.8000 |               0.0105 |             0.0086 |              0.0000 |                    0.0019 |                  5.1430 |              1 |
| O3             | Explicit blocking | provisional_blocked_b | SSA only              |           1 |              12.1000 |                 nan |                nan |             12.1000 |             12.1000 |        219.1000 |        207.0000 |           1.6000 |        573.9000 |       23.2000 |               0.0091 |             0.0081 |              0.0000 |                    0.0010 |                  6.1544 |              2 |

## Swap vs SSA paired comparison

| scenario   |   K | objective_id   | objective_name    | objective_validity    |   n_pairs |   swap_minus_ssa |   ci_low |   ci_high |   p_value |   fraction_swap_better |   cohens_dz |   ssa_net_advantage |   swap_net_advantage |
|:-----------|----:|:---------------|:------------------|:----------------------|----------:|-----------------:|---------:|----------:|----------:|-----------------------:|------------:|--------------------:|---------------------:|
| B1         |  10 | O2             | Competition-aware | validated             |        10 |           0.0000 |   0.0000 |    0.0000 |  nan      |                 0.0000 |    nan      |             14.5000 |              14.5000 |
| B1         |  10 | O3             | Explicit blocking | provisional_blocked_b |        10 |           1.2000 | -12.7325 |   14.2125 |    0.8891 |                 0.6000 |      0.0453 |             12.1000 |              13.3000 |

## Best configuration per scenario/K

| scenario   |   K | objective_id   | objective_name    | mode     | objective_validity   |   net_advantage |   spread_A |   spread_B |   blocked_B |
|:-----------|----:|:---------------|:------------------|:---------|:---------------------|----------------:|-----------:|-----------:|------------:|
| B1         |  10 | O2             | Competition-aware | SSA only | validated            |         14.5000 |   207.1000 |   192.6000 |     16.0000 |

## Robustness across scenarios

| objective_id   | objective_name    | mode                  |      B1 |   across_scenario_mean |   across_scenario_std |   across_scenario_range |
|:---------------|:------------------|:----------------------|--------:|-----------------------:|----------------------:|------------------------:|
| O2             | Competition-aware | SSA + Swap refinement | 14.5000 |                14.5000 |                   nan |                  0.0000 |
| O2             | Competition-aware | SSA only              | 14.5000 |                14.5000 |                   nan |                  0.0000 |
| O3             | Explicit blocking | SSA + Swap refinement | 13.3000 |                13.3000 |                   nan |                  0.0000 |
| O3             | Explicit blocking | SSA only              | 12.1000 |                12.1000 |                   nan |                  0.0000 |

## Runtime

| scenario   | mode                  |   n_configs |   net_advantage_mean |   net_advantage_std |   net_advantage_se |   net_advantage_min |   net_advantage_max |   spread_A_mean |   spread_B_mean |   blocked_B_mean |   ignorant_mean |   rounds_mean |   total_runtime_mean |   ssa_runtime_mean |   swap_runtime_mean |   final_eval_runtime_mean |   net_advantage_se_mean | objective_validity   |
|:-----------|:----------------------|------------:|---------------------:|--------------------:|-------------------:|--------------------:|--------------------:|----------------:|----------------:|-----------------:|----------------:|--------------:|---------------------:|-------------------:|--------------------:|--------------------------:|------------------------:|:---------------------|
| B1         | SSA + Swap refinement |           2 |              13.9000 |              0.8485 |             0.6000 |             13.3000 |             14.5000 |        210.6000 |        196.7000 |          11.9000 |        592.7000 |       23.7000 |               0.0169 |             0.0084 |              0.0080 |                    0.0005 |                  4.8458 | validated            |
| B1         | SSA only              |           2 |              13.3000 |              1.6971 |             1.2000 |             12.1000 |             14.5000 |        213.1000 |        199.8000 |           8.8000 |        587.1000 |       23.5000 |               0.0098 |             0.0084 |              0.0000 |                    0.0014 |                  5.6487 | validated            |
