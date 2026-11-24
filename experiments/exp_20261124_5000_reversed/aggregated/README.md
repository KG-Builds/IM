# Aggregated results

Generated from `raw_results/run_20261124_5000_reversed/` only. No old result is
used.

| File | Contents |
|---|---|
| `overall.csv` | Mean Net Advantage etc. by objective and mode |
| `per_scenario.csv` | Same, split by B scenario |
| `per_k.csv` | Same, split by K |
| `mode_comparison.csv` | Paired SSA vs SSA+Swap differences |
| `objective_vs_reference.csv` | Paired objective-vs-O2 differences |
| `objective_ranking.csv` | Objective ranking by final Net Advantage |
| `best_average.csv` | Mean Net Advantage per objective/mode |
| `runtime.csv` | Runtime by scenario and mode |
| `variability.csv` | MC uncertainty summaries |
| `robustness.csv` | Objective × scenario means |
| `best_config.csv` | Best configuration per scenario/K |
| `winner_table_SSA_only.csv` | 20-row B×K winner table, SSA-only |
| `winner_table_SSA_plus_Swap_refinement.csv` | 20-row B×K winner table, SSA+Swap |
| `winner_table_Combined.csv` | 20-row B×K winner table, 12 candidates |
| `winner_counts.csv` | Strict wins / ties / losses per candidate |
| `winner_summary.md` | Human-readable winner analysis |
| `summary.md` | General analysis report |

The full per-variant values for every B×K comparison are in the
`winner_table_*.csv` files.
