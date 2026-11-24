# Experiment — 5,000-Node Reverse-Influence Rerun

This directory is the clean, self-contained rerun requested after the
influence-direction correction.

## What is here

| Folder | Contents |
|---|---|
| `config/` | Full/smoke experiment configuration JSON |
| `network/` | 5,000-node / 37,500-edge trust CSV, metadata, validation |
| `raw_results/` | Full run and smoke run outputs, per-seed MC files, SSA histories, per-scenario checkpoints |
| `aggregated/` | Analysis tables, winner tables, win/tie/loss counts, best-average table, winner summary |
| `final_figures/` | Exactly four final Net Advantage PNGs |
| `reports/` | `FINAL_REPORT.md` |
| `logs/` | Run and console logs |
| `checkpoints/` | Copies of the 120 SSA checkpoint JSON files |
| `validation/` | Smoke results, benchmark, test output, validation summary |
| `to_be_discarded/` | Old pre-correction outputs, moved with a manifest; **not used anywhere** |

## Key facts

- Trust network: 5,000 nodes, 37,500 directed trust arcs (`source → target`).
- Influence network: exact reverse (`target → source`), 37,500 arcs.
- Direction check: all 37,500 trust edges reversed; 0 same-direction edges.
- Generation seed: 42; seed-42 regeneration is identical.
- Full run: `run_20261124_5000_reversed`, 240/240 configurations.
- Simulations: 202,055; final evaluation 100 MC seeds per configuration.
- Wall time: ≈2m37s; summed per-config compute 225.6 s.
- Tests: 121 passing.
- Replay check: 5/5 sampled configurations reproduced their stored MC arrays
  exactly; one SSA checkpoint reproduced its seed set and score exactly.
- Old results: moved to `to_be_discarded/`, excluded from all calculations.

## How to reproduce

```bash
# From the project root
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode plan
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode smoke
python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode full --confirm-full --run-id run_20261124_5000_reversed
python experiments/exp_20261124_5000_reversed/analyze_5000.py
```

The runner never deletes files. It writes only inside this experiment directory
(plus its configured log folder).

## Final figures

- `final_figures/Net_Advantage_B1_Random.png`
- `final_figures/Net_Advantage_B2_Degree.png`
- `final_figures/Net_Advantage_B3_Positive_PageRank.png`
- `final_figures/Net_Advantage_B4_Community.png`

Each figure has two panels (SSA-only and SSA+Swap), six objective lines,
95% Monte Carlo CIs, consistent colors/markers and a shared legend.

## Main result

**No single SSA variant consistently wins every B scenario and every K value.**
The closest alternative by strict wins is **O1** (6/20 in SSA-only; tied with
O4 at 5/20 in SSA+Swap). The highest mean Net Advantage is **O6** in SSA+Swap
and **O1** in SSA-only; best average is not the same as consistent winning.

See `reports/FINAL_REPORT.md` and `aggregated/winner_summary.md`.
