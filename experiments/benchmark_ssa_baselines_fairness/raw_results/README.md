# Raw results

- `results.csv` — one row per method × scenario × K (240 rows when complete).
  Contains the selected seed set, opponent seed set, mean spreads, Net
  Advantage, Blocked_B, 95% Monte Carlo CIs, selection/evaluation runtimes and
  the configuration signature.
- `ledger.csv` — computation ledger. Each cell is `completed_valid`, `reused`,
  `invalidated`, `failed` or `incomplete`, with the reason.
- `manifest.json` — expected/completed/missing cells, network identity,
  environment and ledger status counts.
- `seed_sets.csv` — A and B seed sets in a flat table.
- `opponent_seeds.csv` — fixed B seed sets and order signatures.
- `mc/` — per-cell 100-row Monte Carlo tables (`seed`, `spread_A`,
  `spread_B`, `ignorant`, `rounds`, `baseline_B`, `blocked_B_run`,
  `net_advantage_run`).
- `checkpoints/` — per-cell JSON with the selected seed set, selection
  metadata and configuration signature.
- `preprocessing/` — cached one-time PageRank, NCGE centrality and FIBMM
  community structures.
- `fibmm_diagnostics/` — FIBMM internal `Π`, `τ`, `σ`, per-community saving
  ratios and CRP sample-size plan.

The per-run MC tables are the durable source for paired comparisons. The
aggregated CSVs are derived from them.
