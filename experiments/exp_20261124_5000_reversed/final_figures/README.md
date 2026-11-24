# Final figures — exactly four

These are the only final presentation figures for the 5,000-node
reverse-influence experiment.

| File | Scenario |
|---|---|
| `Net_Advantage_B1_Random.png` | B1 — Random opponent |
| `Net_Advantage_B2_Degree.png` | B2 — Degree-based opponent |
| `Net_Advantage_B3_Positive_PageRank.png` | B3 — Positive PageRank-based opponent |
| `Net_Advantage_B4_Community.png` | B4 — Community-concentrated opponent |

Each PNG contains two panels:

- left: SSA-only
- right: SSA + Swap refinement

Both panels show Net Advantage vs K for the six objectives O1–O6, with 95%
Monte Carlo confidence intervals over 100 evaluation seeds. Colors, markers,
line styles, axis formatting and the shared legend are consistent across all
four figures.

All four were inspected programmatically and visually; no clipped titles,
overlapping legends, or unreadable labels were found.
