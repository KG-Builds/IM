# Figures

All figures are generated read-only from validated aggregated results by
`analyze_benchmark.py`.  Regenerating a figure never reruns a simulation.

| Figure | Purpose |
|---|---|
| `Net_Advantage_B1_Random.png` | Primary B1 chart: two panels (SSA/fairness and traditional baselines), Net Advantage vs K. |
| `Net_Advantage_B2_Degree.png` | Primary B2 chart. |
| `Net_Advantage_B3_Positive_PageRank.png` | Primary B3 chart. |
| `Net_Advantage_B4_Community.png` | Primary B4 chart. |
| `Net_Advantage_SSA_and_Fairness_All_Scenarios.png` | Focus chart for SSA O1/O4/O6, SSA-Notebook and FIBMM across all scenarios. |
| `Mean_Net_Advantage_by_Algorithm.png` | Descriptive mean across the 20 scenario × K cells. |
| `Selection_Runtime_by_Algorithm.png` | Mean selection runtime per configuration (log scale). |

Error bars in the line charts are 95% Monte Carlo CIs over the 100 common final
evaluation seeds.  They do not represent optimizer variability.  The full
figure manifest is `../aggregated/figure_manifest.csv`.
