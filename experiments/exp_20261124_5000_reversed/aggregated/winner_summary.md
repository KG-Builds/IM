# Winner analysis — 5,000-node reverse-influence experiment

A strict win means the candidate has the highest observed mean Net Advantage
for that B × K comparison. Ties are exact/near-exact (tolerance 1e-9).

## SSA only

| scenario   |   K | winner   |   winner_net_advantage | is_tie   | runner_up   |   margin |
|:-----------|----:|:---------|-----------------------:|:---------|:------------|---------:|
| B1         |  10 | O4       |                 19.080 | False    | O5          |    5.050 |
| B1         |  20 | O1       |                 53.000 | False    | O6          |    2.820 |
| B1         |  30 | O1       |                 55.460 | False    | O6          |    4.710 |
| B1         |  40 | O2       |                 74.240 | False    | O3          |    8.560 |
| B1         |  50 | O1       |                 81.540 | False    | O5          |    0.610 |
| B2         |  10 | O5       |                 22.370 | False    | O2          |    0.970 |
| B2         |  20 | O1       |                 48.930 | False    | O2          |   10.570 |
| B2         |  30 | O3       |                 56.530 | False    | O6          |    5.180 |
| B2         |  40 | O1       |                 68.920 | False    | O6          |    2.470 |
| B2         |  50 | O5       |                 67.850 | False    | O4          |    0.310 |
| B3         |  10 | O1       |                 39.080 | False    | O5          |    0.280 |
| B3         |  20 | O3       |                 46.740 | False    | O4          |    5.130 |
| B3         |  30 | O2       |                 68.850 | False    | O6          |    5.710 |
| B3         |  40 | O4       |                 60.910 | False    | O6          |    4.250 |
| B3         |  50 | O5       |                 76.540 | False    | O6          |    3.530 |
| B4         |  10 | O3       |                 29.330 | False    | O5          |    1.030 |
| B4         |  20 | O3       |                 56.190 | False    | O1          |    9.960 |
| B4         |  30 | O5       |                 68.300 | False    | O2          |    3.840 |
| B4         |  40 | O3       |                 66.630 | False    | O5          |    2.660 |
| B4         |  50 | O2       |                 77.550 | False    | O1          |    2.830 |

## SSA + Swap refinement

| scenario   |   K | winner   |   winner_net_advantage | is_tie   | runner_up   |   margin |
|:-----------|----:|:---------|-----------------------:|:---------|:------------|---------:|
| B1         |  10 | O4       |                 19.080 | False    | O2          |    4.590 |
| B1         |  20 | O1       |                 61.750 | False    | O6          |   12.830 |
| B1         |  30 | O1       |                 52.260 | False    | O6          |    1.510 |
| B1         |  40 | O2       |                 77.870 | False    | O6          |    4.180 |
| B1         |  50 | O4       |                 85.300 | False    | O6          |    1.910 |
| B2         |  10 | O4       |                 32.700 | False    | O1          |    5.710 |
| B2         |  20 | O2       |                 45.430 | False    | O5          |    4.470 |
| B2         |  30 | O3       |                 56.530 | False    | O4          |    2.550 |
| B2         |  40 | O1       |                 67.240 | False    | O6          |    0.790 |
| B2         |  50 | O4       |                 67.540 | False    | O5          |    0.160 |
| B3         |  10 | O1       |                 33.500 | False    | O5          |    2.560 |
| B3         |  20 | O4       |                 46.820 | False    | O3          |    0.400 |
| B3         |  30 | O5       |                 61.270 | False    | O6          |    1.750 |
| B3         |  40 | O6       |                 64.160 | False    | O4          |    3.250 |
| B3         |  50 | O6       |                 74.820 | False    | O1          |    8.370 |
| B4         |  10 | O3       |                 28.050 | False    | O5          |    1.690 |
| B4         |  20 | O3       |                 57.820 | False    | O4          |    0.070 |
| B4         |  30 | O2       |                 64.460 | False    | O5          |    9.350 |
| B4         |  40 | O3       |                 66.630 | False    | O2          |    4.290 |
| B4         |  50 | O1       |                 85.850 | False    | O6          |   15.430 |

## Combined

| scenario   |   K | winner                                    |   winner_net_advantage | is_tie   | runner_up                  |   margin |
|:-----------|----:|:------------------------------------------|-----------------------:|:---------|:---------------------------|---------:|
| B1         |  10 | O4 | SSA + Swap refinement; O4 | SSA only |                 19.080 | True     | O2 | SSA + Swap refinement |    4.590 |
| B1         |  20 | O1 | SSA + Swap refinement                |                 61.750 | False    | O1 | SSA only              |    8.750 |
| B1         |  30 | O1 | SSA only                             |                 55.460 | False    | O1 | SSA + Swap refinement |    3.200 |
| B1         |  40 | O2 | SSA + Swap refinement                |                 77.870 | False    | O2 | SSA only              |    3.630 |
| B1         |  50 | O4 | SSA + Swap refinement                |                 85.300 | False    | O6 | SSA + Swap refinement |    1.910 |
| B2         |  10 | O4 | SSA + Swap refinement                |                 32.700 | False    | O1 | SSA + Swap refinement |    5.710 |
| B2         |  20 | O1 | SSA only                             |                 48.930 | False    | O2 | SSA + Swap refinement |    3.500 |
| B2         |  30 | O3 | SSA + Swap refinement; O3 | SSA only |                 56.530 | True     | O4 | SSA + Swap refinement |    2.550 |
| B2         |  40 | O1 | SSA only                             |                 68.920 | False    | O1 | SSA + Swap refinement |    1.680 |
| B2         |  50 | O5 | SSA only                             |                 67.850 | False    | O4 | SSA only              |    0.310 |
| B3         |  10 | O1 | SSA only                             |                 39.080 | False    | O5 | SSA only              |    0.280 |
| B3         |  20 | O4 | SSA + Swap refinement                |                 46.820 | False    | O3 | SSA only              |    0.080 |
| B3         |  30 | O2 | SSA only                             |                 68.850 | False    | O6 | SSA only              |    5.710 |
| B3         |  40 | O6 | SSA + Swap refinement                |                 64.160 | False    | O4 | SSA only              |    3.250 |
| B3         |  50 | O5 | SSA only                             |                 76.540 | False    | O6 | SSA + Swap refinement |    1.720 |
| B4         |  10 | O3 | SSA only                             |                 29.330 | False    | O5 | SSA only              |    1.030 |
| B4         |  20 | O3 | SSA + Swap refinement                |                 57.820 | False    | O4 | SSA + Swap refinement |    0.070 |
| B4         |  30 | O5 | SSA only                             |                 68.300 | False    | O2 | SSA only              |    3.840 |
| B4         |  40 | O3 | SSA + Swap refinement; O3 | SSA only |                 66.630 | True     | O5 | SSA only              |    2.660 |
| B4         |  50 | O1 | SSA + Swap refinement                |                 85.850 | False    | O2 | SSA only              |    8.300 |

## Win / tie / loss counts

| mode                  | candidate                  |   strict_wins |   ties |   losses | always_wins   |
|:----------------------|:---------------------------|--------------:|-------:|---------:|:--------------|
| SSA only              | O1                         |             6 |      0 |       14 | False         |
| SSA only              | O3                         |             5 |      0 |       15 | False         |
| SSA only              | O5                         |             4 |      0 |       16 | False         |
| SSA only              | O2                         |             3 |      0 |       17 | False         |
| SSA only              | O4                         |             2 |      0 |       18 | False         |
| SSA only              | O6                         |             0 |      0 |       20 | False         |
| SSA + Swap refinement | O1                         |             5 |      0 |       15 | False         |
| SSA + Swap refinement | O4                         |             5 |      0 |       15 | False         |
| SSA + Swap refinement | O3                         |             4 |      0 |       16 | False         |
| SSA + Swap refinement | O2                         |             3 |      0 |       17 | False         |
| SSA + Swap refinement | O6                         |             2 |      0 |       18 | False         |
| SSA + Swap refinement | O5                         |             1 |      0 |       19 | False         |
| Combined              | O1 | SSA only              |             4 |      0 |       16 | False         |
| Combined              | O4 | SSA + Swap refinement |             3 |      1 |       16 | False         |
| Combined              | O5 | SSA only              |             3 |      0 |       17 | False         |
| Combined              | O1 | SSA + Swap refinement |             2 |      0 |       18 | False         |
| Combined              | O3 | SSA only              |             1 |      2 |       17 | False         |
| Combined              | O3 | SSA + Swap refinement |             1 |      2 |       17 | False         |
| Combined              | O2 | SSA only              |             1 |      0 |       19 | False         |
| Combined              | O2 | SSA + Swap refinement |             1 |      0 |       19 | False         |
| Combined              | O6 | SSA + Swap refinement |             1 |      0 |       19 | False         |
| Combined              | O4 | SSA only              |             0 |      1 |       19 | False         |
| Combined              | O6 | SSA only              |             0 |      0 |       20 | False         |
| Combined              | O5 | SSA + Swap refinement |             0 |      0 |       20 | False         |

**No single SSA variant consistently wins every B scenario and every K value.**
