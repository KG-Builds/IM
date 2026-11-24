# Workspace and Safety Compliance

- All new files and modifications are inside
  `IM_Synthetic_Experiment/experiments/benchmark_ssa_baselines_fairness/`.
- The source notebook
  `old work/competitive_im_no_celf_csv_persist.ipynb` and the paper PDF were
  opened read-only and were not modified.
- The authoritative prior experiment
  `experiments/exp_20261124_5000_reversed/` was read only.  Its results and
  checkpoints were not modified.  SSA-only MC rows were copied into the new
  experiment's `raw_results/mc/` after validation.
- The network CSV was read only; it was not regenerated or duplicated.
- No files or directories were deleted.
- No packages were installed or upgraded.
- No `sudo`, system settings, shell configuration, environment settings or
  permissions were changed.
- No git operations were performed.
- No internet/network access was used.
- No symlinks outside the permitted workspace were followed.
- Long-running work is bounded (fixed 4 × 5 × 12 matrix, fixed SSA/CRIE/
  Signed-GA/FIBMM budgets) and checkpointed per cell.
- CSV/JSON writes use atomic temporary-file replacement where they represent
  durable results or checkpoints.
