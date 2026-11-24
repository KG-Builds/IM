# Validation

- `tests.txt` — unit/integration test output.
- `verify.txt` — output of `python run_benchmark.py --mode verify`
  (network identity, opponent orders, prior SSA reuse and deterministic SSA
  reproduction).
- `crosscheck.txt` — cross-table consistency between raw results, algorithm
  summary, winner table and FIBMM fairness table.
- `../aggregated/validation_summary.json` — machine-readable result-integrity
  validation.
- `../aggregated/validation_report.md` — human-readable validation report.
- `../network/network_validation.json` — network counts, hashes and direction
  proof.

- `notebook_validation.txt` — structural and execution validation of `competitive_im_benchmark_with_fibm.ipynb` (load-existing mode; no simulations).

- `notebook_1000_validation.txt` — structural and execution validation of the 1,000-node notebook (load-existing mode; no simulations).
- `benchmark_1000_validation.json` — result-integrity, signature, MC-file and opponent-consistency validation for the 1,000-node run.
