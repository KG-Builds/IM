# Logs

- `full_run_console.log` — stdout/stderr of the full run.
- `raw_results/logs/run.log` — structured run log (one line per cell).
- `../validation/verify.txt` — network/opponent/reuse verification.
- `../validation/tests.txt` — unit/integration tests.

The full run is checkpointed and resumable.  On restart, valid completed cells
are skipped; the ledger records `reused`/`completed_valid` and the reason.
