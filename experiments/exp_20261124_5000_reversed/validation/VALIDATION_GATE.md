# Validation Gate — 5,000-Node Reverse-Influence Rerun

Date: 2026-11-24  
Decision: **PASS — full experiment completed.**

## 1. Direction correction

| Check | Result |
|---|---|
| Trust CSV direction | `source → target` |
| Influence graph direction | `target → source` |
| Reversal applied once in `load_influence_network` | PASS |
| No component reverses again | PASS |
| All 37,500 trust edges reversed | PASS |
| Missing / extra / same-direction influence edges | 0 / 0 / 0 |
| Deterministic positive propagation | PASS |
| Deterministic negative propagation (rumour flip) | PASS |
| Multiple-edge chain | PASS |
| Candidate pool uses influence direction | PASS |
| B2 degree uses influence out-degree | PASS |
| B3 PageRank uses influence direction | PASS |
| C++ / Python consistency on tiny graphs | PASS |

## 2. Network

| Check | Result |
|---|---|
| Nodes | 5,000 |
| Trust arcs | 37,500 |
| Influence arcs | 37,500 |
| Self-loops | 0 |
| Duplicate directed arcs | 0 |
| Positive / negative | 18,706 / 18,794 |
| Seed-42 regeneration | identical DataFrame and semantic hash |
| Trust file SHA-256 | `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e` |
| Trust semantic SHA-256 | `576d00314530fc82c86d5a433e13a2ea64dbe135723350d936df21b082e2b20f` |
| Influence semantic SHA-256 | `b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9` |

## 3. Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

**Result: 121 tests, 0 failures.**  
Captured output: `validation/tests_5000.txt`.

## 4. Smoke test

- Run: `SMOKE_5000_reversed_20261124`
- 4/4 configurations (B1, K=10, O2+O3, both modes)
- Resume: re-running skips all completed rows
- Analysis ingestion: PASS

## 5. Full run

- Run: `run_20261124_5000_reversed`
- 240/240 configurations
- 202,055 simulations
- 240 MC files ×100 rows
- 120 checkpoints, 120 SSA histories
- Validation: PASS (no duplicates, no core NaNs, one run/fingerprint/direction/count, valid paired seed sets)
- Replay check: 5/5 sampled configurations reproduced their stored per-seed MC arrays exactly; one SSA checkpoint reproduced its seed set and objective score exactly (`replay_check.json`)
- Wall time: ≈2m37s

## 6. Old results

- All pre-correction 1,000-node same-direction results, analysis, logs and
  derived reports were moved (not deleted) to `to_be_discarded/`.
- Manifest: `to_be_discarded/MOVE_MANIFEST.csv`.
- No old result is used in any new table, figure, comparison or conclusion.

## 7. Benchmark

- 5,000-node influence graph: ~7,101 sims/s at 8 threads (4.64× serial).
- Full-design upper bound: 443,900 simulations; actual run 202,055.
- Benchmark: `validation/benchmark_5000.md`.

## 8. Final figures

Exactly four final PNGs exist in `final_figures/`; all were visually inspected
and show no clipped titles, overlapping legends, or unreadable labels.

## 9. Conclusion

All gates pass. The full experiment is complete and the winner analysis is
based only on the new 5,000-node reverse-influence results.
