# Validation Summary — 5,000-node reverse-influence experiment

Generated: 2026-11-23T20:26:39+00:00

## Test suite

- Command: `python -m unittest discover -s tests -p 'test_*.py'`
- Result: **121 tests, 0 failures**.
- Test artefacts are under `tests_tmp/`; no deletion was performed.

## Direction

- Trust CSV direction: `source → target`.
- Influence graph direction: `target → source`.
- Direction check: all 37,500 trust edges reversed; 0 missing, 0 extra, 0 same-direction, 0 reciprocal trust pairs.
- Deterministic tests cover positive, negative, multiple-edge, candidate-pool, B2 degree and B3 PageRank direction.

## Network

- Nodes: 5,000; trust arcs: 37,500; influence arcs: 37,500.
- Positive/negative: 18,706 / 18,794.
- Seed-42 regeneration: identical DataFrame and semantic hash.
- Trust file SHA-256: `d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e`.
- Trust semantic SHA-256: `576d00314530fc82c86d5a433e13a2ea64dbe135723350d936df21b082e2b20f`.
- Influence semantic SHA-256: `b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9`.

## Smoke

- Run: `SMOKE_5000_reversed_20261124`.
- 4/4 configurations, resume and analysis ingestion pass.

## Full run

- Run: `run_20261124_5000_reversed`.
- 240/240 configurations; 202,055 simulations.
- 240 MC files ×100 rows; 120 checkpoints; 120 SSA histories.
- Validation PASS: no duplicates, no core NaNs, one run/fingerprint/direction/count, seed sets valid and paired.

## Replay check

- 5/5 sampled configurations reproduced their stored per-seed MC arrays exactly.
- One SSA checkpoint re-ran with the identical seed set and objective score.
- File: `validation/replay_check.json`.

## Old results

- All pre-correction results, analysis, logs and derived reports moved to `to_be_discarded/`.
- Manifest: `to_be_discarded/MOVE_MANIFEST.csv`.
- No old result is used in this analysis.

## Benchmark

- 5,000-node influence graph: ~7,101 simulations/s at 8 threads (4.64× serial).
- Full-design upper bound: 443,900 simulations; actual run: 202,055.

