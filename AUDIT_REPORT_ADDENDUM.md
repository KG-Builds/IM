# Audit Report Addendum — Session 2

> **Update:** the reference notebooks were subsequently supplied in `old work/`.
> The "reference notebooks absent" limitation in this addendum is therefore
> superseded. See `REFERENCE_NOTEBOOK_AUDIT.md` for the line-by-line comparison
> and the corrections/rerun triggered by the supplied notebooks.
>
> **Update (2026-11-24):** the current authoritative experiment uses a 5,000-node
> network and reverses the influence direction relative to the trust edges. See
> `experiments/exp_20261124_5000_reversed/reports/FINAL_REPORT.md`. Old
> 1,000-node same-direction results are in that experiment's
> `to_be_discarded/` folder and must not be used.

This addendum verifies and, where necessary, corrects the prior
`AUDIT_REPORT.md`. The original report is preserved unchanged.

---

## 1. Prior defect status

| ID | Prior finding | Session-2 status |
|---|---|---|
| D1 | `cpp_adapter` imported the extension before adding `cpp_engine/` to `sys.path` | **Fixed**; import order corrected and tested |
| D2 | Notebook code cells contained literal `\n` | **Fixed**; all six notebooks rebuilt and compile-checked |
| D3 | Objective score cache keyed only by `(A,B,seeds)` | **Fixed**; metric cache is objective-independent, objective values recomputed |
| D4 | `metric_cache_eval` referenced missing `blocked_B` | **Fixed**; explicit paired-baseline implementation + tests |
| D5 | No network fingerprint / incomplete metadata | **Fixed**; file + semantic SHA-256, metadata v2 |
| D6 | Incomplete resume signature, non-atomic CSV | **Fixed**; full signature, fingerprint check, atomic writes, manifest |
| D7 | CSV schema missing required fields | **Fixed**; 72-column schema with seed sets/protocols/uncertainty/runtimes |
| D8 | Graph re-serialised per simulation | **Fixed**; persistent engine, ~15.5× faster than one-shot |
| D9 | Runtime included cache hits | **Fixed**; per-config simulation counts/time + cache counters |
| D10 | Dead/incorrect helper API | **Fixed**; compatibility facade returns correct metrics |
| D11 | Fragile network generator | **Preserved intentionally**; pinned by byte-identical seed-42 test |
| D12 | Analysis notebook not read-only / no validation | **Fixed**; read-only module with validation, run selection, tables/figures |
| D13 | B notebooks ran 60 configs immediately | **Fixed**; default no-run + confirmation + CLI |
| D14 | README/Makefile install instructions | **Fixed**; manual setup documented, Makefile builds only |

---

## 2. Corrections to the prior audit's descriptive statistics

The prior audit reported a few network statistics that the delivered file does
not support. The current, test-backed values are:

| Quantity | Prior report | Verified value |
|---|---|---|
| Underlying unordered pairs | 7,500 | **7,498** |
| Reciprocal arc pairs | 4 pairs (8 arcs implied) | **2 pairs (4 reciprocal arcs)** |
| Out-degree minimum | 1 | **0** |
| Out-degree mean | 7.52 | **7.50** |
| In-degree mean | 7.50 | **7.50** |
| Sinks (out-degree 0) | not stated | **2** |
| Isolated nodes | 0 | **0** (confirmed) |
| Positive / negative arcs | 3,720 / 3,780 | **3,720 / 3,780** (confirmed) |
| Probability class counts | 2,278 / 2,950 / 2,272 | **2,278 / 2,950 / 2,272** (confirmed) |

The edge-count convention is now explicit: 7,500 unique directed arcs over
7,498 unordered pairs plus 2 reciprocal pairs. The directed-arc target is met
exactly; the unordered-pair count is not 7,500. This is documented in
`data/network_metadata_v2.json` and `CONFIG.md`.

File SHA-256:
`659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`

Semantic SHA-256:
`461febf43a90332f04f13d706cf8f34579202aab120a7716886f5f7e9f54fc25`

---

## 3. Source-of-truth verification

- No reference notebooks named in the master prompt exist inside the workspace
  (confirmed again by recursive search).
- The starter code and master prompt remain the only in-workspace authorities.
- The C++ engine was the only executable diffusion implementation; a Python
  reference was added for parity testing.
- The refactored C++ engine is bit-identical to the backed-up original engine on
  the same seeds (30 full-network runs), demonstrating that the optimization did
  not change model behaviour.

---

## 4. Scientific decisions recorded in session 2

1. **Blocked_B** remains provisional. The explicit paired B-only protocol is
   implemented and tested, but only O2 is fully validated. The full matrix is
   run with O1/O3/O4/O5/O6 labelled `provisional_blocked_b`.
2. **SSA/Swap** are preserved from the starter; no replacement optimizer and no
   silent change of acceptance policy.
3. **Network** is not regenerated or overwritten. Seed-42 regeneration is
   byte-identical and pinned by a regression test.
4. **Statistical unit** is the evaluation replicate (common random seed), not an
   independent optimizer run. Between-optimizer-run variance is not estimated.
5. **Smoke results** are isolated under `results/smoke/` and excluded from
   default analysis selection.

---

## 5. Session-2 evidence

- `python -m unittest discover -s tests -p 'test_*.py'` → **102 tests OK**
- C++ build/import → **PASS**, OpenMP enabled
- Benchmark → `BENCHMARK.md` (40,255 sims/s at 8 threads; serial/parallel
  identical; persistent engine 15.5× faster than one-shot)
- Smoke run `SMOKE_TEST_20261123_202100` → **4/4 configurations**, resume and
  analysis ingestion **PASS**
- Full-run gate → `VALIDATION_GATE.md`

Full-run status and result summary are in `FINAL_AUDIT.md`.
