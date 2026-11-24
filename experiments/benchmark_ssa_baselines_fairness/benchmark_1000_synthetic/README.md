# 1,000-Node Synthetic Benchmark

This directory contains the separate 1,000-node run used by
`competitive_im_benchmark_1000_with_fibm.ipynb`.

- Network: `data/synthetic_network_1000_7500.csv`
- Nodes: 1,000
- Directed trust arcs: 7,500
- Influence direction: exact reverse of trust (`target → source`)
- Trust file SHA-256: `659d4fa65a3904c96e5653c16f3357b95a43a00440844157b203eb0d3e3087b5`
- Influence semantic SHA-256: `0c0f4e1f1d8a3abc0e2765941e94cfe233efbe32246c5b56c960f0c1ee7f7197`
- Scenarios: B1–B4
- K: 10, 20, 30, 40, 50
- Methods: SSA O1/O4/O6, SSA-Notebook, Random, Top Degree, Signed PageRank,
  CRIEG, NCGE, Signed-GA, CELF, FIBM
- Optimization MC: 15
- Final evaluation MC: 100
- FIBM CRP cap: 100,000 per configuration

The run was produced with
`config/benchmark_1000_synthetic_config.json` and is independent of the
5,000-node benchmark.  The 5,000-node results are not reused or mixed with
this run.

The notebook reads `raw_results/results.csv`, the per-cell MC files, the
ledger, the manifest and the FIBM diagnostics.  Its default execution mode is
`load_existing`; it does not launch simulations.

## Reproduction

```bash
python run_1000_synthetic.py
```

The runner disables reuse of the 5,000-node SSA results, but it still skips
valid completed 1,000-node cells.  Use `--force` only when a dependency has
changed; use `--only-methods`, `--only-scenarios` and `--only-k` for a
bounded rerun.
