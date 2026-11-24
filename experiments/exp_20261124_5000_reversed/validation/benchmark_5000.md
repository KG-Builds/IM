# Benchmark — 5000-node reversed influence graph

Generated: 2026-11-24T01:42:34+05:30

- Nodes: 5000
- Influence edges: 37500
- Direction: reverse_of_trust

| Threads | Sims | Seconds | Sims/s | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 1000 | 0.6530 | 1531 | 1.00x |
| 2 | 1000 | 0.3353 | 2982 | 1.95x |
| 4 | 1000 | 0.1821 | 5491 | 3.59x |
| 8 | 1000 | 0.1408 | 7102 | 4.64x |

Serial/parallel checksums identical: **True**

- Full-design upper-bound simulations: 443,900
- Serial estimate: 290s
- 8-thread estimate: 63s

The estimate is an upper bound; SSA uses small batches and adds Python overhead.
