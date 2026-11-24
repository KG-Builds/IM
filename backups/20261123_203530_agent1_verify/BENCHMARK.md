# Benchmark — Signed Competitive Diffusion Engine

Generated: 2026-11-23T20:20:23+05:30

- Network: 1000 nodes, 7500 directed arcs
- OpenMP compiled: True
- Workload: A=50 seeds, B=50 seeds

## Persistent engine (graph built once)

| Threads | Simulations | Seconds | Sims/s | Speedup vs serial |
|---:|---:|---:|---:|---:|
| 1 | 2000 | 0.1862 | 10739 | 1.00× |
| 2 | 2000 | 0.1074 | 18620 | 1.73× |
| 4 | 2000 | 0.0684 | 29239 | 2.72× |
| 8 | 2000 | 0.0497 | 40255 | 3.75× |

Serial and parallel checksums identical: **True**

## One-shot API vs persistent engine

| API | Sims/s |
|---|---:|
| One-shot `simulate(...)` (rebuilds graph) | 692 |
| Persistent 1 thread | 10739 |
| Persistent 8 threads | 40255 |

## Python reference

- 30 simulations in 0.157s (190.8 sims/s).

## Full-design workload estimate

- SSA simulations: 237,600
- Swap simulations: 180,000
- Final evaluation simulations: 24,000
- Baseline simulations: 2,300
- **Total: 443,900 simulations**
- Serial estimate: 41s
- 8-thread estimate: 11s

The estimate is an upper bound based on full-batch throughput; SSA uses small 15-simulation batches and adds Python overhead, so the actual wall time may differ. It is comfortably inside an overnight window.

