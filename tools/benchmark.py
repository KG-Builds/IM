"""Bounded performance benchmark (no experiment run).

Run from the project root:

    python tools/benchmark.py

Writes BENCHMARK.md and logs/benchmark_<timestamp>.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "common", ROOT / "cpp_engine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cpp_adapter import CppSimulator, openmp_available  # noqa: E402
from diffusion import simulate_python  # noqa: E402
from network import NETWORK_CSV, build_adjacency, load_network  # noqa: E402
from opponents import opponent_order  # noqa: E402
from utils import atomic_write_json, local_now_iso  # noqa: E402

import simulation_engine  # noqa: E402


def checksum(results):
    return int(sum(r["spread_A"] * 1000003 + r["spread_B"] * 1009 + r["rounds"] for r in results))


def main():
    df, G, nodes = load_network(NETWORK_CSV)
    B = opponent_order(G, nodes, "B1")[:50]
    Bset = set(B)
    A = [int(u) for u in nodes if u not in Bset][:50]
    index, adj, probs, pol, strength = build_adjacency(G, nodes)

    report = {
        "timestamp": local_now_iso(),
        "network": {"nodes": len(nodes), "directed_arcs": G.number_of_edges()},
        "openmp": bool(openmp_available()),
        "workload": {"A_seeds": len(A), "B_seeds": len(B)},
        "persistent_engine": {},
        "one_shot_engine": {},
        "python_reference": {},
        "estimate": {},
    }

    n_sims = 2000
    seeds = list(range(n_sims))
    checksums = {}
    for threads in (1, 2, 4, 8):
        sim = CppSimulator(nodes, G, n_threads=threads)
        sim.simulate_many(A, B, list(range(50)), threads)  # warm-up
        t0 = time.perf_counter()
        results = sim.simulate_many(A, B, seeds, threads)
        dt = time.perf_counter() - t0
        checksums[threads] = checksum(results)
        report["persistent_engine"][str(threads)] = {
            "simulations": n_sims,
            "seconds": dt,
            "simulations_per_second": n_sims / dt,
        }
    serial = report["persistent_engine"]["1"]["simulations_per_second"]
    for threads in (2, 4, 8):
        report["persistent_engine"][str(threads)]["speedup_vs_serial"] = (
            report["persistent_engine"][str(threads)]["simulations_per_second"] / serial
        )
    report["persistent_engine"]["checksums_identical"] = len(set(checksums.values())) == 1

    # One-shot API: rebuilds/validates the graph on every call.
    n_one = 300
    t0 = time.perf_counter()
    one_results = []
    for s in range(n_one):
        r = simulation_engine.simulate(nodes, adj, probs, pol, strength, A, B, s)
        one_results.append({"spread_A": r.spread_A, "spread_B": r.spread_B, "rounds": r.rounds})
    dt = time.perf_counter() - t0
    report["one_shot_engine"] = {
        "simulations": n_one,
        "seconds": dt,
        "simulations_per_second": n_one / dt,
        "checksum": checksum(one_results),
    }

    # Python reference (bounded).
    n_py = 30
    t0 = time.perf_counter()
    py_results = [
        simulate_python(nodes, adj, probs, pol, strength, A, B, s) for s in range(n_py)
    ]
    dt = time.perf_counter() - t0
    report["python_reference"] = {
        "simulations": n_py,
        "seconds": dt,
        "simulations_per_second": n_py / dt,
        "spread_A_mean": float(np.mean([r["spread_A"] for r in py_results])),
    }

    # Workload estimate for the full 240-configuration design.
    n_configs_ssa = 4 * 5 * 6
    n_configs_total = 4 * 5 * 6 * 2
    ssa_evals = 12 * (10 + 1)
    opt_mc = 15
    final_mc = 100
    swap_tests = 100
    sims_ssa = n_configs_ssa * ssa_evals * opt_mc
    sims_swap = n_configs_ssa * swap_tests * opt_mc
    sims_final = n_configs_total * final_mc
    sims_baseline = 4 * 5 * (opt_mc + final_mc)
    total_sims = sims_ssa + sims_swap + sims_final + sims_baseline
    serial_tp = serial
    parallel_tp = report["persistent_engine"]["8"]["simulations_per_second"]
    report["estimate"] = {
        "configurations_ssa": n_configs_ssa,
        "configurations_total": n_configs_total,
        "simulations_ssa": sims_ssa,
        "simulations_swap": sims_swap,
        "simulations_final": sims_final,
        "simulations_baseline": sims_baseline,
        "simulations_total": total_sims,
        "serial_seconds_at_measured_throughput": total_sims / serial_tp,
        "parallel8_seconds_at_measured_throughput": total_sims / parallel_tp,
        "note": "Upper-bound estimate; actual SSA calls are smaller batches and add Python overhead.",
    }

    out_json = ROOT / "logs" / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    atomic_write_json(out_json, report)

    lines = [
        "# Benchmark — Signed Competitive Diffusion Engine",
        "",
        f"Generated: {report['timestamp']}",
        "",
        f"- Network: {len(nodes)} nodes, {G.number_of_edges()} directed arcs",
        f"- OpenMP compiled: {report['openmp']}",
        f"- Workload: A={len(A)} seeds, B={len(B)} seeds",
        "",
        "## Persistent engine (graph built once)",
        "",
        "| Threads | Simulations | Seconds | Sims/s | Speedup vs serial |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threads in ("1", "2", "4", "8"):
        row = report["persistent_engine"][threads]
        lines.append(
            f"| {threads} | {row['simulations']} | {row['seconds']:.4f} | "
            f"{row['simulations_per_second']:.0f} | {row.get('speedup_vs_serial', 1.0):.2f}× |"
        )
    lines += [
        "",
        f"Serial and parallel checksums identical: **{report['persistent_engine']['checksums_identical']}**",
        "",
        "## One-shot API vs persistent engine",
        "",
        "| API | Sims/s |",
        "|---|---:|",
        f"| One-shot `simulate(...)` (rebuilds graph) | {report['one_shot_engine']['simulations_per_second']:.0f} |",
        f"| Persistent 1 thread | {report['persistent_engine']['1']['simulations_per_second']:.0f} |",
        f"| Persistent 8 threads | {report['persistent_engine']['8']['simulations_per_second']:.0f} |",
        "",
        "## Python reference",
        "",
        f"- {n_py} simulations in {report['python_reference']['seconds']:.3f}s "
        f"({report['python_reference']['simulations_per_second']:.1f} sims/s).",
        "",
        "## Full-design workload estimate",
        "",
        f"- SSA simulations: {sims_ssa:,}",
        f"- Swap simulations: {sims_swap:,}",
        f"- Final evaluation simulations: {sims_final:,}",
        f"- Baseline simulations: {sims_baseline:,}",
        f"- **Total: {total_sims:,} simulations**",
        f"- Serial estimate: {report['estimate']['serial_seconds_at_measured_throughput']:.0f}s",
        f"- 8-thread estimate: {report['estimate']['parallel8_seconds_at_measured_throughput']:.0f}s",
        "",
        "The estimate is an upper bound based on full-batch throughput; SSA uses small "
        "15-simulation batches and adds Python overhead, so the actual wall time may differ. "
        "It is comfortably inside an overnight window.",
        "",
    ]
    out_md = ROOT / "BENCHMARK.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote", out_md)
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
