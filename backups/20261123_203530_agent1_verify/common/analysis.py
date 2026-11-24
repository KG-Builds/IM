"""Read-only analysis of stored experiment results.

This module never imports the diffusion engine and never launches simulations.
It reads ``experiment_results.csv`` plus the per-configuration ``mc_runs`` CSVs,
validates them, and produces tables and publication-quality figures.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Keep matplotlib's cache inside the workspace (no writes outside the project).
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "analysis" / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # package imports
    from .objectives import VARIANTS
    from .stats import holm_bonferroni, mean_ci, paired_diff
    from .utils import ANALYSIS_DIR, RESULTS_DIR, ensure_dir, utc_now_iso
except ImportError:  # top-level imports
    from objectives import VARIANTS  # type: ignore
    from stats import holm_bonferroni, mean_ci, paired_diff  # type: ignore
    from utils import ANALYSIS_DIR, RESULTS_DIR, ensure_dir, utc_now_iso  # type: ignore

MODE_SSA = "SSA only"
MODE_SWAP = "SSA + Swap refinement"
MODES = (MODE_SSA, MODE_SWAP)
SCENARIOS = ("B1", "B2", "B3", "B4")
K_VALUES = (10, 20, 30, 40, 50)
OBJECTIVE_IDS = tuple(VARIANTS.keys())

REQUIRED_COLUMNS = [
    "run_id", "run_kind", "scenario", "K", "objective_id", "objective_name",
    "mode", "objective_validity", "network_fingerprint", "spread_A", "spread_B",
    "net_advantage", "gain_A", "blocked_B", "ignorant_nodes", "average_rounds",
    "net_advantage_se", "ssa_seed_set", "final_seed_set", "opponent_seed_set",
    "final_evaluation_objective_score", "total_runtime_sec", "mc_runs_file", "status",
]

OBJECTIVE_COLORS = {
    "O1": "#1f77b4", "O2": "#d62728", "O3": "#2ca02c",
    "O4": "#9467bd", "O5": "#ff7f0e", "O6": "#8c564b",
}
SCENARIO_TITLES = {
    "B1": "B1 Random", "B2": "B2 Degree", "B3": "B3 Positive PageRank",
    "B4": "B4 Community",
}


# ---------------------------------------------------------------------------
# Run discovery and loading
# ---------------------------------------------------------------------------
def discover_runs(results_root: Path | str = RESULTS_DIR, include_smoke: bool = False) -> list[dict]:
    """Find run directories with a run_metadata.json or manifest.json."""
    results_root = Path(results_root)
    found: list[dict] = []
    if not results_root.exists():
        return found
    for meta_path in results_root.rglob("run_metadata.json"):
        if ".ipynb_checkpoints" in meta_path.parts:
            continue
        run_dir = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_kind = meta.get("run_kind", "unknown")
        if run_kind == "SMOKE_TEST" and not include_smoke:
            continue
        manifest = {}
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        found.append({
            "run_id": meta.get("run_id", run_dir.name),
            "run_kind": run_kind,
            "status": meta.get("status", manifest.get("status", "unknown")),
            "path": str(run_dir),
            "started_utc": meta.get("started_utc"),
            "finished_utc": meta.get("finished_utc"),
            "expected_configurations": manifest.get("expected_configurations"),
            "completed_configurations": manifest.get("completed_configurations"),
            "mtime": run_dir.stat().st_mtime,
        })
    found.sort(key=lambda r: r["mtime"], reverse=True)
    return found


def select_run(run_id: str | None = None, results_root: Path | str = RESULTS_DIR,
               include_smoke: bool = False) -> Path:
    """Select a run directory explicitly, or the latest non-smoke run."""
    results_root = Path(results_root)
    if run_id:
        candidate = Path(run_id)
        if candidate.is_dir():
            return candidate.resolve()
        for info in discover_runs(results_root, include_smoke=True):
            if info["run_id"] == run_id:
                return Path(info["path"])
        # Also accept a relative subpath such as smoke/<run_id>.
        candidate = results_root / run_id
        if candidate.is_dir():
            return candidate.resolve()
        raise FileNotFoundError(f"No run {run_id!r} under {results_root}")
    runs = discover_runs(results_root, include_smoke=include_smoke)
    if not runs:
        raise FileNotFoundError(f"No completed runs found under {results_root}")
    return Path(runs[0]["path"])


def load_run_metadata(run_dir: Path | str) -> dict:
    run_dir = Path(run_dir)
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(run_dir: Path | str) -> dict:
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: Path | str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    frames = []
    for path in sorted(run_dir.glob("*/experiment_results.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No scenario experiment_results.csv found under {run_dir}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _safe_mc_path(run_dir: Path, rel_path: str) -> Path:
    path = (run_dir / str(rel_path)).resolve()
    root = run_dir.resolve()
    if not str(path).startswith(str(root) + os.sep):
        raise ValueError(f"mc_runs path escapes the run directory: {rel_path}")
    return path


def load_mc_runs(run_dir: Path | str, df: pd.DataFrame) -> tuple[dict, list[str]]:
    """Load every referenced per-seed MC file, keyed by configuration."""
    run_dir = Path(run_dir)
    cache: dict = {}
    missing: list[str] = []
    if "mc_runs_file" not in df.columns:
        return cache, ["mc_runs_file column missing"]
    for row in df.itertuples(index=False):
        rel = getattr(row, "mc_runs_file")
        if not isinstance(rel, str) or not rel:
            missing.append(f"{row.scenario}/K{row.K}/{row.objective_id}/{row.mode}: empty path")
            continue
        key = (row.scenario, int(row.K), row.objective_id, row.mode)
        if key in cache:
            continue
        try:
            path = _safe_mc_path(run_dir, rel)
            if not path.exists():
                missing.append(str(rel))
                continue
            cache[key] = pd.read_csv(path)
        except Exception as exc:
            missing.append(f"{rel}: {exc}")
    return cache, missing


def validate_run(run_dir: Path | str, df: pd.DataFrame | None = None) -> dict:
    run_dir = Path(run_dir)
    if df is None:
        df = load_run(run_dir)
    issues: list[str] = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"missing required columns: {missing_cols}")
    if df.empty:
        issues.append("result table is empty")
    key_cols = ["scenario", "K", "objective_id", "mode"]
    if all(c in df.columns for c in key_cols):
        dup = int(df.duplicated(key_cols).sum())
        if dup:
            issues.append(f"{dup} duplicate configuration rows")
    for col in ("network_fingerprint", "run_id", "run_kind"):
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            issues.append(f"multiple {col} values in one run")
    if all(c in df.columns for c in ("spread_A", "spread_B", "net_advantage")):
        err = (df["spread_A"] - df["spread_B"] - df["net_advantage"]).abs()
        if (err > 1e-6).any():
            issues.append("net_advantage != spread_A - spread_B in some rows")
    if "blocked_B" in df.columns and (df["blocked_B"] < -1e-9).any():
        issues.append("negative blocked_B values")
    if "status" in df.columns and (df["status"] != "completed").any():
        issues.append("some rows are not marked completed")

    meta = load_run_metadata(run_dir)
    manifest = load_manifest(run_dir)
    expected = None
    cfg = meta.get("config", {})
    if cfg:
        expected = {
            (sid, int(k), oid, mode)
            for sid in cfg.get("scenarios", [])
            for k in cfg.get("k_values", [])
            for oid in cfg.get("objective_ids", [])
            for mode in cfg.get("modes", [])
        }
    missing_configs: list[dict] = []
    if expected is not None and all(c in df.columns for c in key_cols):
        got = {(r.scenario, int(r.K), r.objective_id, r.mode) for r in df.itertuples(index=False)}
        missing_keys = sorted(expected - got)
        missing_configs = [
            {"scenario": s, "K": k, "objective_id": o, "mode": m}
            for (s, k, o, m) in missing_keys
        ]
        if missing_keys:
            issues.append(f"{len(missing_keys)} expected configurations missing")
        extra = got - expected
        if extra:
            issues.append(f"{len(extra)} unexpected configurations present")

    mc_cache, mc_missing = load_mc_runs(run_dir, df)
    if mc_missing:
        issues.append(f"{len(mc_missing)} missing/unreadable mc_runs files")
    seed_alignment_issues = 0
    for (sid, k, oid), group in df.groupby(["scenario", "K", "objective_id"]):
        if MODE_SSA in set(group["mode"]) and MODE_SWAP in set(group["mode"]):
            a = mc_cache.get((sid, int(k), oid, MODE_SSA))
            b = mc_cache.get((sid, int(k), oid, MODE_SWAP))
            if a is not None and b is not None and set(a["seed"]) != set(b["seed"]):
                seed_alignment_issues += 1
    if seed_alignment_issues:
        issues.append(f"{seed_alignment_issues} configurations have unpaired SSA/Swap MC seeds")

    network_fingerprint = None
    if "network_fingerprint" in df.columns and len(df):
        network_fingerprint = str(df["network_fingerprint"].iloc[0])
    meta_fp = meta.get("network", {}).get("file_sha256")
    if meta_fp and network_fingerprint and meta_fp != network_fingerprint:
        issues.append("result network fingerprint differs from run metadata")

    report = {
        "ok": not issues,
        "issues": issues,
        "n_rows": int(len(df)),
        "run_id": str(df["run_id"].iloc[0]) if "run_id" in df.columns and len(df) else None,
        "run_kind": str(df["run_kind"].iloc[0]) if "run_kind" in df.columns and len(df) else None,
        "scenarios": sorted(df["scenario"].dropna().unique().tolist()) if "scenario" in df.columns else [],
        "k_values": sorted(int(x) for x in df["K"].dropna().unique()) if "K" in df.columns else [],
        "objective_ids": sorted(df["objective_id"].dropna().unique().tolist()) if "objective_id" in df.columns else [],
        "modes": sorted(df["mode"].dropna().unique().tolist()) if "mode" in df.columns else [],
        "network_fingerprint": network_fingerprint,
        "missing_configurations": missing_configs,
        "mc_files_loaded": len(mc_cache),
        "mc_files_missing": mc_missing,
        "manifest_status": manifest.get("status"),
        "expected_configurations": manifest.get("expected_configurations"),
        "completed_configurations": manifest.get("completed_configurations"),
    }
    return report


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _group_summary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        vals = group["net_advantage"].astype(float).to_numpy()
        row.update({
            "n_configs": int(len(group)),
            "net_advantage_mean": float(vals.mean()),
            "net_advantage_std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "net_advantage_se": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
            "net_advantage_min": float(vals.min()),
            "net_advantage_max": float(vals.max()),
            "spread_A_mean": float(group["spread_A"].mean()),
            "spread_B_mean": float(group["spread_B"].mean()),
            "blocked_B_mean": float(group["blocked_B"].mean()),
            "ignorant_mean": float(group["ignorant_nodes"].mean()) if "ignorant_nodes" in group else float("nan"),
            "rounds_mean": float(group["average_rounds"].mean()) if "average_rounds" in group else float("nan"),
            "total_runtime_mean": float(group["total_runtime_sec"].mean()) if "total_runtime_sec" in group else float("nan"),
            "ssa_runtime_mean": float(group["ssa_runtime_sec"].mean()) if "ssa_runtime_sec" in group else float("nan"),
            "swap_runtime_mean": float(group["swap_runtime_sec"].mean()) if "swap_runtime_sec" in group else float("nan"),
            "final_eval_runtime_mean": float(group["final_evaluation_runtime_sec"].mean()) if "final_evaluation_runtime_sec" in group else float("nan"),
            "net_advantage_se_mean": float(group["net_advantage_se"].mean()) if "net_advantage_se" in group else float("nan"),
        })
        if "objective_id" in keys and "objective_validity" in group:
            row["objective_validity"] = group["objective_validity"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True) if rows else pd.DataFrame()


def _runtime_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, mode), group in df.groupby(["scenario", "mode"]):
        rows.append({
            "scenario": sid,
            "mode": mode,
            "n_configs": int(len(group)),
            "total_runtime_mean": float(group["total_runtime_sec"].mean()),
            "total_runtime_std": float(group["total_runtime_sec"].std(ddof=1)) if len(group) > 1 else float("nan"),
            "ssa_runtime_mean": float(group["ssa_runtime_sec"].mean()),
            "swap_runtime_mean": float(group["swap_runtime_sec"].mean()),
            "final_eval_runtime_mean": float(group["final_evaluation_runtime_sec"].mean()),
            "simulations_mean": float(group["simulations_for_config"].mean()) if "simulations_for_config" in group else float("nan"),
            "simulation_seconds_mean": float(group["simulation_seconds_for_config"].mean()) if "simulation_seconds_for_config" in group else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["scenario", "mode"]).reset_index(drop=True) if rows else pd.DataFrame()


def _variability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (oid, name, mode), group in df.groupby(["objective_id", "objective_name", "mode"]):
        ci_width = (group["net_advantage_ci_high"] - group["net_advantage_ci_low"]) if (
            "net_advantage_ci_high" in group and "net_advantage_ci_low" in group
        ) else pd.Series([np.nan])
        rows.append({
            "objective_id": oid,
            "objective_name": name,
            "mode": mode,
            "n_configs": int(len(group)),
            "net_advantage_mean": float(group["net_advantage"].mean()),
            "net_advantage_std": float(group["net_advantage"].std(ddof=1)) if len(group) > 1 else float("nan"),
            "mc_se_mean": float(group["net_advantage_se"].mean()) if "net_advantage_se" in group else float("nan"),
            "mc_ci_width_mean": float(ci_width.mean()),
        })
    return pd.DataFrame(rows).sort_values(["objective_id", "mode"]).reset_index(drop=True) if rows else pd.DataFrame()


def _mode_comparison(df: pd.DataFrame, mc_cache: dict, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for (sid, k, oid), group in df.groupby(["scenario", "K", "objective_id"]):
        modes = set(group["mode"])
        if MODE_SSA not in modes or MODE_SWAP not in modes:
            continue
        ssa = mc_cache.get((sid, int(k), oid, MODE_SSA))
        swap = mc_cache.get((sid, int(k), oid, MODE_SWAP))
        if ssa is None or swap is None:
            continue
        merged = pd.merge(
            ssa[["seed", "net_advantage_run"]],
            swap[["seed", "net_advantage_run"]],
            on="seed", suffixes=("_ssa", "_swap"),
        )
        if merged.empty:
            continue
        stats = paired_diff(merged["net_advantage_run_swap"], merged["net_advantage_run_ssa"],
                            n_boot=n_boot, seed=1234)
        row = {
            "scenario": sid, "K": int(k), "objective_id": oid,
            "objective_name": group["objective_name"].iloc[0],
            "objective_validity": group["objective_validity"].iloc[0],
            "n_pairs": int(len(merged)),
            "swap_minus_ssa": float(stats["mean_diff"]),
            "ci_low": float(stats["ci_low"]),
            "ci_high": float(stats["ci_high"]),
            "p_value": float(stats["p_value"]),
            "fraction_swap_better": float(stats["fraction_a_better"]),
            "cohens_dz": float(stats["cohens_dz"]),
            "ssa_net_advantage": float(merged["net_advantage_run_ssa"].mean()),
            "swap_net_advantage": float(merged["net_advantage_run_swap"].mean()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["scenario", "K", "objective_id"]).reset_index(drop=True)
    return out


def _objective_vs_reference(df: pd.DataFrame, mc_cache: dict, reference: str = "O2",
                            n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for (sid, k, mode), group in df.groupby(["scenario", "K", "mode"]):
        ref_rows = group[group["objective_id"] == reference]
        if ref_rows.empty:
            continue
        ref_mc = mc_cache.get((sid, int(k), reference, mode))
        if ref_mc is None:
            continue
        p_values = {}
        local = []
        for _, row in group[group["objective_id"] != reference].iterrows():
            oid = row["objective_id"]
            other_mc = mc_cache.get((sid, int(k), oid, mode))
            if other_mc is None:
                continue
            merged = pd.merge(
                other_mc[["seed", "net_advantage_run"]],
                ref_mc[["seed", "net_advantage_run"]],
                on="seed", suffixes=("_other", "_ref"),
            )
            if merged.empty:
                continue
            stats = paired_diff(merged["net_advantage_run_other"], merged["net_advantage_run_ref"],
                                n_boot=n_boot, seed=4321)
            local.append({
                "scenario": sid, "K": int(k), "mode": mode,
                "objective_id": oid,
                "objective_name": row["objective_name"],
                "objective_validity": row["objective_validity"],
                "reference": reference,
                "n_pairs": int(len(merged)),
                "mean_diff_vs_reference": float(stats["mean_diff"]),
                "ci_low": float(stats["ci_low"]),
                "ci_high": float(stats["ci_high"]),
                "p_value": float(stats["p_value"]),
                "fraction_better": float(stats["fraction_a_better"]),
            })
            p_values[oid] = float(stats["p_value"])
        adjusted = holm_bonferroni(p_values)
        for item in local:
            item.update(adjusted.get(item["objective_id"], {}))
            rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["scenario", "K", "mode", "objective_id"]).reset_index(drop=True)
    return out


def build_tables(df: pd.DataFrame, run_dir: Path | str, n_boot: int = 2000) -> dict:
    run_dir = Path(run_dir)
    mc_cache, _missing = load_mc_runs(run_dir, df)
    tables = {
        "overall": _group_summary(df, ["objective_id", "objective_name", "objective_validity", "mode"]),
        "per_scenario": _group_summary(df, ["scenario", "objective_id", "objective_name", "mode"]),
        "per_k": _group_summary(df, ["K", "objective_id", "objective_name", "mode"]),
        "mode_comparison": _mode_comparison(df, mc_cache, n_boot=n_boot),
        "objective_vs_reference": _objective_vs_reference(df, mc_cache, n_boot=n_boot),
        "objective_ranking": _group_summary(df, ["objective_id", "objective_name", "objective_validity", "mode"]),
        "runtime": _runtime_table(df),
        "variability": _variability_table(df),
    }

    # Robustness: objective x scenario mean net advantage, plus across-scenario summary.
    robustness = _group_summary(df, ["objective_id", "objective_name", "mode", "scenario"])
    if not robustness.empty:
        pivot = robustness.pivot_table(
            index=["objective_id", "objective_name", "mode"],
            columns="scenario", values="net_advantage_mean",
        ).reset_index()
        scenario_cols = [c for c in pivot.columns if c in SCENARIOS]
        if scenario_cols:
            pivot["across_scenario_mean"] = pivot[scenario_cols].mean(axis=1)
            pivot["across_scenario_std"] = pivot[scenario_cols].std(axis=1, ddof=1)
            pivot["across_scenario_range"] = pivot[scenario_cols].max(axis=1) - pivot[scenario_cols].min(axis=1)
        tables["robustness"] = pivot
    else:
        tables["robustness"] = pd.DataFrame()

    # Best configuration per scenario/K (ranked by final Net Advantage).
    best_rows = []
    for (sid, k), group in df.groupby(["scenario", "K"]):
        best = group.loc[group["net_advantage"].astype(float).idxmax()]
        best_rows.append({
            "scenario": sid, "K": int(k), "objective_id": best["objective_id"],
            "objective_name": best["objective_name"], "mode": best["mode"],
            "objective_validity": best["objective_validity"],
            "net_advantage": float(best["net_advantage"]),
            "spread_A": float(best["spread_A"]), "spread_B": float(best["spread_B"]),
            "blocked_B": float(best["blocked_B"]),
        })
    tables["best_config"] = pd.DataFrame(best_rows).sort_values(["scenario", "K"]).reset_index(drop=True) if best_rows else pd.DataFrame()

    # Objective ranking by final Net Advantage (not optimizer fitness).
    ranking = tables["objective_ranking"].copy()
    if not ranking.empty:
        ranking["rank_in_mode"] = ranking.groupby("mode")["net_advantage_mean"].rank(
            ascending=False, method="min"
        ).astype(int)
        ranking = ranking.sort_values(["mode", "rank_in_mode", "objective_id"]).reset_index(drop=True)
        tables["objective_ranking"] = ranking

    return tables


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _objective_label(oid: str) -> str:
    spec = VARIANTS.get(oid, {})
    return f"{oid} {spec.get('name', '')}".strip()


def build_figures(df: pd.DataFrame, tables: dict, out_dir: Path | str) -> list[Path]:
    out_dir = ensure_dir(out_dir)
    paths: list[Path] = []

    # 1. Net Advantage vs K by objective, per scenario and mode.
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for ax, sid in zip(axes.ravel(), SCENARIOS):
            s = sub[sub["scenario"] == sid]
            if s.empty:
                ax.set_visible(False)
                continue
            for oid in OBJECTIVE_IDS:
                o = s[s["objective_id"] == oid].sort_values("K")
                if o.empty:
                    continue
                ax.errorbar(o["K"], o["net_advantage"], yerr=o["net_advantage_se"],
                            marker="o", capsize=3, label=_objective_label(oid),
                            color=OBJECTIVE_COLORS.get(oid), linewidth=1.8)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_title(SCENARIO_TITLES.get(sid, sid))
            ax.set_xlabel("K")
            ax.set_ylabel("Net Advantage (Spread_A - Spread_B)")
            ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(f"Net Advantage vs K — {mode}", y=1.02)
        paths.append(_save(fig, out_dir / f"net_advantage_vs_K_{_mode_slug(mode)}.png"))

    # 2. Swap vs SSA paired differences.
    mc = tables.get("mode_comparison", pd.DataFrame())
    if not mc.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        grouped = mc.groupby("objective_id")
        xs, means, errs, colors = [], [], [], []
        for oid in OBJECTIVE_IDS:
            g = grouped.get_group(oid) if oid in grouped.groups else None
            if g is None or g.empty:
                continue
            xs.append(_objective_label(oid))
            means.append(float(g["swap_minus_ssa"].mean()))
            errs.append(float(g["swap_minus_ssa"].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0)
            colors.append(OBJECTIVE_COLORS.get(oid))
        ax.bar(range(len(xs)), means, yerr=errs, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=20, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Swap - SSA Net Advantage (mean across configs)")
        ax.set_title("Paired SSA+Swap vs SSA-only difference (per-seed CRN, descriptive CI)")
        ax.grid(axis="y", alpha=0.25)
        paths.append(_save(fig, out_dir / "swap_vs_ssa_paired_differences.png"))

    # 3. Spread_A vs Spread_B trade-off.
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["spread_A"], df["spread_B"], c=df["net_advantage"],
                         cmap="coolwarm", s=28, alpha=0.8, edgecolor="black", linewidth=0.2)
    ax.set_xlabel("Spread_A")
    ax.set_ylabel("Spread_B")
    ax.set_title("Spread_A vs Spread_B trade-off (color = Net Advantage)")
    fig.colorbar(scatter, ax=ax, label="Net Advantage")
    ax.grid(alpha=0.25)
    paths.append(_save(fig, out_dir / "spread_A_vs_spread_B.png"))

    # 4. Blocked_B (provisional) vs K.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, sid in zip(axes.ravel(), SCENARIOS):
        s = df[df["scenario"] == sid]
        if s.empty:
            ax.set_visible(False)
            continue
        for oid in OBJECTIVE_IDS:
            o = s[s["objective_id"] == oid].groupby("K")["blocked_B"].mean().reset_index()
            if o.empty:
                continue
            ax.plot(o["K"], o["blocked_B"], marker="o", label=_objective_label(oid),
                    color=OBJECTIVE_COLORS.get(oid), linewidth=1.6)
        ax.set_title(SCENARIO_TITLES.get(sid, sid))
        ax.set_xlabel("K")
        ax.set_ylabel("Blocked_B (mean)")
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Blocked_B vs K — PROVISIONAL protocol (not a validated six-objective result)", y=1.02)
    fig.text(0.5, 0.5, "PROVISIONAL", fontsize=40, color="red", alpha=0.12,
             ha="center", va="center", transform=fig.transFigure)
    paths.append(_save(fig, out_dir / "blocked_B_provisional.png"))

    # 5. Runtime summary.
    rt = tables.get("runtime", pd.DataFrame())
    if not rt.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{r.scenario}\n{r.mode}" for r in rt.itertuples(index=False)]
        x = np.arange(len(rt))
        bottom = np.zeros(len(rt))
        for col, label, color in (
            ("ssa_runtime_mean", "SSA", "#4c72b0"),
            ("swap_runtime_mean", "Swap", "#dd8452"),
            ("final_eval_runtime_mean", "Final evaluation", "#55a868"),
        ):
            vals = rt[col].fillna(0).to_numpy()
            ax.bar(x, vals, bottom=bottom, label=label, color=color)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Mean runtime (seconds per configuration)")
        ax.set_title("Runtime breakdown by scenario and mode")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        paths.append(_save(fig, out_dir / "runtime_breakdown.png"))

    # 6. Objective ranking by final Net Advantage.
    rank = tables.get("objective_ranking", pd.DataFrame())
    if not rank.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
        for ax, mode in zip(axes, MODES):
            r = rank[rank["mode"] == mode].sort_values("net_advantage_mean")
            if r.empty:
                ax.set_visible(False)
                continue
            colors = ["#2ca02c" if v == "validated" else "#d62728" for v in r["objective_validity"]]
            ax.barh([_objective_label(o) for o in r["objective_id"]], r["net_advantage_mean"],
                    xerr=r["net_advantage_se"], color=colors, capsize=4, alpha=0.85)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(mode)
            ax.set_xlabel("Mean final Net Advantage")
            ax.grid(axis="x", alpha=0.25)
        fig.suptitle("Objective ranking by final Net Advantage (green=validated O2; red=provisional Blocked_B)")
        paths.append(_save(fig, out_dir / "objective_ranking_net_advantage.png"))

    # 7. Robustness across B1-B4.
    rob = tables.get("robustness", pd.DataFrame())
    if not rob.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, mode in zip(axes, MODES):
            r = rob[rob["mode"] == mode]
            if r.empty:
                ax.set_visible(False)
                continue
            x = np.arange(len(OBJECTIVE_IDS))
            width = 0.2
            for i, sid in enumerate(SCENARIOS):
                if sid not in r.columns:
                    continue
                vals = []
                for oid in OBJECTIVE_IDS:
                    row = r[(r["objective_id"] == oid)]
                    vals.append(float(row[sid].iloc[0]) if not row.empty else np.nan)
                ax.bar(x + (i - 1.5) * width, vals, width=width, label=SCENARIO_TITLES[sid])
            ax.set_xticks(x)
            ax.set_xticklabels(OBJECTIVE_IDS)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(mode)
            ax.set_ylabel("Mean Net Advantage")
            ax.grid(axis="y", alpha=0.25)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Robustness across opponent scenarios")
        paths.append(_save(fig, out_dir / "robustness_across_scenarios.png"))

    # 8. Variability: configuration-level net advantage by objective and mode.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mode in zip(axes, MODES):
        s = df[df["mode"] == mode]
        data = [s[s["objective_id"] == oid]["net_advantage"].to_numpy() for oid in OBJECTIVE_IDS]
        if not any(len(d) for d in data):
            ax.set_visible(False)
            continue
        ax.boxplot(data, labels=OBJECTIVE_IDS, showmeans=True)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(mode)
        ax.set_ylabel("Configuration-level Net Advantage")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Variability across configurations (not optimizer-run variance)")
    paths.append(_save(fig, out_dir / "variability_configurations.png"))

    # 9. Heatmap: objective x scenario mean Net Advantage.
    if not rob.empty:
        for mode in MODES:
            r = rob[rob["mode"] == mode]
            if r.empty:
                continue
            mat = []
            for oid in OBJECTIVE_IDS:
                row = r[r["objective_id"] == oid]
                mat.append([float(row[sid].iloc[0]) if sid in row and not row.empty else np.nan
                            for sid in SCENARIOS])
            fig, ax = plt.subplots(figsize=(7, 4.5))
            im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
            ax.set_xticks(range(len(SCENARIOS)))
            ax.set_xticklabels(SCENARIOS)
            ax.set_yticks(range(len(OBJECTIVE_IDS)))
            ax.set_yticklabels([_objective_label(o) for o in OBJECTIVE_IDS], fontsize=8)
            for i in range(len(OBJECTIVE_IDS)):
                for j in range(len(SCENARIOS)):
                    if not np.isnan(mat[i][j]):
                        ax.text(j, i, f"{mat[i][j]:.1f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"Mean Net Advantage — {mode}")
            fig.colorbar(im, ax=ax, label="Net Advantage")
            paths.append(_save(fig, out_dir / f"objective_scenario_heatmap_{_mode_slug(mode)}.png"))

    return paths


def _mode_slug(mode: str) -> str:
    return "ssa_only" if mode == MODE_SSA else "ssa_swap"


# ---------------------------------------------------------------------------
# Report and orchestration
# ---------------------------------------------------------------------------
def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_report(tables: dict, report: dict, out_dir: Path | str, run_dir: Path) -> Path:
    out_dir = ensure_dir(out_dir)
    meta = load_run_metadata(run_dir)
    lines = [
        f"# Analysis report — {report.get('run_id')}",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Run kind: `{report.get('run_kind')}`",
        f"- Generated: {utc_now_iso()}",
        f"- Rows: {report.get('n_rows')}",
        f"- Network fingerprint: `{report.get('network_fingerprint')}`",
        f"- Validation: {'PASS' if report.get('ok') else 'ISSUES'}",
        "",
        "## Validation issues",
        "",
    ]
    if report.get("issues"):
        lines += [f"- {issue}" for issue in report["issues"]]
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Data provenance",
        "",
        f"- Config: `{json.dumps(meta.get('config', {}), sort_keys=True)}`",
        f"- Evaluation protocol: `{meta.get('evaluation_protocol')}`",
        "",
        "## Caveats",
        "",
        "- Objective O2 (Competition-aware) is the only fully validated objective.",
        "- O1/O3/O4/O5/O6 depend on the provisional paired B-only Blocked_B protocol.",
        "- Net Advantage is the ranking metric; optimizer fitness is not used for ranking.",
        "- Monte Carlo replicates are common random numbers, not independent optimizer runs.",
        "",
        "## Overall summary",
        "",
        _to_markdown(tables.get("overall", pd.DataFrame())),
        "",
        "## Objective ranking (final Net Advantage)",
        "",
        _to_markdown(tables.get("objective_ranking", pd.DataFrame())),
        "",
        "## Swap vs SSA paired comparison",
        "",
        _to_markdown(tables.get("mode_comparison", pd.DataFrame()).head(40)),
        "",
        "## Best configuration per scenario/K",
        "",
        _to_markdown(tables.get("best_config", pd.DataFrame())),
        "",
        "## Robustness across scenarios",
        "",
        _to_markdown(tables.get("robustness", pd.DataFrame())),
        "",
        "## Runtime",
        "",
        _to_markdown(tables.get("runtime", pd.DataFrame())),
    ]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_analysis(run_id: str | None = None, run_dir: Path | str | None = None,
                 results_root: Path | str = RESULTS_DIR, include_smoke: bool = False,
                 make_figures: bool = True, n_boot: int = 2000) -> dict:
    """Load, validate, tabulate and plot one run. Read-only w.r.t. experiment data."""
    if run_dir is None:
        run_dir = select_run(run_id, results_root=results_root, include_smoke=include_smoke)
    run_dir = Path(run_dir)
    df = load_run(run_dir)
    report = validate_run(run_dir, df)
    tables = build_tables(df, run_dir, n_boot=n_boot)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_label = report.get("run_id") or run_dir.name
    table_dir = ensure_dir(ANALYSIS_DIR / "tables" / f"{run_label}_{stamp}")
    figure_dir = ensure_dir(ANALYSIS_DIR / "figures" / f"{run_label}_{stamp}")

    table_paths = {}
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            path = table_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            table_paths[name] = str(path)
    report_path = write_report(tables, report, table_dir, run_dir)
    figure_paths = build_figures(df, tables, figure_dir) if make_figures else []
    return {
        "run_dir": str(run_dir),
        "run_id": run_label,
        "report": report,
        "tables": tables,
        "table_paths": table_paths,
        "report_path": str(report_path),
        "figure_dir": str(figure_dir),
        "figure_paths": [str(p) for p in figure_paths],
    }
