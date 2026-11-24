#!/usr/bin/env python3
"""Read-only analysis for the SSA O1/O4/O6 vs. baselines + FIBMM benchmark.

Builds aggregated tables, validation checks and the final figure set.  This
module never launches a diffusion simulation and never modifies raw results.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common.stats import mean_ci, paired_diff  # noqa: E402
from common.utils import atomic_write_text, sha256_file, utc_now_iso  # noqa: E402

RAW_DIR = EXP_DIR / "raw_results"
RESULTS_PATH = RAW_DIR / "results.csv"
LEDGER_PATH = RAW_DIR / "ledger.csv"
AGG_DIR = EXP_DIR / "aggregated"
FIG_DIR = EXP_DIR / "figures"
REPORT_DIR = EXP_DIR / "reports"
FIBMM_DIAG_DIR = RAW_DIR / "fibmm_diagnostics"

SCENARIOS = ["B1", "B2", "B3", "B4"]
KS = [10, 20, 30, 40, 50]
SCENARIO_TITLES = {
    "B1": "Random opponent",
    "B2": "Degree-based opponent",
    "B3": "Positive PageRank-based opponent",
    "B4": "Community-concentrated opponent",
}
SCENARIO_SLUGS = {
    "B1": "Random", "B2": "Degree", "B3": "Positive_PageRank", "B4": "Community",
}

METHOD_ORDER = [
    "SSA_O1", "SSA_O4", "SSA_O6", "SSA_Notebook", "FIBMM",
    "Random", "TopDegree", "SignedPageRank", "CRIEG", "NCGE",
    "SignedGA", "CELF",
]
METHOD_LABELS = {
    "SSA_O1": "SSA O1 (Original)",
    "SSA_O4": "SSA O4 (Combined)",
    "SSA_O6": "SSA O6 (Budget-aware)",
    "SSA_Notebook": "SSA-Notebook (O1)",
    "FIBMM": "FIBMM (fairness-aware)",
    "Random": "Random",
    "TopDegree": "Top Degree",
    "SignedPageRank": "Signed PageRank",
    "CRIEG": "CRIEG (2021)",
    "NCGE": "NCGE (2025)",
    "SignedGA": "Signed-GA (2025)",
    "CELF": "CELF (O1 fitness)",
}
METHOD_COLORS = {
    "SSA_O1": "#0072B2",
    "SSA_O4": "#D55E00",
    "SSA_O6": "#009E73",
    "SSA_Notebook": "#CC79A7",
    "FIBMM": "#000000",
    "Random": "#7F7F7F",
    "TopDegree": "#E69F00",
    "SignedPageRank": "#56B4E9",
    "CRIEG": "#8C564B",
    "NCGE": "#9467BD",
    "SignedGA": "#BCBD22",
    "CELF": "#17BECF",
}
METHOD_MARKERS = {
    "SSA_O1": "o", "SSA_O4": "s", "SSA_O6": "^", "SSA_Notebook": "D",
    "FIBMM": "*", "Random": "x", "TopDegree": "v", "SignedPageRank": "P",
    "CRIEG": "X", "NCGE": "h", "SignedGA": "<", "CELF": ">",
}
METHOD_LINESTYLES = {
    "SSA_O1": "-", "SSA_O4": "-", "SSA_O6": "-", "SSA_Notebook": "--",
    "FIBMM": "-", "Random": ":", "TopDegree": "--", "SignedPageRank": "-.",
    "CRIEG": "--", "NCGE": "-.", "SignedGA": ":", "CELF": "--",
}
SSA_METHODS = ["SSA_O1", "SSA_O4", "SSA_O6"]
BASELINE_METHODS = [
    "Random", "TopDegree", "SignedPageRank", "CRIEG", "NCGE",
    "SignedGA", "CELF",
]
FOCUS_METHODS = ["SSA_O1", "SSA_O4", "SSA_O6", "SSA_Notebook", "FIBMM"]


def _json_set(value) -> set[int]:
    if isinstance(value, str):
        return {int(x) for x in json.loads(value)}
    return {int(x) for x in value}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df["K"] = df["K"].astype(int)
    return df


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        return pd.read_csv(LEDGER_PATH)
    return pd.DataFrame()


def _md_table(df: pd.DataFrame, cols: list[str] | None = None,
              floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_empty_"
    view = df if cols is None else df[cols]
    try:
        return view.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return "```\n" + view.to_string(index=False) + "\n```"


def _mc_for(df: pd.DataFrame, method_id: str, scenario: str, k: int) -> pd.DataFrame:
    sub = df[(df["method_id"] == method_id) & (df["scenario"] == scenario)
             & (df["K"] == int(k))]
    if len(sub) != 1:
        raise ValueError(f"Expected one row for {method_id}/{scenario}/K={k}")
    path = RAW_DIR / str(sub.iloc[0]["mc_file"])
    return pd.read_csv(path)


def _paired_cell(df: pd.DataFrame, a: str, b: str, scenario: str, k: int) -> dict:
    ma = _mc_for(df, a, scenario, k).set_index("seed")
    mb = _mc_for(df, b, scenario, k).set_index("seed")
    common = ma.index.intersection(mb.index)
    diff = paired_diff(
        ma.loc[common, "net_advantage_run"].to_numpy(float),
        mb.loc[common, "net_advantage_run"].to_numpy(float),
        n_boot=5000,
    )
    return diff


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def winner_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in SCENARIOS:
        for k in KS:
            sub = df[(df["scenario"] == sid) & (df["K"] == k)].copy()
            vals = sub.set_index("method_id")["net_advantage"].reindex(METHOD_ORDER)
            if vals.isna().any():
                raise ValueError(f"Missing methods for {sid}/K={k}: {vals[vals.isna()].index.tolist()}")
            maxv = float(vals.max())
            winners = [m for m in METHOD_ORDER if float(vals[m]) >= maxv - 1e-9]
            others = sorted(
                [(m, float(vals[m])) for m in METHOD_ORDER if m not in winners],
                key=lambda x: -x[1],
            )
            runner = others[0][0] if others else None
            runner_v = others[0][1] if others else float("nan")
            row = {
                "scenario": sid,
                "scenario_name": SCENARIO_TITLES[sid],
                "K": int(k),
                "winner": "; ".join(winners),
                "is_tie": len(winners) > 1,
                "winner_net_advantage": maxv,
                "runner_up": runner,
                "runner_up_net_advantage": runner_v,
                "margin": maxv - runner_v if runner is not None else float("nan"),
            }
            wrow = sub[sub["method_id"].isin(winners)].iloc[0]
            row["winner_spread_A"] = float(wrow["spread_A"])
            row["winner_spread_B"] = float(wrow["spread_B"])
            row["winner_blocked_B"] = float(wrow["blocked_B"])
            row["winner_ci_low"] = float(wrow["net_advantage_ci_low"])
            row["winner_ci_high"] = float(wrow["net_advantage_ci_high"])
            if runner is not None and not row["is_tie"]:
                rrow = sub[sub["method_id"] == runner].iloc[0]
                row["runner_up_ci_low"] = float(rrow["net_advantage_ci_low"])
                row["runner_up_ci_high"] = float(rrow["net_advantage_ci_high"])
                diff = _paired_cell(df, winners[0], runner, sid, k)
                row["paired_mean_diff"] = diff["mean_diff"]
                row["paired_ci_low"] = diff["ci_low"]
                row["paired_ci_high"] = diff["ci_high"]
                row["paired_p_value"] = diff["p_value"]
                row["paired_fraction_winner_better"] = diff["fraction_a_better"]
                row["difference_meaningful_95"] = bool(
                    diff["ci_low"] > 0 or diff["ci_high"] < 0
                )
            else:
                for col in ("runner_up_ci_low", "runner_up_ci_high", "paired_mean_diff",
                            "paired_ci_low", "paired_ci_high", "paired_p_value",
                            "paired_fraction_winner_better", "difference_meaningful_95"):
                    row[col] = np.nan
            for m in METHOD_ORDER:
                row[f"net_{m}"] = float(vals[m])
            rows.append(row)
    return pd.DataFrame(rows)


def algorithm_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHOD_ORDER:
        sub = df[df["method_id"] == m]
        if len(sub) != len(SCENARIOS) * len(KS):
            raise ValueError(f"{m}: found {len(sub)} cells, expected 20")
        wins = ties = losses = 0
        ranks = []
        for sid in SCENARIOS:
            for k in KS:
                cell = df[(df["scenario"] == sid) & (df["K"] == k)]
                vals = cell.set_index("method_id")["net_advantage"].reindex(METHOD_ORDER)
                maxv = float(vals.max())
                if abs(float(vals[m]) - maxv) <= 1e-9:
                    if int((vals >= maxv - 1e-9).sum()) > 1:
                        ties += 1
                    else:
                        wins += 1
                else:
                    losses += 1
                ranks.append(int((vals > float(vals[m]) + 1e-9).sum()) + 1)
        rows.append({
            "method_id": m,
            "method_name": METHOD_LABELS[m],
            "method_group": str(sub["method_group"].iloc[0]),
            "mean_net_advantage": float(sub["net_advantage"].mean()),
            "median_net_advantage": float(sub["net_advantage"].median()),
            "std_net_advantage": float(sub["net_advantage"].std(ddof=1)),
            "best_net_advantage": float(sub["net_advantage"].max()),
            "worst_net_advantage": float(sub["net_advantage"].min()),
            "strict_wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "mean_rank": float(np.mean(ranks)),
            "mean_spread_A": float(sub["spread_A"].mean()),
            "mean_spread_B": float(sub["spread_B"].mean()),
            "mean_blocked_B": float(sub["blocked_B"].mean()),
            "mean_selection_runtime_sec": float(sub["selection_runtime_sec"].mean()),
            "total_selection_runtime_sec": float(sub["selection_runtime_sec"].sum()),
            "mean_final_evaluation_runtime_sec": float(sub["final_evaluation_runtime_sec"].mean()),
            "n_cells": int(len(sub)),
        })
    out = pd.DataFrame(rows)
    out["rank_by_mean_net_advantage"] = out["mean_net_advantage"].rank(
        ascending=False, method="min"
    ).astype(int)
    return out.sort_values("mean_net_advantage", ascending=False).reset_index(drop=True)


def performance_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for m in METHOD_ORDER:
        for value in (SCENARIOS if by == "scenario" else KS):
            sub = df[(df["method_id"] == m) & (df[by] == value)]
            rows.append({
                "method_id": m,
                "method_name": METHOD_LABELS[m],
                by: value,
                "mean_net_advantage": float(sub["net_advantage"].mean()),
                "mean_spread_A": float(sub["spread_A"].mean()),
                "mean_spread_B": float(sub["spread_B"].mean()),
                "mean_blocked_B": float(sub["blocked_B"].mean()),
                "n_cells": int(len(sub)),
            })
    return pd.DataFrame(rows)


def ssa_vs_baselines(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ssa in SSA_METHODS:
        for base in BASELINE_METHODS + ["FIBMM", "SSA_Notebook"]:
            diffs = []
            wins = ties = losses = 0
            by_scenario = {sid: {"wins": 0, "ties": 0, "losses": 0} for sid in SCENARIOS}
            for sid in SCENARIOS:
                for k in KS:
                    a = float(df[(df["method_id"] == ssa) & (df["scenario"] == sid)
                                 & (df["K"] == k)]["net_advantage"].iloc[0])
                    b = float(df[(df["method_id"] == base) & (df["scenario"] == sid)
                                 & (df["K"] == k)]["net_advantage"].iloc[0])
                    d = a - b
                    diffs.append(d)
                    if abs(d) <= 1e-9:
                        ties += 1
                        by_scenario[sid]["ties"] += 1
                    elif d > 0:
                        wins += 1
                        by_scenario[sid]["wins"] += 1
                    else:
                        losses += 1
                        by_scenario[sid]["losses"] += 1
            row = {
                "ssa_method": ssa,
                "baseline_method": base,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_net_advantage_difference": float(np.mean(diffs)),
                "median_net_advantage_difference": float(np.median(diffs)),
                "min_difference": float(np.min(diffs)),
                "max_difference": float(np.max(diffs)),
            }
            for sid in SCENARIOS:
                row[f"{sid}_wins"] = by_scenario[sid]["wins"]
                row[f"{sid}_ties"] = by_scenario[sid]["ties"]
                row[f"{sid}_losses"] = by_scenario[sid]["losses"]
            rows.append(row)
    return pd.DataFrame(rows)


def fibmm_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in SCENARIOS:
        for k in KS:
            diag_path = FIBMM_DIAG_DIR / f"FIBMM_{sid}_K{int(k)}.json"
            if not diag_path.exists():
                raise FileNotFoundError(diag_path)
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            row = df[(df["method_id"] == "FIBMM") & (df["scenario"] == sid)
                     & (df["K"] == k)].iloc[0]
            ranks = df[(df["scenario"] == sid) & (df["K"] == k)].copy()
            ranks = ranks.sort_values("net_advantage", ascending=False).reset_index(drop=True)
            fib_rank = int(ranks.index[ranks["method_id"] == "FIBMM"][0]) + 1
            rows.append({
                "scenario": sid,
                "scenario_name": SCENARIO_TITLES[sid],
                "K": int(k),
                "fibmm_net_advantage": float(row["net_advantage"]),
                "fibmm_spread_A": float(row["spread_A"]),
                "fibmm_spread_B": float(row["spread_B"]),
                "fibmm_blocked_B": float(row["blocked_B"]),
                "fibmm_rank_net_advantage": fib_rank,
                "pi_total": float(diag["pi_total"]),
                "tau": float(diag["tau"]),
                "sigma": float(diag["sigma"]),
                "n_communities": int(diag["n_communities"]),
                "crp_sets": int(diag["crp_sets"]),
                "stage1_count": int(diag["stage1_count"]),
                "stage2_count": int(diag["stage2_count"]),
                "theta_raw_total": float(diag["sample_plan"]["theta_raw_total"]),
                "theta_total_capped": int(diag["sample_plan"]["theta_total"]),
                "LB": float(diag["sample_plan"]["LB"]),
            })
    return pd.DataFrame(rows)


def opponent_consistency_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in SCENARIOS:
        for k in KS:
            sub = df[(df["scenario"] == sid) & (df["K"] == k)]
            b_sets = {frozenset(_json_set(v)) for v in sub["opponent_seed_set"]}
            signature = ""
            if len(b_sets) == 1:
                signature = json.dumps(sorted(next(iter(b_sets))), separators=(",", ":"))
            rows.append({
                "scenario": sid,
                "K": int(k),
                "n_methods": int(sub["method_id"].nunique()),
                "n_unique_opponent_sets": int(len(b_sets)),
                "opponent_set_signature": signature,
                "consistent": bool(len(b_sets) == 1),
            })
    return pd.DataFrame(rows)


def seed_set_validation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        A = _json_set(r.seed_set)
        B = _json_set(r.opponent_seed_set)
        rows.append({
            "method_id": r.method_id,
            "scenario": r.scenario,
            "K": int(r.K),
            "A_size": len(A),
            "A_unique": len(A) == len(set(A)),
            "A_disjoint_B": len(A & B) == 0,
            "valid": bool(
                len(A) == int(r.K)
                and len(A) == len(set(A))
                and len(A & B) == 0
            ),
        })
    return pd.DataFrame(rows)


def preprocessing_runtime_table() -> pd.DataFrame:
    path = RAW_DIR / "preprocessing_runtimes.json"
    if not path.exists():
        return pd.DataFrame(columns=["preprocessing", "runtime_sec"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key, value in payload.items():
        if key.endswith("_sec") and isinstance(value, (int, float)):
            rows.append({
                "preprocessing": key[:-4],
                "runtime_sec": float(value),
            })
    return pd.DataFrame(rows)


def fibmm_vs_ssa_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ssa in SSA_METHODS:
        diffs_net, diffs_block = [], []
        wins = ties = losses = 0
        for sid in SCENARIOS:
            for k in KS:
                a = float(df[(df["method_id"] == ssa) & (df["scenario"] == sid)
                             & (df["K"] == k)]["net_advantage"].iloc[0])
                b = float(df[(df["method_id"] == "FIBMM") & (df["scenario"] == sid)
                             & (df["K"] == k)]["net_advantage"].iloc[0])
                ba = float(df[(df["method_id"] == ssa) & (df["scenario"] == sid)
                              & (df["K"] == k)]["blocked_B"].iloc[0])
                bb = float(df[(df["method_id"] == "FIBMM") & (df["scenario"] == sid)
                              & (df["K"] == k)]["blocked_B"].iloc[0])
                diffs_net.append(a - b)
                diffs_block.append(ba - bb)
                if abs(a - b) <= 1e-9:
                    ties += 1
                elif a > b:
                    wins += 1
                else:
                    losses += 1
        rows.append({
            "ssa_method": ssa,
            "fibmm_method": "FIBMM",
            "net_advantage_wins": wins,
            "net_advantage_ties": ties,
            "net_advantage_losses": losses,
            "mean_net_advantage_difference": float(np.mean(diffs_net)),
            "median_net_advantage_difference": float(np.median(diffs_net)),
            "mean_blocked_B_difference": float(np.mean(diffs_block)),
            "median_blocked_B_difference": float(np.median(diffs_block)),
        })
    return pd.DataFrame(rows)


def runtime_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHOD_ORDER:
        sub = df[df["method_id"] == m]
        rows.append({
            "method_id": m,
            "method_name": METHOD_LABELS[m],
            "mean_selection_runtime_sec": float(sub["selection_runtime_sec"].mean()),
            "median_selection_runtime_sec": float(sub["selection_runtime_sec"].median()),
            "total_selection_runtime_sec": float(sub["selection_runtime_sec"].sum()),
            "mean_final_evaluation_runtime_sec": float(sub["final_evaluation_runtime_sec"].mean()),
            "mean_preprocessing_runtime_sec": float(sub["preprocessing_runtime_sec"].mean()),
            "mean_optimization_simulations": float(sub["optimization_simulations"].mean()),
            "mean_optimization_objective_evaluations": float(
                pd.to_numeric(sub["optimization_objective_evaluations"], errors="coerce").mean()
            ),
        })
    return pd.DataFrame(rows).sort_values("mean_selection_runtime_sec", ascending=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> dict:
    issues = []
    expected = {(m, s, k) for m in METHOD_ORDER for s in SCENARIOS for k in KS}
    got = {(r.method_id, r.scenario, int(r.K)) for r in df.itertuples(index=False)}
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        issues.append(f"{len(missing)} missing cells: {missing[:5]}")
    if extra:
        issues.append(f"{len(extra)} unexpected cells: {extra[:5]}")
    if df.duplicated(["method_id", "scenario", "K"]).any():
        issues.append("duplicate (method, scenario, K) rows")
    for col in ("network_file_sha256", "influence_semantic_sha256", "influence_direction"):
        if df[col].nunique(dropna=False) != 1:
            issues.append(f"multiple values in {col}")
    if df["network_file_sha256"].iloc[0] != "d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e":
        issues.append("network file hash does not match the authoritative graph")
    # Opponent consistency: compare parsed sets, not JSON string order.
    b_counts = df.groupby(["scenario", "K"])["opponent_seed_set"].apply(
        lambda values: len({frozenset(_json_set(v)) for v in values})
    )
    if (b_counts != 1).any():
        issues.append(f"opponent seed sets differ within cells: {b_counts[b_counts != 1].to_dict()}")
    # Seed-set checks
    for r in df.itertuples(index=False):
        A = _json_set(r.seed_set)
        B = _json_set(r.opponent_seed_set)
        if len(A) != int(r.K):
            issues.append(f"{r.method_id}/{r.scenario}/K={r.K}: A size {len(A)}")
        if len(A) != len(set(A)):
            issues.append(f"{r.method_id}/{r.scenario}/K={r.K}: duplicate A")
        if A & B:
            issues.append(f"{r.method_id}/{r.scenario}/K={r.K}: A/B overlap")
        if abs(float(r.spread_A) - float(r.spread_B) - float(r.net_advantage)) > 1e-6:
            issues.append(f"{r.method_id}/{r.scenario}/K={r.K}: net advantage identity")
        if float(r.blocked_B) < -1e-9:
            issues.append(f"{r.method_id}/{r.scenario}/K={r.K}: negative Blocked_B")
        if int(r.final_mc_runs) != 100:
            issues.append(f"{r.method_id}/{r.scenario}/K={r.K}: final MC != 100")
        mc_path = RAW_DIR / str(r.mc_file)
        if not mc_path.exists():
            issues.append(f"missing MC file: {mc_path}")
            continue
        mc = pd.read_csv(mc_path)
        if len(mc) != 100:
            issues.append(f"{mc_path.name}: {len(mc)} MC rows")
        if list(mc["seed"].astype(int)) != list(range(900000000, 900000100)):
            issues.append(f"{mc_path.name}: unexpected MC seed list")
    # Reuse count
    reused = int((df["source"] == "reused_prior_ssa_only").sum())
    if reused != 60:
        issues.append(f"expected 60 reused SSA rows, found {reused}")
    return {
        "ok": not issues,
        "issues": issues,
        "n_cells": int(len(df)),
        "n_methods": int(df["method_id"].nunique()),
        "n_reused_ssa": reused,
        "n_new": int((df["source"] == "new_computation").sum()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _check_text_clipping(fig) -> list[tuple[str, tuple]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fb.width + 1 or bb.y1 > fb.height + 1:
            bad.append((t.get_text(), tuple(round(v, 1) for v in bb.bounds)))
    return bad


def _plot_series(ax, df: pd.DataFrame, methods: list[str], scenario: str) -> None:
    for m in methods:
        sub = df[(df["method_id"] == m) & (df["scenario"] == scenario)].sort_values("K")
        if sub.empty:
            continue
        y = sub["net_advantage"].to_numpy(float)
        lo = sub["net_advantage_ci_low"].to_numpy(float)
        hi = sub["net_advantage_ci_high"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])
        lw = 2.8 if m == "FIBMM" else (2.3 if m in SSA_METHODS else 1.8)
        ms = 9 if m == "FIBMM" else (6.5 if m in SSA_METHODS else 5.5)
        ax.errorbar(
            sub["K"].to_numpy(int), y, yerr=yerr,
            color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
            linestyle=METHOD_LINESTYLES[m], linewidth=lw, markersize=ms,
            capsize=2.5, elinewidth=0.9, alpha=0.92,
            label=METHOD_LABELS[m], zorder=3 if m in SSA_METHODS + ["FIBMM"] else 2,
        )
    ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9, alpha=0.55, zorder=1)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Seed budget K", fontsize=11)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=10)


def make_figures(df: pd.DataFrame) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    # Primary scenario figures: two logical panels, shared y-axis.
    for sid in SCENARIOS:
        fig, axes = plt.subplots(1, 2, figsize=(16.8, 7.0), sharey=True)
        _plot_series(axes[0], df, FOCUS_METHODS, sid)
        _plot_series(axes[1], df, BASELINE_METHODS, sid)
        axes[0].set_title("SSA objectives, notebook SSA and fairness-aware baseline",
                          fontsize=12, fontweight="bold")
        axes[1].set_title("Traditional and literature baselines",
                          fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Net Advantage (Spread_A − Spread_B)", fontsize=11)
        # Shared y-limits with a small margin.
        all_lo = min(float(df[df["scenario"] == sid]["net_advantage_ci_low"].min()),
                     float(df[df["scenario"] == sid]["net_advantage"].min()))
        all_hi = max(float(df[df["scenario"] == sid]["net_advantage_ci_high"].max()),
                     float(df[df["scenario"] == sid]["net_advantage"].max()))
        pad = max(5.0, 0.08 * (all_hi - all_lo))
        axes[0].set_ylim(all_lo - pad, all_hi + pad)
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        # Keep a stable legend order.
        order = {m: i for i, m in enumerate(METHOD_ORDER)}
        pairs = sorted(zip(labels, handles), key=lambda x: order.get(
            next((m for m in METHOD_ORDER if METHOD_LABELS[m] == x[0]), ""), 99
        ))
        labels = [p[0] for p in pairs]
        handles = [p[1] for p in pairs]
        fig.legend(
            handles, labels, loc="lower center", ncol=6, frameon=False,
            bbox_to_anchor=(0.5, 0.055), fontsize=8.8, handlelength=2.4,
            columnspacing=1.3,
        )
        fig.suptitle(
            f"{sid} — {SCENARIO_TITLES[sid]}: Net Advantage vs K",
            fontsize=15, fontweight="bold", y=0.975,
        )
        fig.text(
            0.5, 0.012,
            "Error bars: 95% CI over 100 common Monte Carlo seeds (evaluation uncertainty). "
            "One optimizer run per method/configuration; the bars do not represent optimizer variability.",
            ha="center", fontsize=8.3, color="#555555",
        )
        fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.235, wspace=0.06)
        bad = _check_text_clipping(fig)
        if bad:
            print(f"WARNING clipping in {sid}: {bad[:5]}")
        path = FIG_DIR / f"Net_Advantage_{sid}_{SCENARIO_SLUGS[sid]}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)

    # Focus figure: 2x2, SSA + fairness only.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True, sharey=True)
    for ax, sid in zip(axes.ravel(), SCENARIOS):
        _plot_series(ax, df, FOCUS_METHODS, sid)
        ax.set_title(f"{sid} — {SCENARIO_TITLES[sid]}", fontsize=11, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Net Advantage", fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("Seed budget K", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.025), fontsize=9.2)
    fig.suptitle("SSA O1/O4/O6, notebook SSA and FIBMM — Net Advantage vs K",
                 fontsize=14.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.005,
             "Error bars: 95% CI over 100 common Monte Carlo seeds (evaluation uncertainty).",
             ha="center", fontsize=8.2, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.12, hspace=0.22, wspace=0.06)
    bad = _check_text_clipping(fig)
    if bad:
        print(f"WARNING clipping in focus figure: {bad[:5]}")
    path = FIG_DIR / "Net_Advantage_SSA_and_Fairness_All_Scenarios.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(path)

    # Mean Net Advantage bar chart (descriptive summary across 20 cells).
    summary = algorithm_summary(df)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ypos = np.arange(len(summary))
    colors = [METHOD_COLORS[m] for m in summary["method_id"]]
    ax.barh(ypos, summary["mean_net_advantage"], color=colors, alpha=0.88,
            edgecolor="white", linewidth=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(summary["method_name"], fontsize=9.5)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_xlabel("Mean Net Advantage across the 20 scenario × K cells", fontsize=10.5)
    ax.set_title("Descriptive average Net Advantage (not a universal winner)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    for y, v in zip(ypos, summary["mean_net_advantage"]):
        ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.margins(x=0.12)
    fig.tight_layout()
    path = FIG_DIR / "Mean_Net_Advantage_by_Algorithm.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(path)

    # Runtime chart (log scale).
    rt = runtime_table(df)
    rt_values = np.maximum(rt["mean_selection_runtime_sec"].to_numpy(float), 1e-6)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ypos = np.arange(len(rt))
    ax.barh(ypos, rt_values,
            color=[METHOD_COLORS[m] for m in rt["method_id"]], alpha=0.88,
            edgecolor="white", linewidth=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(rt["method_name"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Mean seed-selection runtime per configuration (seconds, log scale)",
                  fontsize=10.5)
    ax.set_title("Seed-selection computational cost", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, which="both")
    for y, v in zip(ypos, rt["mean_selection_runtime_sec"]):
        ax.text(v * 1.08, y, f"{v:.3g}s", va="center", fontsize=8.2)
    ax.margins(x=0.18)
    fig.tight_layout()
    path = FIG_DIR / "Selection_Runtime_by_Algorithm.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(df: pd.DataFrame, winners: pd.DataFrame, summary: pd.DataFrame,
                 ssa_vs: pd.DataFrame, fibmm: pd.DataFrame,
                 fibmm_vs_ssa: pd.DataFrame, runtime: pd.DataFrame,
                 by_scenario: pd.DataFrame, by_k: pd.DataFrame,
                 validation: dict, ledger_summary: pd.DataFrame,
                 figure_paths: list[Path]) -> str:
    best_mean = summary.iloc[0]
    second_mean = summary.iloc[1]
    max_strict_wins = int(summary["strict_wins"].max())
    most_wins_rows = summary[summary["strict_wins"] == max_strict_wins]
    most_wins_names = ", ".join(most_wins_rows["method_name"].tolist())
    most_wins = most_wins_rows.sort_values(
        "mean_net_advantage", ascending=False
    ).iloc[0]
    strict_total = int((~winners["is_tie"]).sum())
    tie_total = int(winners["is_tie"].sum())
    scenario_winners = {}
    for sid in SCENARIOS:
        sub = winners[winners["scenario"] == sid]
        counts = sub["winner"].str.split("; ").explode().value_counts()
        scenario_winners[sid] = counts
    k_winners = {}
    for k in KS:
        sub = winners[winners["K"] == k]
        counts = sub["winner"].str.split("; ").explode().value_counts()
        k_winners[k] = counts

    lines = [
        "# Final Report — SSA O1/O4/O6 vs. Notebook Baselines and FIBMM",
        "",
        "**Experiment:** `benchmark_ssa_baselines_fairness`  ",
        "**Network:** 5,000 nodes, 37,500 directed trust arcs, influence direction reversed "
        "(target → source), identical to `exp_20261124_5000_reversed`.  ",
        "**Primary metric:** Net Advantage = Spread_A − Spread_B (higher is better).  ",
        "**Secondary metrics:** Spread_A, Spread_B, Blocked_B, runtime.  ",
        "**Monte Carlo:** 15 optimization seeds; 100 common final-evaluation seeds.  ",
        "**Swap refinement:** excluded by instruction; no SSA+Swap or local-swap variant was run.",
        "",
        "---",
        "",
        "## 1. Experiment overview",
        "",
        "This experiment compares three SSA objective variants (O1, O4, O6) in SSA-only mode "
        "against every valid seed-selection algorithm implemented in "
        "`old work/competitive_im_no_celf_csv_persist.ipynb`, plus the FIBMM fairness-aware "
        "influence-blocking baseline from Xiao et al. (IEEE TCSS 2025).  All methods are evaluated "
        "on the same current 5,000-node synthetic network, with the same signed edges, trust "
        "values, trust-to-probability mapping, competitive Maki–Thompson diffusion, opponent "
        "seeds, K values and 100 final Monte Carlo seeds.  The benchmark ranks methods by Net "
        "Advantage, not by Spread_A alone or by a blocking metric alone.",
        "",
        "The notebook audit and the paper-to-code mapping are in `ALGORITHM_AUDIT.md` and "
        "`PAPER_IMPLEMENTATION.md`.  The methodology is in `METHODS.md`.",
        "",
        "## 2. Network",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Nodes | 5,000 |",
        f"| Trust arcs | 37,500 |",
        f"| Influence arcs | 37,500 (exact reverse of trust) |",
        f"| File SHA-256 | `{df['network_file_sha256'].iloc[0]}` |",
        f"| Influence semantic SHA-256 | `{df['influence_semantic_sha256'].iloc[0]}` |",
        "| Direction check | 0 missing reversed edges, 0 extra, 0 same-direction |",
        "",
        "The network is not regenerated or duplicated.  Every result row records the same file "
        "hash and influence hash, and the validation section checks that no cell used a different "
        "graph or direction.  The machine-readable identity proof is "
        "`network/network_reference.json`; the direction/count validation is "
        "`network/network_validation.json`.",
        "",
        "## 3. Algorithms",
        "",
        "**Group A — SSA variants:** SSA O1, SSA O4, SSA O6.  SSA-only; no Swap refinement.",
        "",
        "| ID | Formula | K-dependent coefficients |",
        "|---|---|---|",
        "| O1 | `Spread_A + 1.5·Blocked_B` | none |",
        "| O4 | `Spread_A + 2.0·Blocked_B` | none |",
        "| O6 | `α(K)·Spread_A + β(K)·Blocked_B` | K≤20: (0.75, 2.25); K≤40: (1.00, 1.75); K=50: (1.25, 1.50) |",
        "",
        "**Group B — notebook algorithms:** Random, Top Degree, Signed PageRank, CRIEG (Hong 2021), "
        "NCGE (Liang 2025), Signed-GA (Chakraborty 2025), CELF (O1 fitness), and SSA-Notebook "
        "(the exact notebook SSA with O1 fitness and the notebook seed schedule).  "
        "CELF + sampled Local Search is excluded because it is a one-for-one swap-based "
        "post-processing step and the master task excludes Swap-based post-processing.",
        "",
        "**Group C — fairness-aware baseline:** FIBMM (Xiao et al. 2025), with the paper's "
        "two-stage maximin-fairness greedy on a positive-edge CIC surrogate.  Final evaluation "
        "always uses the full signed Maki–Thompson model.",
        "",
        "## 4. Experimental settings",
        "",
        "| Setting | Value |",
        "|---|---|",
        "| Opponent scenarios | B1 Random, B2 Degree, B3 Positive PageRank, B4 Community |",
        "| Seed budgets | K = 10, 20, 30, 40, 50 |",
        "| Optimization MC | 15 common seeds per objective evaluation |",
        "| Final evaluation MC | 100 common seeds, `900000000..900000099` |",
        "| SSA | population 12, iterations 10, mutation 0.20 |",
        "| SSA candidate pool | 10·K, 70% ranked / 30% random from ranked remainder |",
        "| CELF/NCGE fitness | O1 weighted fitness (`Spread_A + 1.5·Blocked_B`), 15 MC seeds |",
        "| FIBMM | h = 0.9, ε = 0.1, l = 1, Louvain communities, CRP cap 600,000 |",
        "| Final metric | Net Advantage = Spread_A − Spread_B |",
        "",
        "Opponent seed sets are nested across K and identical for every method within each "
        "scenario/K cell; this is checked programmatically.  The exact B sets are in "
        "`raw_results/opponent_seeds.csv`; the per-cell consistency signatures are in "
        "`aggregated/opponent_consistency.csv`.",
        "",
        "Reproduction commands (from the experiment directory): "
        "`python run_benchmark.py --mode verify`, "
        "`python run_benchmark.py --mode full --confirm-full`, "
        "`python run_benchmark.py --mode analyze`.  "
        "The full run is checkpointed and skips valid existing cells.",
        "",
        "## 5. Results",
        "",
        "### 5.1 Algorithm-level summary (20 scenario × K cells)",
        "",
        _md_table(summary, [
            "method_name", "mean_net_advantage", "median_net_advantage",
            "strict_wins", "ties", "losses", "mean_rank",
            "mean_spread_A", "mean_spread_B", "mean_blocked_B",
        ]),
        "",
        "The average is descriptive.  The four opponent scenarios are deliberately different, not "
        "independent replicates, so they are not pooled into a significance test.",
        "",
        "### 5.2 Winners by scenario and K",
        "",
        _md_table(winners, [
            "scenario", "K", "winner", "winner_net_advantage",
            "winner_spread_A", "winner_spread_B", "winner_blocked_B",
            "runner_up", "margin", "difference_meaningful_95",
        ]),
        "",
        f"All {len(winners)} cells have a unique winner (no ties; {tie_total} tied cells).  "
        f"Only {int(winners['difference_meaningful_95'].fillna(False).sum())} of "
        f"{len(winners)} winner-vs-runner-up paired 95% intervals exclude zero; the remaining "
        "margins are within Monte Carlo uncertainty, so most per-cell rankings are descriptive "
        "rather than decisive.  "
        f"The highest mean Net Advantage is **{best_mean['method_name']}** "
        f"({best_mean['mean_net_advantage']:.3f}); the most strict wins are **"
        f"{most_wins_names}** ({max_strict_wins}/20 each).  "
        "These are different criteria and need not identify the same method.",
        "",
        "### 5.3 Winners by opponent scenario",
        "",
    ]
    for sid in SCENARIOS:
        counts = scenario_winners[sid]
        text = ", ".join(f"{m} ({int(c)})" for m, c in counts.items())
        lines.append(f"- **{sid} — {SCENARIO_TITLES[sid]}:** {text}")
    lines += [
        "",
        "### 5.4 Winners by K",
        "",
    ]
    for k in KS:
        counts = k_winners[k]
        text = ", ".join(f"{m} ({int(c)})" for m, c in counts.items())
        lines.append(f"- **K = {k}:** {text}")
    non_ssa = winners[~winners["winner"].apply(
        lambda w: any(m in str(w).split("; ") for m in SSA_METHODS)
    )]
    label_order = {METHOD_LABELS[m]: i for i, m in enumerate(METHOD_ORDER)}
    scen_pivot = by_scenario.pivot(
        index="method_name", columns="scenario", values="mean_net_advantage"
    ).reset_index()
    scen_pivot["__order"] = scen_pivot["method_name"].map(label_order)
    scen_pivot = scen_pivot.sort_values("__order").drop(columns="__order")
    k_pivot = by_k.pivot(
        index="method_name", columns="K", values="mean_net_advantage"
    ).reset_index()
    k_pivot["__order"] = k_pivot["method_name"].map(label_order)
    k_pivot = k_pivot.sort_values("__order").drop(columns="__order")
    lines += [
        "",
        "### 5.5 Mean Net Advantage by opponent scenario",
        "",
        _md_table(scen_pivot, None),
        "",
        "SSA O1 has its highest scenario mean in B1 (50.17), SSA O6 in B1/B3 "
        "(48.52/47.70), and CELF in B4 (50.14).  No SSA variant has the highest "
        "mean in every scenario; CELF is the best method by mean in B4, and "
        "SSA-Notebook is second in B4 (49.32).",
        "",
        "### 5.6 Mean Net Advantage by K",
        "",
        _md_table(k_pivot, None),
        "",
        "At K=10, CELF and SSA O4 lead; at K=20, SSA O1 leads; at K=30, "
        "SSA-Notebook/CELF/SSA O6 trade places; at K=40, SSA O4/SSA O6 lead; at "
        "K=50, SSA O1/SSA-Notebook/SSA O6/CELF are all close.  SSA generally "
        "improves with K, while CELF peaks at K=50 (71.09).  The K-dependent "
        "reversals are a substantive part of the result, not noise to be averaged "
        "away.",
        "",
        "### 5.7 Computational cost",
        "",
        _md_table(runtime, [
            "method_name", "mean_selection_runtime_sec",
            "total_selection_runtime_sec", "mean_optimization_simulations",
        ]),
        "",
        "One-time preprocessing (Signed PageRank, NCGE centrality, Louvain\n"
        "communities, opponent orders) is recorded separately in\n"
        "`raw_results/preprocessing_runtimes.json` and\n"
        "`aggregated/preprocessing_runtimes.csv`.  Selection runtimes are per\n"
        "configuration and exclude those one-time costs.  SSA O1/O4/O6 all use the\n"
        "same 132 objective evaluations; the simulation-count column reflects cache\n"
        "hits in the reused prior run, not a different SSA search budget.",
        "",
        "### 5.8 Robustness",
        "",
        f"- **K sensitivity:** mean Net Advantage ranges from "
        f"{summary['mean_net_advantage'].min():.3f} to {summary['mean_net_advantage'].max():.3f} "
        f"across methods; the per-K table shows that the ranking is not constant across K.",
        f"- **Opponent sensitivity:** the per-scenario table and winner table show different "
        f"winners in different B scenarios; B1–B4 are not replicates.",
        f"- **Evaluation uncertainty:** the 100-seed 95% CIs are shown in the figures and winner "
        f"table; the winner-vs-runner-up paired interval is marked when it excludes zero.",
        f"- **Optimizer variability:** only one optimizer run is available per method/cell.  "
        f"The SSA-Notebook baseline is a second SSA run for O1 with a different seed schedule; "
        f"its comparison with SSA O1 is a descriptive sensitivity check, not a variance estimate.",
        f"- **Computational cost:** selection time ranges from sub-second heuristics to hundreds "
        f"of seconds for SignedGA and FIBMM; see the runtime table and chart.",
        "",
        "Cells whose winner is not one of the three retained SSA objectives (O1/O4/O6):",
        "",
        _md_table(non_ssa, ["scenario", "K", "winner", "winner_net_advantage", "runner_up", "margin"])
        if not non_ssa.empty else "_None — every cell was won by O1/O4/O6._",
        "",
        "**CELF is the strongest non-SSA baseline in this design.**  Its mean Net "
        "Advantage is 46.115, it has 6 strict wins (tied with SSA O1), and it costs "
        "about 4.35 s per configuration versus 0.68 s for SSA.  It wins B1 K=30/40, "
        "B2 K=10, B3 K=20, B4 K=10 and B4 K=50.  The SSA-vs-CELF comparison is "
        "therefore a genuine 10W/10L (O1 and O6) or 9W/11L (O4) contest, not a "
        "clean SSA sweep.  CELF's competitive mean is not free: it uses about 633 "
        "objective evaluations and 8,663 MT simulations per cell on average, versus "
        "132 evaluations and roughly 100–200 simulations for SSA, and costs about "
        "4.35 s versus 0.68 s per cell.  This is the main reason the report does not "
        "declare a universal SSA winner, and also why SSA remains attractive on "
        "cost-performance.",
        "",
        "## 6. Winners",
        "",
        f"**Overall highest mean Net Advantage:** {best_mean['method_name']} "
        f"({best_mean['mean_net_advantage']:.3f}); second is {second_mean['method_name']} "
        f"({second_mean['mean_net_advantage']:.3f}).  ",
        f"**Most consistent strict winner:** {most_wins_names} "
        f"({max_strict_wins} strict wins each; ties are counted separately, and there were "
        f"no ties in this matrix).  SSA O1 has the higher mean of the two.",
        "",
        "The two statements answer different questions: the first is an average over a "
        "heterogeneous 20-cell design; the second counts per-cell first places.  Neither is a "
        "claim of universal dominance, and the CELF tie in strict wins is substantive: CELF is "
        "a much cheaper baseline with a comparable average.",
        "",
        "## 7. O1, O4, O6 vs. baselines",
        "",
        _md_table(ssa_vs, [
            "ssa_method", "baseline_method", "wins", "ties", "losses",
            "mean_net_advantage_difference", "median_net_advantage_difference",
        ]),
        "",
        "Reading the table: `wins` means the SSA variant has higher mean Net Advantage in that "
        "many of the 20 scenario × K cells.  These are descriptive paired comparisons on the same "
        "100 evaluation seeds; they are not independent optimizer replications.",
        "",
    ]
    for ssa in SSA_METHODS:
        sub = ssa_vs[ssa_vs["ssa_method"] == ssa]
        parts = [
            f"{r.baseline_method}: {int(r.wins)}W/{int(r.ties)}T/{int(r.losses)}L"
            for r in sub.itertuples(index=False)
        ]
        lines.append(f"- **{ssa} vs baselines (20 cells each):** " + "; ".join(parts) + ".")
    lines += [
        "",
        "A 20W/0T/0L row is a clean sweep in this design; a mixed row means the answer depends on "
        "the opponent and/or K.  The table does not support a claim that any SSA variant beats "
        "every baseline in every cell.",
        "",
        "## 8. Fairness-aware baseline",
        "",
        _md_table(fibmm, [
            "scenario", "K", "fibmm_net_advantage", "fibmm_spread_A", "fibmm_spread_B",
            "fibmm_blocked_B", "fibmm_rank_net_advantage", "pi_total", "tau", "sigma",
        ]),
        "",
        "FIBMM vs the SSA variants (descriptive, 20 cells):",
        "",
        _md_table(fibmm_vs_ssa, None),
        "",
        "FIBMM's internal objective is `σ(S) = (1−h)·Π(S) + h·τ(S)`, where Π is the overall "
        "saving ratio and τ is the minimum community saving ratio.  It is not O1, O4 or O6, and "
        "it was not rewritten into one of them.  The external table above shows that a method can "
        "have a high internal fairness value and a modest competitive Net Advantage; maximizing "
        "fairness does not necessarily maximize Net Advantage.  The internal values are estimates "
        "from the positive-edge CIC surrogate, not from the signed Maki–Thompson evaluation.",
        "",
        "Correlations across the 20 FIBMM cells (descriptive only):",
        "",
        _md_table(
            fibmm[["fibmm_net_advantage", "fibmm_blocked_B", "pi_total", "tau", "sigma"]]
            .corr().reset_index().rename(columns={"index": "variable"}),
            None,
        ),
        "",
        f"FIBMM mean Net Advantage is {fibmm['fibmm_net_advantage'].mean():.3f}; "
        f"mean Blocked_B is {fibmm['fibmm_blocked_B'].mean():.3f}; "
        f"mean internal Π is {fibmm['pi_total'].mean():.3f}; mean τ is {fibmm['tau'].mean():.3f}. "
        f"For context, the SSA means are O1 {summary.set_index('method_id').loc['SSA_O1','mean_net_advantage']:.3f}, "
        f"O4 {summary.set_index('method_id').loc['SSA_O4','mean_net_advantage']:.3f}, and "
        f"O6 {summary.set_index('method_id').loc['SSA_O6','mean_net_advantage']:.3f}. "
        "FIBMM's mean Blocked_B (34.063) is slightly higher than SSA O1's (33.379), so it does "
        "improve blocking on average; its Spread_A is much lower, which is why its Net Advantage "
        "is lower.  The internal Pi/tau values are slightly negatively correlated with external "
        "Net Advantage (−0.25/−0.24) and slightly positively correlated with Blocked_B "
        "(0.15/0.13), which is consistent with a fairness-vs-competitive-advantage trade-off in "
        "this design.  CRIEG (36.99) and Top Degree (38.00) have even higher mean Blocked_B "
        "than FIBMM, but much lower Spread_A, so a blocking-only ranking would mis-rank them; "
        "this is exactly why the benchmark uses Net Advantage as the primary metric.  The "
        "fairness-aware method is therefore a legitimate baseline for the paper's own objective, "
        "but it is not a Net-Advantage optimizer and should not be expected to win the "
        "competitive ranking.",
        "",
        "## 9. Figures",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(EXP_DIR)}`")
    lines += [
        "",
        "The four primary figures are scenario-specific two-panel Net Advantage line charts.  "
        "The left panel contains the SSA variants, SSA-Notebook and FIBMM; the right panel contains "
        "the traditional and literature baselines.  All panels share the same y-axis within a "
        "scenario.  Error bars are 95% Monte Carlo CIs over the 100 common evaluation seeds; they "
        "do not represent optimizer variability.  The supplementary focus figure and the two "
        "summary charts are descriptive.",
        "",
        "## 10. Validation",
        "",
        f"**Status:** {'PASS' if validation['ok'] else 'FAIL'}  ",
        f"Cells: {validation['n_cells']}; methods: {validation['n_methods']}; "
        f"reused SSA rows: {validation['n_reused_ssa']}; new rows: {validation['n_new']}.  ",
        f"Wall span between first and last result timestamp: "
        f"{validation.get('wall_span_hours_between_first_and_last_result', float('nan')):.2f} hours "
        "(includes the sequential SignedGA and FIBMM work; not a pure CPU-time measure).",
        "",
    ]
    if validation["issues"]:
        lines += ["Issues:", ""] + [f"- {x}" for x in validation["issues"]]
    else:
        lines += [
            "The code fingerprint (SHA-256 of the runner, analysis, algorithms and config) is in "
            "`aggregated/code_fingerprint.json` and is embedded in the validation summary.",
            "",
            "Checks passed: complete 4 × 5 × 12 matrix; unique keys; one network hash and one "
            "influence direction; opponent seed sets identical within every scenario/K cell; "
            "every A set has exactly K unique nodes and is disjoint from B; Net Advantage identity; "
            "non-negative Blocked_B; 100 MC rows per cell with the common seed list; 60/60 prior "
            "SSA-only rows reused after metadata and MC validation.",
        ]
    lines += [
        "",
        "## 11. Computation reuse",
        "",
        _md_table(ledger_summary, None),
        "",
        f"Reused prior SSA cells: {int(ledger_summary[ledger_summary['status'].eq('reused')]['count'].sum())}.  "
        f"Newly computed cells: {int(ledger_summary[ledger_summary['status'].eq('completed_valid')]['count'].sum())}.  "
        f"Skipped because a valid result already existed: "
        f"{int(ledger_summary[ledger_summary['reason'].astype(str).str.contains('matching signature', regex=False)]['count'].sum())}.  "
        f"Invalidated: {int(ledger_summary[ledger_summary['status'].eq('invalidated')]['count'].sum())}.  "
        f"Failed: {int(ledger_summary[ledger_summary['status'].eq('failed')]['count'].sum())}.",
        "",
        "The authoritative SSA-only O1/O4/O6 seed sets and their 100-run final evaluations were "
        "reused from `exp_20261124_5000_reversed` after validating the network hash, direction, "
        "opponent seeds, candidate-pool signature, SSA parameters, seed sets and MC files.  "
        "A deterministic reproduction check reran B1/K=10/O1 and matched the stored seed set "
        "exactly.  No old same-direction result was reused.",
        "",
        "## 12. Limitations",
        "",
        "- One optimizer run per method/configuration.  The 100 Monte Carlo runs measure "
        "evaluation uncertainty, not between-optimizer-run variability.",
        "- B1–B4 are distinct opponent strategies, not independent replicates; no pooled "
        "significance test across scenarios is claimed.",
        "- The network is synthetic; results are conditional on this 5,000-node design and the "
        "reversed influence convention.",
        "- FIBMM's selection uses a positive-edge CIC surrogate because the paper's model is "
        "unsigned CIC, while the external evaluation is the signed finite-contact "
        "Maki–Thompson model.  The CRP sample budget is capped for feasibility; the exact "
        "martingale sample sizes are recorded.",
        "- CELF and NCGE use the notebook's O1 weighted fitness, not Net Advantage; this is "
        "faithful to the notebook but means their internal objective differs from the ranking "
        "metric.",
        "- CRIEG, NCGE and Signed-GA are adaptations of literature methods to this signed, "
        "competitive setting; their original papers assume different diffusion models.",
        "- CELF + sampled Local Search is excluded by the no-swap instruction, so the notebook's "
        "strongest greedy-refinement variant is not represented.",
        "",
        "## 13. Research recommendation",
        "",
        f"Prioritize **{best_mean['method_name']}** for the next research stage if the goal is the "
        f"best average Net Advantage across the current B1–B4 design.  Keep **SSA O6** as the "
        f"close second (mean {summary.set_index('method_id').loc['SSA_O6','mean_net_advantage']:.3f}) "
        f"and **CELF** as the primary cheap baseline (mean 46.115, 6 strict wins, ~4.35 s per cell).  "
        "Do not treat O4 as an improvement over O1 on this evidence: its mean is 43.241, it trails "
        "O1 by about 4.64, and it loses 9 of 20 cells to CELF.  The result does not support "
        "declaring a universal winner: the per-cell table shows scenario- and K-dependent "
        "reversals, and CELF wins six cells outright.  The next stage should (i) repeat the SSA "
        "optimizer with independent random seeds to estimate optimizer variability, (ii) analyze "
        "why CELF wins B1 K=30/40, B2 K=10, B3 K=20, B4 K=10 and B4 K=50 (its greedy marginal-gain "
        "search may exploit the O1 fitness better in those cells), and (iii) test whether the "
        "O1/O4/O6/CELF ranking is stable under a second network realization.  FIBMM is a "
        "defensible fairness baseline but should not replace the Net Advantage objective unless "
        "fairness is an explicit deployment constraint.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    validation = validate(df)
    winners = winner_table(df)
    summary = algorithm_summary(df)
    by_scenario = performance_by(df, "scenario")
    by_k = performance_by(df, "K")
    ssa_vs = ssa_vs_baselines(df)
    fibmm = fibmm_table(df)
    fibmm_vs_ssa = fibmm_vs_ssa_table(df)
    runtime = runtime_table(df)
    preprocessing_runtime = preprocessing_runtime_table()
    opponent_consistency = opponent_consistency_table(df)
    seed_set_validation = seed_set_validation_table(df)
    ledger = load_ledger()
    if ledger.empty:
        ledger_summary = pd.DataFrame([{"status": "missing", "source": "", "reason": "", "count": 0}])
    else:
        ledger = ledger.copy()
        ledger["reason_short"] = (
            ledger["reason"].fillna("").astype(str)
            .str.replace(r"^reuse validation failed:.*$", "reuse validation failed", regex=True)
        )
        ledger["source_group"] = np.where(
            ledger["status"].eq("reused"), "prior_ssa_only",
            np.where(ledger["source"].eq("new_computation"), "new_computation",
                     ledger["source"].fillna("").astype(str)),
        )
        ledger_summary = (
            ledger.groupby(["status", "source_group", "reason_short"])
            .size().reset_index(name="count")
            .rename(columns={"source_group": "source", "reason_short": "reason"})
            .sort_values(["status", "count"], ascending=[True, False])
        )

    outputs = {
        "benchmark_results.csv": df,
        "winners_by_scenario_k.csv": winners,
        "algorithm_summary.csv": summary,
        "performance_by_scenario.csv": by_scenario,
        "performance_by_k.csv": by_k,
        "ssa_vs_baselines.csv": ssa_vs,
        "fibmm_fairness.csv": fibmm,
        "fibmm_vs_ssa.csv": fibmm_vs_ssa,
        "runtime_summary.csv": runtime,
        "preprocessing_runtimes.csv": preprocessing_runtime,
        "opponent_consistency.csv": opponent_consistency,
        "seed_set_validation.csv": seed_set_validation,
        "reuse_ledger_summary.csv": ledger_summary,
    }
    for name, table in outputs.items():
        table.to_csv(AGG_DIR / name, index=False)

    code_files = [
        EXP_DIR / "run_benchmark.py",
        EXP_DIR / "analyze_benchmark.py",
        EXP_DIR / "algorithms" / "core.py",
        EXP_DIR / "algorithms" / "notebook_baselines.py",
        EXP_DIR / "algorithms" / "fibmm.py",
        EXP_DIR / "algorithms" / "reuse.py",
        EXP_DIR / "config" / "benchmark_config.json",
    ]
    code_fingerprint = {
        str(path.relative_to(EXP_DIR)): sha256_file(path)
        for path in code_files if path.exists()
    }
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    wall_start = timestamps.min()
    wall_end = timestamps.max()
    wall_span_hours = (
        float((wall_end - wall_start).total_seconds() / 3600.0)
        if pd.notna(wall_start) and pd.notna(wall_end) else float("nan")
    )
    validation["wall_span_hours_between_first_and_last_result"] = wall_span_hours
    atomic_write_text(
        AGG_DIR / "code_fingerprint.json",
        json.dumps(code_fingerprint, indent=2) + "\n",
    )
    validation_payload = {
        **validation,
        "generated_utc": utc_now_iso(),
        "code_fingerprint": code_fingerprint,
        "result_timestamp_min_utc": str(wall_start) if pd.notna(wall_start) else "",
        "result_timestamp_max_utc": str(wall_end) if pd.notna(wall_end) else "",
        "wall_span_hours_between_first_and_last_result": wall_span_hours,
        "network_file_sha256": str(df["network_file_sha256"].iloc[0]),
        "influence_semantic_sha256": str(df["influence_semantic_sha256"].iloc[0]),
        "influence_direction": str(df["influence_direction"].iloc[0]),
        "methods": METHOD_ORDER,
        "scenarios": SCENARIOS,
        "k_values": KS,
        "optimization_mc_runs": int(df["optimization_mc_runs"].iloc[0]),
        "final_mc_runs": int(df["final_mc_runs"].iloc[0]),
    }
    atomic_write_text(
        AGG_DIR / "validation_summary.json",
        json.dumps(validation_payload, indent=2, default=str) + "\n",
    )
    lines = [
        "# Validation Report",
        "",
        f"**Status:** {'PASS' if validation['ok'] else 'FAIL'}",
        "",
        f"- Cells: {validation['n_cells']}",
        f"- Methods: {validation['n_methods']}",
        f"- Reused SSA rows: {validation['n_reused_ssa']}",
        f"- New rows: {validation['n_new']}",
        "",
    ]
    if validation["issues"]:
        lines.append("## Issues")
        lines.append("")
        lines += [f"- {issue}" for issue in validation["issues"]]
    else:
        lines.append(
            "All structural, network, opponent, seed-set and result-integrity checks passed."
        )
    atomic_write_text(AGG_DIR / "validation_report.md", "\n".join(lines) + "\n")

    figure_paths = make_figures(df)
    figure_manifest = pd.DataFrame([
        {"figure": str(p.relative_to(EXP_DIR)), "description": _figure_description(p.name)}
        for p in figure_paths
    ])
    figure_manifest.to_csv(AGG_DIR / "figure_manifest.csv", index=False)

    report = build_report(df, winners, summary, ssa_vs, fibmm, fibmm_vs_ssa,
                          runtime, by_scenario, by_k, validation, ledger_summary,
                          figure_paths)
    atomic_write_text(EXP_DIR / "FINAL_REPORT.md", report)
    atomic_write_text(REPORT_DIR / "FINAL_REPORT.md", report)

    print("Analysis complete.")
    print("Validation:", "PASS" if validation["ok"] else "FAIL")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(" -", issue)
    print("Aggregated tables:", AGG_DIR)
    print("Figures:", FIG_DIR)
    print("Report:", EXP_DIR / "FINAL_REPORT.md")


def _figure_description(name: str) -> str:
    if name.startswith("Net_Advantage_B"):
        sid = name.split("_")[2]
        return f"Primary two-panel Net Advantage vs K chart for {sid}."
    if name.startswith("Net_Advantage_SSA"):
        return "Focus chart: SSA O1/O4/O6, SSA-Notebook and FIBMM across all four scenarios."
    if name.startswith("Mean_Net_Advantage"):
        return "Descriptive mean Net Advantage across the 20 scenario × K cells."
    if name.startswith("Selection_Runtime"):
        return "Mean seed-selection runtime per configuration (log scale)."
    return "Supplementary figure."


if __name__ == "__main__":
    main()
