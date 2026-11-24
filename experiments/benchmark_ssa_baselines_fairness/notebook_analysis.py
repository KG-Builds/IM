"""Read-only analysis helpers for the 1,000-node notebook.

The functions accept an explicit ``raw_dir`` and result table, so the same
code can analyze either the 1,000-node or 5,000-node experiment without
mixing their files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.stats import paired_diff

SCENARIOS = ["B1", "B2", "B3", "B4"]
KS = [10, 20, 30, 40, 50]
SCENARIO_TITLES = {
    "B1": "Random opponent",
    "B2": "Degree-based opponent",
    "B3": "Positive PageRank-based opponent",
    "B4": "Community-concentrated opponent",
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
    "FIBMM": "FIBM (fairness-aware)",
    "Random": "Random",
    "TopDegree": "Top Degree",
    "SignedPageRank": "Signed PageRank",
    "CRIEG": "CRIEG (2021)",
    "NCGE": "NCGE (2025)",
    "SignedGA": "Signed-GA (2025)",
    "CELF": "CELF (O1 fitness)",
}
METHOD_COLORS = {
    "SSA_O1": "#0072B2", "SSA_O4": "#D55E00", "SSA_O6": "#009E73",
    "SSA_Notebook": "#CC79A7", "FIBMM": "#000000", "Random": "#7F7F7F",
    "TopDegree": "#E69F00", "SignedPageRank": "#56B4E9", "CRIEG": "#8C564B",
    "NCGE": "#9467BD", "SignedGA": "#BCBD22", "CELF": "#17BECF",
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
BASELINE_METHODS = ["Random", "TopDegree", "SignedPageRank", "CRIEG", "NCGE", "SignedGA", "CELF"]
FOCUS_METHODS = ["SSA_O1", "SSA_O4", "SSA_O6", "SSA_Notebook", "FIBMM"]


def _json_set(value) -> set[int]:
    if isinstance(value, str):
        return {int(x) for x in json.loads(value)}
    return {int(x) for x in value}


def _mc_for(df: pd.DataFrame, raw_dir: Path, method_id: str,
            scenario: str, k: int) -> pd.DataFrame:
    sub = df[(df["method_id"] == method_id) & (df["scenario"] == scenario)
             & (df["K"].astype(int) == int(k))]
    if len(sub) != 1:
        raise ValueError(f"Expected one row for {method_id}/{scenario}/K={k}")
    path = Path(raw_dir) / str(sub.iloc[0]["mc_file"])
    return pd.read_csv(path)


def _paired_cell(df: pd.DataFrame, raw_dir: Path, a: str, b: str,
                 scenario: str, k: int) -> dict:
    ma = _mc_for(df, raw_dir, a, scenario, k).set_index("seed")
    mb = _mc_for(df, raw_dir, b, scenario, k).set_index("seed")
    common = ma.index.intersection(mb.index)
    return paired_diff(
        ma.loc[common, "net_advantage_run"].to_numpy(float),
        mb.loc[common, "net_advantage_run"].to_numpy(float),
        n_boot=5000,
    )


def winner_table(df: pd.DataFrame, raw_dir: Path, tie_tol: float = 1e-9) -> pd.DataFrame:
    rows = []
    for sid in SCENARIOS:
        for k in KS:
            sub = df[(df["scenario"] == sid) & (df["K"].astype(int) == int(k))].copy()
            vals = sub.set_index("method_id")["net_advantage"].reindex(METHOD_ORDER)
            if vals.isna().any():
                raise ValueError(f"Missing methods for {sid}/K={k}")
            maxv = float(vals.max())
            winners = [m for m in METHOD_ORDER if float(vals[m]) >= maxv - tie_tol]
            others = sorted(
                [(m, float(vals[m])) for m in METHOD_ORDER if m not in winners],
                key=lambda x: -x[1],
            )
            runner = others[0][0] if others else None
            runner_v = others[0][1] if others else float("nan")
            wrow = sub[sub["method_id"].isin(winners)].iloc[0]
            row = {
                "scenario": sid,
                "scenario_name": SCENARIO_TITLES[sid],
                "K": int(k),
                "winner": "; ".join(winners),
                "is_tie": len(winners) > 1,
                "winner_net_advantage": maxv,
                "winner_spread_A": float(wrow["spread_A"]),
                "winner_spread_B": float(wrow["spread_B"]),
                "winner_blocked_B": float(wrow["blocked_B"]),
                "winner_ci_low": float(wrow["net_advantage_ci_low"]),
                "winner_ci_high": float(wrow["net_advantage_ci_high"]),
                "runner_up": runner,
                "runner_up_net_advantage": runner_v,
                "margin": maxv - runner_v if runner is not None else float("nan"),
            }
            if runner is not None and len(winners) == 1:
                rrow = sub[sub["method_id"] == runner].iloc[0]
                row["runner_up_ci_low"] = float(rrow["net_advantage_ci_low"])
                row["runner_up_ci_high"] = float(rrow["net_advantage_ci_high"])
                diff = _paired_cell(df, raw_dir, winners[0], runner, sid, int(k))
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


def algorithm_summary(df: pd.DataFrame, tie_tol: float = 1e-9) -> pd.DataFrame:
    rows = []
    for m in METHOD_ORDER:
        sub = df[df["method_id"] == m]
        if len(sub) != len(SCENARIOS) * len(KS):
            raise ValueError(f"{m}: found {len(sub)} cells, expected 20")
        wins = ties = losses = 0
        ranks = []
        for sid in SCENARIOS:
            for k in KS:
                cell = df[(df["scenario"] == sid) & (df["K"].astype(int) == int(k))]
                vals = cell.set_index("method_id")["net_advantage"].reindex(METHOD_ORDER)
                maxv = float(vals.max())
                if abs(float(vals[m]) - maxv) <= tie_tol:
                    if int((vals >= maxv - tie_tol).sum()) > 1:
                        ties += 1
                    else:
                        wins += 1
                else:
                    losses += 1
                ranks.append(int((vals > float(vals[m]) + tie_tol).sum()) + 1)
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
            "n_cells": int(len(sub)),
        })
    out = pd.DataFrame(rows)
    out["rank_by_mean_net_advantage"] = out["mean_net_advantage"].rank(
        ascending=False, method="min"
    ).astype(int)
    return out.sort_values("mean_net_advantage", ascending=False).reset_index(drop=True)


def ssa_vs_baselines(df: pd.DataFrame, tie_tol: float = 1e-9) -> pd.DataFrame:
    rows = []
    for ssa in SSA_METHODS:
        for base in BASELINE_METHODS + ["FIBMM", "SSA_Notebook"]:
            diffs = []
            wins = ties = losses = 0
            for sid in SCENARIOS:
                for k in KS:
                    a = float(df[(df["method_id"] == ssa) & (df["scenario"] == sid)
                                 & (df["K"].astype(int) == int(k))]["net_advantage"].iloc[0])
                    b = float(df[(df["method_id"] == base) & (df["scenario"] == sid)
                                 & (df["K"].astype(int) == int(k))]["net_advantage"].iloc[0])
                    d = a - b
                    diffs.append(d)
                    if abs(d) <= tie_tol:
                        ties += 1
                    elif d > 0:
                        wins += 1
                    else:
                        losses += 1
            rows.append({
                "ssa_method": ssa,
                "baseline_method": base,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_net_advantage_difference": float(np.mean(diffs)),
                "median_net_advantage_difference": float(np.median(diffs)),
                "min_difference": float(np.min(diffs)),
                "max_difference": float(np.max(diffs)),
            })
    return pd.DataFrame(rows)


def fibm_table(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    rows = []
    for sid in SCENARIOS:
        for k in KS:
            diag_path = Path(raw_dir) / "fibmm_diagnostics" / f"FIBMM_{sid}_K{int(k)}.json"
            if not diag_path.exists():
                raise FileNotFoundError(diag_path)
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            row = df[(df["method_id"] == "FIBMM") & (df["scenario"] == sid)
                     & (df["K"].astype(int) == int(k))].iloc[0]
            ranks = df[(df["scenario"] == sid) & (df["K"].astype(int) == int(k))].copy()
            ranks = ranks.sort_values("net_advantage", ascending=False).reset_index(drop=True)
            fib_rank = int(ranks.index[ranks["method_id"] == "FIBMM"][0]) + 1
            rows.append({
                "scenario": sid,
                "scenario_name": SCENARIO_TITLES[sid],
                "K": int(k),
                "fibm_net_advantage": float(row["net_advantage"]),
                "fibm_spread_A": float(row["spread_A"]),
                "fibm_spread_B": float(row["spread_B"]),
                "fibm_blocked_B": float(row["blocked_B"]),
                "fibm_rank_net_advantage": fib_rank,
                "pi_total": float(diag["pi_total"]),
                "tau": float(diag["tau"]),
                "sigma": float(diag["sigma"]),
                "n_communities": int(diag["n_communities"]),
                "crp_sets": int(diag["crp_sets"]),
                "stage1_count": int(diag["stage1_count"]),
                "stage2_count": int(diag["stage2_count"]),
            })
    return pd.DataFrame(rows)


def fibm_vs_ssa_table(df: pd.DataFrame, tie_tol: float = 1e-9) -> pd.DataFrame:
    rows = []
    for ssa in SSA_METHODS:
        diffs_net, diffs_block = [], []
        wins = ties = losses = 0
        for sid in SCENARIOS:
            for k in KS:
                a = float(df[(df["method_id"] == ssa) & (df["scenario"] == sid)
                             & (df["K"].astype(int) == int(k))]["net_advantage"].iloc[0])
                b = float(df[(df["method_id"] == "FIBMM") & (df["scenario"] == sid)
                             & (df["K"].astype(int) == int(k))]["net_advantage"].iloc[0])
                ba = float(df[(df["method_id"] == ssa) & (df["scenario"] == sid)
                              & (df["K"].astype(int) == int(k))]["blocked_B"].iloc[0])
                bb = float(df[(df["method_id"] == "FIBMM") & (df["scenario"] == sid)
                              & (df["K"].astype(int) == int(k))]["blocked_B"].iloc[0])
                diffs_net.append(a - b)
                diffs_block.append(ba - bb)
                if abs(a - b) <= tie_tol:
                    ties += 1
                elif a > b:
                    wins += 1
                else:
                    losses += 1
        rows.append({
            "ssa_method": ssa,
            "fibm_method": "FIBMM",
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
            "mean_optimization_simulations": float(sub["optimization_simulations"].mean()),
            "mean_optimization_objective_evaluations": float(
                pd.to_numeric(sub["optimization_objective_evaluations"], errors="coerce").mean()
            ),
        })
    return pd.DataFrame(rows).sort_values("mean_selection_runtime_sec", ascending=False)
