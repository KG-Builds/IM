"""Move clearly identified old/incorrect result artefacts into to_be_discarded/.

No file is deleted. Every move is recorded in a manifest. Destination names are
made unique if a collision would occur.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "exp_20261124_5000_reversed"
DEST = EXP / "to_be_discarded"
MANIFEST_CSV = DEST / "MOVE_MANIFEST.csv"
MANIFEST_JSON = DEST / "MOVE_MANIFEST.json"
README = DEST / "README.md"

REASON = (
    "Produced from the pre-correction 1000-node, same-direction simulation "
    "(or derived from those results). Not valid for the 5000-node, "
    "reverse-influence experiment."
)


def unique_dest(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.name}__dup{i}")
        if not candidate.exists():
            return candidate
        i += 1


def move(source: Path, dest_parent: Path, reason: str, confirmed: str) -> dict | None:
    if not source.exists():
        return None
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = unique_dest(dest_parent / source.name)
    source.rename(dest)
    return {
        "original_path": str(source.relative_to(ROOT)),
        "new_path": str(dest.relative_to(ROOT)),
        "reason": reason,
        "confirmed_old_or_derived": confirmed,
        "moved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # Old full/smoke result directories.
    old_results = DEST / "old_results"
    for name in ("run_20261123_202600", "run_20261123_205500_reference_aligned", "smoke"):
        rec = move(ROOT / "results" / name, old_results, REASON, "yes")
        if rec:
            records.append(rec)

    # Old analysis tables and figures.
    for kind in ("tables", "figures"):
        src_root = ROOT / "analysis" / kind
        if src_root.exists():
            for child in sorted(src_root.iterdir()):
                if child.is_dir():
                    rec = move(child, DEST / "old_analysis" / kind, REASON, "yes")
                    if rec:
                        records.append(rec)

    # Old run/console logs.
    old_logs = DEST / "old_logs"
    for pattern in (
        "run_20261123_*.log",
        "full_run_console.log",
        "full_run_reference_aligned_console.log",
        "SMOKE_TEST_*.log",
    ):
        for child in sorted((ROOT / "logs").glob(pattern)):
            rec = move(child, old_logs, REASON, "yes")
            if rec:
                records.append(rec)

    # Old result summaries / gates / audits that are derived from old results.
    old_reports = DEST / "old_reports"
    for name in (
        "RESULTS_SUMMARY.md",
        "RESULTS_SUMMARY_REFERENCE_ALIGNED.md",
        "FINAL_AUDIT.md",
        "FINAL_AUDIT_REFERENCE_ALIGNED.md",
        "VALIDATION_GATE.md",
        "VALIDATION_GATE_REFERENCE_ALIGNED.md",
    ):
        rec = move(ROOT / name, old_reports, REASON, "yes (derived from old results)")
        if rec:
            records.append(rec)

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()) if records else [
            "original_path", "new_path", "reason", "confirmed_old_or_derived", "moved_utc"
        ])
        writer.writeheader()
        writer.writerows(records)
    MANIFEST_JSON.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# to_be_discarded — old/incorrect outputs",
        "",
        "These files came from the pre-correction 1000-node, same-direction",
        "simulation, or were derived from those results. They are retained only",
        "for audit and must not be used in any new calculation, plot, comparison,",
        "conclusion, or statistical analysis.",
        "",
        "Nothing here was deleted. See `MOVE_MANIFEST.csv` and",
        "`MOVE_MANIFEST.json` for the original path, new path, reason and",
        "confirmation status of every move.",
        "",
        f"Total moved items: {len(records)}",
        "",
    ]
    README.write_text("\n".join(lines), encoding="utf-8")
    print(f"Moved {len(records)} items. Manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
