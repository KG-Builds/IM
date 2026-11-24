"""Small shared utilities: paths, hashing, deterministic seeds, atomic writes.

All writes are project-local and atomic (write to a temporary file inside the
same directory, then ``os.replace``), per the project safety rules.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
CPP_DIR = PROJECT_ROOT / "cpp_engine"
TESTS_DIR = PROJECT_ROOT / "tests"

NETWORK_CSV = DATA_DIR / "synthetic_network_1000_7500.csv"
NETWORK_METADATA = DATA_DIR / "network_metadata.json"
NETWORK_METADATA_V2 = DATA_DIR / "network_metadata_v2.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: Any, bits: int = 64) -> int:
    """Deterministic integer seed derived from arbitrary parts.

    Uses SHA-256 so the value is stable across Python processes and platforms.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big")
    if bits >= 64:
        return value
    return value % (1 << bits)


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_json(path: Path | str, obj: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, default=_json_default) + "\n")


def atomic_write_csv(df, path: Path | str, index: bool = False) -> Path:
    """Atomic CSV write. Uses pandas to_csv into a sibling temp file."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=index)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any):
    """JSON fallback for numpy scalars / arrays and Paths."""
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "timestamp": local_now_iso(),
    }
    for mod in ("numpy", "pandas", "networkx", "matplotlib", "scipy", "pybind11"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - environment dependent
            info[mod] = f"MISSING ({type(exc).__name__})"
    return info
