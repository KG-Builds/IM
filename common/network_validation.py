"""Load, validate and fingerprint the shared signed network.

The network file is treated as immutable input. Nothing in this module writes
to it; metadata is written to a separate versioned file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    from . import config
    from .util import atomic_write_json, sha256_file
except ImportError:  # flat import when common/ is on sys.path
    import config
    from util import atomic_write_json, sha256_file

REQUIRED_COLUMNS = ["source", "target", "rating", "probability", "strength", "polarity"]


def _expected_probability(magnitude: int) -> float:
    if 1 <= magnitude <= 3:
        return 0.1
    if 4 <= magnitude <= 7:
        return 0.5
    if 8 <= magnitude <= 10:
        return 0.9
    raise ValueError(f"Trust magnitude {magnitude} outside 1..10")


def network_fingerprint(df: pd.DataFrame) -> str:
    """Content-based fingerprint, independent of row order and file formatting."""
    ordered = df[["source", "target", "rating", "probability"]].sort_values(
        ["source", "target"], kind="mergesort"
    )
    h = hashlib.sha256()
    h.update(b"src,dst,rating,probability\n")
    for row in ordered.itertuples(index=False):
        h.update(f"{int(row.source)},{int(row.target)},{int(row.rating)},{float(row.probability):.6f}\n".encode())
    return h.hexdigest()


def validate_network(
    df: pd.DataFrame,
    n_nodes: int | None = None,
    n_arcs: int | None = None,
    strict: bool = True,
) -> dict:
    """Validate structural and attribute invariants. Returns a statistics dict.

    Raises ``ValueError`` with a precise message when an invariant fails and
    ``strict`` is true.
    """
    n_nodes = config.NETWORK_NODES if n_nodes is None else int(n_nodes)
    n_arcs = config.NETWORK_ARCS if n_arcs is None else int(n_arcs)
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Network missing required columns: {missing}")
    if len(df) != n_arcs:
        problems.append(f"expected {n_arcs} rows/arcs, found {len(df)}")

    src = df["source"].astype(int)
    dst = df["target"].astype(int)
    nodes_present = set(src.unique()) | set(dst.unique())
    if src.min() < 0 or dst.min() < 0 or src.max() >= n_nodes or dst.max() >= n_nodes:
        problems.append(f"node ids outside 0..{n_nodes - 1}")
    if len(nodes_present) != n_nodes:
        problems.append(f"expected {n_nodes} distinct nodes, found {len(nodes_present)}")

    if (src == dst).any():
        problems.append("self-loops present")
    duplicates = df.duplicated(subset=["source", "target"]).sum()
    if duplicates:
        problems.append(f"{int(duplicates)} duplicate directed arcs")

    strength = df["strength"].astype(int)
    polarity = df["polarity"].astype(int)
    rating = df["rating"].astype(int)
    probability = df["probability"].astype(float)

    if ((strength < 1) | (strength > 10)).any():
        problems.append("trust magnitude outside 1..10")
    if (~polarity.isin([-1, 1])).any():
        problems.append("polarity must be -1 or +1")
    if (rating != polarity * strength).any():
        problems.append("rating != polarity * strength")
    expected = strength.map(_expected_probability)
    if not np.allclose(probability.to_numpy(), expected.to_numpy()):
        problems.append("probability does not follow the 1-3/4-7/8-10 mapping")

    if problems and strict:
        raise ValueError("Network validation failed: " + "; ".join(problems))

    reciprocals = len(set(zip(src, dst)) & set(zip(dst, src)))
    return {
        "rows": int(len(df)),
        "nodes": int(len(nodes_present)),
        "unique_directed_arcs": int(df.drop_duplicates(["source", "target"]).shape[0]),
        "self_loops": int((src == dst).sum()),
        "reciprocal_arc_pairs": int(reciprocals),
        "positive_edges": int((polarity > 0).sum()),
        "negative_edges": int((polarity < 0).sum()),
        "probability_counts": {str(k): int(v) for k, v in probability.value_counts().sort_index().items()},
        "magnitude_counts": {str(int(k)): int(v) for k, v in strength.value_counts().sort_index().items()},
        "out_degree": {
            "min": int(df.groupby("source").size().min()),
            "max": int(df.groupby("source").size().max()),
            "mean": float(df.groupby("source").size().mean()),
        },
        "in_degree": {
            "min": int(df.groupby("target").size().min()),
            "max": int(df.groupby("target").size().max()),
            "mean": float(df.groupby("target").size().mean()),
        },
        "fingerprint": network_fingerprint(df),
        "problems": problems,
    }


def load_network_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    return df


def load_network(path: str | Path):
    """Return ``(df, DiGraph, sorted_node_ids)``.

    Edges carry ``probability``, ``polarity`` and ``strength``. The graph is a
    directed graph with exactly one arc per row; ``nodes`` contains every node
    id, including nodes that only appear as targets.
    """
    df = load_network_df(path)
    nodes = sorted(set(df["source"]) | set(df["target"]))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for row in df.itertuples(index=False):
        g.add_edge(
            int(row.source),
            int(row.target),
            probability=float(row.probability),
            polarity=int(row.polarity),
            strength=int(row.strength),
            rating=int(row.rating),
        )
    return df, g, nodes


def graph_summary(df: pd.DataFrame) -> dict:
    """Topology summary used in metadata/reporting (Louvain is deterministic)."""
    _, g, nodes = load_network_from_df(df)
    und = nx.Graph()
    und.add_nodes_from(nodes)
    und.add_edges_from(g.edges())
    weak = list(nx.weakly_connected_components(g))
    strong = list(nx.strongly_connected_components(g))
    communities = list(nx.community.louvain_communities(und, seed=config.NETWORK_SEED))
    sizes = sorted((len(c) for c in communities), reverse=True)
    return {
        "weak_components": len(weak),
        "largest_weak_component": max(len(c) for c in weak) if weak else 0,
        "strong_components": len(strong),
        "largest_strong_component": max(len(c) for c in strong) if strong else 0,
        "louvain_communities": len(communities),
        "louvain_sizes_top10": sizes[:10],
    }


def load_network_from_df(df: pd.DataFrame):
    nodes = sorted(set(df["source"].astype(int)) | set(df["target"].astype(int)))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for row in df.itertuples(index=False):
        g.add_edge(
            int(row.source),
            int(row.target),
            probability=float(row.probability),
            polarity=int(row.polarity),
            strength=int(row.strength),
            rating=int(row.rating),
        )
    return df, g, nodes


def write_metadata_v2(network_path: str | Path, metadata_path: str | Path | None = None) -> dict:
    """Write a versioned metadata file; never touches the original metadata file."""
    network_path = Path(network_path)
    metadata_path = Path(metadata_path) if metadata_path else network_path.parent / "network_metadata_v2.json"
    df = load_network_df(network_path)
    stats = validate_network(df)
    meta = {
        "network_file": network_path.name,
        "file_sha256": sha256_file(network_path),
        "content_fingerprint": stats["fingerprint"],
        "fingerprint_definition": "sha256 over sorted 'source,target,rating,probability' rows",
        "edge_count_semantics": (
            "7500 rows = 7500 unique directed arcs. Because each underlying unordered "
            "pair of the BA backbone is oriented exactly once, 7500 arcs mostly equal "
            "7500 unordered ties; 4 reciprocal arc pairs were added by the final "
            "random-arc fill step. No self-loops. No duplicate directed arcs."
        ),
        "nodes": stats["nodes"],
        "arcs": stats["rows"],
        "generation_seed": config.NETWORK_SEED,
        "topology": "barabasi_albert_style_backbone_plus_random_oriented_pairs",
        "edge_orientation": "uniformly random per undirected pair, then random directed fill",
        "positive_sign_probability": config.NETWORK_SIGN_POSITIVE_PROB,
        "trust_magnitude_distribution": "uniform integer 1..10",
        "rating_to_transmission_probability": config.PROBABILITY_MAPPING,
        "positive_edges": stats["positive_edges"],
        "negative_edges": stats["negative_edges"],
        "probability_counts": stats["probability_counts"],
        "magnitude_counts": stats["magnitude_counts"],
        "out_degree": stats["out_degree"],
        "in_degree": stats["in_degree"],
        "reciprocal_arc_pairs": stats["reciprocal_arc_pairs"],
        "reproducibility": "regenerating with seed 42 from common/network_generator.py yields a byte-identical CSV",
        **graph_summary(df),
    }
    atomic_write_json(metadata_path, meta)
    return meta
