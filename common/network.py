"""Shared signed-network loading, validation, fingerprinting and metadata.

The generated CSV is treated as immutable source data.  This module never
regenerates or overwrites it during normal experiment runs; it only reads it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

try:  # package import
    from .utils import (
        NETWORK_CSV,
        NETWORK_METADATA,
        NETWORK_METADATA_V2,
        atomic_write_json,
        sha256_bytes,
        sha256_file,
        utc_now_iso,
    )
except ImportError:  # top-level import with common/ on sys.path
    from utils import (  # type: ignore
        NETWORK_CSV,
        NETWORK_METADATA,
        NETWORK_METADATA_V2,
        atomic_write_json,
        sha256_bytes,
        sha256_file,
        utc_now_iso,
    )

REQUIRED_COLUMNS = ["source", "target", "rating", "probability", "strength", "polarity"]

# Project assumption, not a canonical model parameter (see master prompt).
PROBABILITY_MAPPING = {"1-3": 0.1, "4-7": 0.5, "8-10": 0.9}


def probability_for_strength(strength: int) -> float:
    strength = int(strength)
    if 1 <= strength <= 3:
        return 0.1
    if 4 <= strength <= 7:
        return 0.5
    if 8 <= strength <= 10:
        return 0.9
    raise ValueError(f"strength must be in 1..10, got {strength}")


def load_network(path: Path | str = NETWORK_CSV, validate: bool = True,
                 n_nodes: int = 1000, n_edges: int = 7500):
    """Load the signed trust network.

    Returns ``(df, G, nodes)`` where ``G`` is the **trust-direction**
    ``networkx.DiGraph`` with ``probability``, ``polarity`` and ``strength``
    edge attributes and ``nodes`` is the sorted list of integer node ids.

    This function does **not** reverse edges. Use ``load_influence_network`` for
    the diffusion/experiment graph.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Network CSV is missing required columns: {missing}")
    for col in ("source", "target", "rating", "strength", "polarity"):
        df[col] = df[col].astype(int)
    df["probability"] = df["probability"].astype(float)

    if validate:
        validate_network(df, n_nodes=n_nodes, n_edges=n_edges)

    nodes = sorted(set(df["source"].tolist()) | set(df["target"].tolist()))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for row in df.itertuples(index=False):
        G.add_edge(
            int(row.source),
            int(row.target),
            probability=float(row.probability),
            polarity=int(row.polarity),
            strength=int(row.strength),
            rating=int(row.rating),
        )
    return df, G, nodes


def load_influence_network(path: Path | str = NETWORK_CSV, validate: bool = True,
                           n_nodes: int = 1000, n_edges: int = 7500,
                           reverse: bool = True):
    """Load the network and build the **influence-direction** graph.

    The CSV records trust edges ``source -> target``.  By default the influence
    graph reverses every edge to ``target -> source``, preserving the same
    probability, polarity, strength and rating attributes.  The reversal is
    applied exactly once here; downstream components must use the returned
    graph as-is.

    Returns ``(trust_df, influence_G, nodes)``.
    """
    df, _trust_G, nodes = load_network(path, validate=validate,
                                       n_nodes=n_nodes, n_edges=n_edges)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for row in df.itertuples(index=False):
        u = int(row.source)
        v = int(row.target)
        src, dst = (v, u) if reverse else (u, v)
        G.add_edge(
            src,
            dst,
            probability=float(row.probability),
            polarity=int(row.polarity),
            strength=int(row.strength),
            rating=int(row.rating),
        )
    return df, G, nodes


def influence_semantic_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of the edge table after applying the trust->influence reversal."""
    required = ["source", "target", "rating", "probability", "strength", "polarity"]
    view = df[required].copy()
    view["source"], view["target"] = df["target"].astype(int), df["source"].astype(int)
    return semantic_fingerprint(view)


def direction_check(df: pd.DataFrame, G_influence: nx.DiGraph) -> dict:
    """Verify that every trust edge appears exactly reversed in the influence graph."""
    trust_edges = {(int(r.source), int(r.target)) for r in df.itertuples(index=False)}
    influence_edges = {(int(u), int(v)) for u, v in G_influence.edges()}
    expected_influence = {(v, u) for (u, v) in trust_edges}
    missing = sorted(expected_influence - influence_edges)
    extra = sorted(influence_edges - expected_influence)
    same_direction = sorted(trust_edges & influence_edges)
    reciprocal_trust = 0
    for u, v in trust_edges:
        if (v, u) in trust_edges:
            reciprocal_trust += 1
    reciprocal_trust //= 2
    return {
        "trust_edges": len(trust_edges),
        "influence_edges": len(influence_edges),
        "expected_influence_edges": len(expected_influence),
        "missing_reversed_edges": len(missing),
        "extra_influence_edges": len(extra),
        "same_direction_edges": len(same_direction),
        "reciprocal_trust_pairs": int(reciprocal_trust),
        "all_trust_edges_reversed": len(missing) == 0 and len(extra) == 0,
        "influence_direction": "reverse_of_trust",
        "examples_missing": missing[:5],
        "examples_extra": extra[:5],
    }


def build_influence_metadata_v2(df: pd.DataFrame, trust_path: Path | str,
                                out_path: Path | str, n_nodes: int, n_edges: int) -> dict:
    """Write complete metadata for the trust network and its reversed influence graph."""
    trust_path = Path(trust_path)
    report = validate_network(df, n_nodes=n_nodes, n_edges=n_edges)
    _df, G_inf, nodes = load_influence_network(
        trust_path, validate=True, n_nodes=n_nodes, n_edges=n_edges, reverse=True
    )
    dc = direction_check(df, G_inf)
    out_degree = pd.Series([G_inf.out_degree(u) for u in nodes])
    in_degree = pd.Series([G_inf.in_degree(u) for u in nodes])
    weak = list(nx.weakly_connected_components(G_inf))
    strong = list(nx.strongly_connected_components(G_inf))
    meta = {
        "schema_version": 3,
        "generated_utc": utc_now_iso(),
        "trust_file": trust_path.name,
        "trust_file_sha256": sha256_file(trust_path),
        "trust_semantic_sha256": semantic_fingerprint(df),
        "influence_semantic_sha256": influence_semantic_fingerprint(df),
        "trust_direction": "source_to_target (as recorded in the CSV)",
        "influence_direction": "reverse_of_trust (target_to_source)",
        "node_count": int(len(nodes)),
        "trust_arc_count": int(len(df)),
        "influence_arc_count": int(G_inf.number_of_edges()),
        "underlying_unordered_pairs": int(report["n_underlying_unordered_pairs"]),
        "reciprocal_trust_pairs": int(dc["reciprocal_trust_pairs"]),
        "direction_check": dc,
        "trust_validation": report,
        "influence_degree": {
            "out": _degree_stats(np.asarray([G_inf.out_degree(u) for u in nodes])),
            "in": _degree_stats(np.asarray([G_inf.in_degree(u) for u in nodes])),
        },
        "influence_connectivity": {
            "n_weak_components": len(weak),
            "largest_weak_component": int(max((len(c) for c in weak), default=0)),
            "n_strong_components": len(strong),
            "largest_strong_component": int(max((len(c) for c in strong), default=0)),
            "is_weakly_connected": bool(nx.is_weakly_connected(G_inf)),
        },
        "notes": [
            "The CSV is the trust network; influence edges are the exact reverse.",
            "All downstream graph operations (diffusion, degree, PageRank, "
            "community ranking, candidate pool, SSA, Swap) use the influence graph.",
            "The reversal is applied once at load time; no component reverses it again.",
        ],
    }
    atomic_write_json(out_path, meta)
    return meta


def semantic_fingerprint(df: pd.DataFrame) -> str:
    """Order-independent SHA-256 of the edge table's semantic content."""
    required = ["source", "target", "rating", "probability", "strength", "polarity"]
    view = df[required].copy()
    view = view.sort_values(["source", "target"], kind="mergesort").reset_index(drop=True)
    lines = []
    for row in view.itertuples(index=False):
        lines.append(
            f"{int(row.source)},{int(row.target)},{int(row.rating)},"
            f"{float(row.probability):.12g},{int(row.strength)},{int(row.polarity)}"
        )
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def validate_network(df: pd.DataFrame, n_nodes: int = 1000, n_edges: int = 7500) -> dict:
    """Validate the shared network against the declared design.

    Raises ``ValueError`` on any violation and returns a report dictionary.
    """
    required = ["source", "target", "rating", "probability", "strength", "polarity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    src = df["source"].astype(int).to_numpy()
    dst = df["target"].astype(int).to_numpy()
    rating = df["rating"].astype(int).to_numpy()
    prob = df["probability"].astype(float).to_numpy()
    strength = df["strength"].astype(int).to_numpy()
    polarity = df["polarity"].astype(int).to_numpy()

    errors: list[str] = []

    if len(df) != n_edges:
        errors.append(f"expected {n_edges} rows, found {len(df)}")

    nodes = sorted(set(src.tolist()) | set(dst.tolist()))
    if len(nodes) != n_nodes:
        errors.append(f"expected {n_nodes} distinct nodes, found {len(nodes)}")
    if nodes and nodes != list(range(n_nodes)):
        errors.append("node ids are not contiguous 0..N-1")

    if np.any(src == dst):
        errors.append(f"{int(np.sum(src == dst))} self-loops found")

    pairs = pd.DataFrame({"source": src, "target": dst})
    dup = int(pairs.duplicated().sum())
    if dup:
        errors.append(f"{dup} duplicate directed arcs found")

    if not np.all(np.isin(polarity, [-1, 1])):
        errors.append("polarity values outside {-1, +1}")
    if not np.all((strength >= 1) & (strength <= 10)):
        errors.append("strength values outside 1..10")
    if not np.all(rating == polarity * strength):
        errors.append("rating != polarity * strength")

    expected_prob = np.array([probability_for_strength(s) for s in strength])
    if not np.allclose(prob, expected_prob, atol=1e-12, rtol=0):
        errors.append("probability does not match the declared 1-3/4-7/8-10 mapping")

    if errors:
        raise ValueError("Network validation failed:\n  - " + "\n  - ".join(errors))

    # ---- descriptive statistics -------------------------------------------
    out_deg = pd.Series(src).value_counts()
    in_deg = pd.Series(dst).value_counts()
    out_full = np.array([int(out_deg.get(u, 0)) for u in nodes])
    in_full = np.array([int(in_deg.get(u, 0)) for u in nodes])

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(zip(src.tolist(), dst.tolist()))

    undirected_pairs = {tuple(sorted((int(u), int(v)))) for u, v in zip(src, dst)}
    reciprocal = 0
    arc_set = set(zip(src.tolist(), dst.tolist()))
    for u, v in arc_set:
        if (v, u) in arc_set:
            reciprocal += 1
    reciprocal //= 2

    weak = list(nx.weakly_connected_components(G))
    strong = list(nx.strongly_connected_components(G))

    report = {
        "n_nodes": len(nodes),
        "n_directed_arcs": int(len(df)),
        "n_underlying_unordered_pairs": int(len(undirected_pairs)),
        "n_reciprocal_arc_pairs": int(reciprocal),
        "n_duplicate_arcs": dup,
        "n_self_loops": int(np.sum(src == dst)),
        "n_positive": int(np.sum(polarity > 0)),
        "n_negative": int(np.sum(polarity < 0)),
        "probability_class_counts": {
            "0.1": int(np.sum(np.isclose(prob, 0.1))),
            "0.5": int(np.sum(np.isclose(prob, 0.5))),
            "0.9": int(np.sum(np.isclose(prob, 0.9))),
        },
        "strength_distribution": {int(k): int(v) for k, v in pd.Series(strength).value_counts().sort_index().items()},
        "out_degree": _degree_stats(out_full),
        "in_degree": _degree_stats(in_full),
        "n_isolated": int(np.sum((out_full == 0) & (in_full == 0))),
        "n_sources": int(np.sum(in_full == 0)),
        "n_sinks": int(np.sum(out_full == 0)),
        "n_weak_components": len(weak),
        "largest_weak_component": int(max((len(c) for c in weak), default=0)),
        "n_strong_components": len(strong),
        "largest_strong_component": int(max((len(c) for c in strong), default=0)),
        "is_weakly_connected": bool(nx.is_weakly_connected(G)),
    }
    return report


def _degree_stats(values: np.ndarray) -> dict:
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
    }


def build_adjacency(G: nx.DiGraph, nodes: list[int]):
    """Build the deterministic per-node adjacency representation.

    Neighbours are sorted by node id so the C++ engine sees a stable edge order.
    """
    index = {node: i for i, node in enumerate(nodes)}
    adj, probs, pol, strength = [], [], [], []
    for u in nodes:
        nbrs = sorted(G.successors(u))
        adj.append([index[v] for v in nbrs])
        probs.append([float(G[u][v]["probability"]) for v in nbrs])
        pol.append([int(G[u][v]["polarity"]) for v in nbrs])
        strength.append([int(G[u][v]["strength"]) for v in nbrs])
    return index, adj, probs, pol, strength


def build_metadata_v2(df: pd.DataFrame, path: Path | str = NETWORK_CSV) -> dict:
    """Construct the complete v2 metadata record (does not write)."""
    path = Path(path)
    report = validate_network(df)
    meta = {
        "schema_version": 2,
        "generated_utc": utc_now_iso(),
        "network_file": path.name,
        "file_sha256": sha256_file(path),
        "semantic_sha256": semantic_fingerprint(df),
        "file_size_bytes": path.stat().st_size,
        "node_id_convention": "contiguous integers 0..999",
        "edge_count_convention": (
            "One CSV row = one directed arc. The design target is 7500 unique "
            "directed arcs. The generator first builds an undirected BA-style "
            "backbone of 7500 unordered pairs and orients each pair once; because "
            "the pair de-duplication filter is order-sensitive, a small number of "
            "unordered pairs can be added twice, collapse after orientation, and "
            "be replaced by the random directed fill loop. On the delivered file "
            "this yields 7500 unique directed arcs over 7498 underlying unordered "
            "pairs plus 2 reciprocal arc pairs (4 reciprocal arcs). The directed-"
            "arc target is therefore satisfied exactly; the unordered-pair count "
            "is 7498, not 7500. This dual meaning is explicit, not silently "
            "reinterpreted."
        ),
        "generation": {
            "seed": 42,
            "topology": "barabasi_albert_style_backbone_plus_random_directed_edges",
            "ba_m": 7,
            "ba_initial_clique": 8,
            "orientation": "each undirected pair oriented uniformly at random",
            "sign_positive_probability": 0.5,
            "trust_magnitude_distribution": "uniform integer 1..10",
            "rating_definition": "rating = polarity * strength",
            "rating_to_transmission_probability": PROBABILITY_MAPPING,
            "edge_count_target": 7500,
            "node_count_target": 1000,
        },
        "validation": report,
        "reproduction": {
            "generator": "common/network_generator.py::generate_synthetic_network",
            "parameters": {
                "n_nodes": 1000,
                "n_edges": 7500,
                "seed": 42,
                "topology": "barabasi_albert",
                "sign_positive_probability": 0.5,
            },
            "note": (
                "Regeneration with seed 42 reproduces the CSV byte-for-byte. "
                "The generator is pinned by tests and is intentionally not "
                "refactored, because changing its RNG call order would invalidate "
                "the shared network."
            ),
        },
        "notes": [
            "The network file is read-only source data for all experiments.",
            "No self-loops; duplicate directed arcs are rejected.",
            "Probability mapping is a project assumption, not a canonical model parameter.",
        ],
    }
    return meta


def write_metadata_v2(df: pd.DataFrame, path: Path | str = NETWORK_CSV,
                      out_path: Path | str = NETWORK_METADATA_V2) -> dict:
    meta = build_metadata_v2(df, path)
    atomic_write_json(out_path, meta)
    return meta


def verify_fingerprint(path: Path | str = NETWORK_CSV,
                       metadata_path: Path | str = NETWORK_METADATA_V2) -> dict:
    """Compare the network file against stored v2 metadata."""
    path = Path(path)
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {"ok": False, "reason": "metadata_v2_missing", "metadata_path": str(metadata_path)}
    import json

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_file = sha256_file(path)
    df = pd.read_csv(path)
    actual_semantic = semantic_fingerprint(df)
    ok = (
        actual_file == meta.get("file_sha256")
        and actual_semantic == meta.get("semantic_sha256")
    )
    return {
        "ok": bool(ok),
        "file_sha256": actual_file,
        "expected_file_sha256": meta.get("file_sha256"),
        "semantic_sha256": actual_semantic,
        "expected_semantic_sha256": meta.get("semantic_sha256"),
        "metadata_path": str(metadata_path),
    }


def regenerate_network_df():
    """Regenerate the network in memory with the pinned generator."""
    try:
        from .network_generator import generate_synthetic_network
    except ImportError:
        from network_generator import generate_synthetic_network  # type: ignore

    return generate_synthetic_network(n_nodes=1000, n_edges=7500, seed=42,
                                      topology="barabasi_albert")
