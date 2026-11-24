import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_network(n_nodes=1000, n_edges=7500, seed=42,
                              topology="barabasi_albert", sign_positive_probability=0.5):
    """Create a fixed directed synthetic signed network with unique edges and trust ratings [-10,10] excluding zero."""
    rng = np.random.default_rng(seed)
    nodes = np.arange(n_nodes, dtype=int)
    edges = set()

    if topology == "erdos_renyi":
        # Directed edge sampling without self-loops; exact edge count.
        possible = n_nodes * (n_nodes - 1)
        picks = rng.choice(possible, size=n_edges, replace=False)
        for x in picks:
            src, dst = divmod(int(x), n_nodes-1)
            if dst >= src: dst += 1
            edges.add((src, dst))
    elif topology == "barabasi_albert":
        # Generate undirected BA backbone, then orient each unique edge randomly.
        m = max(1, min(7, n_edges//n_nodes))
        m0 = min(n_nodes, m+1)
        # Simple preferential attachment.
        degree = np.zeros(n_nodes, dtype=int)
        for u in range(1, m0):
            for v in range(u):
                edges.add((u, v))
                degree[u] += 1; degree[v] += 1
        for u in range(m0, n_nodes):
            candidates = np.arange(u)
            weights = degree[:u].astype(float)
            if weights.sum() == 0: weights[:] = 1
            targets = rng.choice(candidates, size=min(m,u), replace=False, p=weights/weights.sum())
            for v in targets:
                edges.add((u, int(v)))
                degree[u] += 1; degree[v] += 1
        # Trim or add randomly to get exactly n_edges undirected pairs, then randomly orient.
        edge_list = list(edges)
        if len(edge_list) > n_edges:
            edge_list = [edge_list[i] for i in rng.choice(len(edge_list), n_edges, replace=False)]
        elif len(edge_list) < n_edges:
            all_candidates = [(u,v) for u in range(n_nodes) for v in range(u+1,n_nodes) if (u,v) not in edges]
            extra = rng.choice(len(all_candidates), n_edges-len(edge_list), replace=False)
            edge_list += [all_candidates[int(i)] for i in extra]
        edges = set()
        for u,v in edge_list:
            if rng.random() < .5: candidate=(u,v)
            else: candidate=(v,u)
            edges.add(candidate)
        # Orientation may create duplicate directed pairs or reverse-pair collisions.
        # Add randomly sampled directed pairs until the requested exact edge count is reached.
        while len(edges) < n_edges:
            src = int(rng.integers(0, n_nodes))
            dst = int(rng.integers(0, n_nodes))
            if src != dst:
                edges.add((src,dst))
        if len(edges) > n_edges:
            edges = set(list(edges)[:n_edges])
    else:
        raise ValueError("topology must be 'barabasi_albert' or 'erdos_renyi'")

    if len(edges) != n_edges:
        raise RuntimeError(f"Expected exactly {n_edges} unique edges, got {len(edges)}")
    rows=[]
    for src,dst in sorted(edges):
        positive = rng.random() < sign_positive_probability
        magnitude = int(rng.integers(1,11))
        rating = magnitude if positive else -magnitude
        prob = 0.1 if magnitude <= 3 else (0.5 if magnitude <= 7 else 0.9)
        rows.append((src,dst,rating,prob,abs(rating),1 if rating>0 else -1))
    return pd.DataFrame(rows, columns=["source","target","rating","probability","strength","polarity"])

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    network = generate_synthetic_network()
    network.to_csv(out/"synthetic_network_1000_7500.csv", index=False)
    meta = {"nodes":1000,"edges":7500,"seed":42,"topology":"barabasi_albert","sign_positive_probability":0.5,
            "trust_magnitude_distribution":"uniform integer 1..10",
            "probability_mapping":{"1-3":0.1,"4-7":0.5,"8-10":0.9}}
    import json
    (out/"network_metadata.json").write_text(json.dumps(meta, indent=2))
    print(network.head(), network.shape)
