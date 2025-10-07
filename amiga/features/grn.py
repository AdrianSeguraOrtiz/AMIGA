"""
GRN (Gene Regulatory Network) feature extraction utilities.

This module provides:
- Builders and basic measures for weighted directed graphs (DiGraph).
- Global and node-level weighted descriptors (density, strengths, Gini).
- Path-based, clustering, community, and reciprocity metrics.
- Optional advanced compact summaries (entropies).

Notes
-----
- Edges are expected to carry a "Confidence" attribute (float weight).
- Community detection uses Louvain on an undirected projection with weights.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import typer

app = typer.Typer(add_completion=False)
EPS = 1e-9


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def gini(x: np.ndarray) -> float:
    """
    Gini coefficient for non-negative values.

    Parameters
    ----------
    x : np.ndarray
        1D array of non-negative values.

    Returns
    -------
    float
        Gini coefficient in [0, 1]. Returns NaN for empty input and 0.0 if the sum is 0.
    """
    x = x.astype(float)
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    total = cumx[-1]
    if total <= 0:
        return 0.0
    # Standard discrete Gini with sorted values
    return (n + 1 - 2 * np.sum(cumx) / total) / n


def build_digraph(
    df: pd.DataFrame,
    source: str = "Source",
    target: str = "Target",
    weight: str = "Confidence",
) -> nx.DiGraph:
    """
    Build a weighted directed graph from an edge list DataFrame.

    If duplicate edges appear, their weights are accumulated.

    Parameters
    ----------
    df : pd.DataFrame
        Edge list with columns [source, target, weight].
    source, target, weight : str
        Column names for source node, target node, and edge weight.

    Returns
    -------
    nx.DiGraph
        Directed graph with 'Confidence' as edge attribute.
    """
    G = nx.DiGraph()
    for _, r in df.iterrows():
        u, v, w = r[source], r[target], float(r[weight])
        if G.has_edge(u, v):
            G[u][v]["Confidence"] += w  # accumulate duplicates
        else:
            G.add_edge(u, v, Confidence=w)
    return G


def strengths(G: nx.DiGraph, mode: str = "in") -> Dict[Any, float]:
    """
    Compute weighted in-/out-strengths per node.

    Parameters
    ----------
    G : nx.DiGraph
        Weighted directed graph.
    mode : {"in", "out"}
        Strength type: sum of incoming ("in") or outgoing ("out") weights.

    Returns
    -------
    Dict[Any, float]
        Mapping node -> strength.
    """
    if mode == "in":
        return {n: sum(G[u][n]["Confidence"] for u in G.predecessors(n)) for n in G.nodes()}
    if mode == "out":
        return {n: sum(G[n][v]["Confidence"] for v in G.successors(n)) for n in G.nodes()}
    raise ValueError("mode must be 'in' or 'out'")


def weighted_density(G: nx.DiGraph) -> float:
    """
    Weighted density for directed graphs without self-loops.

    Defined as total edge weight divided by n*(n-1).

    Parameters
    ----------
    G : nx.DiGraph

    Returns
    -------
    float
    """
    W = sum(d["Confidence"] for _, _, d in G.edges(data=True))
    n = G.number_of_nodes()
    return W / max(n * (n - 1), 1)  # directed, no self-loops


def reciprocity_weighted(G: nx.DiGraph) -> float:
    """
    Weighted reciprocity using min of reciprocal edge weights.

    Returns
    -------
    float
        Sum over pairs min(w(u,v), w(v,u)) divided by total weight.
        If total weight is zero, returns 0.0.
    """
    w_pairs = 0.0
    for u, v, d in G.edges(data=True):
        if G.has_edge(v, u):
            w_pairs += min(d["Confidence"], G[v][u]["Confidence"])
    total = sum(d["Confidence"] for *_, d in G.edges(data=True))
    return w_pairs / total if total > 0 else 0.0


def shortest_path_stats(G: nx.DiGraph) -> Dict[str, float]:
    """
    Shortest-path descriptors on the giant strongly connected component (GSCC).

    Distances use inverse weight: dist(u,v) = 1 / (Confidence + EPS).

    Returns
    -------
    dict
        {
            "w_avg_spl": weighted average shortest path length,
            "w_diam_p95": 95th percentile of pairwise weighted distances
        }
        NaNs if graph is too small or disconnected.
    """
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return {"w_avg_spl": float("nan"), "w_diam_p95": float("nan")}
    giant = max(sccs, key=len)
    H = G.subgraph(giant).copy()
    if H.number_of_edges() == 0 or H.number_of_nodes() < 2:
        return {"w_avg_spl": float("nan"), "w_diam_p95": float("nan")}

    # Define edge distances as inverse of weight
    for u, v in H.edges():
        H[u][v]["dist"] = 1.0 / (H[u][v]["Confidence"] + EPS)

    dists = []
    for s in H:
        lengths = nx.single_source_dijkstra_path_length(H, s, weight="dist")
        dists.extend([l for t, l in lengths.items() if t != s])

    if not dists:
        return {"w_avg_spl": float("nan"), "w_diam_p95": float("nan")}

    dists = np.array(dists, dtype=float)
    return {"w_avg_spl": float(np.mean(dists)), "w_diam_p95": float(np.percentile(dists, 95))}


def community_stats_louvain(G: nx.DiGraph) -> Dict[str, float]:
    """
    Community statistics from Louvain on the undirected weighted projection.

    Parameters
    ----------
    G : nx.DiGraph

    Returns
    -------
    dict
        {
          "comm_n": number of communities,
          "comm_modularity_w": weighted modularity,
          "comm_size_mean": mean community size,
          "comm_size_max": max community size
        }
    """
    # Project to undirected with accumulated weights
    UG = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d["Confidence"]
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += w
        else:
            UG.add_edge(u, v, weight=w)

    if UG.number_of_edges() == 0:
        return {
            "comm_n": 0.0,
            "comm_modularity_w": float("nan"),
            "comm_size_mean": float("nan"),
            "comm_size_max": float("nan"),
        }

    part = community_louvain.best_partition(UG, weight="weight")
    comms: Dict[int, list] = {}
    for n, cid in part.items():
        comms.setdefault(cid, []).append(n)

    sizes = [len(nodes) for nodes in comms.values()]
    Q = community_louvain.modularity(part, UG, weight="weight")

    return {
        "comm_n": float(len(comms)),
        "comm_modularity_w": float(Q),
        "comm_size_mean": float(np.mean(sizes)) if sizes else float("nan"),
        "comm_size_max": float(max(sizes)) if sizes else float("nan"),
    }


# ---------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------

def features_grn_global(G: nx.DiGraph, weights: np.ndarray, top_frac: float) -> Dict[str, Any]:
    """
    Global weighted GRN descriptors.

    Parameters
    ----------
    G : nx.DiGraph
    weights : np.ndarray
        Vector of edge weights (Confidence).
    top_frac : float
        Fraction (0,1] used to compute the top-X weight concentration.

    Returns
    -------
    dict
        {
          "num_nodes", "num_edges", "total_weight", "avg_weight",
          "weighted_density", "weight_gini", "weight_top10_ratio"
        }
        Note: 'weight_top10_ratio' uses `top_frac` of edges (e.g., 0.10 → top 10%).
    """
    feats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "total_weight": float(weights.sum()),
        "avg_weight": float(weights.mean() if weights.size else float("nan")),
        "weighted_density": float(weighted_density(G)),
        "weight_gini": float(gini(weights) if weights.size else float("nan")),
    }
    if weights.size:
        k = max(1, int(weights.size * top_frac))
        feats["weight_top10_ratio"] = float(np.sort(weights)[-k:].sum() / weights.sum())
    else:
        feats["weight_top10_ratio"] = float("nan")
    return feats


def features_strength(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Strength-based statistics (in, out, and total).

    Returns
    -------
    dict
        Means, stds, maxima, and Gini indices for in/out/total strengths,
        plus a crude 'hub' count above mean+2*std on total strength.
    """
    sin = np.array(list(strengths(G, "in").values()), dtype=float)
    sout = np.array(list(strengths(G, "out").values()), dtype=float)
    s = sin + sout

    def stats(prefix, arr: np.ndarray) -> Dict[str, Any]:
        if arr.size == 0:
            return {f"{prefix}_{k}": float("nan") for k in ("mean", "std", "max", "gini")}
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_std": float(np.std(arr)),
            f"{prefix}_max": float(np.max(arr)),
            f"{prefix}_gini": float(gini(arr)),
        }

    feats = {}
    feats.update(stats("in_strength", sin))
    feats.update(stats("out_strength", sout))
    feats.update(stats("strength", s))

    if s.size:
        mu, sigma = float(np.mean(s)), float(np.std(s))
        feats["strength_hubs_n"] = int(np.sum(s > (mu + 2 * sigma)))
    else:
        feats["strength_hubs_n"] = 0
    return feats


def features_assortativity(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Weighted out→in degree assortativity.

    Returns
    -------
    dict
        {"assortativity_out_in_w": value or NaN if not computable}
    """
    try:
        r = nx.degree_assortativity_coefficient(G, x="out", y="in", weight="Confidence")
        return {"assortativity_out_in_w": float(r)}
    except Exception:
        return {"assortativity_out_in_w": float("nan")}


def features_paths(G: nx.DiGraph) -> Dict[str, Any]:
    """Convenience wrapper for weighted shortest-path statistics."""
    return shortest_path_stats(G)


def features_clustering(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Average weighted clustering on the undirected projection.

    Returns
    -------
    dict
        {"avg_clustering_w_undirected": value or NaN if graph has no edges}
    """
    if G.number_of_edges() == 0:
        return {"avg_clustering_w_undirected": float("nan")}
    val = nx.average_clustering(G.to_undirected(), weight="Confidence")
    return {"avg_clustering_w_undirected": float(val)}


def features_communities(G: nx.DiGraph) -> Dict[str, Any]:
    """Louvain-based community statistics on the undirected weighted projection."""
    return community_stats_louvain(G)


def features_reciprocity(G: nx.DiGraph) -> Dict[str, Any]:
    """Weighted reciprocity as min-weight overlap of reciprocal edges."""
    return {"reciprocity_weighted": float(reciprocity_weighted(G))}


def features_advanced(G: nx.DiGraph, weights: np.ndarray) -> Dict[str, Any]:
    """
    Optional compact summaries via entropies over distributions:
    - Edge weights (Confidence)
    - Total node strengths (in + out)

    Returns
    -------
    dict
        {"entropy_weights", "entropy_strength"}
    """
    # Total node strength distribution (in + out)
    s_tot = np.array(list(strengths(G, "in").values())) + np.array(list(strengths(G, "out").values()))

    def entropy(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        p = arr / (arr.sum() + EPS)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    return {
        "entropy_weights": entropy(weights),
        "entropy_strength": entropy(s_tot.astype(float)) if s_tot.size else float("nan"),
    }
