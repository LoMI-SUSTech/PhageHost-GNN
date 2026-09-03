from typing import Optional
import numpy as np
from torch_geometric.data import HeteroData


PHAGE_MP = ("phage", "similar_to", "phage")
HOST_MP = ("host", "similar_to", "host")
def same_type_knn_edges(X: np.ndarray, k: int = 10, source_mask: Optional[np.ndarray] = None, target_mask: Optional[np.ndarray] = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n_nodes = X.shape[0]
    if n_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    # L2 normalization makes dot product equal cosine similarity
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    similarity = X_norm @ X_norm.T

    # excluding the self-neighbors
    np.fill_diagonal(similarity, -np.inf)

    if source_mask is None:
        source_idx = np.arange(n_nodes)
    else:
        source_idx = np.flatnonzero(
            np.asarray(source_mask, dtype=bool)
        )

    if target_mask is None:
        target_idx = np.arange(n_nodes)
    else:
        target_idx = np.flatnonzero(
            np.asarray(target_mask, dtype=bool)
        )

    if source_idx.size == 0 or target_idx.size == 0:
        return np.zeros((2, 0), dtype=np.int64)

    edges = []

    for target in target_idx:

        candidate_sources = source_idx[source_idx != target]
        if candidate_sources.size == 0:
            continue

        target_scores = similarity[target, candidate_sources]
        current_k = min(max(1, int(k)), candidate_sources.size)
        top_local = np.argpartition(-target_scores, current_k - 1)[:current_k]
        selected_sources = candidate_sources[top_local]

        for source in selected_sources:
            edges.append(
                (
                    int(source),
                    int(target),
                )
            )

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    return np.asarray(edges, dtype=np.int64).T


def build_heterodata(phage_x, host_x, phage_edges, host_edges):
    data = HeteroData()
    data["phage"].x = phage_x
    data["host"].x = host_x
    data[PHAGE_MP].edge_index = phage_edges
    data[HOST_MP].edge_index = host_edges
    return data

