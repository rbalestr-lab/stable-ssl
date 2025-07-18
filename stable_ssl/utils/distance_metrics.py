"""Distance metric functions for computing pairwise distances between tensors."""

from typing import Literal

import torch


def compute_pairwise_distances(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: Literal[
        "euclidean", "squared_euclidean", "cosine", "manhattan"
    ] = "euclidean",
) -> torch.Tensor:
    """Compute pairwise distances between two sets of vectors.

    Args:
        x: Tensor of shape (n, d) containing n vectors of dimension d
        y: Tensor of shape (m, d) containing m vectors of dimension d
        metric: Distance metric to use. Options:
            - "euclidean": L2 distance
            - "squared_euclidean": Squared L2 distance
            - "cosine": Cosine distance (1 - cosine_similarity)
            - "manhattan": L1 distance

    Returns:
        Distance matrix of shape (n, m) where element (i, j) is the distance
        between x[i] and y[j]
    """
    if metric == "euclidean":
        return torch.cdist(x, y, p=2)

    elif metric == "squared_euclidean":
        return torch.cdist(x, y, p=2).pow(2)

    elif metric == "cosine":
        # Normalize vectors to unit length
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
        # Cosine similarity = dot product of normalized vectors
        cosine_sim = torch.mm(x_norm, y_norm.t())
        # Cosine distance = 1 - cosine similarity
        return 1 - cosine_sim

    elif metric == "manhattan":
        return torch.cdist(x, y, p=1)

    raise ValueError(
        f"Unknown metric: {metric}. Must be one of: euclidean, squared_euclidean, cosine, manhattan"
    )


def compute_pairwise_distances_chunked(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: Literal[
        "euclidean", "squared_euclidean", "cosine", "manhattan"
    ] = "euclidean",
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Memory-efficient computation of pairwise distances using chunking.

    Args:
        x: Tensor of shape (n, d) containing n vectors of dimension d
        y: Tensor of shape (m, d) containing m vectors of dimension d
        metric: Distance metric to use
        chunk_size: Process y in chunks of this size to save memory

    Returns:
        Distance matrix of shape (n, m)
    """
    n = x.shape[0]
    m = y.shape[0]
    device = x.device

    # Allocate output matrix
    distances = torch.empty(n, m, device=device, dtype=x.dtype)

    # Process y in chunks
    for i in range(0, m, chunk_size):
        end_idx = min(i + chunk_size, m)
        y_chunk = y[i:end_idx]
        distances[:, i:end_idx] = compute_pairwise_distances(x, y_chunk, metric)

    return distances
