"""
PatchCore anomaly scoring module.

Computes anomaly scores via kNN distance to memory bank.
Runs on onnxruntime (no PyTorch needed at inference).

Supports optional PCA dimension reduction for speed:
  - Backbone outputs 448-dim features
  - PCA projects to 128-dim (or custom target)
  - kNN runs on reduced dims → much faster on RISC-V
"""
import numpy as np
from typing import Tuple, Optional


def compute_anomaly_scores(
    features: np.ndarray,
    memory_bank: np.ndarray,
    top_k: int = 5,
    pca_matrix: Optional[np.ndarray] = None,
    pca_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute patch-level distances and image-level anomaly score.

    Args:
        features: (N_patches, D) — feature vectors from test image
        memory_bank: (M, D_pca) — L2-normalized memory bank (PCA-reduced if pca_matrix given)
        top_k: number of top distances to average for image score
        pca_matrix: (D, D_pca) — PCA projection matrix (optional)
        pca_mean: (D,) — PCA mean vector (optional, required if pca_matrix given)

    Returns:
        patch_distances: (N_patches,) — min distance for each patch
        image_score: float — top-k average of patch distances (anomaly score)
    """
    # Optional PCA reduction
    if pca_matrix is not None and pca_mean is not None:
        features = (features - pca_mean) @ pca_matrix  # (N, D_pca)

    # L2 normalize
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # Cosine distance via dot product: d = 2 - 2*<a,b>
    N = features_norm.shape[0]
    M = memory_bank.shape[0]
    chunk_size = 8000  # larger chunks since dims are smaller

    min_distances = np.full(N, np.inf, dtype=np.float32)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        bank_chunk = memory_bank[start:end]  # (chunk, D)
        sim = np.dot(features_norm, bank_chunk.T)  # (N, chunk)
        dist = 2.0 - 2.0 * sim
        chunk_min = np.min(dist, axis=1)
        min_distances = np.minimum(min_distances, chunk_min)

    # Image-level anomaly score: average of top-k patch distances
    sorted_distances = np.sort(min_distances)[::-1]
    image_score = float(np.mean(sorted_distances[:top_k]))

    return min_distances, image_score


def compute_distance_matrix(
    features: np.ndarray,
    memory_bank: np.ndarray,
) -> np.ndarray:
    """
    Compute full pairwise cosine distance matrix.
    For advanced analysis (AUROC curves, debugging).

    Args:
        features: (N, D)
        memory_bank: (M, D)

    Returns:
        distances: (N, M)
    """
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(features_norm, memory_bank.T)
    return 2.0 - 2.0 * sim
