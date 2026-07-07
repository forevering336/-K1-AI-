"""
Memory bank construction with coreset subsampling for PatchCore.

Two strategies:
1. Greedy k-center coreset (exact PatchCore, slower)
2. MiniBatchKMeans clustering (approximate, much faster, good enough)

The memory bank is saved as a .npy file and loaded at inference time.
"""
import numpy as np
from typing import Optional
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


def greedy_coreset(
    features: np.ndarray,
    target_size: int,
    batch_size: int = 5000,
) -> np.ndarray:
    """
    Greedy approximation of k-center coreset selection (PatchCore paper).

    Args:
        features: (N, D) — all feature vectors
        target_size: M — desired coreset size
        batch_size: chunk size for distance computation (memory control)

    Returns:
        coreset: (M, D) — selected feature vectors
    """
    N, D = features.shape
    target_size = min(target_size, N)

    # Normalize features for cosine-like distance
    features_norm = normalize(features, norm="l2")

    # Track nearest center distance for each point
    min_distances = np.full(N, np.inf, dtype=np.float32)
    coreset_indices = []

    # First center: random
    first_idx = np.random.randint(0, N)
    coreset_indices.append(first_idx)

    # Update distances to first center
    first_center = features_norm[first_idx:first_idx+1]  # (1, D)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = features_norm[start:end]  # (chunk, D)
        # Cosine distance: 2 - 2 * dot(a, b)
        sim = np.dot(chunk, first_center.T)  # (chunk, 1)
        dist = 2.0 - 2.0 * sim.flatten()
        min_distances[start:end] = np.minimum(min_distances[start:end], dist)

    # Iteratively select remaining centers
    for i in tqdm(range(1, target_size), desc="Coreset sampling"):
        # Select point with maximum min-distance
        next_idx = int(np.argmax(min_distances))
        coreset_indices.append(next_idx)

        # Update min_distances with new center
        new_center = features_norm[next_idx:next_idx+1]  # (1, D)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk = features_norm[start:end]
            sim = np.dot(chunk, new_center.T)
            dist = 2.0 - 2.0 * sim.flatten()
            min_distances[start:end] = np.minimum(min_distances[start:end], dist)

    coreset = features[coreset_indices].astype(np.float32)
    return coreset


def kmeans_coreset(
    features: np.ndarray,
    target_size: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Approximate coreset using MiniBatchKMeans clustering centroids.
    Much faster than greedy coreset, empirically similar performance.

    Args:
        features: (N, D) — all feature vectors
        target_size: M — number of clusters (coreset size)
        random_state: random seed

    Returns:
        coreset: (M, D) — cluster centroids
    """
    target_size = min(target_size, features.shape[0])
    features_norm = normalize(features, norm="l2")

    kmeans = MiniBatchKMeans(
        n_clusters=target_size,
        random_state=random_state,
        batch_size=min(4096, features.shape[0]),
        n_init=3,
        max_iter=100,
    )
    kmeans.fit(features_norm)

    coreset = kmeans.cluster_centers_.astype(np.float32)
    # Re-normalize centroids
    coreset = normalize(coreset, norm="l2")
    return coreset


def build_memory_bank(
    features: np.ndarray,
    coreset_size: int = 10000,
    method: str = "kmeans",
    random_state: int = 42,
) -> np.ndarray:
    """
    Build a PatchCore memory bank from extracted features.

    Args:
        features: (N, D) — all feature vectors from good images
        coreset_size: target memory bank size
        method: "greedy" for exact PatchCore, "kmeans" for fast approximation
        random_state: seed

    Returns:
        memory_bank: (M, D) — L2-normalized coreset
    """
    N, D = features.shape
    print(f"Building memory bank: {N} patches x {D} dims -> coreset of {coreset_size}")

    if method == "greedy":
        coreset = greedy_coreset(features, coreset_size)
    elif method == "kmeans":
        coreset = kmeans_coreset(features, coreset_size, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Final L2 normalization
    coreset = normalize(coreset, norm="l2").astype(np.float32)

    print(f"Memory bank: {coreset.shape}, dtype={coreset.dtype}, "
          f"memory={coreset.nbytes / 1024 / 1024:.1f} MB")

    return coreset


def reduce_dimension_pca(
    features: np.ndarray,
    target_dim: int = 128,
    random_state: int = 42,
):
    """
    Reduce feature dimension with PCA.

    Args:
        features: (N, D) — original features
        target_dim: desired output dimension
        random_state: seed

    Returns:
        reduced_features: (N, target_dim)
        pca_matrix: (D, target_dim) — for inference-time projection
        pca_mean: (D,) — for inference-time centering
    """
    from sklearn.decomposition import PCA

    N, D = features.shape
    target_dim = min(target_dim, D, N)
    print(f"PCA: {D} -> {target_dim} dims")

    pca = PCA(n_components=target_dim, random_state=random_state)
    reduced = pca.fit_transform(features).astype(np.float32)

    # Explained variance
    total_var = np.sum(pca.explained_variance_ratio_)
    print(f"  Explained variance: {total_var:.1%}")

    pca_matrix = pca.components_.T.astype(np.float32)  # (D, target_dim)
    pca_mean = pca.mean_.astype(np.float32)             # (D,)

    return reduced, pca_matrix, pca_mean
