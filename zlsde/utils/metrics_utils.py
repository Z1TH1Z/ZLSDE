"""Metric computation helper functions."""

import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from typing import Optional


def compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score for clustering quality.
    
    The silhouette score measures how similar an object is to its own cluster
    compared to other clusters. Range: [-1, 1], higher is better.
    
    Args:
        embeddings: Data embeddings (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
    
    Returns:
        Silhouette score in range [-1, 1], or 0.0 if computation not possible
    
    Example:
        >>> embeddings = np.random.randn(100, 50)
        >>> labels = np.random.randint(0, 5, 100)
        >>> score = compute_silhouette_score(embeddings, labels)
    """
    # Filter out noise points (label -1)
    mask = labels >= 0
    if np.sum(mask) < 2:
        return 0.0
    
    filtered_embeddings = embeddings[mask]
    filtered_labels = labels[mask]
    
    # Need at least 2 clusters for silhouette score
    unique_labels = np.unique(filtered_labels)
    if len(unique_labels) < 2:
        return 0.0
    
    try:
        return float(silhouette_score(filtered_embeddings, filtered_labels))
    except Exception:
        return 0.0


def compute_cluster_purity(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute average intra-cluster coherence (purity).
    
    Measures how tightly grouped samples are within each cluster using
    average cosine similarity to cluster centroid.
    
    Args:
        embeddings: Data embeddings (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
    
    Returns:
        Average cluster purity in range [0, 1], higher is better
    
    Example:
        >>> embeddings = np.random.randn(100, 50)
        >>> labels = np.random.randint(0, 5, 100)
        >>> purity = compute_cluster_purity(embeddings, labels)
    """
    # Filter out noise points
    mask = labels >= 0
    if np.sum(mask) == 0:
        return 0.0
    
    filtered_embeddings = embeddings[mask]
    filtered_labels = labels[mask]
    
    unique_labels = np.unique(filtered_labels)
    if len(unique_labels) == 0:
        return 0.0
    
    purities = []
    for label in unique_labels:
        cluster_mask = filtered_labels == label
        cluster_embeddings = filtered_embeddings[cluster_mask]
        
        if len(cluster_embeddings) < 2:
            continue
        
        # Compute centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Compute cosine similarities to centroid
        norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
        centroid_norm = np.linalg.norm(centroid)
        
        if centroid_norm == 0 or np.any(norms == 0):
            continue
        
        normalized_embeddings = cluster_embeddings / norms
        normalized_centroid = centroid / centroid_norm
        
        similarities = np.dot(normalized_embeddings, normalized_centroid)
        purities.append(np.mean(similarities))
    
    return float(np.mean(purities)) if purities else 0.0


def compute_label_flip_rate(old_labels: np.ndarray, new_labels: np.ndarray) -> float:
    """
    Compute percentage of labels that changed between iterations.
    
    Args:
        old_labels: Previous iteration labels (n_samples,)
        new_labels: Current iteration labels (n_samples,)
    
    Returns:
        Flip rate in range [0, 1], lower indicates more stability
    
    Example:
        >>> old = np.array([0, 1, 2, 0, 1])
        >>> new = np.array([0, 1, 2, 1, 1])
        >>> rate = compute_label_flip_rate(old, new)
        >>> # Returns: 0.2 (1 out of 5 labels changed)
    """
    if len(old_labels) != len(new_labels):
        raise ValueError("Label arrays must have same length")
    
    if len(old_labels) == 0:
        return 0.0
    
    flipped = np.sum(old_labels != new_labels)
    return float(flipped / len(old_labels))


def compute_noise_ratio(labels: np.ndarray) -> float:
    """
    Compute percentage of samples labeled as noise.
    
    Args:
        labels: Cluster assignments (n_samples,), where -1 indicates noise
    
    Returns:
        Noise ratio in range [0, 1]
    
    Example:
        >>> labels = np.array([0, 1, -1, 0, -1, 2])
        >>> ratio = compute_noise_ratio(labels)
        >>> # Returns: 0.333... (2 out of 6 are noise)
    """
    if len(labels) == 0:
        return 0.0
    
    noise_count = np.sum(labels == -1)
    return float(noise_count / len(labels))


def compute_cluster_distribution(labels: np.ndarray) -> dict:
    """
    Compute distribution of samples across clusters.
    
    Args:
        labels: Cluster assignments (n_samples,)
    
    Returns:
        Dictionary mapping cluster_id to sample count
    
    Example:
        >>> labels = np.array([0, 1, 0, 2, 1, 0])
        >>> dist = compute_cluster_distribution(labels)
        >>> # Returns: {0: 3, 1: 2, 2: 1}
    """
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def compute_quality_statistics(quality_scores: np.ndarray) -> dict:
    """
    Compute statistics for quality scores.
    
    Args:
        quality_scores: Array of quality scores (n_samples,)
    
    Returns:
        Dictionary with mean, std, min, max, median
    
    Example:
        >>> scores = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
        >>> stats = compute_quality_statistics(scores)
        >>> # Returns: {"mean": 0.84, "std": 0.09, ...}
    """
    if len(quality_scores) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }
    
    return {
        "mean": float(np.mean(quality_scores)),
        "std": float(np.std(quality_scores)),
        "min": float(np.min(quality_scores)),
        "max": float(np.max(quality_scores)),
        "median": float(np.median(quality_scores))
    }


def compute_adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index between two label assignments.
    
    Useful for comparing clustering results or measuring label stability.
    Range: [-1, 1], where 1 indicates perfect agreement.
    
    Args:
        labels_true: Ground truth or reference labels
        labels_pred: Predicted or comparison labels
    
    Returns:
        Adjusted Rand Index score
    
    Example:
        >>> true = np.array([0, 0, 1, 1, 2, 2])
        >>> pred = np.array([0, 0, 1, 1, 1, 2])
        >>> ari = compute_adjusted_rand_index(true, pred)
    """
    # Filter out noise points from both
    mask = (labels_true >= 0) & (labels_pred >= 0)
    if np.sum(mask) < 2:
        return 0.0
    
    filtered_true = labels_true[mask]
    filtered_pred = labels_pred[mask]
    
    try:
        return float(adjusted_rand_score(filtered_true, filtered_pred))
    except Exception:
        return 0.0


def compute_normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information between two label assignments.
    
    Measures the mutual information between two clusterings normalized
    by their entropies. Range: [0, 1], where 1 indicates perfect agreement.
    
    Args:
        labels_true: Ground truth or reference labels
        labels_pred: Predicted or comparison labels
    
    Returns:
        Normalized Mutual Information score
    
    Example:
        >>> true = np.array([0, 0, 1, 1, 2, 2])
        >>> pred = np.array([0, 0, 1, 1, 1, 2])
        >>> nmi = compute_normalized_mutual_info(true, pred)
    """
    # Filter out noise points from both
    mask = (labels_true >= 0) & (labels_pred >= 0)
    if np.sum(mask) < 2:
        return 0.0
    
    filtered_true = labels_true[mask]
    filtered_pred = labels_pred[mask]
    
    try:
        return float(normalized_mutual_info_score(filtered_true, filtered_pred))
    except Exception:
        return 0.0
