"""Layer 5: Quality Control Filter - Detect and filter low-quality samples."""

import logging
from typing import Dict, List, Set

import numpy as np

from zlsde.models.data_models import LabeledDataItem, QualityScore

logger = logging.getLogger(__name__)


class QualityControlFilter:
    """Detect and filter low-quality samples, anomalies, and duplicates."""

    def __init__(self, config):
        """Initialize quality control filter with configuration."""
        self.config = config
        self.anomaly_contamination = getattr(config, "anomaly_contamination", 0.1)
        self.duplicate_threshold = getattr(config, "duplicate_threshold", 0.95)

    def filter(self, items: List[LabeledDataItem]) -> List[QualityScore]:
        """
        Compute quality scores for all items.

        Args:
            items: List of labeled data items

        Returns:
            List of quality scores for each item
        """
        if not items:
            return []

        logger.info(f"Computing quality scores for {len(items)} items")

        # Extract embeddings
        embeddings = np.array([item.embedding for item in items])

        # Detect anomalies
        anomaly_scores = self.detect_anomalies(embeddings)

        # Detect duplicates
        duplicate_indices = self.detect_duplicates(embeddings, self.duplicate_threshold)

        # Compute cluster coherence scores
        coherence_scores = self._compute_cluster_coherence_batch(items)

        # Aggregate quality scores
        quality_scores = []
        for i, item in enumerate(items):
            # Invert anomaly score (higher is better)
            anomaly_score = 1.0 - anomaly_scores[i]
            duplicate_score = 0.0 if i in duplicate_indices else 1.0
            coherence_score = coherence_scores[i]

            # Weighted combination
            quality = 0.4 * anomaly_score + 0.3 * duplicate_score + 0.3 * coherence_score

            quality_scores.append(
                QualityScore(
                    score=float(quality),
                    anomaly_flag=anomaly_scores[i] > 0.5,
                    duplicate_flag=i in duplicate_indices,
                )
            )

        logger.info(
            f"Quality filtering complete. Anomalies: {sum(qs.anomaly_flag for qs in quality_scores)}, "
            f"Duplicates: {sum(qs.duplicate_flag for qs in quality_scores)}"
        )

        return quality_scores

    def detect_anomalies(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest and LOF.

        Args:
            embeddings: Array of embeddings (n_samples, n_features)

        Returns:
            Array of anomaly scores in [0, 1] (higher = more anomalous)
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        n_samples = embeddings.shape[0]

        # Need at least 10 samples for anomaly detection
        if n_samples < 10:
            return np.zeros(n_samples)

        try:
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.anomaly_contamination, random_state=42, n_jobs=-1
            )
            iso_predictions = iso_forest.fit_predict(embeddings)
            iso_scores = (iso_predictions == -1).astype(float)

            # Local Outlier Factor
            lof = LocalOutlierFactor(contamination=self.anomaly_contamination, n_jobs=-1)
            lof_predictions = lof.fit_predict(embeddings)
            lof_scores = (lof_predictions == -1).astype(float)

            # Ensemble: average of both methods
            anomaly_scores = (iso_scores + lof_scores) / 2.0

            return anomaly_scores

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return np.zeros(n_samples)

    def detect_duplicates(self, embeddings: np.ndarray, threshold: float = 0.95) -> Set[int]:
        """
        Detect near-duplicates via cosine similarity.

        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            threshold: Cosine similarity threshold for duplicates

        Returns:
            Set of indices that are duplicates
        """
        from sklearn.metrics.pairwise import cosine_similarity

        n_samples = embeddings.shape[0]

        if n_samples < 2:
            return set()

        try:
            # Compute pairwise cosine similarity
            similarity_matrix = cosine_similarity(embeddings)

            # Find duplicates (excluding self-similarity)
            duplicate_indices = set()
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if similarity_matrix[i, j] >= threshold:
                        # Mark the second occurrence as duplicate
                        duplicate_indices.add(j)

            return duplicate_indices

        except Exception as e:
            logger.warning(f"Duplicate detection failed: {e}")
            return set()

    def compute_cluster_coherence(self, cluster_items: List[LabeledDataItem]) -> float:
        """
        Compute intra-cluster coherence score.

        Args:
            cluster_items: List of items in the same cluster

        Returns:
            Coherence score in [0, 1]
        """
        if len(cluster_items) < 2:
            return 1.0

        # Extract embeddings
        embeddings = np.array([item.embedding for item in cluster_items])

        # Compute centroid
        centroid = np.mean(embeddings, axis=0)

        # Compute average distance to centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        avg_distance = np.mean(distances)

        # Convert to coherence score (lower distance = higher coherence)
        # Use exponential decay
        coherence = np.exp(-avg_distance)

        return float(np.clip(coherence, 0.0, 1.0))

    def _compute_cluster_coherence_batch(self, items: List[LabeledDataItem]) -> np.ndarray:
        """
        Compute coherence scores for all items based on their clusters.

        Args:
            items: List of all labeled items

        Returns:
            Array of coherence scores
        """
        # Group items by cluster
        clusters: Dict[int, List[int]] = {}
        for i, item in enumerate(items):
            cluster_id = item.cluster_id
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)

        # Compute coherence for each cluster
        cluster_coherence: Dict[int, float] = {}
        for cluster_id, indices in clusters.items():
            cluster_items = [items[i] for i in indices]
            cluster_coherence[cluster_id] = self.compute_cluster_coherence(cluster_items)

        # Assign coherence scores to items
        coherence_scores = np.zeros(len(items))
        for i, item in enumerate(items):
            coherence_scores[i] = cluster_coherence[item.cluster_id]

        return coherence_scores
