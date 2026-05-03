"""Layer 3: Clustering Engine - Discover natural groupings in embedding space."""

import logging
from typing import Optional

import numpy as np

from zlsde.models.data_models import ClusterResult

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """Discover natural groupings using density-based and partitional clustering.

    Supports:
    - HDBSCAN for automatic cluster detection (primary method)
    - KMeans with optimal k estimation (fallback)
    - Spectral clustering (alternative method)
    - Auto-selection based on silhouette score
    - Noise point detection and handling
    - Cluster quality metrics computation
    """

    def __init__(self, config):
        """Initialize clustering engine with configuration.

        Args:
            config: PipelineConfig with clustering parameters.
        """
        self.config = config
        self.min_cluster_size = config.min_cluster_size
        self.n_clusters = config.n_clusters
        self.random_seed = config.random_seed
        self._hdbscan_available = False
        self._hdbscan_error: Optional[str] = None
        self._detect_hdbscan_availability()

        logger.info(
            f"ClusteringEngine initialized: method={config.clustering_method}, "
            f"min_cluster_size={self.min_cluster_size}, "
            f"n_clusters={self.n_clusters}"
        )
        if not self._hdbscan_available:
            logger.info(
                "HDBSCAN unavailable; auto clustering will use KMeans/Spectral fallback. "
                f"Reason: {self._hdbscan_error}"
            )

    def _detect_hdbscan_availability(self) -> None:
        """Detect HDBSCAN support once at startup.

        This avoids repeated import attempts and warning spam in iterative runs.
        """
        try:
            from sklearn.cluster import HDBSCAN  # noqa: F401

            self._hdbscan_available = True
            self._hdbscan_error = None
        except Exception as e:
            self._hdbscan_available = False
            self._hdbscan_error = str(e).strip() or type(e).__name__

    def cluster(self, embeddings: np.ndarray, method: str = "auto") -> ClusterResult:
        """Cluster embeddings and return assignments with metrics.

        Tries multiple clustering methods and selects the best based on silhouette score
        when method="auto". Handles noise points and computes quality metrics.

        Args:
            embeddings: 2D numpy array of shape (n_samples, n_features).
            method: Clustering method ("auto", "hdbscan", "kmeans", "spectral").

        Returns:
            ClusterResult with cluster assignments and metrics.

        Raises:
            ValueError: If embeddings is invalid or method is unknown.
            RuntimeError: If all clustering methods fail.

        Preconditions:
        - embeddings is 2D numpy array with shape (n_samples, n_features)
        - n_samples >= min_cluster_size
        - method in ["auto", "hdbscan", "kmeans", "spectral"]
        - No NaN or Inf values in embeddings

        Postconditions:
        - Returns ClusterResult with valid cluster assignments
        - labels array has length n_samples
        - All labels are integers >= -1 (where -1 indicates noise)
        - n_clusters matches number of unique non-noise clusters
        - silhouette_score in [-1, 1]
        - If method="auto", best method selected based on silhouette score
        """
        # Validate inputs
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D array, got shape {embeddings.shape}")

        n_samples, n_features = embeddings.shape

        if n_samples < self.min_cluster_size:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= min_cluster_size ({self.min_cluster_size})"
            )

        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("embeddings contains NaN or Inf values")

        if method not in ["auto", "hdbscan", "kmeans", "spectral"]:
            raise ValueError(f"Unknown clustering method: {method}")

        logger.info(f"Clustering {n_samples} samples with method={method}")

        if method == "auto":
            # Try multiple methods and select best
            results = []

            # Try HDBSCAN
            if self._hdbscan_available:
                try:
                    logger.info("Trying HDBSCAN clustering...")
                    hdbscan_result = self._cluster_hdbscan(embeddings)
                    results.append(hdbscan_result)
                    logger.info(
                        f"HDBSCAN: {hdbscan_result.n_clusters} clusters, "
                        f"silhouette={hdbscan_result.silhouette_score:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"HDBSCAN failed: {e}")
            else:
                logger.debug(f"Skipping HDBSCAN in auto mode: {self._hdbscan_error}")

            # Try KMeans with estimated k
            estimated_k = self._estimate_optimal_k(embeddings)
            k_candidates = [estimated_k, max(2, estimated_k - 1), estimated_k + 1]
            if self.n_clusters is not None:
                k_candidates = [self.n_clusters] + k_candidates

            # De-duplicate while preserving order
            seen = set()
            ordered_candidates = []
            for k in k_candidates:
                if k not in seen:
                    seen.add(k)
                    ordered_candidates.append(k)

            for k in ordered_candidates:
                if k >= 2 and k < n_samples:
                    try:
                        logger.info(f"Trying KMeans with k={k}...")
                        kmeans_result = self._cluster_kmeans(embeddings, n_clusters=k)
                        results.append(kmeans_result)
                        logger.info(
                            f"KMeans (k={k}): silhouette={kmeans_result.silhouette_score:.3f}"
                        )
                    except Exception as e:
                        logger.warning(f"KMeans with k={k} failed: {e}")

            # Try Spectral clustering with estimated k
            spectral_k = self.n_clusters if self.n_clusters is not None else estimated_k
            if spectral_k >= 2 and spectral_k < n_samples:
                try:
                    logger.info(f"Trying Spectral clustering with k={spectral_k}...")
                    spectral_result = self._cluster_spectral(embeddings, n_clusters=spectral_k)
                    results.append(spectral_result)
                    logger.info(
                        f"Spectral (k={spectral_k}): silhouette={spectral_result.silhouette_score:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"Spectral clustering failed: {e}")

            if not results:
                raise RuntimeError("All clustering methods failed")

            # Select best result by silhouette score
            best_result = max(results, key=lambda r: r.silhouette_score)
            logger.info(
                f"Selected {best_result.method_used} with silhouette={best_result.silhouette_score:.3f}"
            )

            return best_result

        elif method == "hdbscan":
            if not self._hdbscan_available:
                raise ImportError(
                    "hdbscan is not available in this environment. "
                    f"Details: {self._hdbscan_error}. "
                    "Use clustering_method='auto'/'kmeans'/'spectral' or install a "
                    "Python-compatible hdbscan build."
                )
            return self._cluster_hdbscan(embeddings)

        elif method == "kmeans":
            k = (
                self.n_clusters
                if self.n_clusters is not None
                else self._estimate_optimal_k(embeddings)
            )
            return self._cluster_kmeans(embeddings, n_clusters=k)

        elif method == "spectral":
            k = (
                self.n_clusters
                if self.n_clusters is not None
                else self._estimate_optimal_k(embeddings)
            )
            return self._cluster_spectral(embeddings, n_clusters=k)

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> ClusterResult:
        """HDBSCAN clustering with automatic cluster detection.

        Uses density-based clustering to automatically discover clusters and
        identify noise points.

        Args:
            embeddings: 2D numpy array of embeddings.

        Returns:
            ClusterResult with HDBSCAN assignments.

        Raises:
            ImportError: If hdbscan is not installed.
            RuntimeError: If clustering fails.
        """
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            raise ImportError(
                "scikit-learn>=1.3 is required for HDBSCAN clustering."
            )

        # Create HDBSCAN clusterer
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=max(1, self.min_cluster_size // 2),
            metric="euclidean",
            cluster_selection_method="eom",
        )

        # Fit and predict
        labels = clusterer.fit_predict(embeddings)

        # Get probabilities (cluster membership strength)
        probabilities = clusterer.probabilities_

        # Count clusters and noise
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)

        # Compute silhouette score (only for non-noise points if we have clusters)
        silhouette = self._compute_silhouette_score(embeddings, labels)

        result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            probabilities=probabilities,
            silhouette_score=silhouette,
            method_used="hdbscan",
        )

        # Validate result
        result.validate(n_samples=len(embeddings))

        return result

    def _cluster_kmeans(self, embeddings: np.ndarray, n_clusters: int) -> ClusterResult:
        """KMeans clustering with specified cluster count.

        Uses partitional clustering to assign each point to exactly one cluster.
        No noise points are identified.

        Args:
            embeddings: 2D numpy array of embeddings.
            n_clusters: Number of clusters to create.

        Returns:
            ClusterResult with KMeans assignments.

        Raises:
            ValueError: If n_clusters is invalid.
        """
        from sklearn.cluster import KMeans

        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

        if n_clusters >= len(embeddings):
            raise ValueError(f"n_clusters ({n_clusters}) must be < n_samples ({len(embeddings)})")

        # Create KMeans clusterer
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=self.random_seed, n_init=10, max_iter=300
        )

        # Fit and predict
        labels = kmeans.fit_predict(embeddings)

        # Compute silhouette score
        silhouette = self._compute_silhouette_score(embeddings, labels)

        result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=0,  # KMeans doesn't identify noise
            probabilities=None,
            silhouette_score=silhouette,
            method_used="kmeans",
        )

        # Validate result
        result.validate(n_samples=len(embeddings))

        return result

    def _cluster_spectral(self, embeddings: np.ndarray, n_clusters: int) -> ClusterResult:
        """Spectral clustering as alternative method.

        Uses graph-based clustering to find clusters based on eigenvectors
        of the similarity matrix.

        Args:
            embeddings: 2D numpy array of embeddings.
            n_clusters: Number of clusters to create.

        Returns:
            ClusterResult with Spectral clustering assignments.

        Raises:
            ValueError: If n_clusters is invalid.
        """
        from sklearn.cluster import SpectralClustering

        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

        if n_clusters >= len(embeddings):
            raise ValueError(f"n_clusters ({n_clusters}) must be < n_samples ({len(embeddings)})")

        # Create Spectral clusterer
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            affinity="nearest_neighbors",
            n_neighbors=min(10, len(embeddings) - 1),
            assign_labels="kmeans",
        )

        # Fit and predict
        labels = spectral.fit_predict(embeddings)

        # Compute silhouette score
        silhouette = self._compute_silhouette_score(embeddings, labels)

        result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=0,  # Spectral doesn't identify noise
            probabilities=None,
            silhouette_score=silhouette,
            method_used="spectral",
        )

        # Validate result
        result.validate(n_samples=len(embeddings))

        return result

    def _estimate_optimal_k(self, embeddings: np.ndarray) -> int:
        """Estimate optimal number of clusters using heuristics.

        Uses the elbow method with inertia and a simple heuristic based on
        sample count.

        Args:
            embeddings: 2D numpy array of embeddings.

        Returns:
            Estimated optimal number of clusters.
        """
        n_samples = len(embeddings)

        # Simple heuristic: sqrt(n/2)
        k_heuristic = max(2, int(np.sqrt(n_samples / 2)))

        # Cap at reasonable values
        k_min = 2
        k_max = min(20, n_samples // self.min_cluster_size)

        k_estimated = max(k_min, min(k_heuristic, k_max))

        logger.info(f"Estimated optimal k={k_estimated} for {n_samples} samples")

        return k_estimated

    def _compute_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score for cluster quality.

        Silhouette score measures how similar points are to their own cluster
        compared to other clusters. Range: [-1, 1], higher is better.

        Args:
            embeddings: 2D numpy array of embeddings.
            labels: 1D numpy array of cluster labels.

        Returns:
            Silhouette score in [-1, 1], or 0.0 if cannot be computed.
        """
        from sklearn.metrics import silhouette_score

        # Filter out noise points (label -1) for silhouette computation
        non_noise_mask = labels >= 0
        non_noise_labels = labels[non_noise_mask]
        non_noise_embeddings = embeddings[non_noise_mask]

        # Need at least 2 clusters and 2 samples per cluster
        unique_labels = np.unique(non_noise_labels)

        if len(unique_labels) < 2:
            logger.warning("Cannot compute silhouette score: less than 2 clusters")
            return 0.0

        if len(non_noise_labels) < 2:
            logger.warning("Cannot compute silhouette score: less than 2 non-noise samples")
            return 0.0

        try:
            score = silhouette_score(non_noise_embeddings, non_noise_labels, metric="euclidean")
            return float(score)
        except Exception as e:
            logger.warning(f"Failed to compute silhouette score: {e}")
            return 0.0

    def compute_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Compute silhouette score and cluster quality metrics.

        Args:
            embeddings: 2D numpy array of embeddings.
            labels: 1D numpy array of cluster labels.

        Returns:
            Dictionary with cluster quality metrics.

        Raises:
            ValueError: If inputs are invalid.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D array, got shape {embeddings.shape}")

        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D array, got shape {labels.shape}")

        if len(embeddings) != len(labels):
            raise ValueError(
                f"embeddings and labels length mismatch: {len(embeddings)} vs {len(labels)}"
            )

        # Count clusters and noise
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)
        noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0.0

        # Compute silhouette score
        silhouette = self._compute_silhouette_score(embeddings, labels)

        metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio,
            "silhouette_score": silhouette,
            "n_samples": len(labels),
        }

        logger.info(f"Cluster metrics: {metrics}")

        return metrics
