"""Feature 7: Embedding Drift Detection & Self-Correction.

Monitors the health of the embedding space across self-training iterations.
Detects two failure modes:
  - Collapse: all clusters converge toward the global centroid
  - Divergence: centroids drift apart uncontrollably

When drift is detected the engine recommends (or auto-executes) a rollback
to the last healthy iteration's cluster assignments.

Patent-relevant novelty:
- Real-time embedding health scoring across training iterations
- Automatic collapse/divergence detection with configurable thresholds
- Self-correcting rollback mechanism to preserve label quality
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from zlsde.models.data_models import DriftReport, PipelineConfig

logger = logging.getLogger(__name__)


class DriftDetector:
    """Monitor embedding space health and detect drift across iterations."""

    def __init__(self, config: PipelineConfig):
        self.enabled = config.enable_drift_detection
        self.collapse_threshold = config.drift_collapse_threshold
        self.divergence_threshold = config.drift_divergence_threshold

        # History
        self._prev_centroids: Optional[Dict[int, np.ndarray]] = None
        self._prev_inter_dist: Optional[float] = None
        self._prev_intra_var: Optional[float] = None
        self._history: List[DriftReport] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        iteration: int,
    ) -> DriftReport:
        """Compute drift metrics for current iteration.

        Returns a DriftReport.  If rollback_recommended is True the caller
        should revert to the previous iteration's labels.
        """
        if not self.enabled:
            return DriftReport(iteration=iteration)

        centroids = self._compute_centroids(embeddings, cluster_labels)

        inter_dist = self._inter_cluster_distance(centroids)
        intra_var = self._intra_cluster_variance(embeddings, cluster_labels, centroids)
        centroid_drift = self._centroid_drift(centroids)

        collapse = self._detect_collapse(inter_dist)
        divergence = self._detect_divergence(centroid_drift)

        health = self._health_score(inter_dist, intra_var, collapse, divergence)

        report = DriftReport(
            iteration=iteration,
            inter_cluster_distance=inter_dist,
            intra_cluster_variance=intra_var,
            centroid_drift=centroid_drift,
            collapse_detected=collapse,
            divergence_detected=divergence,
            rollback_recommended=collapse or divergence,
            health_score=health,
        )

        # Update history
        self._prev_centroids = centroids
        self._prev_inter_dist = inter_dist
        self._prev_intra_var = intra_var
        self._history.append(report)

        logger.info(
            f"Drift check iter {iteration}: health={health:.3f}, "
            f"inter_dist={inter_dist:.4f}, intra_var={intra_var:.4f}, "
            f"drift={centroid_drift:.4f}, "
            f"collapse={'YES' if collapse else 'no'}, "
            f"divergence={'YES' if divergence else 'no'}"
        )

        return report

    @property
    def history(self) -> List[DriftReport]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
        centroids: Dict[int, np.ndarray] = {}
        for cid in np.unique(labels):
            if cid < 0:
                continue
            centroids[int(cid)] = embeddings[labels == cid].mean(axis=0)
        return centroids

    @staticmethod
    def _inter_cluster_distance(centroids: Dict[int, np.ndarray]) -> float:
        """Mean pairwise Euclidean distance between cluster centroids."""
        if len(centroids) < 2:
            return 0.0
        vecs = np.array(list(centroids.values()))
        dists = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
        return float(np.mean(dists)) if dists else 0.0

    @staticmethod
    def _intra_cluster_variance(
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: Dict[int, np.ndarray],
    ) -> float:
        """Mean within-cluster variance (avg squared distance to centroid)."""
        variances = []
        for cid, centroid in centroids.items():
            mask = labels == cid
            if mask.sum() < 2:
                continue
            diffs = embeddings[mask] - centroid
            var = float(np.mean(np.sum(diffs**2, axis=1)))
            variances.append(var)
        return float(np.mean(variances)) if variances else 0.0

    def _centroid_drift(self, centroids: Dict[int, np.ndarray]) -> float:
        """Mean displacement of centroids compared to previous iteration."""
        if self._prev_centroids is None:
            return 0.0

        drifts = []
        for cid, vec in centroids.items():
            if cid in self._prev_centroids:
                drift = float(np.linalg.norm(vec - self._prev_centroids[cid]))
                drifts.append(drift)
        return float(np.mean(drifts)) if drifts else 0.0

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _detect_collapse(self, inter_dist: float) -> bool:
        """Collapse = inter-cluster distance dropped dramatically."""
        if self._prev_inter_dist is None or self._prev_inter_dist == 0:
            return False
        ratio = inter_dist / self._prev_inter_dist
        return ratio < self.collapse_threshold

    def _detect_divergence(self, centroid_drift: float) -> bool:
        """Divergence = centroids moved far more than expected."""
        if self._prev_inter_dist is None or self._prev_inter_dist == 0:
            return False
        ratio = centroid_drift / self._prev_inter_dist
        return ratio > self.divergence_threshold

    @staticmethod
    def _health_score(
        inter_dist: float,
        intra_var: float,
        collapse: bool,
        divergence: bool,
    ) -> float:
        """Composite health score in [0, 1]."""
        if collapse or divergence:
            return 0.2

        # Ratio of inter to intra (higher is healthier)
        if intra_var > 0:
            ratio = inter_dist / (intra_var + 1e-8)
            score = min(1.0, ratio / 5.0)  # normalise so ratio=5 -> 1.0
        else:
            score = 1.0 if inter_dist > 0 else 0.5

        return float(np.clip(score, 0.0, 1.0))
