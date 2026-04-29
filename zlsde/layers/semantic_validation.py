"""Feature 3: Cross-Cluster Semantic Validation (CCSV).

After labels are generated, CCSV checks inter-cluster consistency:
  - Label collisions: different clusters that received the same/very-similar label
  - Merge candidates: clusters whose centroids are nearly identical
  - Split candidates: single clusters with bimodal embedding distributions
  - Outlier clusters: tiny clusters far from all others

Flagged issues can trigger automatic relabelling or be surfaced for review.

Patent-relevant novelty:
- Post-labelling semantic consistency verification loop
- Automatic merge/split/relabel remediation
- Combined label-text + embedding-space validation
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from zlsde.models.data_models import (
    Label,
    LabeledDataItem,
    PipelineConfig,
    SemanticValidationResult,
    ValidationFlag,
)

logger = logging.getLogger(__name__)


class SemanticValidator:
    """Validate label consistency across clusters."""

    def __init__(self, config: PipelineConfig):
        self.label_sim_threshold = config.label_similarity_threshold
        self.centroid_sim_threshold = config.centroid_similarity_threshold
        self.enabled = config.enable_semantic_validation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        labeled_data: List[LabeledDataItem],
        pseudo_labels: Dict[int, Label],
    ) -> SemanticValidationResult:
        """Run all validation checks and return aggregated result."""
        if not self.enabled:
            return SemanticValidationResult()

        flags: List[ValidationFlag] = []

        # Group data by cluster
        clusters = self._group_by_cluster(labeled_data)
        cluster_ids = sorted(clusters.keys())
        if not cluster_ids:
            return SemanticValidationResult()

        # Compute centroids
        centroids = self._compute_centroids(clusters)

        # 1. Label collision detection
        flags.extend(self._detect_label_collisions(pseudo_labels, centroids))

        # 2. Merge candidate detection (centroid proximity)
        flags.extend(self._detect_merge_candidates(centroids))

        # 3. Split candidate detection (bimodal clusters)
        flags.extend(self._detect_split_candidates(clusters))

        # 4. Outlier cluster detection
        flags.extend(self._detect_outlier_clusters(centroids, clusters))

        # Aggregate
        result = SemanticValidationResult(
            flags=flags,
            n_merge_candidates=sum(1 for f in flags if f.flag_type == "merge_candidate"),
            n_split_candidates=sum(1 for f in flags if f.flag_type == "split_candidate"),
            n_label_collisions=sum(1 for f in flags if f.flag_type == "label_collision"),
            n_outlier_clusters=sum(1 for f in flags if f.flag_type == "outlier_cluster"),
            semantic_consistency_score=self._consistency_score(flags, len(cluster_ids)),
        )

        logger.info(
            f"Semantic validation: {len(flags)} flags "
            f"(merges={result.n_merge_candidates}, splits={result.n_split_candidates}, "
            f"collisions={result.n_label_collisions}, outliers={result.n_outlier_clusters}), "
            f"consistency={result.semantic_consistency_score:.3f}"
        )
        return result

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _detect_label_collisions(
        self,
        pseudo_labels: Dict[int, Label],
        centroids: Dict[int, np.ndarray],
    ) -> List[ValidationFlag]:
        """Find clusters with identical or near-identical label text."""
        flags: List[ValidationFlag] = []
        cids = [c for c in pseudo_labels if c >= 0]

        for i, cid_a in enumerate(cids):
            for cid_b in cids[i + 1 :]:
                la = pseudo_labels[cid_a].text.lower().strip()
                lb = pseudo_labels[cid_b].text.lower().strip()
                sim = self._text_similarity(la, lb)
                if sim >= self.label_sim_threshold:
                    flags.append(
                        ValidationFlag(
                            flag_type="label_collision",
                            cluster_ids=[cid_a, cid_b],
                            labels=[pseudo_labels[cid_a].text, pseudo_labels[cid_b].text],
                            similarity_score=sim,
                            description=(
                                f"Clusters {cid_a} and {cid_b} have near-identical labels "
                                f"('{pseudo_labels[cid_a].text}' vs '{pseudo_labels[cid_b].text}', "
                                f"sim={sim:.2f}). Consider merging."
                            ),
                        )
                    )
        return flags

    def _detect_merge_candidates(self, centroids: Dict[int, np.ndarray]) -> List[ValidationFlag]:
        """Find clusters whose centroids are very close in embedding space."""
        flags: List[ValidationFlag] = []
        cids = sorted(centroids.keys())
        if len(cids) < 2:
            return flags

        centroid_matrix = np.array([centroids[c] for c in cids])
        sim_matrix = cosine_similarity(centroid_matrix)

        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                if sim_matrix[i, j] >= self.centroid_sim_threshold:
                    flags.append(
                        ValidationFlag(
                            flag_type="merge_candidate",
                            cluster_ids=[cids[i], cids[j]],
                            labels=[],
                            similarity_score=float(sim_matrix[i, j]),
                            description=(
                                f"Clusters {cids[i]} and {cids[j]} have very similar centroids "
                                f"(cosine={sim_matrix[i, j]:.3f}). Consider merging."
                            ),
                        )
                    )
        return flags

    def _detect_split_candidates(
        self, clusters: Dict[int, List[LabeledDataItem]]
    ) -> List[ValidationFlag]:
        """Detect clusters with bimodal distributions (should be split)."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        flags: List[ValidationFlag] = []

        for cid, items in clusters.items():
            if cid < 0 or len(items) < 10:
                continue

            embs = np.array([item.embedding for item in items])
            try:
                km = KMeans(n_clusters=2, random_state=42, n_init=5)
                sub_labels = km.fit_predict(embs)
                sil = silhouette_score(embs, sub_labels)

                # High silhouette for k=2 means bimodal
                if sil > 0.5:
                    flags.append(
                        ValidationFlag(
                            flag_type="split_candidate",
                            cluster_ids=[cid],
                            labels=[],
                            similarity_score=sil,
                            description=(
                                f"Cluster {cid} appears bimodal (sub-silhouette={sil:.3f}). "
                                f"Consider splitting into sub-clusters."
                            ),
                        )
                    )
            except Exception:
                continue

        return flags

    def _detect_outlier_clusters(
        self,
        centroids: Dict[int, np.ndarray],
        clusters: Dict[int, List[LabeledDataItem]],
    ) -> List[ValidationFlag]:
        """Detect tiny clusters far from all others."""
        flags: List[ValidationFlag] = []
        cids = sorted(centroids.keys())
        if len(cids) < 3:
            return flags

        centroid_matrix = np.array([centroids[c] for c in cids])
        sim_matrix = cosine_similarity(centroid_matrix)

        median_size = float(np.median([len(clusters[c]) for c in cids]))

        for idx, cid in enumerate(cids):
            if cid < 0:
                continue
            # Mean similarity to other clusters
            other_sims = [sim_matrix[idx, j] for j in range(len(cids)) if j != idx]
            mean_sim = float(np.mean(other_sims))
            cluster_size = len(clusters[cid])

            if mean_sim < 0.3 and cluster_size < median_size * 0.25:
                flags.append(
                    ValidationFlag(
                        flag_type="outlier_cluster",
                        cluster_ids=[cid],
                        labels=[],
                        similarity_score=mean_sim,
                        description=(
                            f"Cluster {cid} is an outlier (mean_sim={mean_sim:.3f}, "
                            f"size={cluster_size}, median_size={median_size:.0f})."
                        ),
                    )
                )
        return flags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_cluster(
        items: List[LabeledDataItem],
    ) -> Dict[int, List[LabeledDataItem]]:
        clusters: Dict[int, List[LabeledDataItem]] = {}
        for item in items:
            clusters.setdefault(item.cluster_id, []).append(item)
        return clusters

    @staticmethod
    def _compute_centroids(
        clusters: Dict[int, List[LabeledDataItem]],
    ) -> Dict[int, np.ndarray]:
        centroids: Dict[int, np.ndarray] = {}
        for cid, items in clusters.items():
            if cid < 0:
                continue
            embs = np.array([item.embedding for item in items])
            centroids[cid] = embs.mean(axis=0)
        return centroids

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple character-level Jaccard similarity between two strings."""
        if a == b:
            return 1.0
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a and not set_b:
            return 1.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _consistency_score(flags: List[ValidationFlag], n_clusters: int) -> float:
        """Overall consistency: 1.0 means no flags raised."""
        if n_clusters == 0:
            return 1.0
        penalty = len(flags) / max(n_clusters, 1)
        return float(max(0.0, 1.0 - penalty * 0.2))
