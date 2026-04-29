"""Feature 1: Autonomous Label Taxonomy Discovery (ALTD).

Recursively sub-clusters embedding space to discover hierarchical label
taxonomies without any predefined category schema.  Each level of the
taxonomy is validated with silhouette improvement gating so only
statistically meaningful splits are retained.

Patent-relevant novelty:
- Zero-shot hierarchical taxonomy discovery from raw embeddings
- Silhouette-gated recursive splitting prevents over-fragmentation
- Automatic depth control with configurable thresholds
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from zlsde.models.data_models import Label, PipelineConfig, RawDataItem, TaxonomyNode, TaxonomyTree

logger = logging.getLogger(__name__)


class TaxonomyDiscoveryEngine:
    """Discover hierarchical label taxonomies via recursive sub-clustering."""

    def __init__(self, config: PipelineConfig):
        self.max_depth = config.taxonomy_max_depth
        self.min_samples = config.taxonomy_min_samples
        self.silhouette_threshold = config.taxonomy_silhouette_threshold
        self.random_seed = config.random_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        pseudo_labels: Dict[int, Label],
        label_generator=None,
    ) -> TaxonomyTree:
        """Build a full taxonomy tree from top-level clusters.

        Args:
            embeddings: (n_samples, n_features) array.
            cluster_labels: per-sample cluster id (-1 = noise).
            pseudo_labels: cluster_id -> Label mapping from initial labelling.
            label_generator: optional PseudoLabelGenerator for sub-cluster labels.

        Returns:
            TaxonomyTree with hierarchical nodes.
        """
        root_nodes: List[TaxonomyNode] = []
        unique_clusters = sorted(set(cluster_labels[cluster_labels >= 0]))

        for cid in unique_clusters:
            mask = cluster_labels == cid
            cluster_embs = embeddings[mask]
            parent_label = pseudo_labels.get(cid, Label(text=f"cluster_{cid}", confidence=0.5))

            node = self._build_node(
                embeddings=cluster_embs,
                parent_label=parent_label.text,
                level=0,
                cluster_id=cid,
                confidence=parent_label.confidence,
                label_generator=label_generator,
                all_embeddings=embeddings,
                all_labels=cluster_labels,
            )
            root_nodes.append(node)

        total_nodes = self._count_nodes(root_nodes)
        max_depth = self._max_depth_of(root_nodes)

        tree = TaxonomyTree(
            root_nodes=root_nodes,
            max_depth=max_depth,
            total_nodes=total_nodes,
            discovery_method="recursive_splitting",
        )
        logger.info(f"Taxonomy discovered: {total_nodes} nodes, max depth {max_depth}")
        return tree

    # ------------------------------------------------------------------
    # Recursive builder
    # ------------------------------------------------------------------

    def _build_node(
        self,
        embeddings: np.ndarray,
        parent_label: str,
        level: int,
        cluster_id: int,
        confidence: float,
        label_generator,
        all_embeddings: np.ndarray,
        all_labels: np.ndarray,
    ) -> TaxonomyNode:
        """Recursively build a taxonomy node by attempting sub-clustering."""
        sil = self._silhouette(embeddings, np.zeros(len(embeddings)))

        node = TaxonomyNode(
            label=parent_label,
            level=level,
            cluster_id=cluster_id,
            n_samples=len(embeddings),
            silhouette_score=sil,
            confidence=confidence,
            parent_label=None if level == 0 else parent_label,
        )

        # Stop conditions
        if level >= self.max_depth:
            return node
        if len(embeddings) < self.min_samples * 2:
            return node

        # Attempt sub-clustering
        sub_labels, sub_sil, n_sub = self._try_sub_cluster(embeddings)
        if sub_labels is None or n_sub < 2:
            return node

        # Silhouette improvement gating
        if sub_sil - sil < self.silhouette_threshold:
            return node

        # Build children
        children: List[TaxonomyNode] = []
        for sub_id in range(n_sub):
            sub_mask = sub_labels == sub_id
            sub_embs = embeddings[sub_mask]
            if len(sub_embs) < self.min_samples:
                continue

            child_label = f"{parent_label}/{sub_id}"
            child_confidence = confidence * 0.9  # slight decay

            child_node = self._build_node(
                embeddings=sub_embs,
                parent_label=child_label,
                level=level + 1,
                cluster_id=sub_id,
                confidence=child_confidence,
                label_generator=label_generator,
                all_embeddings=all_embeddings,
                all_labels=all_labels,
            )
            child_node.parent_label = parent_label
            children.append(child_node)

        if children:
            node.children = children

        return node

    # ------------------------------------------------------------------
    # Sub-clustering helpers
    # ------------------------------------------------------------------

    def _try_sub_cluster(self, embeddings: np.ndarray) -> Tuple[Optional[np.ndarray], float, int]:
        """Attempt KMeans sub-clustering with k=2..5, pick best silhouette."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_labels = None
        best_sil = -1.0
        best_k = 0

        for k in range(2, min(6, len(embeddings))):
            try:
                km = KMeans(n_clusters=k, random_state=self.random_seed, n_init=5)
                labels = km.fit_predict(embeddings)
                sil = silhouette_score(embeddings, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_labels = labels
                    best_k = k
            except Exception:
                continue

        return best_labels, best_sil, best_k

    def _silhouette(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score; return 0 if impossible."""
        from sklearn.metrics import silhouette_score

        unique = np.unique(labels)
        if len(unique) < 2 or len(embeddings) < 3:
            return 0.0
        try:
            return float(silhouette_score(embeddings, labels))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Tree utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _count_nodes(nodes: List[TaxonomyNode]) -> int:
        total = 0
        for n in nodes:
            total += 1
            total += TaxonomyDiscoveryEngine._count_nodes(n.children)
        return total

    @staticmethod
    def _max_depth_of(nodes: List[TaxonomyNode]) -> int:
        if not nodes:
            return 0
        return max(
            n.level if not n.children else TaxonomyDiscoveryEngine._max_depth_of(n.children)
            for n in nodes
        )
