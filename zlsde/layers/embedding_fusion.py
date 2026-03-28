"""Feature 4: Multi-Granularity Embedding Fusion.

Combines embeddings from multiple models at different granularities
(word-level, sentence-level, paragraph-level) into a single fused
representation using learned or configurable weights.

Patent-relevant novelty:
- Automatic weight learning via silhouette-score feedback
- Cross-granularity fusion for richer cluster separation
- Pluggable model registry with normalized concatenation
"""

import logging
from typing import List, Optional

import numpy as np

from zlsde.models.data_models import PipelineConfig, RawDataItem

logger = logging.getLogger(__name__)


class EmbeddingFusionEngine:
    """Fuse embeddings from multiple models into a single representation."""

    def __init__(self, config: PipelineConfig):
        self.enabled = config.enable_embedding_fusion
        self.fusion_model_names = config.fusion_models
        self.fusion_weights = list(config.fusion_weights)
        self.device = config.device
        self.primary_model_name = config.embedding_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        items: List[RawDataItem],
        primary_embeddings: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Compute fused multi-granularity embeddings.

        If fusion is disabled or no extra models are configured,
        returns the primary embeddings unchanged.

        Args:
            items: raw data items (needed for re-encoding with other models).
            primary_embeddings: embeddings from the primary model.
            batch_size: batch size for secondary model encoding.

        Returns:
            (n_samples, fused_dim) array of L2-normalised fused embeddings.
        """
        if not self.enabled or not self.fusion_model_names:
            return primary_embeddings

        all_embs: List[np.ndarray] = [primary_embeddings]

        for model_name in self.fusion_model_names:
            logger.info(f"Computing secondary embeddings with {model_name}")
            try:
                secondary = self._encode_with_model(items, model_name, batch_size)
                all_embs.append(secondary)
            except Exception as e:
                logger.warning(f"Failed to compute embeddings with {model_name}: {e}")

        if len(all_embs) == 1:
            return primary_embeddings

        # Determine weights
        weights = self._resolve_weights(len(all_embs))

        # Weighted concatenation + normalisation
        fused = self._weighted_concat(all_embs, weights)

        logger.info(
            f"Embedding fusion: {len(all_embs)} models -> dim {fused.shape[1]}"
        )
        return fused

    def auto_learn_weights(
        self,
        embedding_sets: List[np.ndarray],
        cluster_labels: np.ndarray,
    ) -> List[float]:
        """Learn optimal fusion weights using silhouette-score grid search.

        Tries a simplex grid of weight combinations and selects the one
        that maximises the silhouette score with the given cluster labels.

        Returns:
            Optimal weight vector (sums to 1).
        """
        from sklearn.metrics import silhouette_score

        n = len(embedding_sets)
        if n < 2:
            return [1.0]

        best_weights: List[float] = [1.0 / n] * n
        best_sil = -1.0

        # Simple grid search over weight simplex
        steps = 5
        candidates = self._simplex_grid(n, steps)

        for w in candidates:
            fused = self._weighted_concat(embedding_sets, list(w))
            unique = np.unique(cluster_labels[cluster_labels >= 0])
            if len(unique) < 2:
                continue
            mask = cluster_labels >= 0
            try:
                sil = silhouette_score(fused[mask], cluster_labels[mask])
                if sil > best_sil:
                    best_sil = sil
                    best_weights = list(w)
            except Exception:
                continue

        logger.info(f"Auto-learned fusion weights: {best_weights} (sil={best_sil:.3f})")
        self.fusion_weights = best_weights
        return best_weights

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_with_model(
        self, items: List[RawDataItem], model_name: str, batch_size: int
    ) -> np.ndarray:
        """Encode items with a secondary SentenceTransformer model."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=self.device)
        texts = [str(item.content) for item in items]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def _resolve_weights(self, n_models: int) -> List[float]:
        """Return weight vector, falling back to uniform if not configured."""
        if self.fusion_weights and len(self.fusion_weights) == n_models:
            total = sum(self.fusion_weights)
            return [w / total for w in self.fusion_weights] if total > 0 else [1.0 / n_models] * n_models
        return [1.0 / n_models] * n_models

    @staticmethod
    def _weighted_concat(
        embedding_sets: List[np.ndarray], weights: List[float]
    ) -> np.ndarray:
        """Weighted concatenation followed by L2 normalisation."""
        scaled = []
        for emb, w in zip(embedding_sets, weights):
            # Per-model L2 normalise first
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            scaled.append((emb / norms) * w)

        fused = np.hstack(scaled)

        # Global L2 normalisation
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return fused / norms

    @staticmethod
    def _simplex_grid(n: int, steps: int = 5):
        """Generate weight vectors on a regular simplex grid."""
        if n == 1:
            yield (1.0,)
            return
        if n == 2:
            for i in range(steps + 1):
                w1 = i / steps
                yield (w1, 1.0 - w1)
            return
        # General case via recursive partition
        for i in range(steps + 1):
            w = i / steps
            for sub in EmbeddingFusionEngine._simplex_grid(n - 1, steps - i):
                yield (w,) + tuple(s * (1.0 - w) for s in sub)
