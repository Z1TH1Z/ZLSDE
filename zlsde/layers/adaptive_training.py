"""Feature 5: Confidence-Weighted Adaptive Self-Training (CWAST).

Replaces the vanilla self-training loop with a curriculum-learning
strategy: early iterations train only on the highest-confidence samples,
and the confidence threshold gradually decays to include harder samples.

Sample-level confidence weighting ensures that uncertain labels have
less influence on the classifier decision boundary.

Patent-relevant novelty:
- Curriculum-based pseudo-label learning with automatic threshold decay
- Per-sample confidence weighting in loss contribution
- Adaptive percentile gating (not a fixed threshold)
"""

import logging

import numpy as np
from sklearn.neural_network import MLPClassifier

from zlsde.models.data_models import PipelineConfig, LabeledDataItem

logger = logging.getLogger(__name__)


class AdaptiveSelfTrainer:
    """Confidence-weighted adaptive self-training with curriculum learning."""

    def __init__(self, config: PipelineConfig):
        self.enabled = config.enable_adaptive_training
        self.base_confidence = config.confidence_threshold
        self.percentile_decay = config.curriculum_percentile_decay
        self.random_seed = config.random_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_and_refine(
        self,
        embeddings: np.ndarray,
        labeled_data: list,
        iteration: int,
        max_iterations: int,
    ) -> np.ndarray:
        """Train a classifier with curriculum weighting and refine labels.

        Args:
            embeddings: (n_samples, n_features).
            labeled_data: list of LabeledDataItem.
            iteration: current iteration number (0-based).
            max_iterations: total planned iterations.

        Returns:
            Refined label array (n_samples,).
        """
        if not self.enabled:
            return self._fallback_train(embeddings, labeled_data)

        labels = np.array([item.cluster_id for item in labeled_data])
        confidences = np.array([item.confidence for item in labeled_data])

        # Curriculum: start strict, relax over iterations
        percentile = max(
            10.0,
            100.0 - self.percentile_decay * iteration,
        )
        threshold = float(np.percentile(confidences[labels >= 0], 100 - percentile)) if np.any(labels >= 0) else 0.0

        # Select training samples above threshold (exclude noise)
        train_mask = (labels >= 0) & (confidences >= threshold)
        n_train = int(train_mask.sum())

        if n_train < 10:
            logger.warning("Too few samples above curriculum threshold; using all non-noise")
            train_mask = labels >= 0
            n_train = int(train_mask.sum())

        if n_train < 10:
            logger.warning("Still too few samples; returning original labels")
            return labels

        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        w_train = confidences[train_mask]

        # Normalise weights to [0.1, 1.0] to avoid zero-weight samples
        w_min, w_max = w_train.min(), w_train.max()
        if w_max > w_min:
            w_train = 0.1 + 0.9 * (w_train - w_min) / (w_max - w_min)
        else:
            w_train = np.ones_like(w_train)

        logger.info(
            f"CWAST iteration {iteration}: threshold={threshold:.3f}, "
            f"percentile={percentile:.0f}%, training on {n_train}/{len(labels)} samples"
        )

        # Train weighted MLP
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=150,
            random_state=self.random_seed,
            early_stopping=True,
            validation_fraction=0.1,
        )

        try:
            # Duplicate samples proportional to weight (sklearn MLP has no sample_weight)
            X_aug, y_aug = self._augment_by_weight(X_train, y_train, w_train)
            classifier.fit(X_aug, y_aug)
        except Exception as e:
            logger.warning(f"CWAST training failed: {e}; falling back to unweighted")
            try:
                classifier.fit(X_train, y_train)
            except Exception as e2:
                logger.error(f"Fallback training also failed: {e2}")
                return labels

        # Predict all samples
        refined = classifier.predict(embeddings)
        return refined

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_by_weight(
        X: np.ndarray, y: np.ndarray, weights: np.ndarray, max_factor: int = 3
    ):
        """Approximate sample weighting by repeating high-confidence rows."""
        # Scale weights to repeat counts in [1, max_factor]
        repeat_counts = np.clip(
            np.round(weights * max_factor).astype(int), 1, max_factor
        )
        X_aug = np.repeat(X, repeat_counts, axis=0)
        y_aug = np.repeat(y, repeat_counts, axis=0)
        return X_aug, y_aug

    def _fallback_train(self, embeddings: np.ndarray, labeled_data: list) -> np.ndarray:
        """Plain self-training without adaptive weighting."""
        labels = np.array([item.cluster_id for item in labeled_data])
        mask = labels >= 0
        if mask.sum() < 10:
            return labels

        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=100,
            random_state=self.random_seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        try:
            clf.fit(embeddings[mask], labels[mask])
            return clf.predict(embeddings)
        except Exception:
            return labels
