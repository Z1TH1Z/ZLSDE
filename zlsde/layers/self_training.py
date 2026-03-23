"""Layer 6: Self-Training Loop - Iteratively refine labels through self-training."""

import logging
import numpy as np
from typing import List
from zlsde.models.data_models import IterationMetrics

logger = logging.getLogger(__name__)


class SelfTrainingLoop:
    """Iteratively refine labels through self-training and measure convergence."""
    
    def __init__(self, config):
        """Initialize self-training loop with configuration."""
        self.config = config
        self.convergence_threshold = getattr(config, 'convergence_threshold', 0.02)
        self.max_iterations = getattr(config, 'max_iterations', 3)
    
    def train_classifier(self, embeddings: np.ndarray, labels: np.ndarray, 
                        confidence_threshold: float = 0.8):
        """
        Train lightweight classifier on high-confidence samples.
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Array of labels (n_samples,)
            confidence_threshold: Minimum confidence for training samples
            
        Returns:
            Trained classifier
        """
        from sklearn.neural_network import MLPClassifier
        
        # For initial training, use all non-noise samples
        # In later iterations, this would filter by confidence
        valid_mask = labels >= 0  # Exclude noise (-1)
        X_train = embeddings[valid_mask]
        y_train = labels[valid_mask]
        
        if len(X_train) < 10:
            logger.warning(f"Only {len(X_train)} samples for training. Lowering threshold.")
            # Use all samples if too few
            X_train = embeddings
            y_train = labels
        
        if len(X_train) < 10:
            raise ValueError("Insufficient samples for classifier training (< 10)")
        
        logger.info(f"Training classifier on {len(X_train)} samples")
        
        # Train MLP classifier
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        try:
            classifier.fit(X_train, y_train)
            logger.info("Classifier training complete")
        except Exception as e:
            logger.error(f"Classifier training failed: {e}")
            raise
        
        return classifier
    
    def refine_labels(self, classifier, embeddings: np.ndarray) -> np.ndarray:
        """
        Re-predict labels using trained classifier.
        
        Args:
            classifier: Trained classifier
            embeddings: Array of embeddings
            
        Returns:
            Array of refined labels
        """
        try:
            refined_labels = classifier.predict(embeddings)
            logger.info("Label refinement complete")
            return refined_labels
        except Exception as e:
            logger.error(f"Label refinement failed: {e}")
            raise
    
    def compute_stability(self, old_labels: np.ndarray, new_labels: np.ndarray) -> float:
        """
        Compute label flip rate between iterations.
        
        Args:
            old_labels: Labels from previous iteration
            new_labels: Labels from current iteration
            
        Returns:
            Flip rate (percentage of labels that changed)
        """
        if len(old_labels) != len(new_labels):
            raise ValueError("Label arrays must have same length")
        
        # Count label changes
        n_flips = np.sum(old_labels != new_labels)
        flip_rate = n_flips / len(old_labels)
        
        logger.info(f"Label flip rate: {flip_rate:.2%} ({n_flips}/{len(old_labels)} labels changed)")
        
        return float(flip_rate)
    
    def check_convergence(self, metrics_history: List[IterationMetrics]) -> bool:
        """
        Determine if training has converged.
        
        Args:
            metrics_history: List of metrics from all iterations
            
        Returns:
            True if converged, False otherwise
        """
        if len(metrics_history) < 2:
            return False
        
        current = metrics_history[-1]
        previous = metrics_history[-2]
        
        # Criterion 1: Label flip rate below threshold
        label_stable = current.label_flip_rate < self.convergence_threshold
        
        # Criterion 2: Cluster count stable
        cluster_stable = current.n_clusters == previous.n_clusters
        
        # Criterion 3: Silhouette score not decreasing significantly
        silhouette_stable = current.silhouette_score >= previous.silhouette_score - 0.05
        
        # Converged if all criteria met
        converged = label_stable and cluster_stable and silhouette_stable
        
        if converged:
            logger.info("Convergence criteria met!")
        else:
            logger.info(f"Not converged - Label stable: {label_stable}, "
                       f"Cluster stable: {cluster_stable}, "
                       f"Silhouette stable: {silhouette_stable}")
        
        return converged
