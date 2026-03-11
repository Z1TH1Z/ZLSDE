"""Utility functions for ZLSDE pipeline."""

# Seed control
from zlsde.utils.seed_control import (
    set_random_seed,
    get_random_state,
    restore_random_state,
)

# Logging utilities
from zlsde.utils.logging_utils import (
    setup_logger,
    log_metrics,
    log_stage,
    log_iteration,
    create_log_file_path,
)

# Metrics utilities
from zlsde.utils.metrics_utils import (
    compute_silhouette_score,
    compute_cluster_purity,
    compute_label_flip_rate,
    compute_noise_ratio,
    compute_cluster_distribution,
    compute_quality_statistics,
    compute_adjusted_rand_index,
    compute_normalized_mutual_info,
)

# Validation utilities
from zlsde.utils.validation_utils import (
    validate_embeddings,
    validate_labels,
    validate_quality_scores,
    validate_confidence_scores,
    validate_file_path,
    validate_modality,
    validate_device,
    validate_positive_integer,
    validate_probability,
    validate_non_empty_list,
    validate_clustering_method,
    validate_output_format,
)

__all__ = [
    # Seed control
    "set_random_seed",
    "get_random_state",
    "restore_random_state",
    # Logging
    "setup_logger",
    "log_metrics",
    "log_stage",
    "log_iteration",
    "create_log_file_path",
    # Metrics
    "compute_silhouette_score",
    "compute_cluster_purity",
    "compute_label_flip_rate",
    "compute_noise_ratio",
    "compute_cluster_distribution",
    "compute_quality_statistics",
    "compute_adjusted_rand_index",
    "compute_normalized_mutual_info",
    # Validation
    "validate_embeddings",
    "validate_labels",
    "validate_quality_scores",
    "validate_confidence_scores",
    "validate_file_path",
    "validate_modality",
    "validate_device",
    "validate_positive_integer",
    "validate_probability",
    "validate_non_empty_list",
    "validate_clustering_method",
    "validate_output_format",
]
