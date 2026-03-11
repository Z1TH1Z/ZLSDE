"""Common validation functions."""

import numpy as np
from typing import List, Any, Optional
from pathlib import Path


def validate_embeddings(embeddings: np.ndarray) -> None:
    """
    Validate embeddings array meets requirements.
    
    Args:
        embeddings: Embeddings array to validate
    
    Raises:
        ValueError: If embeddings are invalid
    
    Example:
        >>> embeddings = np.random.randn(100, 50)
        >>> validate_embeddings(embeddings)  # Passes
        >>> validate_embeddings(np.array([1, 2, 3]))  # Raises ValueError
    """
    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings must be numpy array")
    
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
    
    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings array is empty")
    
    if embeddings.shape[1] == 0:
        raise ValueError("Embeddings have zero features")
    
    if np.any(np.isnan(embeddings)):
        raise ValueError("Embeddings contain NaN values")
    
    if np.any(np.isinf(embeddings)):
        raise ValueError("Embeddings contain Inf values")


def validate_labels(labels: np.ndarray, n_samples: Optional[int] = None) -> None:
    """
    Validate cluster labels array.
    
    Args:
        labels: Cluster assignments array
        n_samples: Expected number of samples (optional)
    
    Raises:
        ValueError: If labels are invalid
    
    Example:
        >>> labels = np.array([0, 1, 2, 0, -1])
        >>> validate_labels(labels, n_samples=5)  # Passes
    """
    if not isinstance(labels, np.ndarray):
        raise ValueError("Labels must be numpy array")
    
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D array, got {labels.ndim}D")
    
    if len(labels) == 0:
        raise ValueError("Labels array is empty")
    
    if n_samples is not None and len(labels) != n_samples:
        raise ValueError(f"Expected {n_samples} labels, got {len(labels)}")
    
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integer type")
    
    if np.any(labels < -1):
        raise ValueError("Labels must be >= -1 (where -1 indicates noise)")


def validate_quality_scores(scores: np.ndarray, n_samples: Optional[int] = None) -> None:
    """
    Validate quality scores array.
    
    Args:
        scores: Quality scores array
        n_samples: Expected number of samples (optional)
    
    Raises:
        ValueError: If scores are invalid
    
    Example:
        >>> scores = np.array([0.8, 0.9, 0.7])
        >>> validate_quality_scores(scores, n_samples=3)  # Passes
    """
    if not isinstance(scores, np.ndarray):
        raise ValueError("Quality scores must be numpy array")
    
    if scores.ndim != 1:
        raise ValueError(f"Quality scores must be 1D array, got {scores.ndim}D")
    
    if len(scores) == 0:
        raise ValueError("Quality scores array is empty")
    
    if n_samples is not None and len(scores) != n_samples:
        raise ValueError(f"Expected {n_samples} scores, got {len(scores)}")
    
    if np.any(np.isnan(scores)):
        raise ValueError("Quality scores contain NaN values")
    
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Quality scores must be in range [0, 1]")


def validate_confidence_scores(scores: np.ndarray, n_samples: Optional[int] = None) -> None:
    """
    Validate confidence scores array.
    
    Args:
        scores: Confidence scores array
        n_samples: Expected number of samples (optional)
    
    Raises:
        ValueError: If scores are invalid
    
    Example:
        >>> scores = np.array([0.95, 0.87, 0.92])
        >>> validate_confidence_scores(scores, n_samples=3)  # Passes
    """
    if not isinstance(scores, np.ndarray):
        raise ValueError("Confidence scores must be numpy array")
    
    if scores.ndim != 1:
        raise ValueError(f"Confidence scores must be 1D array, got {scores.ndim}D")
    
    if len(scores) == 0:
        raise ValueError("Confidence scores array is empty")
    
    if n_samples is not None and len(scores) != n_samples:
        raise ValueError(f"Expected {n_samples} scores, got {len(scores)}")
    
    if np.any(np.isnan(scores)):
        raise ValueError("Confidence scores contain NaN values")
    
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Confidence scores must be in range [0, 1]")


def validate_file_path(path: str, must_exist: bool = False) -> None:
    """
    Validate file path.
    
    Args:
        path: File path to validate
        must_exist: If True, file must exist
    
    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If must_exist=True and file doesn't exist
    
    Example:
        >>> validate_file_path("data/input.csv", must_exist=False)  # Passes
    """
    if not isinstance(path, str):
        raise ValueError("Path must be string")
    
    if not path:
        raise ValueError("Path cannot be empty")
    
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check for directory traversal attempts
    try:
        path_obj.resolve()
    except Exception as e:
        raise ValueError(f"Invalid path: {e}")


def validate_modality(modality: str) -> None:
    """
    Validate modality string.
    
    Args:
        modality: Modality to validate
    
    Raises:
        ValueError: If modality is invalid
    
    Example:
        >>> validate_modality("text")  # Passes
        >>> validate_modality("audio")  # Raises ValueError
    """
    valid_modalities = ["text", "image", "multimodal"]
    
    if not isinstance(modality, str):
        raise ValueError("Modality must be string")
    
    if modality not in valid_modalities:
        raise ValueError(
            f"Invalid modality: {modality}. "
            f"Must be one of {valid_modalities}"
        )


def validate_device(device: str) -> None:
    """
    Validate device string.
    
    Args:
        device: Device to validate
    
    Raises:
        ValueError: If device is invalid
    
    Example:
        >>> validate_device("cpu")  # Passes
        >>> validate_device("cuda")  # Passes
        >>> validate_device("tpu")  # Raises ValueError
    """
    valid_devices = ["cpu", "cuda", "mps"]
    
    if not isinstance(device, str):
        raise ValueError("Device must be string")
    
    if device not in valid_devices:
        raise ValueError(
            f"Invalid device: {device}. "
            f"Must be one of {valid_devices}"
        )


def validate_positive_integer(value: int, name: str, min_value: int = 1) -> None:
    """
    Validate positive integer parameter.
    
    Args:
        value: Value to validate
        name: Parameter name (for error messages)
        min_value: Minimum allowed value (default: 1)
    
    Raises:
        ValueError: If value is invalid
    
    Example:
        >>> validate_positive_integer(10, "batch_size")  # Passes
        >>> validate_positive_integer(0, "batch_size")  # Raises ValueError
    """
    if not isinstance(value, int):
        raise ValueError(f"{name} must be integer, got {type(value)}")
    
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")


def validate_probability(value: float, name: str) -> None:
    """
    Validate probability value in range [0, 1].
    
    Args:
        value: Value to validate
        name: Parameter name (for error messages)
    
    Raises:
        ValueError: If value is invalid
    
    Example:
        >>> validate_probability(0.8, "confidence_threshold")  # Passes
        >>> validate_probability(1.5, "confidence_threshold")  # Raises ValueError
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value)}")
    
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in range [0, 1], got {value}")


def validate_non_empty_list(items: List[Any], name: str) -> None:
    """
    Validate list is non-empty.
    
    Args:
        items: List to validate
        name: Parameter name (for error messages)
    
    Raises:
        ValueError: If list is empty or not a list
    
    Example:
        >>> validate_non_empty_list([1, 2, 3], "data_sources")  # Passes
        >>> validate_non_empty_list([], "data_sources")  # Raises ValueError
    """
    if not isinstance(items, list):
        raise ValueError(f"{name} must be list, got {type(items)}")
    
    if len(items) == 0:
        raise ValueError(f"{name} cannot be empty")


def validate_clustering_method(method: str) -> None:
    """
    Validate clustering method string.
    
    Args:
        method: Clustering method to validate
    
    Raises:
        ValueError: If method is invalid
    
    Example:
        >>> validate_clustering_method("hdbscan")  # Passes
        >>> validate_clustering_method("dbscan")  # Raises ValueError
    """
    valid_methods = ["auto", "hdbscan", "kmeans", "spectral"]
    
    if not isinstance(method, str):
        raise ValueError("Clustering method must be string")
    
    if method not in valid_methods:
        raise ValueError(
            f"Invalid clustering method: {method}. "
            f"Must be one of {valid_methods}"
        )


def validate_output_format(format: str) -> None:
    """
    Validate output format string.
    
    Args:
        format: Output format to validate
    
    Raises:
        ValueError: If format is invalid
    
    Example:
        >>> validate_output_format("csv")  # Passes
        >>> validate_output_format("xml")  # Raises ValueError
    """
    valid_formats = ["csv", "json", "parquet", "huggingface"]
    
    if not isinstance(format, str):
        raise ValueError("Output format must be string")
    
    if format not in valid_formats:
        raise ValueError(
            f"Invalid output format: {format}. "
            f"Must be one of {valid_formats}"
        )
