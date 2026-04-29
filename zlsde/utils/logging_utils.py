"""Logging utilities for structured logging."""

import logging
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up structured logger with consistent formatting.

    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to (in addition to stdout)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("zlsde.pipeline", level="DEBUG")
        >>> logger.info("Pipeline started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = "") -> None:
    """
    Log metrics in a structured format.

    Args:
        logger: Logger instance
        metrics: Dictionary of metric names to values
        prefix: Optional prefix for metric names

    Example:
        >>> logger = setup_logger("zlsde")
        >>> log_metrics(logger, {"accuracy": 0.95, "loss": 0.05}, prefix="train")
        # Logs: train/accuracy: 0.95, train/loss: 0.05
    """
    metric_str = ", ".join(
        [
            f"{prefix}/{k}: {v:.4f}" if prefix else f"{k}: {v:.4f}"
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        ]
    )
    logger.info(metric_str)


def log_stage(logger: logging.Logger, stage: str, status: str = "started") -> None:
    """
    Log pipeline stage transitions.

    Args:
        logger: Logger instance
        stage: Stage name (e.g., "Data Ingestion", "Clustering")
        status: Status message (e.g., "started", "completed", "failed")

    Example:
        >>> logger = setup_logger("zlsde")
        >>> log_stage(logger, "Data Ingestion", "started")
        >>> # ... perform ingestion ...
        >>> log_stage(logger, "Data Ingestion", "completed")
    """
    separator = "=" * 60
    logger.info(f"\n{separator}")
    logger.info(f"{stage.upper()} - {status.upper()}")
    logger.info(f"{separator}\n")


def log_iteration(logger: logging.Logger, iteration: int, metrics: dict) -> None:
    """
    Log self-training iteration metrics.

    Args:
        logger: Logger instance
        iteration: Iteration number
        metrics: Dictionary of iteration metrics

    Example:
        >>> logger = setup_logger("zlsde")
        >>> log_iteration(logger, 1, {
        ...     "n_clusters": 5,
        ...     "silhouette_score": 0.45,
        ...     "label_flip_rate": 0.15
        ... })
    """
    logger.info(f"\n--- Iteration {iteration} ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def create_log_file_path(output_dir: str, prefix: str = "zlsde") -> str:
    """
    Create timestamped log file path.

    Args:
        output_dir: Directory to store log file
        prefix: Prefix for log file name

    Returns:
        Full path to log file

    Example:
        >>> path = create_log_file_path("./output", "pipeline")
        >>> # Returns: ./output/pipeline_20240115_103045.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{output_dir}/{prefix}_{timestamp}.log"
