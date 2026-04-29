"""Custom exceptions for ZLSDE pipeline."""


class ZLSDEError(Exception):
    """Base exception for ZLSDE pipeline errors."""

    pass


class EmptyDatasetError(ZLSDEError):
    """Raised when data source contains no valid items."""

    pass


class ClusteringError(ZLSDEError):
    """Raised when all clustering methods fail."""

    pass


class InsufficientDataError(ZLSDEError):
    """Raised when insufficient data for operation."""

    pass


class ExportError(ZLSDEError):
    """Raised when dataset export fails."""

    pass


class ValidationError(ZLSDEError):
    """Raised when configuration or data validation fails."""

    pass
