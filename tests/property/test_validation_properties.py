"""Property-based tests for validation helpers."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from zlsde.utils.validation_utils import (
    validate_clustering_method,
    validate_confidence_scores,
    validate_modality,
    validate_output_format,
    validate_positive_integer,
    validate_probability,
)


@pytest.mark.property
@given(value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_validate_probability_accepts_unit_interval(value: float) -> None:
    """Any finite float inside [0, 1] should be valid."""
    validate_probability(value, "p")


@pytest.mark.property
@given(
    value=st.floats(
        min_value=-1_000_000,
        max_value=1_000_000,
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: x < 0.0 or x > 1.0)
)
def test_validate_probability_rejects_out_of_range(value: float) -> None:
    """Any finite float outside [0, 1] should fail."""
    with pytest.raises(ValueError):
        validate_probability(value, "p")


@pytest.mark.property
@given(value=st.integers(min_value=1, max_value=10_000))
def test_validate_positive_integer_accepts_positive_values(value: int) -> None:
    """Positive integers should satisfy positive integer validation."""
    validate_positive_integer(value, "batch_size", min_value=1)


@pytest.mark.property
@given(value=st.integers(max_value=0))
def test_validate_positive_integer_rejects_non_positive_values(value: int) -> None:
    """Values below the lower bound should fail."""
    with pytest.raises(ValueError):
        validate_positive_integer(value, "batch_size", min_value=1)


@pytest.mark.property
@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=64,
    )
)
def test_validate_confidence_scores_accepts_unit_interval_lists(scores: list[float]) -> None:
    """Confidence vectors in [0, 1] should be valid."""
    arr = np.array(scores, dtype=np.float64)
    validate_confidence_scores(arr, n_samples=len(scores))


@pytest.mark.property
@given(method=st.sampled_from(["auto", "hdbscan", "kmeans", "spectral"]))
def test_validate_clustering_method_accepts_known_methods(method: str) -> None:
    """Known clustering methods should always validate."""
    validate_clustering_method(method)


@pytest.mark.property
@given(modality=st.sampled_from(["text", "image", "multimodal"]))
def test_validate_modality_accepts_known_modalities(modality: str) -> None:
    """Known modalities should always validate."""
    validate_modality(modality)


@pytest.mark.property
@given(fmt=st.sampled_from(["csv", "json", "parquet", "huggingface"]))
def test_validate_output_format_accepts_known_formats(fmt: str) -> None:
    """Known output formats should always validate."""
    validate_output_format(fmt)
