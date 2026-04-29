"""Unit tests for ConfigLoader, custom exceptions, and PseudoLabelGenerator."""

import json
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from zlsde.models.data_models import DataSource, Label, PipelineConfig, ProviderConfig, RawDataItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _min_yaml_dict(source_path="dummy.txt", source_type="text"):
    return {
        "data": {
            "sources": [{"type": source_type, "path": source_path}],
            "modality": "text",
        },
        "labeling": {
            "use_llm": False,
        },
    }


def _raw_item(content="hello world", embedding_dim=8):
    rng = np.random.RandomState(42)
    return RawDataItem(
        id=str(uuid.uuid4()),
        content=content,
        modality="text",
        embedding=rng.randn(embedding_dim).astype(np.float32),
        content_hash="abc",
    )


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    """Tests for ConfigLoader (config/config_loader.py)."""

    # ---- from_dict ----

    def test_from_dict_minimal(self):
        from zlsde.config.config_loader import ConfigLoader

        cfg = ConfigLoader.from_dict(_min_yaml_dict())
        assert cfg.modality == "text"
        assert len(cfg.data_sources) == 1

    def test_from_dict_missing_sources_raises(self):
        from zlsde.config.config_loader import ConfigLoader

        with pytest.raises(ValueError, match="data.sources"):
            ConfigLoader.from_dict({"data": {"modality": "text"}})

    def test_from_dict_source_missing_type_raises(self):
        from zlsde.config.config_loader import ConfigLoader

        d = {"data": {"sources": [{"path": "file.txt"}]}}
        with pytest.raises(ValueError, match="type"):
            ConfigLoader.from_dict(d)

    def test_from_dict_source_missing_path_raises(self):
        from zlsde.config.config_loader import ConfigLoader

        d = {"data": {"sources": [{"type": "text"}]}}
        with pytest.raises(ValueError, match="path"):
            ConfigLoader.from_dict(d)

    def test_from_dict_backward_compat_llm_model(self):
        """llm_model field should map to provider_config.local_model."""
        from zlsde.config.config_loader import ConfigLoader

        d = _min_yaml_dict()
        d["labeling"]["llm_model"] = "google/flan-t5-base"
        cfg = ConfigLoader.from_dict(d)
        assert cfg.provider_config.local_model == "google/flan-t5-base"

    def test_from_dict_api_provider_type(self):
        from zlsde.config.config_loader import ConfigLoader

        d = _min_yaml_dict()
        d["labeling"]["provider_config"] = {
            "provider_type": "api",
            "api_providers": ["groq"],
        }
        cfg = ConfigLoader.from_dict(d)
        assert cfg.provider_config.provider_type == "api"

    def test_from_dict_clustering_fields(self):
        from zlsde.config.config_loader import ConfigLoader

        d = _min_yaml_dict()
        d["clustering"] = {"method": "kmeans", "min_cluster_size": 5, "n_clusters": 4}
        cfg = ConfigLoader.from_dict(d)
        assert cfg.clustering_method == "kmeans"
        assert cfg.min_cluster_size == 5
        assert cfg.n_clusters == 4

    def test_from_dict_output_fields(self):
        from zlsde.config.config_loader import ConfigLoader

        d = _min_yaml_dict()
        d["output"] = {"format": "json", "path": "/tmp/out"}
        cfg = ConfigLoader.from_dict(d)
        assert cfg.output_format == "json"
        assert cfg.output_path == "/tmp/out"

    def test_from_dict_embedding_fields(self):
        from zlsde.config.config_loader import ConfigLoader

        d = _min_yaml_dict()
        d["embedding"] = {"model": "bert-base", "batch_size": 64}
        cfg = ConfigLoader.from_dict(d)
        assert cfg.embedding_model == "bert-base"
        assert cfg.batch_size == 64

    def test_from_dict_training_fields(self):
        from zlsde.config.config_loader import ConfigLoader

        d = _min_yaml_dict()
        d["training"] = {"max_iterations": 5, "convergence_threshold": 0.01}
        cfg = ConfigLoader.from_dict(d)
        assert cfg.max_iterations == 5
        assert cfg.convergence_threshold == 0.01

    # ---- from_yaml ----

    def test_from_yaml_loads_file(self, tmp_path):
        from zlsde.config.config_loader import ConfigLoader

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(_min_yaml_dict()), encoding="utf-8")
        cfg = ConfigLoader.from_yaml(str(yaml_file))
        assert isinstance(cfg, PipelineConfig)

    def test_from_yaml_missing_file_raises(self):
        from zlsde.config.config_loader import ConfigLoader

        with pytest.raises(FileNotFoundError):
            ConfigLoader.from_yaml("/no/such/config.yaml")

    def test_from_yaml_empty_file_raises(self, tmp_path):
        from zlsde.config.config_loader import ConfigLoader

        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty"):
            ConfigLoader.from_yaml(str(yaml_file))

    # ---- create_version_snapshot ----

    def test_create_version_snapshot_keys(self):
        from zlsde.config.config_loader import ConfigLoader

        cfg = ConfigLoader.from_dict(_min_yaml_dict())
        snap = ConfigLoader.create_version_snapshot(cfg)
        assert "config_version" in snap
        assert "timestamp" in snap
        assert "config" in snap
        assert "system_info" in snap

    def test_version_snapshot_save_and_load(self, tmp_path):
        from zlsde.config.config_loader import ConfigLoader

        cfg = ConfigLoader.from_dict(_min_yaml_dict())
        snap = ConfigLoader.create_version_snapshot(cfg)
        snap_path = tmp_path / "snapshot.json"
        ConfigLoader.save_version_snapshot(snap, str(snap_path))
        loaded = ConfigLoader.load_version_snapshot(str(snap_path))
        assert loaded["config_version"] == snap["config_version"]

    def test_load_snapshot_missing_file_raises(self):
        from zlsde.config.config_loader import ConfigLoader

        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_version_snapshot("/no/such/snapshot.json")

    def test_load_snapshot_missing_fields_raises(self, tmp_path):
        from zlsde.config.config_loader import ConfigLoader

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required field"):
            ConfigLoader.load_version_snapshot(str(bad))

    # ---- Standalone backward-compat functions ----

    def test_standalone_load_config_from_dict(self):
        from zlsde.config.config_loader import load_config_from_dict

        cfg = load_config_from_dict(_min_yaml_dict())
        assert isinstance(cfg, PipelineConfig)

    def test_standalone_load_config_from_yaml(self, tmp_path):
        from zlsde.config.config_loader import load_config_from_yaml

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(_min_yaml_dict()), encoding="utf-8")
        cfg = load_config_from_yaml(str(yaml_file))
        assert isinstance(cfg, PipelineConfig)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class TestCustomExceptions:
    """Tests for zlsde.exceptions module."""

    def test_zlsde_error_is_exception(self):
        from zlsde.exceptions import ZLSDEError

        assert issubclass(ZLSDEError, Exception)

    def test_empty_dataset_error_inherits(self):
        from zlsde.exceptions import EmptyDatasetError, ZLSDEError

        assert issubclass(EmptyDatasetError, ZLSDEError)

    def test_clustering_error_inherits(self):
        from zlsde.exceptions import ClusteringError, ZLSDEError

        assert issubclass(ClusteringError, ZLSDEError)

    def test_insufficient_data_error_inherits(self):
        from zlsde.exceptions import InsufficientDataError, ZLSDEError

        assert issubclass(InsufficientDataError, ZLSDEError)

    def test_export_error_inherits(self):
        from zlsde.exceptions import ExportError, ZLSDEError

        assert issubclass(ExportError, ZLSDEError)

    def test_validation_error_inherits(self):
        from zlsde.exceptions import ValidationError, ZLSDEError

        assert issubclass(ValidationError, ZLSDEError)

    def test_exceptions_can_be_raised_and_caught(self):
        from zlsde.exceptions import (
            ClusteringError,
            EmptyDatasetError,
            ExportError,
            InsufficientDataError,
            ValidationError,
            ZLSDEError,
        )

        for exc_cls in (
            ZLSDEError,
            EmptyDatasetError,
            ClusteringError,
            InsufficientDataError,
            ExportError,
            ValidationError,
        ):
            with pytest.raises(exc_cls):
                raise exc_cls("test message")

    def test_exception_message_preserved(self):
        from zlsde.exceptions import ClusteringError

        try:
            raise ClusteringError("all methods failed")
        except ClusteringError as e:
            assert "all methods failed" in str(e)

    def test_zlsde_error_catches_subclasses(self):
        from zlsde.exceptions import EmptyDatasetError, ZLSDEError

        with pytest.raises(ZLSDEError):
            raise EmptyDatasetError("no data")


# ---------------------------------------------------------------------------
# PseudoLabelGenerator
# ---------------------------------------------------------------------------


class TestPseudoLabelGenerator:
    """Tests for PseudoLabelGenerator (layers/label_generation.py)."""

    def _make_provider_manager(self, return_value="technology"):
        manager = MagicMock()
        manager.generate_label.return_value = return_value
        return manager

    def _raw_items(self, n=5, content="machine learning model"):
        return [_raw_item(content=content) for _ in range(n)]

    # ---- generate_labels ----

    def test_generate_labels_returns_label_for_each_cluster(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        pm = self._make_provider_manager("technology")
        gen = PseudoLabelGenerator(pm)
        clusters = {0: self._raw_items(3), 1: self._raw_items(3)}
        labels = gen.generate_labels(clusters)
        assert set(labels.keys()) == {0, 1}
        for lbl in labels.values():
            assert isinstance(lbl, Label)

    def test_generate_labels_noise_cluster_gets_noise_label(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        pm = self._make_provider_manager("science")
        gen = PseudoLabelGenerator(pm)
        labels = gen.generate_labels({-1: self._raw_items(3)})
        assert labels[-1].text == "noise"
        assert labels[-1].confidence == 0.0

    def test_generate_labels_fallback_on_provider_failure(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator
        from zlsde.providers.exceptions import AllProvidersFailedError

        pm = MagicMock()
        pm.generate_label.side_effect = AllProvidersFailedError("all failed")
        gen = PseudoLabelGenerator(pm)
        labels = gen.generate_labels({0: self._raw_items(3)})
        assert "cluster_0" in labels[0].text or labels[0].text.startswith("cluster_")

    def test_generate_labels_empty_clusters(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        pm = self._make_provider_manager("foo")
        gen = PseudoLabelGenerator(pm)
        labels = gen.generate_labels({})
        assert labels == {}

    # ---- _select_representatives ----

    def test_select_representatives_k_bounded_by_cluster_size(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        pm = self._make_provider_manager()
        gen = PseudoLabelGenerator(pm)
        items = self._raw_items(3)
        reps = gen._select_representatives(items, k=10)
        assert len(reps) == 3  # can't exceed cluster size

    def test_select_representatives_returns_k_items(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        pm = self._make_provider_manager()
        gen = PseudoLabelGenerator(pm)
        items = self._raw_items(10)
        reps = gen._select_representatives(items, k=5)
        assert len(reps) == 5

    def test_select_representatives_empty_returns_empty(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        assert gen._select_representatives([], k=5) == []

    # ---- _create_prompt ----

    def test_create_prompt_contains_samples(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        items = self._raw_items(3, content="neural networks")
        prompt = gen._create_prompt(items)
        assert "neural networks" in prompt

    def test_create_prompt_empty_returns_fallback(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        prompt = gen._create_prompt([])
        assert "label" in prompt.lower()

    # ---- _validate_label ----

    def test_validate_label_empty_returns_unlabeled(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        assert gen._validate_label("") == "unlabeled"

    def test_validate_label_short_kept_as_is(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        assert gen._validate_label("machine learning") == "machine learning"

    def test_validate_label_long_trimmed(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        result = gen._validate_label("the quick brown fox jumped over")
        # Should be trimmed to <= 3 words
        assert len(result.split()) <= 3

    def test_validate_label_strips_whitespace(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        assert gen._validate_label("  sports  ") == "sports"

    # ---- _compute_confidence ----

    def test_compute_confidence_in_range(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        conf = gen._compute_confidence("technology", self._raw_items(10))
        assert 0.0 <= conf <= 1.0

    def test_compute_confidence_low_for_fallback_labels(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        conf_fallback = gen._compute_confidence("unlabeled", self._raw_items(50))
        conf_good = gen._compute_confidence("technology", self._raw_items(50))
        assert conf_fallback < conf_good

    # ---- _infer_rule_based_label ----

    def test_infer_rule_based_label_known_domain(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        items = [_raw_item("machine learning neural transformer model models")]
        label = gen._infer_rule_based_label(items)
        assert label == "ai ml concepts"

    def test_infer_rule_based_label_empty_items(self):
        from zlsde.layers.label_generation import PseudoLabelGenerator

        gen = PseudoLabelGenerator(self._make_provider_manager())
        assert gen._infer_rule_based_label([]) == "unlabeled"
