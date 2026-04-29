"""Unit tests for utility modules and CLI."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# metrics_utils
# ---------------------------------------------------------------------------


class TestMetricsUtils:
    """Tests for zlsde/utils/metrics_utils.py."""

    # ---- compute_silhouette_score ----

    def test_silhouette_two_clusters(self):
        from zlsde.utils.metrics_utils import compute_silhouette_score

        emb = np.vstack(
            [
                np.random.RandomState(0).randn(20, 4) + np.array([10, 0, 0, 0]),
                np.random.RandomState(1).randn(20, 4) + np.array([-10, 0, 0, 0]),
            ]
        ).astype(np.float32)
        labels = np.array([0] * 20 + [1] * 20)
        score = compute_silhouette_score(emb, labels)
        assert -1.0 <= score <= 1.0

    def test_silhouette_single_cluster_returns_zero(self):
        from zlsde.utils.metrics_utils import compute_silhouette_score

        emb = np.random.randn(10, 4).astype(np.float32)
        labels = np.zeros(10, dtype=int)
        assert compute_silhouette_score(emb, labels) == 0.0

    def test_silhouette_all_noise_returns_zero(self):
        from zlsde.utils.metrics_utils import compute_silhouette_score

        emb = np.random.randn(10, 4).astype(np.float32)
        labels = np.full(10, -1, dtype=int)
        assert compute_silhouette_score(emb, labels) == 0.0

    def test_silhouette_excludes_noise_points(self):
        from zlsde.utils.metrics_utils import compute_silhouette_score

        rng = np.random.RandomState(42)
        emb = np.vstack(
            [
                rng.randn(15, 4) + [5, 0, 0, 0],
                rng.randn(15, 4) + [-5, 0, 0, 0],
                rng.randn(5, 4),  # noise
            ]
        ).astype(np.float32)
        labels = np.array([0] * 15 + [1] * 15 + [-1] * 5)
        score = compute_silhouette_score(emb, labels)
        assert -1.0 <= score <= 1.0

    # ---- compute_cluster_purity ----

    def test_cluster_purity_in_range(self):
        from zlsde.utils.metrics_utils import compute_cluster_purity

        rng = np.random.RandomState(0)
        emb = rng.randn(20, 4).astype(np.float32)
        labels = np.array([i % 4 for i in range(20)], dtype=int)
        purity = compute_cluster_purity(emb, labels)
        assert 0.0 <= purity <= 1.0

    def test_cluster_purity_all_noise_returns_zero(self):
        from zlsde.utils.metrics_utils import compute_cluster_purity

        emb = np.random.randn(10, 4).astype(np.float32)
        labels = np.full(10, -1, dtype=int)
        assert compute_cluster_purity(emb, labels) == 0.0

    # ---- compute_label_flip_rate ----

    def test_flip_rate_no_flips(self):
        from zlsde.utils.metrics_utils import compute_label_flip_rate

        a = np.array([0, 1, 2, 0])
        assert compute_label_flip_rate(a, a.copy()) == 0.0

    def test_flip_rate_all_flips(self):
        from zlsde.utils.metrics_utils import compute_label_flip_rate

        a = np.array([0, 0, 0])
        b = np.array([1, 1, 1])
        assert compute_label_flip_rate(a, b) == 1.0

    def test_flip_rate_partial(self):
        from zlsde.utils.metrics_utils import compute_label_flip_rate

        a = np.array([0, 1, 2, 0, 1])
        b = np.array([0, 1, 2, 1, 1])  # 1 of 5 changed
        assert abs(compute_label_flip_rate(a, b) - 0.2) < 1e-9

    def test_flip_rate_length_mismatch_raises(self):
        from zlsde.utils.metrics_utils import compute_label_flip_rate

        with pytest.raises(ValueError):
            compute_label_flip_rate(np.array([0, 1]), np.array([0]))

    def test_flip_rate_empty_returns_zero(self):
        from zlsde.utils.metrics_utils import compute_label_flip_rate

        a = np.array([], dtype=int)
        assert compute_label_flip_rate(a, a.copy()) == 0.0

    # ---- compute_noise_ratio ----

    def test_noise_ratio_no_noise(self):
        from zlsde.utils.metrics_utils import compute_noise_ratio

        assert compute_noise_ratio(np.array([0, 1, 2])) == 0.0

    def test_noise_ratio_all_noise(self):
        from zlsde.utils.metrics_utils import compute_noise_ratio

        assert compute_noise_ratio(np.array([-1, -1, -1])) == 1.0

    def test_noise_ratio_partial(self):
        from zlsde.utils.metrics_utils import compute_noise_ratio

        labels = np.array([0, 1, -1, 0, -1, 2])  # 2/6 noise
        assert abs(compute_noise_ratio(labels) - (2 / 6)) < 1e-9

    def test_noise_ratio_empty_returns_zero(self):
        from zlsde.utils.metrics_utils import compute_noise_ratio

        assert compute_noise_ratio(np.array([], dtype=int)) == 0.0

    # ---- compute_cluster_distribution ----

    def test_cluster_distribution_counts(self):
        from zlsde.utils.metrics_utils import compute_cluster_distribution

        labels = np.array([0, 1, 0, 2, 1, 0])
        dist = compute_cluster_distribution(labels)
        assert dist[0] == 3
        assert dist[1] == 2
        assert dist[2] == 1

    def test_cluster_distribution_includes_noise(self):
        from zlsde.utils.metrics_utils import compute_cluster_distribution

        labels = np.array([-1, 0, 0, -1])
        dist = compute_cluster_distribution(labels)
        assert dist[-1] == 2
        assert dist[0] == 2

    # ---- compute_quality_statistics ----

    def test_quality_statistics_keys(self):
        from zlsde.utils.metrics_utils import compute_quality_statistics

        scores = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
        stats = compute_quality_statistics(scores)
        assert {"mean", "std", "min", "max", "median"} <= stats.keys()

    def test_quality_statistics_empty_returns_zeros(self):
        from zlsde.utils.metrics_utils import compute_quality_statistics

        stats = compute_quality_statistics(np.array([]))
        assert stats["mean"] == 0.0

    def test_quality_statistics_values_correct(self):
        from zlsde.utils.metrics_utils import compute_quality_statistics

        scores = np.array([1.0, 1.0, 1.0])
        stats = compute_quality_statistics(scores)
        assert stats["mean"] == 1.0
        assert stats["std"] == 0.0
        assert stats["min"] == 1.0
        assert stats["max"] == 1.0

    # ---- compute_adjusted_rand_index ----

    def test_ari_perfect_agreement(self):
        from zlsde.utils.metrics_utils import compute_adjusted_rand_index

        labels = np.array([0, 0, 1, 1, 2, 2])
        assert abs(compute_adjusted_rand_index(labels, labels.copy()) - 1.0) < 1e-6

    def test_ari_too_few_samples_returns_zero(self):
        from zlsde.utils.metrics_utils import compute_adjusted_rand_index

        # All noise → masked → fewer than 2 non-noise
        a = np.array([-1, -1])
        assert compute_adjusted_rand_index(a, a) == 0.0

    # ---- compute_normalized_mutual_info ----

    def test_nmi_perfect_agreement(self):
        from zlsde.utils.metrics_utils import compute_normalized_mutual_info

        labels = np.array([0, 0, 1, 1, 2, 2])
        assert abs(compute_normalized_mutual_info(labels, labels.copy()) - 1.0) < 1e-6

    def test_nmi_returns_zero_for_all_noise(self):
        from zlsde.utils.metrics_utils import compute_normalized_mutual_info

        a = np.array([-1, -1, -1])
        assert compute_normalized_mutual_info(a, a) == 0.0


# ---------------------------------------------------------------------------
# validation_utils
# ---------------------------------------------------------------------------


class TestValidationUtils:
    """Tests for zlsde/utils/validation_utils.py."""

    # ---- validate_embeddings ----

    def test_validate_embeddings_ok(self):
        from zlsde.utils.validation_utils import validate_embeddings

        validate_embeddings(np.random.randn(10, 4).astype(np.float32))  # no error

    def test_validate_embeddings_not_ndarray_raises(self):
        from zlsde.utils.validation_utils import validate_embeddings

        with pytest.raises(ValueError, match="numpy array"):
            validate_embeddings([[1, 2], [3, 4]])

    def test_validate_embeddings_1d_raises(self):
        from zlsde.utils.validation_utils import validate_embeddings

        with pytest.raises(ValueError, match="2D"):
            validate_embeddings(np.array([1.0, 2.0, 3.0]))

    def test_validate_embeddings_empty_raises(self):
        from zlsde.utils.validation_utils import validate_embeddings

        with pytest.raises(ValueError, match="empty"):
            validate_embeddings(np.empty((0, 4)))

    def test_validate_embeddings_nan_raises(self):
        from zlsde.utils.validation_utils import validate_embeddings

        emb = np.ones((5, 4), dtype=np.float32)
        emb[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_embeddings(emb)

    def test_validate_embeddings_inf_raises(self):
        from zlsde.utils.validation_utils import validate_embeddings

        emb = np.ones((5, 4), dtype=np.float32)
        emb[0, 0] = np.inf
        with pytest.raises(ValueError, match="Inf"):
            validate_embeddings(emb)

    def test_validate_embeddings_zero_features_raises(self):
        from zlsde.utils.validation_utils import validate_embeddings

        with pytest.raises(ValueError):
            validate_embeddings(np.empty((5, 0)))

    # ---- validate_labels ----

    def test_validate_labels_ok(self):
        from zlsde.utils.validation_utils import validate_labels

        validate_labels(np.array([0, 1, -1, 2]), n_samples=4)

    def test_validate_labels_not_ndarray_raises(self):
        from zlsde.utils.validation_utils import validate_labels

        with pytest.raises(ValueError, match="numpy array"):
            validate_labels([0, 1, 2])

    def test_validate_labels_2d_raises(self):
        from zlsde.utils.validation_utils import validate_labels

        with pytest.raises(ValueError):
            validate_labels(np.array([[0, 1], [2, 3]]))

    def test_validate_labels_length_mismatch_raises(self):
        from zlsde.utils.validation_utils import validate_labels

        with pytest.raises(ValueError):
            validate_labels(np.array([0, 1, 2]), n_samples=5)


# ---------------------------------------------------------------------------
# CLI (cli.py)
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the CLI entry point (zlsde/cli.py)."""

    def _run_main(self, args):
        """Run main() with given sys.argv args, return exit code."""
        from zlsde.cli import main

        with patch.object(sys, "argv", ["zlsde"] + args):
            try:
                return main()
            except SystemExit as e:
                return e.code

    def test_missing_config_arg_exits_nonzero(self):
        """--config is required; omitting it should cause a non-zero exit."""
        from zlsde.cli import main

        with patch.object(sys, "argv", ["zlsde"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0

    def test_nonexistent_config_file_returns_1(self, tmp_path):
        code = self._run_main(["--config", "/no/such/config.yaml"])
        assert code == 1

    def test_valid_config_invokes_orchestrator(self, tmp_path):
        """A valid config file should reach the orchestrator (mocked)."""
        import yaml as yaml_mod

        from zlsde.config.config_loader import ConfigLoader
        from zlsde.models.data_models import (
            DataSource,
            IterationMetrics,
            PipelineConfig,
            PipelineResult,
        )

        # Write a minimal config YAML
        cfg_dict = {
            "data": {
                "sources": [{"type": "text", "path": str(tmp_path / "data.txt")}],
                "modality": "text",
            },
            "labeling": {"use_llm": False},
        }
        (tmp_path / "data.txt").write_text("sample\n", encoding="utf-8")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_mod.dump(cfg_dict), encoding="utf-8")

        # Build a fake PipelineResult to return
        from datetime import datetime

        fake_metrics = IterationMetrics(
            iteration=1,
            silhouette_score=0.5,
            n_clusters=2,
            noise_ratio=0.0,
            label_flip_rate=0.01,
            cluster_purity=0.8,
            quality_mean=0.7,
            quality_std=0.1,
            timestamp=datetime.now().isoformat(),
        )
        fake_config = ConfigLoader.from_dict(cfg_dict)
        fake_result = PipelineResult(
            status="completed",
            dataset_path=str(tmp_path / "output"),
            n_samples=1,
            n_labeled=1,
            final_metrics=fake_metrics,
            iteration_history=[fake_metrics],
            config_snapshot=fake_config,
            execution_time_seconds=0.1,
        )

        with patch("zlsde.cli.PipelineOrchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = fake_result
            code = self._run_main(["--config", str(cfg_file)])
        assert code == 0

    def test_output_override_sets_config_path(self, tmp_path):
        """--output flag should override config.output_path."""
        import yaml as yaml_mod

        from zlsde.config.config_loader import ConfigLoader
        from zlsde.models.data_models import IterationMetrics, PipelineResult

        cfg_dict = {
            "data": {
                "sources": [{"type": "text", "path": str(tmp_path / "data.txt")}],
                "modality": "text",
            },
            "labeling": {"use_llm": False},
        }
        (tmp_path / "data.txt").write_text("sample\n", encoding="utf-8")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_mod.dump(cfg_dict), encoding="utf-8")

        from datetime import datetime

        fake_metrics = IterationMetrics(
            iteration=1,
            silhouette_score=0.5,
            n_clusters=2,
            noise_ratio=0.0,
            label_flip_rate=0.0,
            cluster_purity=0.8,
            quality_mean=0.7,
            quality_std=0.1,
            timestamp=datetime.now().isoformat(),
        )
        fake_config = ConfigLoader.from_dict(cfg_dict)
        fake_result = PipelineResult(
            status="completed",
            dataset_path="/custom/out",
            n_samples=1,
            n_labeled=1,
            final_metrics=fake_metrics,
            iteration_history=[fake_metrics],
            config_snapshot=fake_config,
            execution_time_seconds=0.1,
        )

        captured_config = {}

        def fake_orchestrator(config):
            captured_config["output_path"] = config.output_path
            m = MagicMock()
            m.run.return_value = fake_result
            return m

        with patch("zlsde.cli.PipelineOrchestrator", side_effect=fake_orchestrator):
            self._run_main(["--config", str(cfg_file), "--output", "/custom/out"])

        assert captured_config.get("output_path") == "/custom/out"
