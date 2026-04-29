"""Unit tests for core pipeline layers: ingestion, clustering, quality_control,
self_training, and exporter."""

import csv
import json
import os
import tempfile
import uuid

import numpy as np
import pytest

from zlsde.models.data_models import DataSource, LabeledDataItem, PipelineConfig, RawDataItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    """Return a minimal PipelineConfig with sensible defaults."""
    defaults = dict(
        data_sources=[DataSource(type="text", path="dummy.txt")],
        modality="text",
        min_cluster_size=2,
        random_seed=42,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_labeled_item(label="cat", cluster_id=0, confidence=0.9, dim=8, content="sample text"):
    embedding = np.random.randn(dim).astype(np.float32)
    return LabeledDataItem(
        id=str(uuid.uuid4()),
        content=content,
        embedding=embedding,
        label=label,
        cluster_id=cluster_id,
        confidence=confidence,
        quality_score=0.8,
        modality="text",
        iteration=1,
    )


# ---------------------------------------------------------------------------
# DataIngestionLayer
# ---------------------------------------------------------------------------


class TestDataIngestionLayer:
    """Tests for DataIngestionLayer (ingestion.py)."""

    def setup_method(self):
        from zlsde.layers.ingestion import DataIngestionLayer

        self.config = _make_config()
        self.layer = DataIngestionLayer(self.config)

    # ---- CSV loading ----

    def test_load_csv_basic(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("content\nhello world\nfoo bar\n", encoding="utf-8")
        source = DataSource(type="csv", path=str(csv_file))
        items = self.layer._load_csv(source)
        assert len(items) == 2
        assert items[0].content == "hello world"
        assert items[1].content == "foo bar"

    def test_load_csv_with_id_column(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,content\nabc,hello\n", encoding="utf-8")
        source = DataSource(type="csv", path=str(csv_file))
        items = self.layer._load_csv(source)
        assert items[0].id == "abc"

    def test_load_csv_skips_empty_rows(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("content\nhello\n\n  \nworld\n", encoding="utf-8")
        source = DataSource(type="csv", path=str(csv_file))
        items = self.layer._load_csv(source)
        assert len(items) == 2

    def test_load_csv_missing_file_raises(self):
        source = DataSource(type="csv", path="/nonexistent/file.csv")
        with pytest.raises(FileNotFoundError):
            self.layer._load_csv(source)

    def test_load_csv_missing_content_column_raises(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,text\nabc,hello\n", encoding="utf-8")
        source = DataSource(type="csv", path=str(csv_file))
        with pytest.raises(ValueError, match="content"):
            self.layer._load_csv(source)

    # ---- JSON loading ----

    def test_load_json_array(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(
            json.dumps([{"content": "alpha"}, {"content": "beta"}]),
            encoding="utf-8",
        )
        source = DataSource(type="json", path=str(json_file))
        items = self.layer._load_json(source)
        assert len(items) == 2
        contents = {i.content for i in items}
        assert contents == {"alpha", "beta"}

    def test_load_json_single_object(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"content": "solo"}), encoding="utf-8")
        source = DataSource(type="json", path=str(json_file))
        items = self.layer._load_json(source)
        assert len(items) == 1
        assert items[0].content == "solo"

    def test_load_json_missing_file_raises(self):
        source = DataSource(type="json", path="/no/such/file.json")
        with pytest.raises(FileNotFoundError):
            self.layer._load_json(source)

    def test_load_json_skips_empty_content(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(
            json.dumps([{"content": ""}, {"content": "valid"}]),
            encoding="utf-8",
        )
        source = DataSource(type="json", path=str(json_file))
        items = self.layer._load_json(source)
        assert len(items) == 1
        assert items[0].content == "valid"

    # ---- Text loading ----

    def test_load_text_each_line_is_item(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("line one\nline two\nline three\n", encoding="utf-8")
        source = DataSource(type="text", path=str(txt_file))
        items = self.layer._load_text(source)
        assert len(items) == 3

    def test_load_text_skips_blank_lines(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("hello\n\n\nworld\n", encoding="utf-8")
        source = DataSource(type="text", path=str(txt_file))
        items = self.layer._load_text(source)
        assert len(items) == 2

    def test_load_text_missing_file_raises(self):
        source = DataSource(type="text", path="/no/such/file.txt")
        with pytest.raises(FileNotFoundError):
            self.layer._load_text(source)

    # ---- Folder loading ----

    def test_load_folder_reads_txt_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("content a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("content b", encoding="utf-8")
        source = DataSource(type="folder", path=str(tmp_path))
        items = self.layer._load_folder(source)
        assert len(items) == 2

    def test_load_folder_ignores_non_txt(self, tmp_path):
        (tmp_path / "a.txt").write_text("good", encoding="utf-8")
        (tmp_path / "b.csv").write_text("bad", encoding="utf-8")
        source = DataSource(type="folder", path=str(tmp_path))
        items = self.layer._load_folder(source)
        assert len(items) == 1

    def test_load_folder_missing_raises(self):
        source = DataSource(type="folder", path="/no/such/folder")
        with pytest.raises(FileNotFoundError):
            self.layer._load_folder(source)

    def test_load_folder_not_a_dir_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        source = DataSource(type="folder", path=str(f))
        with pytest.raises(ValueError):
            self.layer._load_folder(source)

    # ---- load_data dispatch ----

    def test_load_data_unsupported_type_raises(self):
        source = DataSource(type="pdf", path="file.pdf")
        with pytest.raises(ValueError, match="Unsupported source type"):
            self.layer.load_data([source])

    # ---- Deduplication ----

    def test_deduplicate_removes_identical_content(self):
        items = [
            RawDataItem(
                id=str(uuid.uuid4()),
                content="same",
                modality="text",
                content_hash=self.layer._compute_content_hash("same"),
            ),
            RawDataItem(
                id=str(uuid.uuid4()),
                content="same",
                modality="text",
                content_hash=self.layer._compute_content_hash("same"),
            ),
            RawDataItem(
                id=str(uuid.uuid4()),
                content="different",
                modality="text",
                content_hash=self.layer._compute_content_hash("different"),
            ),
        ]
        unique = self.layer.deduplicate(items)
        assert len(unique) == 2

    def test_deduplicate_empty_list(self):
        assert self.layer.deduplicate([]) == []

    def test_deduplicate_all_unique(self):
        items = [
            RawDataItem(
                id=str(uuid.uuid4()),
                content=f"item {i}",
                modality="text",
                content_hash=self.layer._compute_content_hash(f"item {i}"),
            )
            for i in range(5)
        ]
        unique = self.layer.deduplicate(items)
        assert len(unique) == 5

    # ---- Content hash ----

    def test_content_hash_is_deterministic(self):
        h1 = self.layer._compute_content_hash("hello")
        h2 = self.layer._compute_content_hash("hello")
        assert h1 == h2

    def test_content_hash_differs_for_different_content(self):
        assert self.layer._compute_content_hash("a") != self.layer._compute_content_hash("b")

    # ---- Validate ----

    def test_validate_returns_valid_items(self):
        items = [
            RawDataItem(id=str(uuid.uuid4()), content="hello", modality="text", content_hash="abc"),
        ]
        valid = self.layer.validate(items)
        assert len(valid) == 1

    def test_validate_empty_list(self):
        assert self.layer.validate([]) == []


# ---------------------------------------------------------------------------
# ClusteringEngine
# ---------------------------------------------------------------------------


class TestClusteringEngine:
    """Tests for ClusteringEngine (clustering.py)."""

    def setup_method(self):
        from zlsde.layers.clustering import ClusteringEngine

        self.config = _make_config(min_cluster_size=2, n_clusters=None)
        self.engine = ClusteringEngine(self.config)

    def _blobs(self, n=60, n_centers=3, dim=4, seed=0):
        """Generate well-separated blob embeddings."""
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(n_samples=n, centers=n_centers, n_features=dim, random_state=seed)
        return X.astype(np.float32)

    # ---- Input validation ----

    def test_cluster_rejects_1d_array(self):
        with pytest.raises(ValueError, match="2D"):
            self.engine.cluster(np.array([1.0, 2.0, 3.0]))

    def test_cluster_rejects_nan(self):
        emb = self._blobs()
        emb[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            self.engine.cluster(emb)

    def test_cluster_rejects_inf(self):
        emb = self._blobs()
        emb[0, 0] = np.inf
        with pytest.raises(ValueError, match="Inf"):
            self.engine.cluster(emb)

    def test_cluster_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            self.engine.cluster(self._blobs(), method="magic")

    def test_cluster_rejects_too_few_samples(self):
        emb = np.random.randn(1, 4).astype(np.float32)
        with pytest.raises(ValueError, match="min_cluster_size"):
            self.engine.cluster(emb)

    # ---- KMeans ----

    def test_cluster_kmeans_returns_correct_shape(self):
        emb = self._blobs(n=30, n_centers=3)
        config = _make_config(min_cluster_size=2, n_clusters=3)
        from zlsde.layers.clustering import ClusteringEngine

        engine = ClusteringEngine(config)
        result = engine.cluster(emb, method="kmeans")
        assert len(result.labels) == 30
        assert result.n_clusters == 3
        assert result.n_noise == 0

    def test_cluster_kmeans_silhouette_in_range(self):
        emb = self._blobs(n=30, n_centers=3)
        config = _make_config(min_cluster_size=2, n_clusters=3)
        from zlsde.layers.clustering import ClusteringEngine

        result = ClusteringEngine(config).cluster(emb, method="kmeans")
        assert -1.0 <= result.silhouette_score <= 1.0

    def test_cluster_kmeans_method_used(self):
        emb = self._blobs(n=30, n_centers=3)
        config = _make_config(min_cluster_size=2, n_clusters=3)
        from zlsde.layers.clustering import ClusteringEngine

        result = ClusteringEngine(config).cluster(emb, method="kmeans")
        assert result.method_used == "kmeans"

    # ---- Auto method selection ----

    def test_cluster_auto_returns_result(self):
        emb = self._blobs(n=40, n_centers=3)
        result = self.engine.cluster(emb, method="auto")
        assert len(result.labels) == 40
        assert result.method_used in ("hdbscan", "kmeans", "spectral")

    def test_cluster_auto_skips_hdbscan_when_unavailable(self):
        emb = self._blobs(n=40, n_centers=3)
        self.engine._hdbscan_available = False
        self.engine._hdbscan_error = "mock unavailable"
        result = self.engine.cluster(emb, method="auto")
        assert result.method_used in ("kmeans", "spectral")

    def test_cluster_hdbscan_explicit_raises_if_unavailable(self):
        emb = self._blobs(n=40, n_centers=3)
        self.engine._hdbscan_available = False
        self.engine._hdbscan_error = "mock unavailable"
        with pytest.raises(ImportError, match="not available"):
            self.engine.cluster(emb, method="hdbscan")

    # ---- Estimate optimal k ----

    def test_estimate_k_at_least_2(self):
        emb = np.random.randn(10, 4).astype(np.float32)
        k = self.engine._estimate_optimal_k(emb)
        assert k >= 2

    # ---- Silhouette score ----

    def test_silhouette_zero_if_single_cluster(self):
        emb = self._blobs(n=20, n_centers=3)
        labels = np.zeros(20, dtype=int)
        score = self.engine._compute_silhouette_score(emb, labels)
        assert score == 0.0

    def test_silhouette_all_noise_returns_zero(self):
        emb = self._blobs(n=20, n_centers=3)
        labels = np.full(20, -1, dtype=int)
        score = self.engine._compute_silhouette_score(emb, labels)
        assert score == 0.0

    # ---- Compute metrics ----

    def test_compute_metrics_keys(self):
        emb = self._blobs(n=30, n_centers=3)
        labels = np.array([i % 3 for i in range(30)], dtype=int)
        metrics = self.engine.compute_metrics(emb, labels)
        assert {
            "n_clusters",
            "n_noise",
            "noise_ratio",
            "silhouette_score",
            "n_samples",
        } <= metrics.keys()

    def test_compute_metrics_length_mismatch_raises(self):
        emb = np.random.randn(10, 4)
        labels = np.zeros(5, dtype=int)
        with pytest.raises(ValueError):
            self.engine.compute_metrics(emb, labels)

    def test_compute_metrics_noise_ratio(self):
        emb = np.random.randn(10, 4).astype(np.float32)
        labels = np.array([-1, -1, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
        metrics = self.engine.compute_metrics(emb, labels)
        assert abs(metrics["noise_ratio"] - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# QualityControlFilter
# ---------------------------------------------------------------------------


class TestQualityControlFilter:
    """Tests for QualityControlFilter (quality_control.py)."""

    def setup_method(self):
        from zlsde.layers.quality_control import QualityControlFilter

        self.config = _make_config(anomaly_contamination=0.1, duplicate_threshold=0.95)
        self.qc = QualityControlFilter(self.config)

    def _items(self, n=20, dim=8, seed=0):
        rng = np.random.RandomState(seed)
        return [
            _make_labeled_item(cluster_id=i % 3, confidence=rng.uniform(0.5, 1.0)) for i in range(n)
        ]

    # ---- filter ----

    def test_filter_returns_same_count(self):
        items = self._items(20)
        scores = self.qc.filter(items)
        assert len(scores) == 20

    def test_filter_empty_returns_empty(self):
        assert self.qc.filter([]) == []

    def test_filter_scores_in_range(self):
        items = self._items(20)
        scores = self.qc.filter(items)
        for s in scores:
            assert 0.0 <= s.score <= 1.0

    # ---- detect_duplicates ----

    def test_detect_duplicates_identical_embeddings(self):
        emb = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
        dupes = self.qc.detect_duplicates(emb, threshold=0.95)
        # Indices 1-4 should be flagged as duplicates of index 0
        assert len(dupes) == 4

    def test_detect_duplicates_all_unique(self):
        rng = np.random.RandomState(1)
        emb = rng.randn(5, 8).astype(np.float32)
        # Very low threshold — unlikely to flag any
        dupes = self.qc.detect_duplicates(emb, threshold=0.9999)
        # Just check it runs without error (result depends on random data)
        assert isinstance(dupes, set)

    def test_detect_duplicates_single_sample(self):
        emb = np.ones((1, 4), dtype=np.float32)
        assert self.qc.detect_duplicates(emb) == set()

    # ---- detect_anomalies ----

    def test_detect_anomalies_returns_correct_length(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 8).astype(np.float32)
        scores = self.qc.detect_anomalies(emb)
        assert len(scores) == 20

    def test_detect_anomalies_scores_in_range(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 8).astype(np.float32)
        scores = self.qc.detect_anomalies(emb)
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_detect_anomalies_too_few_samples_returns_zeros(self):
        emb = np.ones((5, 4), dtype=np.float32)
        scores = self.qc.detect_anomalies(emb)
        assert np.all(scores == 0.0)

    # ---- compute_cluster_coherence ----

    def test_coherence_single_item_returns_one(self):
        items = [_make_labeled_item()]
        assert self.qc.compute_cluster_coherence(items) == 1.0

    def test_coherence_result_in_range(self):
        items = self._items(10)
        score = self.qc.compute_cluster_coherence(items)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# SelfTrainingLoop
# ---------------------------------------------------------------------------


class TestSelfTrainingLoop:
    """Tests for SelfTrainingLoop (self_training.py)."""

    def setup_method(self):
        from datetime import datetime

        from zlsde.layers.self_training import SelfTrainingLoop
        from zlsde.models.data_models import IterationMetrics

        self.config = _make_config(convergence_threshold=0.02, max_iterations=3)
        self.loop = SelfTrainingLoop(self.config)
        self.IterationMetrics = IterationMetrics
        self.now = datetime.now().isoformat()

    def _make_metrics(self, flip_rate, n_clusters, silhouette):
        return self.IterationMetrics(
            iteration=1,
            silhouette_score=silhouette,
            n_clusters=n_clusters,
            noise_ratio=0.0,
            label_flip_rate=flip_rate,
            cluster_purity=0.8,
            quality_mean=0.7,
            quality_std=0.1,
            timestamp=self.now,
        )

    # ---- compute_stability ----

    def test_stability_all_same_is_zero(self):
        a = np.array([0, 1, 2, 0, 1])
        rate = self.loop.compute_stability(a, a.copy())
        assert rate == 0.0

    def test_stability_all_different_is_one(self):
        a = np.array([0, 0, 0])
        b = np.array([1, 1, 1])
        assert self.loop.compute_stability(a, b) == 1.0

    def test_stability_partial_change(self):
        a = np.array([0, 1, 2, 0, 1])
        b = np.array([0, 1, 2, 1, 1])  # 1 change out of 5
        assert abs(self.loop.compute_stability(a, b) - 0.2) < 1e-9

    def test_stability_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.loop.compute_stability(np.array([0, 1]), np.array([0]))

    # ---- check_convergence ----

    def test_convergence_requires_at_least_2_iterations(self):
        m = self._make_metrics(0.001, 3, 0.5)
        assert self.loop.check_convergence([m]) is False

    def test_convergence_detected_when_all_criteria_met(self):
        m1 = self._make_metrics(flip_rate=0.001, n_clusters=3, silhouette=0.5)
        m2 = self._make_metrics(flip_rate=0.001, n_clusters=3, silhouette=0.5)
        assert self.loop.check_convergence([m1, m2]) is True

    def test_no_convergence_high_flip_rate(self):
        m1 = self._make_metrics(flip_rate=0.5, n_clusters=3, silhouette=0.5)
        m2 = self._make_metrics(flip_rate=0.5, n_clusters=3, silhouette=0.5)
        assert self.loop.check_convergence([m1, m2]) is False

    def test_no_convergence_cluster_count_changed(self):
        m1 = self._make_metrics(flip_rate=0.001, n_clusters=3, silhouette=0.5)
        m2 = self._make_metrics(flip_rate=0.001, n_clusters=4, silhouette=0.5)
        assert self.loop.check_convergence([m1, m2]) is False

    # ---- train_classifier / refine_labels ----

    def test_train_and_refine(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(40, 8).astype(np.float32)
        labels = np.array([i % 3 for i in range(40)], dtype=int)
        clf = self.loop.train_classifier(embeddings, labels)
        refined = self.loop.refine_labels(clf, embeddings)
        assert len(refined) == 40

    def test_train_classifier_insufficient_data_raises(self):
        emb = np.random.randn(3, 4).astype(np.float32)
        labels = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="Insufficient"):
            self.loop.train_classifier(emb, labels)


# ---------------------------------------------------------------------------
# DatasetExporter
# ---------------------------------------------------------------------------


class TestDatasetExporter:
    """Tests for DatasetExporter (exporter.py)."""

    def setup_method(self):
        from zlsde.layers.exporter import DatasetExporter

        self.config = _make_config()
        self.exporter = DatasetExporter(self.config)

    def _dataset(self, n=5):
        return [_make_labeled_item(label=f"label_{i}", cluster_id=i % 2) for i in range(n)]

    # ---- CSV export ----

    def test_export_csv_creates_file(self, tmp_path):
        result = self.exporter.export(self._dataset(), "csv", str(tmp_path))
        assert os.path.exists(result.path)
        assert result.format == "csv"
        assert result.n_samples == 5

    def test_export_csv_correct_columns(self, tmp_path):
        self.exporter.export(self._dataset(), "csv", str(tmp_path))
        import pandas as pd

        df = pd.read_csv(tmp_path / "dataset.csv")
        for col in ("id", "content", "label", "cluster_id", "confidence"):
            assert col in df.columns

    def test_export_csv_saves_embeddings_npy(self, tmp_path):
        self.exporter.export(self._dataset(), "csv", str(tmp_path))
        assert (tmp_path / "embeddings.npy").exists()

    # ---- JSON export ----

    def test_export_json_creates_file(self, tmp_path):
        result = self.exporter.export(self._dataset(), "json", str(tmp_path))
        assert os.path.exists(result.path)
        assert result.format == "json"

    def test_export_json_valid_structure(self, tmp_path):
        result = self.exporter.export(self._dataset(), "json", str(tmp_path))
        with open(result.path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 5
        assert "label" in data[0]
        assert "embedding" in data[0]

    # ---- Parquet export ----

    def test_export_parquet_creates_file(self, tmp_path):
        pytest.importorskip("pyarrow", reason="pyarrow not installed")
        result = self.exporter.export(self._dataset(), "parquet", str(tmp_path))
        assert os.path.exists(result.path)
        assert result.format == "parquet"

    # ---- Unknown format ----

    def test_export_unknown_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.exporter.export(self._dataset(), "xml", str(tmp_path))

    # ---- Metadata ----

    def test_export_generates_metadata_json(self, tmp_path):
        self.exporter.export(self._dataset(), "csv", str(tmp_path))
        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert "dataset_statistics" in meta
        assert meta["dataset_statistics"]["n_samples"] == 5

    # ---- Empty dataset ----

    def test_export_empty_dataset(self, tmp_path):
        result = self.exporter.export([], "csv", str(tmp_path))
        assert result.n_samples == 0

    # ---- generate_metadata ----

    def test_generate_metadata_keys(self):
        meta = self.exporter.generate_metadata(self._dataset(), self.config)
        assert "version" in meta
        assert "dataset_statistics" in meta
        assert "configuration" in meta
