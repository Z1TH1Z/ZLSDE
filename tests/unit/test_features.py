"""Unit tests for the 7 novel feature layers."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from zlsde.models.data_models import (
    DataSource,
    DriftReport,
    Label,
    LabeledDataItem,
    LabelProvenance,
    PipelineConfig,
    ProvenanceReport,
    ProviderConfig,
    RawDataItem,
    SemanticValidationResult,
    TaxonomyNode,
    TaxonomyTree,
    ValidationFlag,
)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_config(**overrides) -> PipelineConfig:
    """Create a PipelineConfig with sensible test defaults."""
    defaults = dict(
        data_sources=[DataSource(type="csv", path="test.csv")],
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        min_cluster_size=2,
        max_iterations=3,
        random_seed=42,
        # Enable all features
        enable_taxonomy=True,
        taxonomy_max_depth=2,
        taxonomy_min_samples=3,
        taxonomy_silhouette_threshold=0.05,
        enable_provenance=True,
        enable_explanations=False,
        enable_semantic_validation=True,
        label_similarity_threshold=0.8,
        centroid_similarity_threshold=0.85,
        enable_embedding_fusion=False,
        enable_adaptive_training=True,
        curriculum_percentile_decay=10.0,
        enable_provider_optimization=False,
        enable_drift_detection=True,
        drift_collapse_threshold=0.3,
        drift_divergence_threshold=2.0,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _random_embeddings(n: int, dim: int = 32, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def _make_labeled_items(n: int, n_clusters: int = 3, dim: int = 32) -> list:
    """Create a list of LabeledDataItem with cluster assignments."""
    embs = _random_embeddings(n, dim)
    items = []
    for i in range(n):
        cid = i % n_clusters
        items.append(
            LabeledDataItem(
                id=f"item_{i}",
                content=f"sample text {i}",
                embedding=embs[i],
                label=f"label_{cid}",
                cluster_id=cid,
                confidence=0.7 + 0.1 * (cid / max(n_clusters - 1, 1)),
                modality="text",
                iteration=0,
            )
        )
    return items


# ══════════════════════════════════════════════════════════════════════
# Feature 1: Autonomous Label Taxonomy Discovery
# ══════════════════════════════════════════════════════════════════════


class TestTaxonomyDiscovery:
    def test_discover_builds_tree(self):
        from zlsde.layers.taxonomy_discovery import TaxonomyDiscoveryEngine

        config = _make_config()
        engine = TaxonomyDiscoveryEngine(config)

        n, dim = 60, 16
        embs = _random_embeddings(n, dim)
        labels = np.array([i % 3 for i in range(n)])
        pseudo_labels = {
            0: Label(text="sports", confidence=0.9),
            1: Label(text="tech", confidence=0.85),
            2: Label(text="food", confidence=0.8),
        }

        tree = engine.discover(embs, labels, pseudo_labels)

        assert isinstance(tree, TaxonomyTree)
        assert tree.total_nodes >= 3  # at least root nodes
        assert tree.max_depth >= 0
        assert len(tree.root_nodes) == 3

    def test_taxonomy_respects_max_depth(self):
        from zlsde.layers.taxonomy_discovery import TaxonomyDiscoveryEngine

        config = _make_config(taxonomy_max_depth=0)
        engine = TaxonomyDiscoveryEngine(config)

        embs = _random_embeddings(30, 16)
        labels = np.array([i % 2 for i in range(30)])
        pseudo = {0: Label(text="a", confidence=0.9), 1: Label(text="b", confidence=0.9)}

        tree = engine.discover(embs, labels, pseudo)
        # Depth 0 means no children
        for node in tree.root_nodes:
            assert len(node.children) == 0

    def test_taxonomy_flatten(self):
        child = TaxonomyNode(label="child", level=1, n_samples=5)
        parent = TaxonomyNode(label="parent", level=0, n_samples=10, children=[child])
        tree = TaxonomyTree(root_nodes=[parent], max_depth=1, total_nodes=2)

        flat = tree.flatten()
        assert "parent" in flat
        assert "child" in flat["parent"]

    def test_taxonomy_to_dict(self):
        node = TaxonomyNode(label="root", level=0, n_samples=10)
        tree = TaxonomyTree(root_nodes=[node], max_depth=0, total_nodes=1)
        d = tree.to_dict()
        assert d["total_nodes"] == 1
        assert len(d["taxonomy"]) == 1
        assert d["taxonomy"][0]["label"] == "root"


# ══════════════════════════════════════════════════════════════════════
# Feature 2: Label Provenance & Explainability
# ══════════════════════════════════════════════════════════════════════


class TestProvenanceTracker:
    def test_record_label_creates_provenance(self):
        from zlsde.layers.provenance import ProvenanceTracker

        config = _make_config()
        tracker = ProvenanceTracker(config)

        record = tracker.record_label(
            cluster_id=0,
            label_text="sports",
            provider_used="Groq",
            prompt_sent="test prompt",
            raw_response="sports",
            confidence=0.9,
        )

        assert isinstance(record, LabelProvenance)
        assert record.cluster_id == 0
        assert record.label_text == "sports"
        assert record.provider_used == "Groq"

    def test_provenance_tracks_iteration_history(self):
        from zlsde.layers.provenance import ProvenanceTracker

        config = _make_config()
        tracker = ProvenanceTracker(config)

        tracker.record_label(cluster_id=0, label_text="sports", provider_used="Groq")
        tracker.record_label(cluster_id=0, label_text="athletics", provider_used="Mistral")

        report = tracker.generate_report()
        record = report.provenance_records[0]
        assert len(record.iteration_history) == 1
        assert record.iteration_history[0]["prev_label"] == "sports"
        assert record.iteration_history[0]["new_label"] == "athletics"

    def test_provenance_report_aggregation(self):
        from zlsde.layers.provenance import ProvenanceTracker

        config = _make_config()
        tracker = ProvenanceTracker(config)

        tracker.record_label(cluster_id=0, label_text="a", provider_used="Groq", confidence=0.8)
        tracker.record_label(cluster_id=1, label_text="b", provider_used="Mistral", confidence=0.6)

        report = tracker.generate_report()
        assert report.total_labels_generated == 2
        assert report.avg_confidence == pytest.approx(0.7, abs=0.01)
        assert report.provider_usage["Groq"] == 1
        assert report.provider_usage["Mistral"] == 1

    def test_provenance_disabled(self):
        from zlsde.layers.provenance import ProvenanceTracker

        config = _make_config(enable_provenance=False)
        tracker = ProvenanceTracker(config)

        record = tracker.record_label(cluster_id=0, label_text="x")
        assert record.provider_used == "unknown"  # minimal record

    def test_add_explanation(self):
        from zlsde.layers.provenance import ProvenanceTracker

        config = _make_config()
        tracker = ProvenanceTracker(config)
        tracker.record_label(cluster_id=0, label_text="sports")
        tracker.add_explanation(0, "These samples are about sports activities")

        report = tracker.generate_report()
        assert (
            report.provenance_records[0].explanation == "These samples are about sports activities"
        )
        assert report.total_labels_explained == 1


# ══════════════════════════════════════════════════════════════════════
# Feature 3: Cross-Cluster Semantic Validation
# ══════════════════════════════════════════════════════════════════════


class TestSemanticValidation:
    def test_label_collision_detection(self):
        from zlsde.layers.semantic_validation import SemanticValidator

        config = _make_config(label_similarity_threshold=0.5)
        validator = SemanticValidator(config)

        items = _make_labeled_items(30, n_clusters=3)
        # Give two clusters identical labels
        pseudo = {
            0: Label(text="sports news", confidence=0.9),
            1: Label(text="sports news", confidence=0.8),
            2: Label(text="cooking recipes", confidence=0.85),
        }

        result = validator.validate(items, pseudo)
        assert result.n_label_collisions >= 1

    def test_no_flags_on_clean_data(self):
        from zlsde.layers.semantic_validation import SemanticValidator

        config = _make_config(
            label_similarity_threshold=0.99,
            centroid_similarity_threshold=0.999,
        )
        validator = SemanticValidator(config)

        # Well-separated clusters
        n = 30
        dim = 32
        rng = np.random.RandomState(42)
        items = []
        for i in range(n):
            cid = i % 3
            base = np.zeros(dim)
            base[cid * 10 : (cid + 1) * 10] = 1.0
            emb = base + rng.randn(dim) * 0.01
            items.append(
                LabeledDataItem(
                    id=f"item_{i}",
                    content=f"text {i}",
                    embedding=emb,
                    label=f"label_{cid}",
                    cluster_id=cid,
                    confidence=0.9,
                    modality="text",
                    iteration=0,
                )
            )

        pseudo = {
            0: Label(text="alpha", confidence=0.9),
            1: Label(text="beta", confidence=0.9),
            2: Label(text="gamma", confidence=0.9),
        }

        result = validator.validate(items, pseudo)
        assert result.n_label_collisions == 0

    def test_validation_disabled(self):
        from zlsde.layers.semantic_validation import SemanticValidator

        config = _make_config(enable_semantic_validation=False)
        validator = SemanticValidator(config)

        result = validator.validate([], {})
        assert result.semantic_consistency_score == 1.0

    def test_consistency_score_range(self):
        from zlsde.layers.semantic_validation import SemanticValidator

        config = _make_config()
        validator = SemanticValidator(config)

        items = _make_labeled_items(30, n_clusters=3)
        pseudo = {i: Label(text=f"label_{i}", confidence=0.9) for i in range(3)}

        result = validator.validate(items, pseudo)
        assert 0.0 <= result.semantic_consistency_score <= 1.0


# ══════════════════════════════════════════════════════════════════════
# Feature 4: Multi-Granularity Embedding Fusion
# ══════════════════════════════════════════════════════════════════════


class TestEmbeddingFusion:
    def test_fusion_disabled_returns_primary(self):
        from zlsde.layers.embedding_fusion import EmbeddingFusionEngine

        config = _make_config(enable_embedding_fusion=False)
        engine = EmbeddingFusionEngine(config)

        embs = _random_embeddings(10, 32)
        items = [RawDataItem(id=f"i{i}", content=f"t{i}", modality="text") for i in range(10)]

        result = engine.fuse(items, embs)
        np.testing.assert_array_equal(result, embs)

    def test_fusion_no_extra_models_returns_primary(self):
        from zlsde.layers.embedding_fusion import EmbeddingFusionEngine

        config = _make_config(enable_embedding_fusion=True, fusion_models=[])
        engine = EmbeddingFusionEngine(config)

        embs = _random_embeddings(10, 32)
        items = [RawDataItem(id=f"i{i}", content=f"t{i}", modality="text") for i in range(10)]

        result = engine.fuse(items, embs)
        np.testing.assert_array_equal(result, embs)

    def test_weighted_concat_normalised(self):
        from zlsde.layers.embedding_fusion import EmbeddingFusionEngine

        a = _random_embeddings(5, 16)
        b = _random_embeddings(5, 16, seed=99)
        fused = EmbeddingFusionEngine._weighted_concat([a, b], [0.5, 0.5])

        assert fused.shape == (5, 32)
        norms = np.linalg.norm(fused, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_auto_learn_weights(self):
        from zlsde.layers.embedding_fusion import EmbeddingFusionEngine

        config = _make_config(enable_embedding_fusion=True)
        engine = EmbeddingFusionEngine(config)

        a = _random_embeddings(30, 16)
        b = _random_embeddings(30, 16, seed=99)
        labels = np.array([i % 3 for i in range(30)])

        weights = engine.auto_learn_weights([a, b], labels)
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 0.01


# ══════════════════════════════════════════════════════════════════════
# Feature 5: Confidence-Weighted Adaptive Self-Training
# ══════════════════════════════════════════════════════════════════════


class TestAdaptiveSelfTraining:
    def test_cwast_refines_labels(self):
        from zlsde.layers.adaptive_training import AdaptiveSelfTrainer

        config = _make_config()
        trainer = AdaptiveSelfTrainer(config)

        items = _make_labeled_items(60, n_clusters=3, dim=16)
        embs = np.array([item.embedding for item in items])

        refined = trainer.train_and_refine(embs, items, iteration=0, max_iterations=3)
        assert refined.shape == (60,)
        assert set(refined).issubset({0, 1, 2})  # same cluster IDs

    def test_cwast_disabled_uses_fallback(self):
        from zlsde.layers.adaptive_training import AdaptiveSelfTrainer

        config = _make_config(enable_adaptive_training=False)
        trainer = AdaptiveSelfTrainer(config)

        items = _make_labeled_items(60, n_clusters=3, dim=16)
        embs = np.array([item.embedding for item in items])

        refined = trainer.train_and_refine(embs, items, iteration=0, max_iterations=3)
        assert refined.shape == (60,)

    def test_curriculum_percentile_decay(self):
        from zlsde.layers.adaptive_training import AdaptiveSelfTrainer

        config = _make_config(curriculum_percentile_decay=20.0)
        trainer = AdaptiveSelfTrainer(config)

        items = _make_labeled_items(60, n_clusters=3, dim=16)
        embs = np.array([item.embedding for item in items])

        # Iteration 0 should be stricter than iteration 4
        r0 = trainer.train_and_refine(embs, items, iteration=0, max_iterations=5)
        r4 = trainer.train_and_refine(embs, items, iteration=4, max_iterations=5)
        assert r0.shape == r4.shape


# ══════════════════════════════════════════════════════════════════════
# Feature 6: Dynamic Provider Cost-Quality Optimizer
# ══════════════════════════════════════════════════════════════════════


class TestProviderOptimizer:
    def _make_mock_provider(self, name: str, label: str = "test label"):
        p = Mock()
        p.get_provider_name.return_value = name
        p.is_available.return_value = True
        p.generate_label.return_value = label
        return p

    def test_ucb1_explores_all_arms(self):
        from zlsde.layers.provider_optimizer import DynamicProviderOptimizer

        config = _make_config(enable_provider_optimization=True)
        p1 = self._make_mock_provider("P1", "label_a")
        p2 = self._make_mock_provider("P2", "label_b")

        opt = DynamicProviderOptimizer(config, [p1, p2])

        # First two calls should explore each arm once
        opt.generate_label("prompt1")
        opt.generate_label("prompt2")

        stats = opt.get_statistics()
        assert stats["P1"]["pulls"] >= 1
        assert stats["P2"]["pulls"] >= 1

    def test_fallback_when_disabled(self):
        from zlsde.layers.provider_optimizer import DynamicProviderOptimizer

        config = _make_config(enable_provider_optimization=False)
        p = self._make_mock_provider("P1")
        opt = DynamicProviderOptimizer(config, [p])

        result = opt.generate_label("prompt")
        assert result == "test label"

    def test_handles_provider_failure(self):
        from zlsde.layers.provider_optimizer import DynamicProviderOptimizer
        from zlsde.providers.exceptions import ProviderError

        config = _make_config(enable_provider_optimization=True)
        p1 = self._make_mock_provider("P1")
        p1.generate_label.side_effect = ProviderError("fail")
        p2 = self._make_mock_provider("P2", "fallback")

        opt = DynamicProviderOptimizer(config, [p1, p2])
        result = opt.generate_label("prompt")
        assert result == "fallback"


# ══════════════════════════════════════════════════════════════════════
# Feature 7: Embedding Drift Detection
# ══════════════════════════════════════════════════════════════════════


class TestDriftDetection:
    def test_first_iteration_no_drift(self):
        from zlsde.layers.drift_detection import DriftDetector

        config = _make_config()
        detector = DriftDetector(config)

        embs = _random_embeddings(30, 16)
        labels = np.array([i % 3 for i in range(30)])

        report = detector.check(embs, labels, iteration=0)
        assert isinstance(report, DriftReport)
        assert not report.collapse_detected
        assert not report.divergence_detected
        assert not report.rollback_recommended

    def test_collapse_detection(self):
        from zlsde.layers.drift_detection import DriftDetector

        config = _make_config(drift_collapse_threshold=0.5)
        detector = DriftDetector(config)

        # Iteration 0: well-separated clusters
        n, dim = 30, 16
        rng = np.random.RandomState(42)
        embs_good = np.zeros((n, dim))
        for i in range(n):
            cid = i % 3
            embs_good[i, cid * 5 : (cid + 1) * 5] = 1.0
            embs_good[i] += rng.randn(dim) * 0.01
        labels = np.array([i % 3 for i in range(n)])

        detector.check(embs_good, labels, iteration=0)

        # Iteration 1: collapsed (everything near origin)
        embs_bad = rng.randn(n, dim) * 0.001
        report = detector.check(embs_bad, labels, iteration=1)
        assert report.collapse_detected
        assert report.rollback_recommended

    def test_health_score_range(self):
        from zlsde.layers.drift_detection import DriftDetector

        config = _make_config()
        detector = DriftDetector(config)

        embs = _random_embeddings(30, 16)
        labels = np.array([i % 3 for i in range(30)])

        report = detector.check(embs, labels, iteration=0)
        assert 0.0 <= report.health_score <= 1.0

    def test_drift_disabled(self):
        from zlsde.layers.drift_detection import DriftDetector

        config = _make_config(enable_drift_detection=False)
        detector = DriftDetector(config)

        embs = _random_embeddings(10, 16)
        labels = np.array([0] * 5 + [1] * 5)

        report = detector.check(embs, labels, iteration=0)
        assert report.health_score == 1.0
        assert not report.rollback_recommended

    def test_history_tracking(self):
        from zlsde.layers.drift_detection import DriftDetector

        config = _make_config()
        detector = DriftDetector(config)

        embs = _random_embeddings(30, 16)
        labels = np.array([i % 3 for i in range(30)])

        detector.check(embs, labels, iteration=0)
        detector.check(embs, labels, iteration=1)

        assert len(detector.history) == 2
        assert detector.history[0].iteration == 0
        assert detector.history[1].iteration == 1


# ══════════════════════════════════════════════════════════════════════
# Data model tests for new types
# ══════════════════════════════════════════════════════════════════════


class TestNewDataModels:
    def test_taxonomy_node_creation(self):
        node = TaxonomyNode(label="test", level=0, n_samples=10)
        assert node.label == "test"
        assert node.children == []

    def test_label_provenance_creation(self):
        prov = LabelProvenance(cluster_id=0, label_text="sports")
        assert prov.cluster_id == 0
        assert prov.iteration_history == []

    def test_validation_flag_creation(self):
        flag = ValidationFlag(
            flag_type="merge_candidate",
            cluster_ids=[0, 1],
            labels=["a", "b"],
            similarity_score=0.95,
        )
        assert flag.flag_type == "merge_candidate"
        assert not flag.resolved

    def test_drift_report_creation(self):
        report = DriftReport(iteration=0)
        assert report.health_score == 1.0
        assert not report.collapse_detected
