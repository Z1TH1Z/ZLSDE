"""Pipeline Orchestrator - Coordinates execution of all pipeline layers.

Integrates the 7 novel features:
  1. Autonomous Label Taxonomy Discovery (ALTD)
  2. Label Provenance & Explainability Engine
  3. Cross-Cluster Semantic Validation (CCSV)
  4. Multi-Granularity Embedding Fusion
  5. Confidence-Weighted Adaptive Self-Training (CWAST)
  6. Dynamic Provider Cost-Quality Optimizer
  7. Embedding Drift Detection & Self-Correction
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from zlsde.layers.adaptive_training import AdaptiveSelfTrainer
from zlsde.layers.clustering import ClusteringEngine
from zlsde.layers.drift_detection import DriftDetector
from zlsde.layers.embedding_fusion import EmbeddingFusionEngine
from zlsde.layers.exporter import DatasetExporter
from zlsde.layers.ingestion import DataIngestionLayer
from zlsde.layers.label_generation import PseudoLabelGenerator
from zlsde.layers.provenance import ProvenanceTracker
from zlsde.layers.quality_control import QualityControlFilter
from zlsde.layers.representation import RepresentationEngine
from zlsde.layers.self_training import SelfTrainingLoop
from zlsde.layers.semantic_validation import SemanticValidator

# Feature layers
from zlsde.layers.taxonomy_discovery import TaxonomyDiscoveryEngine
from zlsde.models.data_models import (
    IterationMetrics,
    Label,
    LabeledDataItem,
    PipelineConfig,
    PipelineResult,
    RawDataItem,
)
from zlsde.utils.seed_control import set_random_seed

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates execution of all pipeline layers and manages iteration loops."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.config.validate()

        # Set random seed for reproducibility
        set_random_seed(config.random_seed)

        # --- Core layers ---
        self.ingestion = DataIngestionLayer(config)
        self.embedder = RepresentationEngine(
            modality=config.modality,
            model_name=config.embedding_model,
            device=config.device,
        )
        self.clusterer = ClusteringEngine(config)

        # Label generator (with optional provider optimizer - Feature 6)
        self.label_generator: Optional[PseudoLabelGenerator] = None
        self.provider_optimizer = None
        if config.use_llm:
            from zlsde.layers.label_generation import create_label_generator

            self.label_generator = create_label_generator(config, device=config.device)

            if config.enable_provider_optimization:
                try:
                    from zlsde.layers.provider_optimizer import DynamicProviderOptimizer

                    providers = list(self.label_generator.provider_manager.providers)
                    self.provider_optimizer = DynamicProviderOptimizer(config, providers)
                    logger.info("Feature 6: Dynamic provider optimizer enabled")
                except Exception as e:
                    logger.warning(f"Provider optimizer init failed: {e}")

        self.quality_filter = QualityControlFilter(config)
        self.self_trainer = SelfTrainingLoop(config)
        self.exporter = DatasetExporter(config)

        # --- Feature layers ---
        # Feature 1: Taxonomy Discovery
        self.taxonomy_engine = TaxonomyDiscoveryEngine(config) if config.enable_taxonomy else None

        # Feature 2: Provenance Tracker
        self.provenance = ProvenanceTracker(config)

        # Feature 3: Semantic Validator
        self.semantic_validator = SemanticValidator(config)

        # Feature 4: Embedding Fusion
        self.fusion_engine = EmbeddingFusionEngine(config)

        # Feature 5: Adaptive Self-Trainer (CWAST)
        self.adaptive_trainer = AdaptiveSelfTrainer(config)

        # Feature 7: Drift Detector
        self.drift_detector = DriftDetector(config)

        logger.info("Pipeline orchestrator initialized (with 7 novel feature layers)")

    # ==================================================================
    # Main entry point
    # ==================================================================

    def run(self) -> PipelineResult:
        """
        Execute complete pipeline from ingestion to export.

        Returns:
            Pipeline execution result with metrics and dataset path
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting ZLSDE Pipeline Execution")
        logger.info("=" * 80)

        try:
            # ── Layer 1: Data Ingestion ──────────────────────────────
            logger.info("\n[Layer 1] Data Ingestion")
            raw_data = self.ingestion.load_data(self.config.data_sources)
            raw_data = self.ingestion.deduplicate(raw_data)
            raw_data = self.ingestion.validate(raw_data)
            logger.info(f"Loaded {len(raw_data)} valid samples")

            if len(raw_data) == 0:
                raise ValueError("No valid data after ingestion")

            # ── Layer 2: Representation Learning ─────────────────────
            logger.info("\n[Layer 2] Representation Learning")
            embeddings = self.embedder.embed(raw_data, batch_size=self.config.batch_size)

            if self.config.use_dimensionality_reduction:
                logger.info("Applying dimensionality reduction")
                embeddings = self.embedder.reduce_dimensions(embeddings, self.config.n_components)

            # Feature 4: Multi-Granularity Embedding Fusion
            embeddings = self.fusion_engine.fuse(
                raw_data, embeddings, batch_size=self.config.batch_size
            )

            logger.info(f"Generated embeddings with shape {embeddings.shape}")

            # Store embeddings in raw data items for later use
            for i, item in enumerate(raw_data):
                item.embedding = embeddings[i]

            # ── Self-training iteration loop ─────────────────────────
            iteration_metrics: List[IterationMetrics] = []
            current_labels: Optional[np.ndarray] = None
            labeled_data: Optional[List[LabeledDataItem]] = None
            prev_labeled_data: Optional[List[LabeledDataItem]] = None
            taxonomy_tree = None

            for iteration in range(self.config.max_iterations):
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
                logger.info(f"{'=' * 80}")

                # Save previous state for potential drift rollback
                prev_labeled_data = labeled_data

                # Execute iteration (all layers inside)
                labeled_data, metrics, pseudo_labels = self._execute_iteration(
                    iteration, raw_data, embeddings, current_labels
                )
                iteration_metrics.append(metrics)

                # Feature 7: Drift Detection
                cluster_arr = np.array([item.cluster_id for item in labeled_data])
                drift_report = self.drift_detector.check(embeddings, cluster_arr, iteration)

                if drift_report.rollback_recommended and prev_labeled_data is not None:
                    logger.warning(
                        f"Drift detected (health={drift_report.health_score:.2f}) — "
                        f"rolling back to previous iteration"
                    )
                    labeled_data = prev_labeled_data
                    break

                # Feature 1: Taxonomy Discovery (run after final or converged iteration)
                is_last = iteration == self.config.max_iterations - 1
                converged = False
                if iteration < self.config.max_iterations - 1:
                    converged = self._check_convergence(iteration_metrics)

                if (is_last or converged) and self.taxonomy_engine:
                    logger.info("\n[Feature 1] Autonomous Label Taxonomy Discovery")
                    taxonomy_tree = self.taxonomy_engine.discover(
                        embeddings,
                        cluster_arr,
                        pseudo_labels,
                        label_generator=self.label_generator,
                    )

                if converged:
                    logger.info(f"\nConverged after {iteration + 1} iterations!")
                    break

                # Update current labels for next iteration
                current_labels = cluster_arr

            # ── Layer 7: Export ───────────────────────────────────────
            logger.info("\n[Layer 7] Dataset Export")
            export_result = self.exporter.export(
                labeled_data,
                format=self.config.output_format,
                output_path=self.config.output_path,
            )

            # Export provenance report (Feature 2)
            self._export_provenance(self.config.output_path)

            # Export taxonomy (Feature 1)
            if taxonomy_tree:
                self._export_taxonomy(taxonomy_tree, self.config.output_path)

            # Export drift history (Feature 7)
            self._export_drift_history(self.config.output_path)

            execution_time = time.time() - start_time

            # Create result
            result = PipelineResult(
                status="completed",
                dataset_path=export_result.path,
                n_samples=len(raw_data),
                n_labeled=len([d for d in labeled_data if d.cluster_id >= 0]),
                final_metrics=iteration_metrics[-1],
                iteration_history=iteration_metrics,
                config_snapshot=self.config,
                execution_time_seconds=execution_time,
            )

            logger.info("\n" + "=" * 80)
            logger.info("Pipeline Execution Complete!")
            logger.info("=" * 80)
            logger.info(f"Status: {result.status}")
            logger.info(f"Total samples: {result.n_samples}")
            logger.info(f"Labeled samples: {result.n_labeled}")
            logger.info(f"Final clusters: {result.final_metrics.n_clusters}")
            logger.info(f"Final silhouette score: {result.final_metrics.silhouette_score:.3f}")
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            logger.info(f"Dataset saved to: {result.dataset_path}")
            logger.info("=" * 80)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)

            return PipelineResult(
                status="failed",
                dataset_path="",
                n_samples=0,
                n_labeled=0,
                final_metrics=IterationMetrics(
                    iteration=0,
                    silhouette_score=0.0,
                    n_clusters=0,
                    noise_ratio=0.0,
                    label_flip_rate=0.0,
                    cluster_purity=0.0,
                    quality_mean=0.0,
                    quality_std=0.0,
                    timestamp=datetime.now().isoformat(),
                ),
                iteration_history=[],
                config_snapshot=self.config,
                execution_time_seconds=execution_time,
                error_message=str(e),
            )

    # ==================================================================
    # Per-iteration execution
    # ==================================================================

    def _execute_iteration(
        self,
        iteration: int,
        raw_data: List[RawDataItem],
        embeddings: np.ndarray,
        previous_labels: Optional[np.ndarray] = None,
    ) -> tuple:
        """Execute single self-training iteration with all feature layers.

        Returns:
            (labeled_data, metrics, pseudo_labels)
        """
        # ── Layer 3: Clustering ──────────────────────────────────────
        logger.info("[Layer 3] Clustering")
        cluster_result = self.clusterer.cluster(embeddings, method=self.config.clustering_method)
        logger.info(
            f"Found {cluster_result.n_clusters} clusters, " f"{cluster_result.n_noise} noise points"
        )
        logger.info(f"Silhouette score: {cluster_result.silhouette_score:.3f}")

        # ── Layer 4: Pseudo-Label Generation ─────────────────────────
        logger.info("[Layer 4] Pseudo-Label Generation")
        if self.config.use_llm and self.label_generator:
            clusters = self._group_by_cluster(raw_data, cluster_result.labels)

            # Feature 2: Provenance-wrapped generation
            if self.provenance.enabled:
                pseudo_labels = self.provenance.wrap_label_generation(
                    clusters,
                    self.label_generator,
                    n_representatives=self.config.n_representatives,
                )
            else:
                pseudo_labels = self.label_generator.generate_labels(clusters)
        else:
            pseudo_labels = self._generate_default_labels(cluster_result)

        logger.info(f"Generated labels for {len(pseudo_labels)} clusters")

        # Create labeled dataset
        labeled_data = self._create_labeled_dataset(
            raw_data, embeddings, cluster_result, pseudo_labels, iteration
        )

        # ── Feature 3: Cross-Cluster Semantic Validation ─────────────
        if self.semantic_validator.enabled:
            logger.info("[Feature 3] Cross-Cluster Semantic Validation")
            validation_result = self.semantic_validator.validate(labeled_data, pseudo_labels)
            # Log any flags
            for flag in validation_result.flags:
                logger.info(f"  CCSV flag: {flag.description}")

        # ── Layer 5: Quality Control ─────────────────────────────────
        logger.info("[Layer 5] Quality Control")
        quality_scores = self.quality_filter.filter(labeled_data)

        for i, item in enumerate(labeled_data):
            item.quality_score = quality_scores[i].score
            item.anomaly_flag = quality_scores[i].anomaly_flag
            item.duplicate_flag = quality_scores[i].duplicate_flag

        # Compute iteration metrics
        metrics = self._compute_iteration_metrics(
            iteration, cluster_result, labeled_data, previous_labels
        )

        logger.info(
            f"Iteration metrics: flip_rate={metrics.label_flip_rate:.2%}, "
            f"quality_mean={metrics.quality_mean:.3f}"
        )

        # ── Layer 6: Self-Training (Feature 5: CWAST) ────────────────
        if iteration < self.config.max_iterations - 1:
            logger.info("[Layer 6] Self-Training (CWAST)")
            try:
                if self.adaptive_trainer.enabled:
                    refined_labels = self.adaptive_trainer.train_and_refine(
                        embeddings,
                        labeled_data,
                        iteration=iteration,
                        max_iterations=self.config.max_iterations,
                    )
                else:
                    # Fallback to original self-training
                    current_cluster_labels = np.array([item.cluster_id for item in labeled_data])
                    classifier = self.self_trainer.train_classifier(
                        embeddings,
                        current_cluster_labels,
                        self.config.confidence_threshold,
                    )
                    refined_labels = self.self_trainer.refine_labels(classifier, embeddings)

                for i, item in enumerate(labeled_data):
                    item.cluster_id = int(refined_labels[i])

                logger.info("Self-training refinement complete")

            except Exception as e:
                logger.warning(f"Self-training failed: {e}. Continuing with current labels.")

        return labeled_data, metrics, pseudo_labels

    # ==================================================================
    # Helpers (unchanged from original)
    # ==================================================================

    def _check_convergence(self, metrics: List[IterationMetrics]) -> bool:
        return self.self_trainer.check_convergence(metrics)

    def _group_by_cluster(
        self, raw_data: List[RawDataItem], labels: np.ndarray
    ) -> Dict[int, List[RawDataItem]]:
        clusters: Dict[int, List[RawDataItem]] = {}
        for i, item in enumerate(raw_data):
            cluster_id = int(labels[i])
            clusters.setdefault(cluster_id, []).append(item)
        return clusters

    def _generate_default_labels(self, cluster_result) -> Dict[int, Label]:
        labels: Dict[int, Label] = {}
        for cluster_id in range(cluster_result.n_clusters):
            labels[cluster_id] = Label(text=f"cluster_{cluster_id}", confidence=0.5)
        labels[-1] = Label(text="noise", confidence=0.0)
        return labels

    def _create_labeled_dataset(
        self,
        raw_data: List[RawDataItem],
        embeddings: np.ndarray,
        cluster_result,
        pseudo_labels: Dict[int, Label],
        iteration: int,
    ) -> List[LabeledDataItem]:
        labeled_data: List[LabeledDataItem] = []
        for i, item in enumerate(raw_data):
            cluster_id = int(cluster_result.labels[i])
            label = pseudo_labels.get(cluster_id, Label(text="unknown", confidence=0.0))
            labeled_item = LabeledDataItem(
                id=item.id,
                content=item.content,
                embedding=embeddings[i],
                label=label.text,
                cluster_id=cluster_id,
                confidence=label.confidence,
                quality_score=0.0,
                modality=item.modality,
                iteration=iteration,
                anomaly_flag=False,
                duplicate_flag=False,
                metadata=item.metadata,
            )
            labeled_data.append(labeled_item)
        return labeled_data

    def _compute_iteration_metrics(
        self,
        iteration: int,
        cluster_result,
        labeled_data: List[LabeledDataItem],
        previous_labels: Optional[np.ndarray] = None,
    ) -> IterationMetrics:
        if previous_labels is not None:
            current_labels = np.array([item.cluster_id for item in labeled_data])
            label_flip_rate = self.self_trainer.compute_stability(previous_labels, current_labels)
        else:
            label_flip_rate = 0.0

        quality_scores = [item.quality_score for item in labeled_data]
        quality_mean = float(np.mean(quality_scores))
        quality_std = float(np.std(quality_scores))

        clusters: Dict[int, List[LabeledDataItem]] = {}
        for item in labeled_data:
            clusters.setdefault(item.cluster_id, []).append(item)

        coherence_scores = []
        for cluster_id, items in clusters.items():
            if cluster_id >= 0:
                coherence = self.quality_filter.compute_cluster_coherence(items)
                coherence_scores.append(coherence)

        cluster_purity = float(np.mean(coherence_scores)) if coherence_scores else 0.0

        n_noise = sum(1 for item in labeled_data if item.cluster_id == -1)
        noise_ratio = n_noise / len(labeled_data)

        return IterationMetrics(
            iteration=iteration,
            silhouette_score=cluster_result.silhouette_score,
            n_clusters=cluster_result.n_clusters,
            noise_ratio=noise_ratio,
            label_flip_rate=label_flip_rate,
            cluster_purity=cluster_purity,
            quality_mean=quality_mean,
            quality_std=quality_std,
            timestamp=datetime.now().isoformat(),
        )

    # ==================================================================
    # Export helpers for new features
    # ==================================================================

    def _export_provenance(self, output_path: str) -> None:
        """Export provenance report as JSON."""
        if not self.provenance.enabled:
            return
        try:
            report = self.provenance.generate_report()
            os.makedirs(output_path, exist_ok=True)
            path = os.path.join(output_path, "provenance_report.json")
            data = {
                "total_labels_generated": report.total_labels_generated,
                "total_labels_explained": report.total_labels_explained,
                "avg_confidence": report.avg_confidence,
                "provider_usage": report.provider_usage,
                "records": [
                    {
                        "cluster_id": r.cluster_id,
                        "label": r.label_text,
                        "provider": r.provider_used,
                        "confidence": r.confidence,
                        "confidence_breakdown": r.confidence_breakdown,
                        "representative_samples": r.representative_samples,
                        "iteration_history": r.iteration_history,
                        "timestamp": r.timestamp,
                    }
                    for r in report.provenance_records
                ],
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Provenance report saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to export provenance report: {e}")

    def _export_taxonomy(self, taxonomy_tree, output_path: str) -> None:
        """Export taxonomy tree as JSON."""
        try:
            os.makedirs(output_path, exist_ok=True)
            path = os.path.join(output_path, "taxonomy.json")
            with open(path, "w") as f:
                json.dump(taxonomy_tree.to_dict(), f, indent=2)
            logger.info(f"Taxonomy tree saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to export taxonomy: {e}")

    def _export_drift_history(self, output_path: str) -> None:
        """Export drift detection history as JSON."""
        if not self.drift_detector.enabled:
            return
        try:
            history = self.drift_detector.history
            if not history:
                return
            os.makedirs(output_path, exist_ok=True)
            path = os.path.join(output_path, "drift_history.json")
            data = [
                {
                    "iteration": r.iteration,
                    "inter_cluster_distance": r.inter_cluster_distance,
                    "intra_cluster_variance": r.intra_cluster_variance,
                    "centroid_drift": r.centroid_drift,
                    "collapse_detected": r.collapse_detected,
                    "divergence_detected": r.divergence_detected,
                    "health_score": r.health_score,
                    "rollback_recommended": r.rollback_recommended,
                }
                for r in history
            ]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Drift history saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to export drift history: {e}")
