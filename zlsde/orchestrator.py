"""Pipeline Orchestrator - Coordinates execution of all pipeline layers."""

import logging
import time
from datetime import datetime
from typing import List, Dict
import numpy as np

from zlsde.models.data_models import (
    PipelineConfig, PipelineResult, IterationMetrics, 
    LabeledDataItem, RawDataItem, Label
)
from zlsde.layers.ingestion import DataIngestionLayer
from zlsde.layers.representation import RepresentationEngine
from zlsde.layers.clustering import ClusteringEngine
from zlsde.layers.label_generation import PseudoLabelGenerator
from zlsde.layers.quality_control import QualityControlFilter
from zlsde.layers.self_training import SelfTrainingLoop
from zlsde.layers.exporter import DatasetExporter
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
        
        # Initialize components
        self.ingestion = DataIngestionLayer(config)
        self.embedder = RepresentationEngine(
            modality=config.modality,
            model_name=config.embedding_model,
            device=config.device
        )
        self.clusterer = ClusteringEngine(config)
        
        # Initialize label generator with provider manager
        if config.use_llm:
            from zlsde.layers.label_generation import create_label_generator
            self.label_generator = create_label_generator(config, device=config.device)
        else:
            self.label_generator = None
        
        self.quality_filter = QualityControlFilter(config)
        self.self_trainer = SelfTrainingLoop(config)
        self.exporter = DatasetExporter(config)
        
        logger.info("Pipeline orchestrator initialized")
    
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
            # Layer 1: Data Ingestion
            logger.info("\n[Layer 1] Data Ingestion")
            raw_data = self.ingestion.load_data(self.config.data_sources)
            raw_data = self.ingestion.deduplicate(raw_data)
            raw_data = self.ingestion.validate(raw_data)
            logger.info(f"Loaded {len(raw_data)} valid samples")
            
            if len(raw_data) == 0:
                raise ValueError("No valid data after ingestion")
            
            # Layer 2: Representation Learning
            logger.info("\n[Layer 2] Representation Learning")
            embeddings = self.embedder.embed(raw_data, batch_size=self.config.batch_size)
            
            if self.config.use_dimensionality_reduction:
                logger.info("Applying dimensionality reduction")
                embeddings = self.embedder.reduce_dimensions(embeddings, self.config.n_components)
            
            logger.info(f"Generated embeddings with shape {embeddings.shape}")
            
            # Store embeddings in raw data items for later use
            for i, item in enumerate(raw_data):
                item.embedding = embeddings[i]
            
            # Self-training iteration loop
            iteration_metrics = []
            current_labels = None
            labeled_data = None
            
            for iteration in range(self.config.max_iterations):
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
                logger.info(f"{'=' * 80}")
                
                # Execute iteration
                labeled_data, metrics = self._execute_iteration(
                    iteration, raw_data, embeddings, current_labels
                )
                iteration_metrics.append(metrics)
                
                # Check convergence (except on last iteration)
                if iteration < self.config.max_iterations - 1:
                    if self._check_convergence(iteration_metrics):
                        logger.info(f"\nConverged after {iteration + 1} iterations!")
                        break
                
                # Update current labels for next iteration
                current_labels = np.array([item.cluster_id for item in labeled_data])
            
            # Layer 7: Export
            logger.info("\n[Layer 7] Dataset Export")
            export_result = self.exporter.export(
                labeled_data,
                format=self.config.output_format,
                output_path=self.config.output_path
            )
            
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
                execution_time_seconds=execution_time
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
            
            # Return failed result
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
                    timestamp=datetime.now().isoformat()
                ),
                iteration_history=[],
                config_snapshot=self.config,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
    
    def _execute_iteration(self, iteration: int, raw_data: List[RawDataItem], 
                          embeddings: np.ndarray, previous_labels: np.ndarray = None) -> tuple:
        """
        Execute single self-training iteration.
        
        Args:
            iteration: Current iteration number
            raw_data: List of raw data items
            embeddings: Array of embeddings
            previous_labels: Labels from previous iteration (None for first iteration)
            
        Returns:
            Tuple of (labeled_data, metrics)
        """
        # Layer 3: Clustering
        logger.info(f"[Layer 3] Clustering")
        cluster_result = self.clusterer.cluster(embeddings, method=self.config.clustering_method)
        logger.info(f"Found {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise points")
        logger.info(f"Silhouette score: {cluster_result.silhouette_score:.3f}")
        
        # Layer 4: Pseudo-Label Generation
        logger.info(f"[Layer 4] Pseudo-Label Generation")
        if self.config.use_llm and self.label_generator:
            clusters = self._group_by_cluster(raw_data, cluster_result.labels)
            pseudo_labels = self.label_generator.generate_labels(clusters)
        else:
            # Generate default labels
            pseudo_labels = self._generate_default_labels(cluster_result)
        
        logger.info(f"Generated labels for {len(pseudo_labels)} clusters")
        
        # Create labeled dataset
        labeled_data = self._create_labeled_dataset(
            raw_data, embeddings, cluster_result, pseudo_labels, iteration
        )
        
        # Layer 5: Quality Control
        logger.info(f"[Layer 5] Quality Control")
        quality_scores = self.quality_filter.filter(labeled_data)
        
        # Apply quality scores to labeled data
        for i, item in enumerate(labeled_data):
            item.quality_score = quality_scores[i].score
            item.anomaly_flag = quality_scores[i].anomaly_flag
            item.duplicate_flag = quality_scores[i].duplicate_flag
        
        # Compute iteration metrics
        metrics = self._compute_iteration_metrics(
            iteration, cluster_result, labeled_data, previous_labels
        )
        
        logger.info(f"Iteration metrics: flip_rate={metrics.label_flip_rate:.2%}, "
                   f"quality_mean={metrics.quality_mean:.3f}")
        
        # Layer 6: Self-Training (if not last iteration)
        if iteration < self.config.max_iterations - 1:
            logger.info(f"[Layer 6] Self-Training")
            try:
                # Train classifier on current labels
                current_cluster_labels = np.array([item.cluster_id for item in labeled_data])
                classifier = self.self_trainer.train_classifier(
                    embeddings,
                    current_cluster_labels,
                    self.config.confidence_threshold
                )
                
                # Refine labels
                refined_labels = self.self_trainer.refine_labels(classifier, embeddings)
                
                # Update cluster assignments
                for i, item in enumerate(labeled_data):
                    item.cluster_id = int(refined_labels[i])
                
                logger.info("Self-training refinement complete")
                
            except Exception as e:
                logger.warning(f"Self-training failed: {e}. Continuing with current labels.")
        
        return labeled_data, metrics
    
    def _check_convergence(self, metrics: List[IterationMetrics]) -> bool:
        """
        Determine if pipeline has converged.
        
        Args:
            metrics: List of iteration metrics
            
        Returns:
            True if converged, False otherwise
        """
        return self.self_trainer.check_convergence(metrics)
    
    def _group_by_cluster(self, raw_data: List[RawDataItem], 
                         labels: np.ndarray) -> Dict[int, List[RawDataItem]]:
        """Group raw data items by cluster ID."""
        clusters = {}
        for i, item in enumerate(raw_data):
            cluster_id = int(labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(item)
        return clusters
    
    def _generate_default_labels(self, cluster_result) -> Dict[int, Label]:
        """Generate default labels when LLM is not used."""
        labels = {}
        for cluster_id in range(cluster_result.n_clusters):
            labels[cluster_id] = Label(text=f"cluster_{cluster_id}", confidence=0.5)
        # Add noise label
        labels[-1] = Label(text="noise", confidence=0.0)
        return labels
    
    def _create_labeled_dataset(self, raw_data: List[RawDataItem], 
                               embeddings: np.ndarray,
                               cluster_result, 
                               pseudo_labels: Dict[int, Label],
                               iteration: int) -> List[LabeledDataItem]:
        """Create labeled dataset from clustering and label generation results."""
        labeled_data = []
        
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
                quality_score=0.0,  # Will be updated by quality filter
                modality=item.modality,
                iteration=iteration,
                anomaly_flag=False,
                duplicate_flag=False,
                metadata=item.metadata
            )
            labeled_data.append(labeled_item)
        
        return labeled_data
    
    def _compute_iteration_metrics(self, iteration: int, cluster_result,
                                   labeled_data: List[LabeledDataItem],
                                   previous_labels: np.ndarray = None) -> IterationMetrics:
        """Compute metrics for current iteration."""
        # Compute label flip rate
        if previous_labels is not None:
            current_labels = np.array([item.cluster_id for item in labeled_data])
            label_flip_rate = self.self_trainer.compute_stability(previous_labels, current_labels)
        else:
            label_flip_rate = 0.0  # First iteration
        
        # Compute quality statistics
        quality_scores = [item.quality_score for item in labeled_data]
        quality_mean = float(np.mean(quality_scores))
        quality_std = float(np.std(quality_scores))
        
        # Compute cluster purity (average coherence)
        clusters = {}
        for item in labeled_data:
            if item.cluster_id not in clusters:
                clusters[item.cluster_id] = []
            clusters[item.cluster_id].append(item)
        
        coherence_scores = []
        for cluster_id, items in clusters.items():
            if cluster_id >= 0:  # Exclude noise
                coherence = self.quality_filter.compute_cluster_coherence(items)
                coherence_scores.append(coherence)
        
        cluster_purity = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
        # Compute noise ratio
        n_noise = sum(1 for item in labeled_data if item.cluster_id == -1)
        noise_ratio = n_noise / len(labeled_data)
        
        metrics = IterationMetrics(
            iteration=iteration,
            silhouette_score=cluster_result.silhouette_score,
            n_clusters=cluster_result.n_clusters,
            noise_ratio=noise_ratio,
            label_flip_rate=label_flip_rate,
            cluster_purity=cluster_purity,
            quality_mean=quality_mean,
            quality_std=quality_std,
            timestamp=datetime.now().isoformat()
        )
        
        return metrics
