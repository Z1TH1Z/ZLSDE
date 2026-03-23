"""Layer 7: Dataset Exporter - Export labeled dataset in multiple formats."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from zlsde.models.data_models import PipelineConfig, LabeledDataItem

logger = logging.getLogger(__name__)


class ExportResult:
    """Result of dataset export operation."""
    def __init__(self, path: str, format: str, n_samples: int):
        self.path = path
        self.format = format
        self.n_samples = n_samples


class DatasetExporter:
    """Export labeled dataset in multiple formats with comprehensive metadata."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize dataset exporter with configuration."""
        self.config = config
    
    def export(self, dataset: List[LabeledDataItem], format: str, output_path: str) -> ExportResult:
        """
        Export dataset in specified format.
        
        Args:
            dataset: List of labeled data items
            format: Export format ('csv', 'json', 'parquet')
            output_path: Output directory path
            
        Returns:
            ExportResult with path and metadata
        """
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        logger.info(f"Exporting {len(dataset)} samples to {format} format")
        
        # Route to appropriate export method
        if format == "csv":
            file_path = self.export_csv(dataset, output_path)
        elif format == "json":
            file_path = self.export_json(dataset, output_path)
        elif format == "parquet":
            file_path = self.export_parquet(dataset, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Generate metadata
        metadata = self.generate_metadata(dataset, self.config)
        metadata_path = os.path.join(output_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Export complete: {file_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        return ExportResult(path=file_path, format=format, n_samples=len(dataset))
    
    def export_csv(self, dataset: List[LabeledDataItem], path: str) -> str:
        """
        Export as CSV file.
        
        Args:
            dataset: List of labeled data items
            path: Output directory path
            
        Returns:
            Path to exported CSV file
        """
        # Convert to DataFrame
        data = []
        for item in dataset:
            row = {
                'id': item.id,
                'content': str(item.content),
                'label': item.label,
                'cluster_id': item.cluster_id,
                'confidence': item.confidence,
                'quality_score': item.quality_score,
                'modality': item.modality,
                'iteration': item.iteration,
                'anomaly_flag': item.anomaly_flag,
                'duplicate_flag': item.duplicate_flag
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = os.path.join(path, "dataset.csv")
        df.to_csv(csv_path, index=False)
        
        # Save embeddings separately (too large for CSV)
        embeddings = np.array([item.embedding for item in dataset])
        embeddings_path = os.path.join(path, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        return csv_path
    
    def export_json(self, dataset: List[LabeledDataItem], path: str) -> str:
        """
        Export as JSON file.
        
        Args:
            dataset: List of labeled data items
            path: Output directory path
            
        Returns:
            Path to exported JSON file
        """
        # Convert to JSON-serializable format
        data = []
        for item in dataset:
            record = {
                'id': item.id,
                'content': str(item.content),
                'label': item.label,
                'cluster_id': int(item.cluster_id),
                'confidence': float(item.confidence),
                'quality_score': float(item.quality_score),
                'modality': item.modality,
                'iteration': int(item.iteration),
                'anomaly_flag': bool(item.anomaly_flag),
                'duplicate_flag': bool(item.duplicate_flag),
                'embedding': item.embedding.tolist() if isinstance(item.embedding, np.ndarray) else item.embedding
            }
            if item.metadata:
                record['metadata'] = item.metadata
            data.append(record)
        
        # Save to JSON
        json_path = os.path.join(path, "dataset.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return json_path
    
    def export_parquet(self, dataset: List[LabeledDataItem], path: str) -> str:
        """
        Export as Parquet file.
        
        Args:
            dataset: List of labeled data items
            path: Output directory path
            
        Returns:
            Path to exported Parquet file
        """
        # Convert to DataFrame
        data = []
        for item in dataset:
            row = {
                'id': item.id,
                'content': str(item.content),
                'label': item.label,
                'cluster_id': item.cluster_id,
                'confidence': item.confidence,
                'quality_score': item.quality_score,
                'modality': item.modality,
                'iteration': item.iteration,
                'anomaly_flag': item.anomaly_flag,
                'duplicate_flag': item.duplicate_flag,
                'embedding': item.embedding.tolist() if isinstance(item.embedding, np.ndarray) else item.embedding
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to Parquet
        parquet_path = os.path.join(path, "dataset.parquet")
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        
        return parquet_path
    
    def generate_metadata(self, dataset: List[LabeledDataItem], config: PipelineConfig) -> Dict:
        """
        Generate comprehensive metadata JSON.
        
        Args:
            dataset: List of labeled data items
            config: Pipeline configuration
            
        Returns:
            Metadata dictionary
        """
        # Compute statistics
        labels = [item.label for item in dataset]
        label_counts = pd.Series(labels).value_counts().to_dict()
        
        quality_scores = [item.quality_score for item in dataset]
        confidences = [item.confidence for item in dataset]
        
        metadata = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "zlsde_version": "0.1.0",
            "dataset_statistics": {
                "n_samples": len(dataset),
                "n_clusters": len(set(item.cluster_id for item in dataset if item.cluster_id >= 0)),
                "n_noise": sum(1 for item in dataset if item.cluster_id == -1),
                "n_anomalies": sum(1 for item in dataset if item.anomaly_flag),
                "n_duplicates": sum(1 for item in dataset if item.duplicate_flag),
                "label_distribution": label_counts,
                "quality_mean": float(np.mean(quality_scores)),
                "quality_std": float(np.std(quality_scores)),
                "confidence_mean": float(np.mean(confidences)),
                "confidence_std": float(np.std(confidences))
            },
            "configuration": {
                "modality": config.modality,
                "embedding_model": config.embedding_model,
                "clustering_method": config.clustering_method,
                "llm_model": config.llm_model if config.use_llm else None,
                "max_iterations": config.max_iterations,
                "random_seed": config.random_seed
            },
            "system_info": {
                "device": config.device,
                "batch_size": config.batch_size
            }
        }
        
        return metadata
