"""Pydantic data models for ZLSDE pipeline configuration."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class DataSource(BaseModel):
    """Configuration for a single data source."""
    type: str = Field(description="Type of data source (e.g., 'csv', 'json', 'pdf')")
    path: str = Field(description="Path or URI to the data source")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about the source")


class ProviderConfig(BaseModel):
    """Configuration for LLM providers."""
    provider_type: str = Field(default='local', description="Type of provider: 'local' runs models locally, 'api' uses external APIs")
    api_providers: List[str] = Field(default_factory=lambda: ['groq', 'mistral', 'openrouter'], description="List of API providers to try in fallback sequence")
    groq_model: str = Field(default='mixtral-8x7b-32768', description="Model to use for Groq provider")
    mistral_model: str = Field(default='mistral-small-latest', description="Model to use for Mistral provider")
    openrouter_model: str = Field(default='mistralai/mixtral-8x7b-instruct', description="Model to use for OpenRouter provider")
    local_model: str = Field(default='google/flan-t5-base', description="Local HuggingFace model identifier")
    timeout: int = Field(default=30, description="Timeout in seconds for API requests")

    def validate(self) -> None:
        """Validate provider configuration."""
        if self.provider_type not in ['local', 'api']:
            raise ValueError(
                f"provider_type must be 'local' or 'api', got '{self.provider_type}'"
            )
        valid_api_providers = {'groq', 'mistral', 'openrouter'}
        for p in self.api_providers:
            if p not in valid_api_providers:
                raise ValueError(
                    f"Invalid API provider: '{p}'. Must be one of {sorted(valid_api_providers)}"
                )
        if self.timeout <= 0:
            raise ValueError(
                f"timeout must be a positive integer, got {self.timeout}"
            )


class PipelineConfig(BaseModel):
    """Full ZLSDE pipeline configuration."""
    data_sources: List[DataSource] = Field(description="List of data sources to process")
    modality: str = Field(default='text', description="Data modality")
    
    # Embedding Configuration
    embedding_model: str = Field(default='sentence-transformers/all-MiniLM-L6-v2', description="Model for generating embeddings")
    use_dimensionality_reduction: bool = Field(default=False, description="Whether to apply dimensionality reduction before clustering")
    n_components: int = Field(default=50, description="Number of components for dimensionality reduction")
    
    # Clustering Configuration
    clustering_method: str = Field(default='auto', description="Method used for clustering data")
    min_cluster_size: int = Field(default=10, description="Minimum number of samples required to form a cluster")
    
    # Labeling Configuration
    provider_config: ProviderConfig = Field(default_factory=ProviderConfig, description="Provider settings for generating labels")
    llm_model: Optional[str] = Field(default=None, description="Legacy field for specifying LLM model. Retained for backward compatibility.")
    use_llm: bool = Field(default=True, description="Whether to use an LLM for labeling")
    n_representatives: int = Field(default=5, description="Number of representative samples to send for labeing per cluster")
    
    # Quality Configuration
    quality_threshold: float = Field(default=0.7, description="Threshold for considering a generated label as high-quality")
    anomaly_contamination: float = Field(default=0.1, description="Expected proportion of anomalies in the dataset")
    duplicate_threshold: float = Field(default=0.95, description="Similarity threshold for flagging duplicate items")
    
    # Training Configuration
    max_iterations: int = Field(default=3, description="Maximum iterations for self-training pipeline")
    convergence_threshold: float = Field(default=0.02, description="Threshold of label changes between iterations indicating convergence")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence for assigning pseudo-labels during self-training")
    
    # Output Configuration
    output_format: str = Field(default='csv', description="Format to save the processed dataset")
    output_path: str = Field(default='./output', description="Directory path for output files")
    
    # System Configuration
    device: str = Field(default='cpu', description="Compute device ('cpu', 'cuda', etc.)")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    log_level: str = Field(default='INFO', description="Logging verbosity level")

    def validate(self) -> None:
        """Validate pipeline configuration."""
        if not self.data_sources:
            raise ValueError("At least one data source must be provided.")
        self.provider_config.validate()


class Label(BaseModel):
    """A generated label."""
    text: str
    confidence: float


class RawDataItem(BaseModel):
    """Raw data item before processing."""
    id: str
    content: Any
    modality: str = "text"
    metadata: Optional[Dict[str, Any]] = None
    embedding: Any = None
    cluster_id: int = -1


class LabeledDataItem(BaseModel):
    """Data item after processing and labeling."""
    id: str
    content: Any
    embedding: Any
    label: str
    cluster_id: int
    confidence: float
    quality_score: float = 0.0
    modality: str
    iteration: int
    anomaly_flag: bool = False
    duplicate_flag: bool = False
    metadata: Optional[Dict[str, Any]] = None


class IterationMetrics(BaseModel):
    """Metrics for a single self-training iteration."""
    iteration: int
    silhouette_score: float
    n_clusters: int
    noise_ratio: float
    label_flip_rate: float
    cluster_purity: float
    quality_mean: float
    quality_std: float
    timestamp: str


class PipelineResult(BaseModel):
    """Result of a pipeline execution."""
    status: str
    dataset_path: str
    n_samples: int
    n_labeled: int
    final_metrics: IterationMetrics
    iteration_history: List[IterationMetrics]
    config_snapshot: PipelineConfig
    execution_time_seconds: float
    error_message: Optional[str] = None


class ClusterResult(BaseModel):
    """Result from a clustering operation."""
    labels: Any = Field(description="Array of cluster labels (-1 for noise)")
    n_clusters: int = Field(description="Number of clusters found")
    n_noise: int = Field(default=0, description="Number of noise points")
    probabilities: Any = Field(default=None, description="Cluster membership probabilities (HDBSCAN only)")
    silhouette_score: float = Field(default=0.0, description="Silhouette score for cluster quality")
    method_used: str = Field(default="unknown", description="Name of clustering algorithm used")

    model_config = {"arbitrary_types_allowed": True}

    def validate(self, n_samples: int) -> None:
        """Validate clustering result."""
        import numpy as np
        labels_arr = np.array(self.labels)
        if len(labels_arr) != n_samples:
            raise ValueError(
                f"labels length ({len(labels_arr)}) does not match n_samples ({n_samples})"
            )


class QualityScore(BaseModel):
    """Quality assessment for a single data item."""
    score: float = Field(description="Overall quality score in [0, 1]")
    anomaly_flag: bool = Field(default=False, description="Whether this item is flagged as an anomaly")
    duplicate_flag: bool = Field(default=False, description="Whether this item is flagged as a duplicate")

