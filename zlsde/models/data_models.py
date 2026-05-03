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
    groq_model: str = Field(default='llama-3.1-8b-instant', description="Model to use for Groq provider")
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
    n_clusters: Optional[int] = Field(default=None, description="Explicit number of clusters for KMeans/Spectral methods")
    
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
    
    # Taxonomy Discovery Configuration (Feature 1: ALTD)
    enable_taxonomy: bool = Field(default=False, description="Enable hierarchical taxonomy discovery")
    taxonomy_max_depth: int = Field(default=3, description="Maximum depth for recursive taxonomy splitting")
    taxonomy_min_samples: int = Field(default=5, description="Minimum samples to attempt sub-clustering")
    taxonomy_silhouette_threshold: float = Field(default=0.1, description="Minimum silhouette improvement to accept sub-clusters")

    # Provenance Configuration (Feature 2: Label Provenance)
    enable_provenance: bool = Field(default=True, description="Enable label provenance tracking")
    enable_explanations: bool = Field(default=False, description="Generate natural language explanations for labels")

    # Semantic Validation Configuration (Feature 3: CCSV)
    enable_semantic_validation: bool = Field(default=True, description="Enable cross-cluster semantic validation")
    label_similarity_threshold: float = Field(default=0.8, description="Threshold for detecting label collisions")
    centroid_similarity_threshold: float = Field(default=0.85, description="Threshold for detecting merge candidates")

    # Multi-Granularity Embedding Configuration (Feature 4)
    enable_embedding_fusion: bool = Field(default=False, description="Enable multi-granularity embedding fusion")
    fusion_models: List[str] = Field(default_factory=lambda: [], description="Additional embedding models for fusion")
    fusion_weights: List[float] = Field(default_factory=lambda: [], description="Weights for each embedding model (auto-learned if empty)")

    # Adaptive Self-Training Configuration (Feature 5: CWAST)
    enable_adaptive_training: bool = Field(default=True, description="Enable confidence-weighted adaptive self-training")
    curriculum_percentile_decay: float = Field(default=10.0, description="Percentile decrease per iteration for curriculum training")

    # Dynamic Provider Optimization (Feature 6)
    enable_provider_optimization: bool = Field(default=False, description="Enable cost-quality-aware dynamic provider routing")
    provider_quality_weight: float = Field(default=0.5, description="Weight for quality vs cost in provider selection")
    provider_exploration_rate: float = Field(default=0.1, description="Rate of exploring non-optimal providers for better estimates")

    # Drift Detection Configuration (Feature 7)
    enable_drift_detection: bool = Field(default=True, description="Enable embedding drift detection across iterations")
    drift_collapse_threshold: float = Field(default=0.3, description="Inter-cluster distance drop ratio triggering collapse detection")
    drift_divergence_threshold: float = Field(default=2.0, description="Centroid drift ratio triggering divergence detection")

    # System Configuration
    device: str = Field(default='cpu', description="Compute device ('cpu', 'cuda', etc.)")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    log_level: str = Field(default='INFO', description="Logging verbosity level")

    def validate(self) -> None:
        """Validate pipeline configuration."""
        if not self.data_sources:
            raise ValueError("At least one data source must be provided.")
        if self.n_clusters is not None and self.n_clusters < 2:
            raise ValueError("n_clusters must be >= 2 when provided.")
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
    content_hash: Optional[str] = None
    source: Optional[str] = None


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


# --- Feature 1: Autonomous Label Taxonomy Discovery (ALTD) ---

class TaxonomyNode(BaseModel):
    """A single node in the discovered label taxonomy tree."""
    label: str = Field(description="Label text for this node")
    level: int = Field(description="Depth level in the taxonomy (0 = root)")
    cluster_id: int = Field(default=-1, description="Associated cluster ID at this level")
    n_samples: int = Field(default=0, description="Number of samples under this node")
    silhouette_score: float = Field(default=0.0, description="Clustering quality at this node")
    confidence: float = Field(default=0.0, description="Label confidence for this node")
    children: List["TaxonomyNode"] = Field(default_factory=list, description="Child taxonomy nodes")
    parent_label: Optional[str] = Field(default=None, description="Parent node's label")

    model_config = {"arbitrary_types_allowed": True}


class TaxonomyTree(BaseModel):
    """Complete discovered label taxonomy."""
    root_nodes: List[TaxonomyNode] = Field(default_factory=list, description="Top-level taxonomy nodes")
    max_depth: int = Field(default=0, description="Maximum depth of the taxonomy")
    total_nodes: int = Field(default=0, description="Total number of nodes in the tree")
    discovery_method: str = Field(default="recursive_splitting", description="Method used for taxonomy discovery")

    def flatten(self) -> Dict[str, List[str]]:
        """Flatten taxonomy into parent->children mapping."""
        result: Dict[str, List[str]] = {}
        def _traverse(node: TaxonomyNode):
            child_labels = [c.label for c in node.children]
            if child_labels:
                result[node.label] = child_labels
            for child in node.children:
                _traverse(child)
        for root in self.root_nodes:
            _traverse(root)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize taxonomy to nested dictionary."""
        def _node_to_dict(node: TaxonomyNode) -> Dict[str, Any]:
            return {
                "label": node.label,
                "level": node.level,
                "n_samples": node.n_samples,
                "confidence": node.confidence,
                "silhouette_score": node.silhouette_score,
                "children": [_node_to_dict(c) for c in node.children]
            }
        return {
            "max_depth": self.max_depth,
            "total_nodes": self.total_nodes,
            "taxonomy": [_node_to_dict(r) for r in self.root_nodes]
        }


# --- Feature 2: Label Provenance & Explainability ---

class LabelProvenance(BaseModel):
    """Complete provenance record for a single label decision."""
    cluster_id: int = Field(description="Cluster this label was generated for")
    label_text: str = Field(description="The generated label")
    provider_used: str = Field(default="unknown", description="LLM provider that generated this label")
    prompt_sent: str = Field(default="", description="Exact prompt sent to the LLM")
    raw_response: str = Field(default="", description="Raw LLM response before parsing")
    representative_samples: List[str] = Field(default_factory=list, description="Representative samples used in prompt")
    confidence: float = Field(default=0.0, description="Confidence score")
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict, description="Component scores contributing to confidence")
    explanation: str = Field(default="", description="LLM-generated natural language explanation for the label")
    iteration_history: List[Dict[str, Any]] = Field(default_factory=list, description="Label changes across iterations")
    timestamp: str = Field(default="", description="When this label was generated")


class ProvenanceReport(BaseModel):
    """Aggregated provenance report for the entire pipeline run."""
    provenance_records: List[LabelProvenance] = Field(default_factory=list)
    provider_usage: Dict[str, int] = Field(default_factory=dict, description="Call count per provider")
    total_labels_generated: int = Field(default=0)
    total_labels_explained: int = Field(default=0)
    avg_confidence: float = Field(default=0.0)


# --- Feature 3: Cross-Cluster Semantic Validation (CCSV) ---

class ValidationFlag(BaseModel):
    """A flag raised by cross-cluster semantic validation."""
    flag_type: str = Field(description="Type: 'merge_candidate', 'split_candidate', 'label_collision', 'outlier_cluster'")
    cluster_ids: List[int] = Field(description="Cluster IDs involved")
    labels: List[str] = Field(description="Labels of the involved clusters")
    similarity_score: float = Field(default=0.0, description="Similarity metric that triggered the flag")
    description: str = Field(default="", description="Human-readable description of the issue")
    resolved: bool = Field(default=False, description="Whether this flag was resolved by relabeling")


class SemanticValidationResult(BaseModel):
    """Result of cross-cluster semantic validation."""
    flags: List[ValidationFlag] = Field(default_factory=list)
    n_merge_candidates: int = Field(default=0)
    n_split_candidates: int = Field(default=0)
    n_label_collisions: int = Field(default=0)
    n_outlier_clusters: int = Field(default=0)
    n_relabeled: int = Field(default=0, description="Number of clusters relabeled after validation")
    semantic_consistency_score: float = Field(default=1.0, description="Overall consistency score [0, 1]")


# --- Feature 7: Embedding Drift Detection ---

class DriftReport(BaseModel):
    """Report on embedding space health across iterations."""
    iteration: int = Field(description="Iteration this report covers")
    inter_cluster_distance: float = Field(default=0.0, description="Mean distance between cluster centroids")
    intra_cluster_variance: float = Field(default=0.0, description="Mean within-cluster variance")
    centroid_drift: float = Field(default=0.0, description="Mean centroid displacement from previous iteration")
    collapse_detected: bool = Field(default=False, description="Whether embedding collapse was detected")
    divergence_detected: bool = Field(default=False, description="Whether embedding divergence was detected")
    rollback_recommended: bool = Field(default=False, description="Whether rollback to previous iteration is recommended")
    health_score: float = Field(default=1.0, description="Overall embedding space health [0, 1]")

