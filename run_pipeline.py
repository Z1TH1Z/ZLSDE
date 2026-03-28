#!/usr/bin/env python
"""Quick pipeline runner for testing"""
from zlsde import PipelineConfig, PipelineOrchestrator
from zlsde.models import DataSource, ProviderConfig
import json

# Build configuration
config = PipelineConfig(
    data_sources=[
        DataSource(type="csv", path="examples/data/sample_text.csv")
    ],
    modality="text",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    clustering_method="auto",
    min_cluster_size=3,
    provider_config=ProviderConfig(
        provider_type="local",
        local_model="google/flan-t5-base",
    ),
    use_llm=True,
    max_iterations=2,
    output_format="csv",
    output_path="./output/dataset",
    # Enable features
    enable_taxonomy=True,
    enable_provenance=True,
    enable_semantic_validation=True,
    enable_drift_detection=True,
)

print("=" * 70)
print("Starting ZLSDE Pipeline...")
print("=" * 70)

# Run pipeline
orchestrator = PipelineOrchestrator(config)
result = orchestrator.run()

print("\n" + "=" * 70)
print("Pipeline Execution Complete!")
print("=" * 70)
print(f"Status:           {result.status}")
print(f"Samples Processed: {result.n_samples}")
print(f"Labeled Samples:  {result.n_labeled}")
print(f"Number of Clusters: {result.final_metrics.n_clusters}")
print(f"Output Location:  {result.dataset_path}")
print("=" * 70)
