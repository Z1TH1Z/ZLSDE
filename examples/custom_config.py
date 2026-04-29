"""Example of programmatic configuration for ZLSDE pipeline."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from zlsde import DataSource, PipelineConfig, PipelineOrchestrator


def create_custom_config():
    """Create a custom pipeline configuration programmatically."""

    config = PipelineConfig(
        # Data sources - can specify multiple sources
        data_sources=[
            DataSource(type="csv", path="examples/data/sample_text.csv"),
            # DataSource(type="json", path="examples/data/sample_data.json"),
        ],
        # Modality: "text", "image", or "multimodal"
        modality="text",
        # Embedding configuration
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_dimensionality_reduction=True,  # Enable UMAP reduction
        n_components=50,  # Reduce to 50 dimensions
        # Clustering configuration
        clustering_method="auto",  # Auto-select best method
        min_cluster_size=10,  # Minimum samples per cluster
        # Label generation configuration
        llm_model="google/flan-t5-base",
        use_llm=True,  # Use LLM for semantic labels
        n_representatives=5,  # Number of samples to show LLM
        # Quality control configuration
        quality_threshold=0.75,  # Stricter quality filtering
        anomaly_contamination=0.05,  # Expect 5% anomalies
        duplicate_threshold=0.98,  # High similarity threshold
        # Self-training configuration
        max_iterations=3,  # Maximum refinement iterations
        convergence_threshold=0.015,  # 1.5% flip rate for convergence
        confidence_threshold=0.85,  # High confidence for training
        # Export configuration
        output_format="parquet",  # Use Parquet for efficiency
        output_path="./output/custom_dataset",
        # System configuration
        device="cpu",  # Use "cuda" for GPU acceleration
        batch_size=16,  # Smaller batches for CPU
        random_seed=42,  # For reproducibility
        log_level="INFO",
    )

    return config


def main():
    """Run pipeline with custom configuration."""

    print("=" * 80)
    print("ZLSDE Custom Configuration Example")
    print("=" * 80)

    # Create custom configuration
    config = create_custom_config()

    # Validate configuration
    try:
        config.validate()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return 1

    # Display configuration summary
    print("\nConfiguration Summary:")
    print("-" * 80)
    print(f"Modality: {config.modality}")
    print(f"Embedding model: {config.embedding_model}")
    print(
        f"Dimensionality reduction: {config.use_dimensionality_reduction} "
        f"(n_components={config.n_components})"
    )
    print(f"Clustering: {config.clustering_method} (min_size={config.min_cluster_size})")
    print(f"LLM: {config.llm_model if config.use_llm else 'Disabled'}")
    print(f"Quality threshold: {config.quality_threshold}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Output format: {config.output_format}")
    print(f"Output path: {config.output_path}")
    print("-" * 80)

    # Run pipeline
    print("\nStarting pipeline execution...")
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run()

    # Display results
    if result.status == "completed":
        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Dataset: {result.dataset_path}")
        print(f"Samples: {result.n_labeled}/{result.n_samples}")
        print(f"Clusters: {result.final_metrics.n_clusters}")
        print(
            f"Quality: {result.final_metrics.quality_mean:.3f} ± {result.final_metrics.quality_std:.3f}"
        )
        print(f"Time: {result.execution_time_seconds:.2f}s")
        print("=" * 80)
        return 0
    else:
        print(f"\nFailed: {result.error_message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
