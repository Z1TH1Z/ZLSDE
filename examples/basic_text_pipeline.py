"""Basic text dataset pipeline example for ZLSDE."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from zlsde import DataSource, PipelineConfig, PipelineOrchestrator


def main():
    """Run basic text pipeline example."""

    # Configure pipeline for text data
    config = PipelineConfig(
        data_sources=[DataSource(type="csv", path="examples/data/sample_text.csv")],
        modality="text",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_dimensionality_reduction=False,
        clustering_method="auto",
        min_cluster_size=5,
        llm_model="google/flan-t5-base",
        use_llm=True,
        n_representatives=5,
        quality_threshold=0.7,
        max_iterations=3,
        convergence_threshold=0.02,
        confidence_threshold=0.8,
        output_format="csv",
        output_path="./output/text_dataset",
        device="cpu",
        batch_size=32,
        random_seed=42,
        log_level="INFO",
    )

    print("=" * 80)
    print("ZLSDE Basic Text Pipeline Example")
    print("=" * 80)
    print(f"Input: {config.data_sources[0].path}")
    print(f"Output: {config.output_path}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"LLM model: {config.llm_model}")
    print("=" * 80)
    print()

    # Create and run pipeline
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run()

    # Display results
    if result.status == "completed":
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
        print(f"Dataset saved to: {result.dataset_path}")
        print(f"Total samples: {result.n_samples}")
        print(f"Labeled samples: {result.n_labeled}")
        print(f"Number of clusters: {result.final_metrics.n_clusters}")
        print(f"Silhouette score: {result.final_metrics.silhouette_score:.3f}")
        print(f"Quality mean: {result.final_metrics.quality_mean:.3f}")
        print(f"Execution time: {result.execution_time_seconds:.2f} seconds")
        print("=" * 80)

        # Display iteration history
        print("\nIteration History:")
        print("-" * 80)
        for metrics in result.iteration_history:
            print(
                f"Iteration {metrics.iteration}: "
                f"clusters={metrics.n_clusters}, "
                f"silhouette={metrics.silhouette_score:.3f}, "
                f"flip_rate={metrics.label_flip_rate:.2%}"
            )
        print("=" * 80)
    else:
        print(f"\nPipeline failed: {result.error_message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
