"""Command-line interface for ZLSDE pipeline."""

import argparse
import logging
import sys
import yaml
from pathlib import Path

from zlsde.orchestrator import PipelineOrchestrator
from zlsde.config.config_loader import ConfigLoader


def setup_logging(verbose: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ZLSDE - Zero-Label Self-Discovering Dataset Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline with config file
  zlsde --config config.yaml
  
  # Run with custom output directory
  zlsde --config config.yaml --output ./my_output
  
  # Run with verbose logging
  zlsde --config config.yaml --verbose
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="ZLSDE 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config_path = Path(args.config)
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {args.config}")
            return 1
        
        # Load config using ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.from_yaml(str(config_path))
        
        # Override output path if provided
        if args.output:
            logger.info(f"Overriding output path: {args.output}")
            config.output_path = args.output
        
        # Validate configuration
        logger.info("Validating configuration...")
        config.validate()
        logger.info("Configuration valid")
        
        # Display configuration summary
        logger.info("\n" + "=" * 80)
        logger.info("Configuration Summary")
        logger.info("=" * 80)
        logger.info(f"Modality: {config.modality}")
        logger.info(f"Embedding model: {config.embedding_model}")
        logger.info(f"Clustering method: {config.clustering_method}")
        logger.info(f"LLM model: {config.llm_model if config.use_llm else 'None (disabled)'}")
        logger.info(f"Max iterations: {config.max_iterations}")
        logger.info(f"Output format: {config.output_format}")
        logger.info(f"Output path: {config.output_path}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Random seed: {config.random_seed}")
        logger.info("=" * 80 + "\n")
        
        # Create and run pipeline
        logger.info("Initializing pipeline...")
        orchestrator = PipelineOrchestrator(config)
        
        logger.info("Starting pipeline execution...")
        result = orchestrator.run()
        
        # Display results
        if result.status == "completed":
            logger.info("\n" + "=" * 80)
            logger.info("SUCCESS: Pipeline completed successfully!")
            logger.info("=" * 80)
            logger.info(f"Dataset path: {result.dataset_path}")
            logger.info(f"Total samples: {result.n_samples}")
            logger.info(f"Labeled samples: {result.n_labeled}")
            logger.info(f"Final clusters: {result.final_metrics.n_clusters}")
            logger.info(f"Final silhouette score: {result.final_metrics.silhouette_score:.3f}")
            logger.info(f"Execution time: {result.execution_time_seconds:.2f} seconds")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("\n" + "=" * 80)
            logger.error("FAILED: Pipeline execution failed")
            logger.error("=" * 80)
            if result.error_message:
                logger.error(f"Error: {result.error_message}")
            logger.error("=" * 80)
            return 1
    
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
