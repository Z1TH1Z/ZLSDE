"""Example: Using API Providers for Label Generation

This example demonstrates how to use cloud-based LLM APIs (Groq, Mistral, OpenRouter)
instead of local models for faster label generation with automatic fallback.

Prerequisites:
1. Set up your API keys in a .env file:
   GROQ_API_KEY=your_key_here
   MISTRAL_API_KEY=your_key_here
   OPENROUTER_API_KEY=your_key_here

2. Install required dependencies:
   pip install -r requirements.txt
"""

import os
from dotenv import load_dotenv
from zlsde.models.data_models import PipelineConfig, DataSource, ProviderConfig
from zlsde.orchestrator import PipelineOrchestrator

# Load environment variables from .env file
load_dotenv()

print("=" * 80)
print("ZLSDE Pipeline with API Providers")
print("=" * 80)

# Check API keys
print("\nChecking API keys...")
groq_key = os.getenv('GROQ_API_KEY')
mistral_key = os.getenv('MISTRAL_API_KEY')
openrouter_key = os.getenv('OPENROUTER_API_KEY')

print(f"  GROQ_API_KEY: {'✓' if groq_key else '✗ (will be skipped)'}")
print(f"  MISTRAL_API_KEY: {'✓' if mistral_key else '✗ (will be skipped)'}")
print(f"  OPENROUTER_API_KEY: {'✓' if openrouter_key else '✗ (will be skipped)'}")

if not any([groq_key, mistral_key, openrouter_key]):
    print("\n⚠ Warning: No API keys found. Pipeline will use local model only.")
    print("  To use API providers, create a .env file with your API keys.")

# Configure pipeline with API providers
config = PipelineConfig(
    # Data source
    data_sources=[
        DataSource(type="csv", path="examples/data/sample_text.csv")
    ],
    modality="text",
    
    # Embedding configuration
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    batch_size=32,
    
    # Clustering configuration
    clustering_method="auto",
    min_cluster_size=5,
    
    # API Provider configuration
    provider_config=ProviderConfig(
        provider_type="api",  # Use API services with fallback
        api_providers=["groq", "mistral", "openrouter"],  # Try in this order
        groq_model="mixtral-8x7b-32768",
        mistral_model="mistral-small-latest",
        openrouter_model="mistralai/mixtral-8x7b-instruct",
        local_model="google/flan-t5-base",  # Fallback if all APIs fail
        timeout=30
    ),
    use_llm=True,
    n_representatives=5,
    
    # Quality control
    quality_threshold=0.7,
    
    # Self-training
    max_iterations=2,
    convergence_threshold=0.02,
    
    # Output
    output_format="csv",
    output_path="./output/api_example",
    
    # System
    random_seed=42,
    log_level="INFO"
)

print("\n" + "=" * 80)
print("Running Pipeline...")
print("=" * 80)

# Run pipeline
orchestrator = PipelineOrchestrator(config)
result = orchestrator.run()

print("\n" + "=" * 80)
print("Pipeline Complete!")
print("=" * 80)
print(f"\nStatus: {result.status}")
print(f"Samples processed: {result.n_samples}")
print(f"Labeled samples: {result.n_labeled}")
print(f"Clusters found: {result.n_clusters}")
print(f"Iterations: {result.n_iterations}")
print(f"Dataset saved to: {result.dataset_path}")

# Print provider statistics if available
if hasattr(orchestrator.label_generator, 'provider_manager'):
    print("\n" + "=" * 80)
    print("Provider Statistics")
    print("=" * 80)
    print(orchestrator.label_generator.provider_manager.get_summary())

print("\n✓ Example complete!")
print("\nNext steps:")
print("  1. Check the output directory for your labeled dataset")
print("  2. Review the provider statistics to see which API was used")
print("  3. Try different provider configurations in the config")
