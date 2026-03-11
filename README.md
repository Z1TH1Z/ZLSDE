# ZLSDE - Zero-Label Self-Discovering Dataset Engine

An autonomous end-to-end pipeline that transforms raw unlabeled multimodal data into structured, high-quality labeled datasets without human annotation.

## Overview

ZLSDE orchestrates seven modular layers:
1. **Data Ingestion** - Load and validate raw data from multiple sources
2. **Representation Learning** - Transform data into semantic embeddings
3. **Clustering** - Discover natural groupings in embedding space
4. **Pseudo-Label Generation** - Generate labels using zero-shot LLMs
5. **Quality Control** - Filter low-quality samples and anomalies
6. **Self-Training Loop** - Iteratively refine labels
7. **Dataset Export** - Export structured labeled datasets

## Features

- **Zero Human Annotation**: Fully autonomous labeling pipeline
- **Multimodal Support**: Text, images, and multimodal data
- **CPU Compatible**: Runs on CPU (GPU optional for acceleration)
- **Scalable**: Handles up to 1M samples on a single machine
- **Reproducible**: Deterministic execution with seed control
- **Configurable**: YAML-based configuration system

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/zlsde.git
cd zlsde

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from zlsde import PipelineOrchestrator, PipelineConfig
from zlsde.models import DataSource

# Configure pipeline
config = PipelineConfig(
    data_sources=[DataSource(type="csv", path="data/unlabeled.csv")],
    modality="text",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    clustering_method="auto",
    llm_model="google/flan-t5-base",
    max_iterations=3,
    output_format="csv",
    output_path="./output/dataset",
    device="cpu",
    random_seed=42
)

# Run pipeline
orchestrator = PipelineOrchestrator(config)
result = orchestrator.run()

print(f"Status: {result.status}")
print(f"Labeled samples: {result.n_labeled}/{result.n_samples}")
print(f"Dataset saved to: {result.dataset_path}")
```

## Configuration

Create a YAML configuration file:

```yaml
data:
  sources:
    - type: csv
      path: data/unlabeled.csv
  modality: text

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32

clustering:
  method: auto
  min_cluster_size: 10

labeling:
  llm_model: google/flan-t5-base
  use_llm: true
  n_representatives: 5

quality:
  threshold: 0.7
  anomaly_contamination: 0.1
  duplicate_threshold: 0.95

training:
  max_iterations: 3
  convergence_threshold: 0.02
  confidence_threshold: 0.8

output:
  format: csv
  path: ./output/dataset

system:
  random_seed: 42
  log_level: INFO
```

Run with CLI:

```bash
zlsde --config config.yaml --verbose
```

## API Provider Configuration

ZLSDE supports both local models and cloud-based LLM APIs for label generation. Using API providers can significantly reduce setup time and resource requirements.

### Supported API Providers

- **Groq**: Fast inference with Mixtral and Llama models
- **Mistral AI**: Official Mistral models API
- **OpenRouter**: Access to multiple LLM providers

### Setting Up API Keys

1. Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

2. Add your API keys to the `.env` file:
```bash
# Get your keys from:
# Groq: https://console.groq.com/keys
# Mistral: https://console.mistral.ai/api-keys
# OpenRouter: https://openrouter.ai/keys

GROQ_API_KEY=your_groq_key_here
MISTRAL_API_KEY=your_mistral_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

3. **Important**: Never commit the `.env` file to version control. It's already in `.gitignore`.

### Using API Providers in Configuration

Update your configuration to use API providers:

```yaml
labeling:
  provider_config:
    # Use API services with automatic fallback
    provider_type: api
    
    # Providers are tried in order
    api_providers:
      - groq
      - mistral
      - openrouter
    
    # Model configurations
    groq_model: mixtral-8x7b-32768
    mistral_model: mistral-small-latest
    openrouter_model: mistralai/mixtral-8x7b-instruct
    
    # Local model as fallback
    local_model: google/flan-t5-base
    
    # API timeout in seconds
    timeout: 30
  
  use_llm: true
  n_representatives: 5
```

### Fallback Chain

When `provider_type: api` is set, ZLSDE automatically implements a fallback chain:

1. **Groq API** - Tried first (fast inference)
2. **Mistral AI API** - Tried if Groq fails
3. **OpenRouter API** - Tried if Mistral fails
4. **Local Model** - Final fallback if all APIs fail

This ensures your pipeline continues working even if one service is unavailable.

### Using Local Models Only

To use only local models (no API calls):

```yaml
labeling:
  provider_config:
    provider_type: local
    local_model: google/flan-t5-base
  use_llm: true
```

### Backward Compatibility

Old configurations using `llm_model` directly are still supported:

```yaml
labeling:
  llm_model: google/flan-t5-base  # Automatically uses local provider
  use_llm: true
```

## Requirements

- Python >= 3.10
- See `requirements.txt` for full dependency list

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run property-based tests
pytest tests/property/

# Format code
black zlsde/ tests/

# Type checking
mypy zlsde/
```

## Project Structure

```
zlsde/
├── zlsde/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── cli.py
│   ├── exceptions.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── data_models.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── representation.py
│   │   ├── clustering.py
│   │   ├── label_generation.py
│   │   ├── quality_control.py
│   │   ├── self_training.py
│   │   └── exporter.py
│   └── utils/
│       ├── __init__.py
│       ├── seed_control.py
│       ├── logging_utils.py
│       ├── metrics_utils.py
│       └── validation_utils.py
├── tests/
│   ├── unit/
│   ├── property/
│   └── integration/
├── setup.py
├── requirements.txt
└── README.md
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Citation

If you use ZLSDE in your research, please cite:

```bibtex
@software{zlsde2024,
  title={ZLSDE: Zero-Label Self-Discovering Dataset Engine},
  author={ZLSDE Team},
  year={2024},
  url={https://github.com/yourusername/zlsde}
}
```
