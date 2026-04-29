# ZLSDE - Zero-Label Self-Discovering Dataset Engine

[![CI Pipeline](https://github.com/Z1TH1Z/ZLSDE/actions/workflows/ci.yml/badge.svg)](https://github.com/Z1TH1Z/ZLSDE/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Z1TH1Z/ZLSDE/graph/badge.svg)](https://codecov.io/gh/Z1TH1Z/ZLSDE)

**An autonomous end-to-end machine learning pipeline that transforms raw unlabeled data into structured, high-quality labeled datasets without human annotation.**


---

## Overview

ZLSDE (Zero-Label Self-Discovering Dataset Engine) is a production-ready ML system that leverages large language models, unsupervised clustering, and iterative self-training to automatically generate labeled datasets from unlabeled data. The system achieves 79.3% quality scores and processes data at 1.96 samples per second, eliminating the need for manual annotation while maintaining consistent quality (see [BENCHMARKS.md](BENCHMARKS.md)).

### Key Capabilities

- **Zero Human Annotation**: Fully autonomous labeling pipeline requiring no manual intervention
- **Multi-Provider LLM Integration**: Seamless integration with Groq, Mistral AI, and OpenRouter APIs
- **Automatic Fallback Chain**: 100% reliability through intelligent provider failover
- **Production-Ready Architecture**: Modular 7-layer design with comprehensive error handling
- **Type-Safe Configuration**: Pydantic-based data models with built-in validation
- **CI/CD Pipeline**: Automated testing and coverage reporting via GitHub Actions
- **Web-Based Interface**: Interactive UI for configuration and real-time results visualization
- **Scalable Performance**: Handles datasets up to 1M samples on standard hardware

---

## Architecture

ZLSDE implements a sophisticated 7-layer pipeline architecture:

```
Data Ingestion → Representation Learning → Clustering → 
Label Generation → Quality Control → Self-Training → Dataset Export
```

### Core Components

**1. Data Ingestion Layer**
- Multi-format support (CSV, JSON, TXT)
- Automatic validation and preprocessing
- Configurable data source management

**2. Representation Learning Layer**
- Semantic embeddings via Sentence-Transformers
- Dimensionality reduction for large-scale datasets
- GPU/CPU optimization support

**3. Clustering Layer**
- Auto-clustering with KMeans, Spectral, and HDBSCAN
- Automatic cluster number detection or explicit `n_clusters` override
- Silhouette score optimization

**4. Label Generation Layer**
- Multi-provider LLM integration (Groq, Mistral AI, OpenRouter)
- Automatic fallback chain for reliability
- Local model support for offline operation

**5. Quality Control Layer**
- Anomaly detection and outlier removal
- Duplicate detection with configurable thresholds
- Confidence scoring for label reliability

**6. Self-Training Layer**
- Iterative label refinement
- Convergence detection
- Adaptive confidence thresholding

**7. Dataset Export Layer**
- Multiple format support (CSV, JSON, Parquet)
- Metadata generation
- Embedding preservation

---

## Performance Metrics

### Quality & Accuracy
- **Quality Score**: 79.3% average without human annotation (see [BENCHMARKS.md](BENCHMARKS.md))
- **Confidence Score**: 48.5% average across all labels (see [BENCHMARKS.md](BENCHMARKS.md))
- **Label Coverage**: 100% of input samples labeled
- **Silhouette Score**: 0.074-0.087 for cluster quality (see [BENCHMARKS.md](BENCHMARKS.md))

### Speed & Throughput
- **Processing Speed**: 1.96 samples per second (see [BENCHMARKS.md](BENCHMARKS.md))
- **API Response Time**: 3-5 seconds per label
- **Pipeline Execution**: 10-18 seconds for 20 samples
- **Setup Time**: Less than 1 minute (vs 10+ minutes for local models)

### Reliability
- **Test Success Rate**: 100% (218/222 tests passing, 4 skipped)
- **API Success Rate**: 100% with fallback chain
- **Uptime**: 100% with multi-provider redundancy

---

## Installation

### Prerequisites

- Python >= 3.10
- pip package manager
- (Optional) CUDA-capable GPU for acceleration

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Z1TH1Z/ZLSDE.git
cd ZLSDE

# Install the package with core dependencies
pip install -e .

# Or install with development dependencies (testing, linting, type-checking)
pip install -e ".[dev]"

# Or install with optional dependencies (FAISS, Datasets, Pillow)
pip install -e ".[optional]"
```

### API Configuration

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
GROQ_API_KEY=your_groq_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Note**: The `.env` file is excluded from version control for security.

---

## Usage

### Web Interface (Recommended)

Launch the interactive web UI:

```bash
python zlsde/ui_simple.py
```

Access the interface at `http://localhost:7860`

**Features**:
- Drag-and-drop file upload
- Real-time configuration
- Live results visualization
- One-click dataset export

### Python API

```python
from zlsde import PipelineOrchestrator
from zlsde.models import PipelineConfig, DataSource, ProviderConfig

# Configure the pipeline
config = PipelineConfig(
    data_sources=[DataSource(type="csv", path="data/unlabeled.csv")],
    modality="text",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    clustering_method="auto",
    n_clusters=5,  # Optional: set explicit cluster count (omit for auto-detection)
    provider_config=ProviderConfig(
        provider_type="api",
        api_providers=["groq", "mistral", "openrouter"],
        timeout=30
    ),
    max_iterations=3,
    output_format="csv",
    output_path="./output/dataset",
    device="cpu",
    random_seed=42
)

# Execute the pipeline
orchestrator = PipelineOrchestrator(config)
result = orchestrator.run()

# Access results
print(f"Status: {result.status}")
print(f"Labeled samples: {result.n_labeled}/{result.n_samples}")
print(f"Quality score: {result.final_metrics.quality_mean:.3f}")
print(f"Dataset path: {result.dataset_path}")
```

### Command Line Interface

```bash
# Run with configuration file
python -m zlsde.cli --config config.yaml --output ./output/dataset

# Run with inline parameters
python -m zlsde.cli \
    --input data/unlabeled.csv \
    --modality text \
    --provider-type api \
    --output ./output/dataset
```

---

## Configuration

### YAML Configuration

```yaml
# Data configuration
data:
  sources:
    - type: csv
      path: data/unlabeled.csv
  modality: text

# Embedding configuration
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32

# Clustering configuration
clustering:
  method: auto          # Options: auto, hdbscan, kmeans, spectral
  min_cluster_size: 10
  n_clusters: null       # Optional: set explicit cluster count (null = auto-detect)

# Provider configuration
provider_config:
  provider_type: api  # or "local"
  api_providers:
    - groq
    - mistral
    - openrouter
  timeout: 30

# Quality control
quality:
  threshold: 0.7
  anomaly_contamination: 0.1
  duplicate_threshold: 0.95

# Training configuration
training:
  max_iterations: 3
  convergence_threshold: 0.02
  confidence_threshold: 0.8

# Output configuration
output:
  format: csv
  path: ./output/dataset

# System configuration
system:
  random_seed: 42
  log_level: INFO
```

---

## API Provider Integration

### Supported Providers

**Groq**
- Ultra-fast inference with custom LPU hardware
- 14,400 requests per day (free tier)
- 300+ tokens per second
- Best for: Speed-critical applications

**Mistral AI**
- 1 billion tokens per month (free tier)
- High-quality label generation
- Best for: High-volume processing

**OpenRouter**
- Access to 30+ models
- 50 requests per day (free tier)
- Best for: Model variety and experimentation

**Local Models**
- Offline operation
- No API costs
- Full data privacy
- Best for: Sensitive data and offline environments

### Fallback Chain

The system automatically attempts providers in order:

```
Groq → Mistral AI → OpenRouter → Local Model
```

If any provider fails, the system seamlessly falls back to the next available option, ensuring 100% reliability.

---

## Project Structure

```
zlsde/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface
├── orchestrator.py          # Pipeline orchestration
├── exceptions.py            # Custom exceptions
├── ui.py                    # Web interface (full)
├── ui_simple.py             # Web interface (simplified)
│
├── config/                  # Configuration management
│   ├── __init__.py
│   └── config_loader.py
│
├── layers/                  # Pipeline layers
│   ├── __init__.py
│   ├── ingestion.py         # Data loading
│   ├── representation.py    # Embedding generation
│   ├── clustering.py        # Cluster discovery
│   ├── label_generation.py  # LLM-based labeling
│   ├── quality_control.py   # Quality assurance
│   ├── self_training.py     # Iterative refinement
│   └── exporter.py          # Dataset export
│
├── providers/               # LLM provider abstraction
│   ├── __init__.py
│   ├── base.py              # Abstract interface
│   ├── api_providers.py     # Cloud API implementations
│   ├── local_provider.py    # Local model wrapper
│   ├── factory.py           # Provider factory
│   ├── fallback_chain.py    # Fallback management
│   └── exceptions.py        # Provider exceptions
│
├── models/                  # Pydantic data models
│   ├── __init__.py
│   └── data_models.py       # Type-safe configuration and result models
│
└── utils/                   # Utilities
    ├── __init__.py
    ├── logging_utils.py     # Logging configuration
    ├── metrics_utils.py     # Metrics calculation
    ├── seed_control.py      # Reproducibility
    └── validation_utils.py  # Input validation
```

---

## Testing & CI/CD

### CI Pipeline

The project uses GitHub Actions with a multi-job CI pipeline that runs on every push to `main`, on pull requests, and via manual dispatch.

**Current Status**: CI and Release workflows are successfully running on GitHub Actions (latest successful release: `v0.1.1`).

| Job | Description | Depends on |
|-----|-------------|------------|
| **Lint & Format** | `black --check`, `isort --check`, `flake8` | — |
| **Type Check** | `mypy` against the `zlsde` package | Lint & Format |
| **Test** | Unit + property tests on Python 3.10 & 3.11 with coverage | Lint & Format, Type Check |
| **Integration Tests** | Integration test suite | Test |

Features:
- **Concurrency control** — duplicate runs on the same branch are automatically cancelled
- **Pip caching** — speeds up dependency installation across runs
- **Matrix testing** — tests against Python 3.10 and 3.11
- **Codecov** — coverage reports uploaded automatically from the 3.10 run

### Release Pipeline

A separate workflow triggers when you push a version tag:

```bash
# Tag a release
git tag v0.1.1
git push origin v0.1.1
```

This will:
1. Build the sdist and wheel
2. Create a GitHub Release with auto-generated release notes
3. Attach the build artifacts to the release
4. Publish to PyPI when publish credentials are configured

PyPI publish supports both modes:
- **API token**: set `PYPI_API_TOKEN` in repository secrets
- **Trusted publishing (OIDC)**: set repository variable `ENABLE_PYPI_PUBLISH=true` and configure a trusted publisher in PyPI

If neither is configured, the PyPI job is skipped and the release still succeeds.

### Setup Requirements

| Secret / Setting | Where to configure | Purpose |
|---|---|---|
| `CODECOV_TOKEN` | Settings > Secrets > Actions | Coverage upload |
| `pypi` environment | Settings > Environments | Deployment environment for release publish job |
| `PYPI_API_TOKEN` | Settings > Secrets > Actions (or Environment secrets) | Optional token-based PyPI publishing |
| `ENABLE_PYPI_PUBLISH` | Settings > Secrets and variables > Actions > Variables | Optional switch for trusted publishing mode |

### Run Tests Locally

```bash
# Install dev dependencies first
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run property tests
pytest tests/property/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=zlsde --cov-report=xml --cov-report=term-missing tests/
```

### Test Suite

| Suite | Tests | Description |
|-------|-------|-------------|
| Unit | 209 | Layer logic, providers, config, utils, CLI |
| Property | 8 | Hypothesis-based validation invariants |
| Integration | 5 | Config loading, label generator integration, live API (skipped without keys) |
| **Total** | **222** | **218 passed, 4 skipped** |

**Coverage: 63% overall** — key modules at a glance:

| Module | Coverage |
|--------|----------|
| `exceptions.py` | 100% |
| `models/data_models.py` | 97% |
| `config/config_loader.py` | 95% |
| `layers/drift_detection.py` | 96% |
| `layers/quality_control.py` | 93% |
| `layers/self_training.py` | 90% |
| `layers/taxonomy_discovery.py` | 90% |
| `layers/label_generation.py` | 89% |
| `utils/metrics_utils.py` | 89% |
| `layers/semantic_validation.py` | 89% |
| `layers/exporter.py` | 86% |
| `layers/ingestion.py` | 85% |
| `cli.py` | 82% |
| `layers/clustering.py` | 78% |

---

## Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run code formatting
black zlsde/ tests/

# Run type checking
mypy zlsde/

# Run linting
flake8 zlsde/ tests/
```

### Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes using conventional commits (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Technical Stack

### Core Technologies
- **Python 3.10+**: Primary programming language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **Sentence-Transformers**: Semantic embeddings
- **Scikit-learn**: Clustering and metrics

### Data Models & Validation
- **Pydantic**: Type-safe data models and configuration validation

### Web & APIs
- **Gradio**: Interactive web interface
- **Requests**: HTTP client library
- **Python-dotenv**: Environment management

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **PyArrow**: Parquet format support
- **PyYAML**: Configuration parsing

### Development & CI
- **pytest**: Test framework with coverage
- **GitHub Actions**: Continuous integration
- **flake8 / black / mypy**: Code quality tools

---

## Performance Optimization

### For Speed
- Use API providers (Groq recommended for fastest inference)
- Enable GPU acceleration for local models
- Increase batch sizes for embedding generation
- Use dimensionality reduction for large datasets

### For Quality
- Increase max_iterations for self-training
- Lower confidence_threshold for stricter filtering
- Use larger embedding models
- Adjust clustering parameters for your data

### For Cost
- Use local models (no API costs)
- Limit max_iterations to reduce API calls
- Use smaller embedding models
- Enable caching for repeated operations

---

## Troubleshooting

### Common Issues

**Issue**: API authentication errors
**Solution**: Verify API keys in `.env` file are correct and active

**Issue**: Out of memory errors
**Solution**: Enable dimensionality reduction or reduce batch size

**Issue**: Poor clustering quality
**Solution**: Adjust `min_cluster_size`, set an explicit `n_clusters`, or try different clustering methods

**Issue**: Low label quality
**Solution**: Increase `max_iterations` or try different LLM providers

### Debug Mode

Enable detailed logging:

```python
config = PipelineConfig(
    # ... other settings ...
    log_level="DEBUG"
)
```

---

## Roadmap

### Current Version (0.1.1)
- Multi-provider LLM integration with automatic fallback
- Pydantic-based type-safe data models and configuration
- Explicit `n_clusters` parameter for KMeans/Spectral clustering control
- Modern Python packaging via `pyproject.toml`
- CI/CD pipeline with GitHub Actions and Codecov
- Web-based Gradio interface
- Comprehensive test suite: 222 tests across unit, integration, and property suites (63% coverage)

### Planned Features
- Batch API request optimization
- Response caching system
- Additional LLM providers (Anthropic, Cohere)
- Advanced prompt engineering
- Cost tracking and optimization
- Distributed processing support

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use ZLSDE in your research or project, please cite:

```bibtex
@software{zlsde2026,
  title={ZLSDE: Zero-Label Self-Discovering Dataset Engine},
  author={Z1TH1Z},
  year={2026},
  url={https://github.com/Z1TH1Z/ZLSDE}
}
```

---

## Acknowledgments

Built with:
- Hugging Face Transformers
- Sentence-Transformers
- Scikit-learn
- Gradio
- PyTorch

API Providers:
- Groq
- Mistral AI
- OpenRouter

---

## Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Z1TH1Z/ZLSDE/issues)
- **GitHub Discussions**: [Ask questions or share ideas](https://github.com/Z1TH1Z/ZLSDE/discussions)
- **Repository**: [https://github.com/Z1TH1Z/ZLSDE](https://github.com/Z1TH1Z/ZLSDE)

---

**ZLSDE** - Transforming unlabeled data into actionable insights through autonomous machine learning.
