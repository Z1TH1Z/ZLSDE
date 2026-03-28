# ZLSDE — Setup, Run & Test Guide

**Zero-Label Self-Discovering Dataset Engine** · v0.1.0

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [API Keys (Optional)](#3-api-keys-optional)
4. [Project Structure](#4-project-structure)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Configuration Reference](#6-configuration-reference)
7. [Enabling the 7 Novel Features](#7-enabling-the-7-novel-features)
8. [Running Tests](#8-running-tests)
9. [Output Files](#9-output-files)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| Python | 3.10+ | 3.11 recommended |
| pip | 23+ | `pip install --upgrade pip` |
| Git | any | for cloning |
| RAM | 4 GB | 8 GB+ recommended for large datasets |
| Disk | 2 GB free | for models and outputs |

> **GPU (optional):** Set `device: cuda` in config to use NVIDIA GPU for faster embeddings.

---

## 2. Installation

### Clone the repository

```bash
git clone https://github.com/Z1TH1Z/ZLSDE.git
cd ZLSDE
```

### Create a virtual environment

```bash
# Create venv
python -m venv .venv

# Activate — Linux/macOS
source .venv/bin/activate

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Windows (CMD)
.venv\Scripts\activate.bat
```

### Install the package

```bash
# Core dependencies only
pip install -e .

# Core + development tools (needed for testing)
pip install -e ".[dev]"

# Core + optional extras (FAISS, HuggingFace Datasets, image support)
pip install -e ".[dev,optional]"
```

> First install downloads embedding models (~90 MB) automatically on first run.

---

## 3. API Keys (Optional)

ZLSDE defaults to a **local model** (`google/flan-t5-base`) for label generation — no API keys required.

To use faster cloud LLMs, create a `.env` file in the project root:

```bash
# .env  — copy this template and fill in your keys
GROQ_API_KEY=your_groq_key_here
MISTRAL_API_KEY=your_mistral_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

> Get free keys at: [console.groq.com](https://console.groq.com) · [console.mistral.ai](https://console.mistral.ai) · [openrouter.ai](https://openrouter.ai)

---

## 4. Project Structure

```
ZLSDE/
├── zlsde/                        # Main package
│   ├── layers/                   # Pipeline layer modules
│   │   ├── ingestion.py          # Layer 1 — data loading
│   │   ├── representation.py     # Layer 2 — embeddings
│   │   ├── clustering.py         # Layer 3 — clustering
│   │   ├── label_generation.py   # Layer 4 — LLM labelling
│   │   ├── quality_control.py    # Layer 5 — quality filtering
│   │   ├── self_training.py      # Layer 6 — self-training loop
│   │   ├── exporter.py           # Layer 7 — dataset export
│   │   │
│   │   ├── taxonomy_discovery.py # Feature 1 — ALTD
│   │   ├── provenance.py         # Feature 2 — provenance engine
│   │   ├── semantic_validation.py# Feature 3 — CCSV
│   │   ├── embedding_fusion.py   # Feature 4 — multi-granularity fusion
│   │   ├── adaptive_training.py  # Feature 5 — CWAST
│   │   ├── provider_optimizer.py # Feature 6 — UCB1 optimizer
│   │   └── drift_detection.py    # Feature 7 — drift detection
│   │
│   ├── models/
│   │   └── data_models.py        # All Pydantic data models
│   ├── providers/                # LLM provider implementations
│   ├── config/                   # YAML config loader
│   ├── orchestrator.py           # Main pipeline coordinator
│   └── cli.py                    # Command-line interface
│
├── tests/
│   ├── unit/
│   │   ├── test_providers.py     # Provider tests (18 tests)
│   │   └── test_features.py      # Feature tests (32 tests)
│   └── integration/
│       └── test_api_integration.py
│
├── examples/
│   ├── config.yaml               # Local model config
│   ├── config_api.yaml           # API provider config
│   ├── basic_text_pipeline.py    # Python API example
│   └── api_provider_example.py   # API usage example
│
├── pyproject.toml                # Package config & dependencies
├── pytest.ini                    # Test configuration
└── SETUP.md                      # This file
```

---

## 5. Running the Pipeline

### Option A — Command-Line Interface

#### Step 1: Prepare your data

Create a CSV file with a `content` column:

```csv
content
"The quarterly earnings report showed a 15% increase in revenue."
"Machine learning models require large amounts of training data."
"The recipe calls for two cups of flour and one egg."
"Scientists discovered a new species of deep-sea fish."
```

Save it as `my_data.csv`.

#### Step 2: Create a config file

```yaml
# my_config.yaml

data:
  sources:
    - type: csv
      path: my_data.csv
  modality: text

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32
  use_dimensionality_reduction: false

clustering:
  method: auto
  min_cluster_size: 5

labeling:
  provider_config:
    provider_type: local          # "api" to use cloud LLMs
    local_model: google/flan-t5-base
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
  format: csv                     # csv | json | parquet
  path: ./output/my_dataset

system:
  random_seed: 42
  log_level: INFO
```

#### Step 3: Run

```bash
# Basic run
zlsde --config my_config.yaml

# With verbose logging
zlsde --config my_config.yaml --verbose

# Override output directory
zlsde --config my_config.yaml --output ./results

# Check version
zlsde --version
```

---

### Option B — Python API

```python
from zlsde import PipelineConfig, PipelineOrchestrator
from zlsde.models import DataSource, ProviderConfig

# Build configuration programmatically
config = PipelineConfig(
    data_sources=[
        DataSource(type="csv", path="my_data.csv")
    ],
    modality="text",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    clustering_method="auto",
    min_cluster_size=5,
    provider_config=ProviderConfig(
        provider_type="local",          # or "api"
        local_model="google/flan-t5-base",
    ),
    use_llm=True,
    max_iterations=3,
    output_format="csv",
    output_path="./output",

    # Enable novel features
    enable_taxonomy=True,
    enable_provenance=True,
    enable_semantic_validation=True,
    enable_adaptive_training=True,
    enable_drift_detection=True,
)

# Run
orchestrator = PipelineOrchestrator(config)
result = orchestrator.run()

print(f"Status:   {result.status}")
print(f"Samples:  {result.n_samples}")
print(f"Labeled:  {result.n_labeled}")
print(f"Clusters: {result.final_metrics.n_clusters}")
print(f"Output:   {result.dataset_path}")
```

---

### Option C — Use API Providers

```bash
# 1. Set API keys in .env
echo "GROQ_API_KEY=your_key" >> .env

# 2. Use the API config
zlsde --config examples/config_api.yaml
```

Or in Python:

```python
config = PipelineConfig(
    data_sources=[DataSource(type="csv", path="my_data.csv")],
    provider_config=ProviderConfig(
        provider_type="api",
        api_providers=["groq", "mistral", "openrouter"],
        groq_model="mixtral-8x7b-32768",
    ),
    # Feature 6: Dynamic provider optimizer
    enable_provider_optimization=True,
    provider_quality_weight=0.6,
)
```

---

## 6. Configuration Reference

### Data Sources

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `type` | str | `csv`, `json`, `text`, `folder` | Source format |
| `path` | str | any path | Path to file or directory |
| `metadata` | dict | optional | Extra metadata attached to all items |

**CSV format** — requires a `content` column. Optional: `id`, `modality`.

**JSON format** — array of objects or single object with `content` field.

**Folder format** — recursively loads all `.txt` files.

---

### Embedding

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `device` | `cpu` | `cpu`, `cuda`, or `mps` |
| `batch_size` | `32` | Items per embedding batch |
| `use_dimensionality_reduction` | `false` | Apply UMAP reduction |
| `n_components` | `50` | UMAP target dimensions |

---

### Clustering

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `method` | `auto` | `auto`, `hdbscan`, `kmeans`, `spectral` | `auto` tries all and picks best silhouette |
| `min_cluster_size` | `10` | int ≥ 2 | Minimum samples per cluster |

---

### Labeling (Provider Config)

| Key | Default | Description |
|-----|---------|-------------|
| `provider_type` | `local` | `local` (no API key) or `api` |
| `api_providers` | `[groq, mistral, openrouter]` | Tried in order with auto-fallback |
| `groq_model` | `mixtral-8x7b-32768` | Model for Groq |
| `mistral_model` | `mistral-small-latest` | Model for Mistral |
| `local_model` | `google/flan-t5-base` | HuggingFace local model |
| `timeout` | `30` | API request timeout (seconds) |
| `use_llm` | `true` | Disable for default `cluster_N` labels |
| `n_representatives` | `5` | Samples shown to LLM per cluster |

---

### Quality Control

| Key | Default | Description |
|-----|---------|-------------|
| `threshold` | `0.7` | Minimum quality score [0–1] |
| `anomaly_contamination` | `0.1` | Expected anomaly fraction |
| `duplicate_threshold` | `0.95` | Cosine similarity for duplicate detection |

---

### Training

| Key | Default | Description |
|-----|---------|-------------|
| `max_iterations` | `3` | Max self-training iterations |
| `convergence_threshold` | `0.02` | Label flip rate to stop early |
| `confidence_threshold` | `0.8` | Minimum confidence for pseudo-labels |

---

### Output

| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `format` | `csv` | `csv`, `json`, `parquet` | Export format |
| `path` | `./output` | any path | Output directory |

---

## 7. Enabling the 7 Novel Features

All features are disabled-by-default safe — they activate via config flags.

### Feature 1 — Autonomous Label Taxonomy Discovery (ALTD)

Recursively sub-clusters each top-level cluster to discover a hierarchical label tree.

```python
config = PipelineConfig(
    ...
    enable_taxonomy=True,
    taxonomy_max_depth=3,          # how deep to recurse
    taxonomy_min_samples=5,        # min samples to attempt split
    taxonomy_silhouette_threshold=0.1,  # min improvement to keep split
)
```

Output: `output/taxonomy.json`

---

### Feature 2 — Label Provenance & Explainability

Records the full audit trail for every label: provider, prompt, response, confidence breakdown, and iteration history.

```python
config = PipelineConfig(
    ...
    enable_provenance=True,
    enable_explanations=False,     # set True to generate NL explanations
)
```

Output: `output/provenance_report.json`

---

### Feature 3 — Cross-Cluster Semantic Validation (CCSV)

Detects label collisions, merge/split candidates, and outlier clusters after labelling.

```python
config = PipelineConfig(
    ...
    enable_semantic_validation=True,
    label_similarity_threshold=0.8,     # Jaccard sim for collision detection
    centroid_similarity_threshold=0.85, # cosine sim for merge candidates
)
```

---

### Feature 4 — Multi-Granularity Embedding Fusion

Combines embeddings from multiple models into one richer representation.

```python
config = PipelineConfig(
    ...
    enable_embedding_fusion=True,
    fusion_models=[
        "sentence-transformers/all-mpnet-base-v2",  # second model
    ],
    fusion_weights=[0.6, 0.4],     # empty list = auto-learn weights
)
```

---

### Feature 5 — Confidence-Weighted Adaptive Self-Training (CWAST)

Curriculum learning: starts training only on high-confidence samples, gradually includes harder ones.

```python
config = PipelineConfig(
    ...
    enable_adaptive_training=True,
    curriculum_percentile_decay=10.0,  # threshold drops 10% per iteration
)
```

---

### Feature 6 — Dynamic Provider Cost-Quality Optimizer

UCB1 multi-armed bandit that routes labelling requests to the best cost/quality provider.

```python
config = PipelineConfig(
    ...
    enable_provider_optimization=True,
    provider_quality_weight=0.5,     # 0=speed-only, 1=quality-only
    provider_exploration_rate=0.1,   # exploration bonus weight
)
```

---

### Feature 7 — Embedding Drift Detection & Self-Correction

Monitors embedding space health each iteration. Rolls back to last good state on collapse or divergence.

```python
config = PipelineConfig(
    ...
    enable_drift_detection=True,
    drift_collapse_threshold=0.3,      # inter-cluster drop ratio trigger
    drift_divergence_threshold=2.0,    # centroid drift ratio trigger
)
```

Output: `output/drift_history.json`

---

### All Features On — Full Config Example

```python
config = PipelineConfig(
    data_sources=[DataSource(type="csv", path="my_data.csv")],
    modality="text",
    # Features
    enable_taxonomy=True,
    taxonomy_max_depth=3,
    enable_provenance=True,
    enable_semantic_validation=True,
    enable_embedding_fusion=False,     # needs extra models
    enable_adaptive_training=True,
    enable_provider_optimization=False,# needs API providers
    enable_drift_detection=True,
)
```

---

## 8. Running Tests

### Run all tests

```bash
pytest
```

### Run only feature tests (32 tests)

```bash
pytest tests/unit/test_features.py -v
```

### Run only provider tests (18 tests)

```bash
pytest tests/unit/test_providers.py -v
```

### Run with coverage report

```bash
pytest --cov=zlsde --cov-report=term-missing
```

### Run a specific test class

```bash
pytest tests/unit/test_features.py::TestTaxonomyDiscovery -v
pytest tests/unit/test_features.py::TestDriftDetection -v
pytest tests/unit/test_features.py::TestProvenanceTracker -v
```

### Run a single test

```bash
pytest tests/unit/test_features.py::TestDriftDetection::test_collapse_detection -v
```

### Expected output

```
============================= test session starts =============================
collected 50 items

tests/unit/test_features.py::TestTaxonomyDiscovery::test_discover_builds_tree PASSED
tests/unit/test_features.py::TestTaxonomyDiscovery::test_taxonomy_respects_max_depth PASSED
...
tests/unit/test_providers.py::TestFallbackChainManager::test_generate_label_success_first_provider PASSED
...
============================== 50 passed in ~45s =============================
```

---

## 9. Output Files

After a successful run, the output directory contains:

```
output/
├── dataset.csv            # Labeled dataset (main output)
├── embeddings.npy         # Raw embeddings (n_samples × dim)
├── metadata.json          # Pipeline statistics and config snapshot
├── provenance_report.json # Feature 2: full label audit trail
├── taxonomy.json          # Feature 1: hierarchical label tree (if enabled)
└── drift_history.json     # Feature 7: per-iteration drift metrics (if enabled)
```

### `dataset.csv` columns

| Column | Description |
|--------|-------------|
| `id` | Unique item identifier |
| `content` | Original text content |
| `label` | Generated semantic label |
| `cluster_id` | Cluster assignment (-1 = noise) |
| `confidence` | Label confidence score [0–1] |
| `quality_score` | Quality score [0–1] |
| `modality` | Data modality |
| `iteration` | Iteration when labeled |
| `anomaly_flag` | True if flagged as anomaly |
| `duplicate_flag` | True if near-duplicate |

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'zlsde'`

```bash
# Re-install in editable mode from project root
pip install -e .
```

### `No providers available`

Either set API keys in `.env` or switch to local mode:

```yaml
labeling:
  provider_config:
    provider_type: local
    local_model: google/flan-t5-base
```

### `hdbscan` import error

```bash
pip install hdbscan==0.8.33
```

### `umap` import error (only needed with `use_dimensionality_reduction: true`)

```bash
pip install umap-learn==0.5.5
```

### Out of memory on large datasets

Reduce batch size and disable embedding fusion:

```python
config = PipelineConfig(
    batch_size=8,
    enable_embedding_fusion=False,
    ...
)
```

### Clustering finds only 1 cluster

Lower `min_cluster_size` or increase dataset size:

```yaml
clustering:
  min_cluster_size: 3
```

### Tests fail with `ImportError`

```bash
pip install -e ".[dev]"
```

---

## Quick Reference

```bash
# Install
pip install -e ".[dev]"

# Run pipeline (local model, no API key needed)
zlsde --config examples/config.yaml

# Run pipeline (cloud LLMs, requires .env)
zlsde --config examples/config_api.yaml

# Run all tests
pytest

# Run feature tests only
pytest tests/unit/test_features.py -v

# Run with coverage
pytest --cov=zlsde --cov-report=term-missing
```
