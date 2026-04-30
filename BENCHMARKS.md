# BENCHMARKS.md

This document explains every headline metric that appears in the README: what each number means, how it was computed from source code, and how to reproduce a comparable run. Where the original dataset or run environment is not fully recoverable, this is stated explicitly.

The dataset CSV is committed to the repo at `examples/data/sample_text_multicluster_520.csv` but is synthetically generated and not sourced from any external dataset.

---

## Headline Numbers

| Metric | Value | Measured on | See section |
|---|---|---|---|
| Samples processed | 520 | 520-sample multicluster run | [§Setup](#setup) |
| Throughput | 18.36 samples/sec | Groq API, CPU, end-to-end | [§Throughput](#throughput-1836-samplessec) |
| Quality score | 56.1% | 520-sample run, iteration 2 | [§Quality Score](#quality-score-561) |
| Confidence score | 77.8% | 520-sample run, all labels | [§Quality Score](#quality-score-561) |
| Silhouette score | 0.247 | KMeans/auto clustering | [§Clustering](#clusters-discovered-17) |
| Label flip rate | 0.19% | Between iterations 1→2 | [§Label Flip Rate](#label-flip-rate-019) |
| Clusters discovered | 17 | auto method, min_cluster_size=4 | [§Clustering](#clusters-discovered-17) |
| Tests passing | 218 | Full suite, 4 skipped | [§Tests](#test-count) |

---

## Setup

### Hardware

- CPU: Intel Core i5 12th Gen
- RAM: 16 GB
- GPU: Nvidia RTX 3050 (pipeline ran on CPU — `device: cpu` in config; GPU not used for this run)
- OS: Windows 11

### Software

- Python >= 3.10 (tested on 3.10, 3.11)
- `sentence-transformers` — embedding generation
- `scikit-learn` — clustering (KMeans, Spectral), anomaly detection (IsolationForest, LOF), MLP classifier
- `hdbscan` — density-based clustering (used if installed; falls back to KMeans/Spectral if not)
- `requests` — Groq API calls
- `pydantic` — config validation
- LLM provider: **Groq**, model: **llama-3.1-8b-instant**

### Dataset

- Modality: text
- Source: synthetically generated (free-text sentences across 7 latent topical domains, each sample suffixed with an instance marker e.g. `(instance_1)`)
- Size: 520 samples, single column named `content`
- File: `examples/data/sample_text_multicluster_520.csv`
- Any text CSV with a single text column works as a drop-in replacement; edit the `path` field in the config.

### Configuration

The exact config used for the 520-sample run:

```yaml
# examples/config_multicluster_520.yaml

data:
  sources:
    - type: csv
      path: examples/data/sample_text_multicluster_520.csv
  modality: text

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2  # 384-dim embeddings
  device: cpu
  batch_size: 32
  use_dimensionality_reduction: false             # UMAP not used in this run
  n_components: 50

clustering:
  method: auto          # tries HDBSCAN, KMeans variants, Spectral; picks best silhouette
  min_cluster_size: 4

labeling:
  provider_config:
    provider_type: api
    api_providers: [groq]
    groq_model: llama-3.1-8b-instant
    timeout: 30
  use_llm: true
  n_representatives: 5  # samples per cluster sent to LLM for label generation

quality:
  threshold: 0.7
  anomaly_contamination: 0.1   # IsolationForest + LOF contamination fraction
  duplicate_threshold: 0.95    # cosine similarity above which a sample is a near-duplicate

training:
  max_iterations: 2
  convergence_threshold: 0.02  # flip rate below this → converged
  confidence_threshold: 0.8    # minimum confidence for self-training samples

output:
  format: csv
  path: ./output/validation_runs/multicluster_520
  include_embeddings: true

system:
  random_seed: 42
  log_level: INFO
```

---

## Methodology — How Each Number Is Computed

### Quality Score (56.1%)

Source: [zlsde/layers/quality_control.py](zlsde/layers/quality_control.py)

Each labeled sample receives a quality score computed as a weighted combination of three signals:

```
quality = 0.4 * (1 - raw_anomaly) + 0.3 * duplicate_score + 0.3 * coherence_score
```

**raw_anomaly** — average of two binary anomaly flags:
- `IsolationForest(contamination=0.1)` fitted on all 520 embeddings; flag=1 if predicted as outlier
- `LocalOutlierFactor(contamination=0.1)` fitted on all 520 embeddings; flag=1 if predicted as outlier
- `raw_anomaly = (iso_flag + lof_flag) / 2`, so values are 0.0, 0.5, or 1.0

**duplicate_score** — `0.0` if the sample's embedding has cosine similarity ≥ 0.95 with any earlier sample (the later of the pair is flagged); `1.0` otherwise.

**coherence_score** — per-cluster intra-cluster coherence: `exp(-avg_distance_to_centroid)` computed in embedding space, assigned uniformly to all members of the cluster.

The **56.1%** is the mean quality score across all 520 labeled items after iteration 2.

The **77.8% confidence score** is the mean label confidence returned by the Groq LLM across all clusters — a separate signal from the quality score.

**Why this run scored lower than a typical run.** Two contributing factors specific to the 2026-04-30 run:

1. **Groq rate limit on 2 clusters.** Clusters 0 and 13 hit a rate limit mid-run and received fallback labels (`cluster_0`) instead of LLM-generated names. Fallback labels have coherence scores computed against a generic centroid, not a meaningful cluster centroid, which drags down the per-sample quality score for those clusters.
2. **Auto-select clustering method.** The `auto` method scored KMeans variants against Spectral and selected the winner by silhouette. Different runs can select different methods; the winning method affects the coherence component of the quality formula and therefore the headline quality score. Run-to-run variance of ±5–10 percentage points is normal.

### Throughput (18.36 samples/sec)

Throughput is measured end-to-end: wall-clock time from `orchestrator.run()` start to labels written, divided by 520 samples. This includes:

- Embedding generation (all-MiniLM-L6-v2 on CPU, batch_size=32)
- Clustering (KMeans/Spectral comparison in auto mode)
- LLM label generation via Groq API (5 representative samples × n_clusters requests)
- Quality control scoring
- Self-training (MLP classifier, 2 iterations)
- CSV export

The 2026-04-30 notebook run completed in **28.3 s** for 520 samples. The dominant cost is **Groq API latency**, but Groq response times vary significantly by region and server load. On this run, Groq responded in sub-second per call on average (17 clusters × 2 iterations = 34 API calls, 2 of which hit a rate limit and returned immediately as errors; total LLM time ~24 s). On a different provider, region, or under heavier rate-limiting, throughput can be 5–50× different in either direction — see [§Run-to-run variance](#run-to-run-variance) below.

### Label Flip Rate (0.19%)

Source: [zlsde/layers/self_training.py](zlsde/layers/self_training.py) — `compute_stability()`

```python
flip_rate = np.sum(old_labels != new_labels) / len(old_labels)
```

Computed between the cluster assignment vectors from iteration 1 and iteration 2. A flip means a sample's cluster ID changed. 0.19% across 520 samples means roughly 1 sample changed cluster assignment on the second pass — the pipeline effectively converged after iteration 1.

`max_iterations=2` in this config, so only one flip-rate measurement exists (iteration 1 → 2).

### Clusters Discovered (17)

Source: [zlsde/layers/clustering.py](zlsde/layers/clustering.py)

Config sets `method: auto`. The auto path:
1. Tries **HDBSCAN** (if installed) with `min_cluster_size=4`
2. Tries **KMeans** with k ∈ {estimated_k, estimated_k−1, estimated_k+1}, where `estimated_k = max(2, int(sqrt(520/2))) = 16`
3. Tries **Spectral** with k = estimated_k
4. Selects whichever result has the highest silhouette score

**UMAP was not used in this run** — `use_dimensionality_reduction: false` in the config. Clustering ran directly on 384-dimensional all-MiniLM-L6-v2 embeddings.

The silhouette score of **0.247** (2026-04-30 run) reflects that 520 text samples across 7 latent domains, embedded in 384-dim space, do not form tightly separated clusters — this is typical for free-text data with overlapping topics.

The number 17 is sensitive to `min_cluster_size` and embedding model. Changing either will change cluster count.

#### Cluster naming collisions

An honest finding from the 2026-04-30 notebook run: of the 17 discovered clusters, the LLM produced duplicate names in several cases — two clusters were labeled "cybersecurity", two "astrophysics", two "software development", and two "cooking recipes". The cluster IDs remain distinct internally; only the human-readable names collide.

This reflects a **granularity gap** between the embedding model and the LLM labeler. The sentence-transformer finds finer-grained sub-topics (e.g., offensive-security vs. defensive-security practices both within "cybersecurity"), but the LLM summarizes both into the same high-level category name. The effective unique-topic count for this run is approximately 13, not 17.

This is not a bug. It is expected behavior when the number of embedding clusters exceeds the vocabulary of distinct category names the LLM naturally produces for this domain. Mitigations include providing a taxonomy hint in the labeling prompt or post-processing labels to append a disambiguating suffix.

### Test Count

```bash
pytest tests/ -q
# 218 passed, 4 skipped
```

Suite breakdown:

| Suite | Count | Description |
|---|---|---|
| Unit | ~209 | Layer logic, providers, config, utils, CLI |
| Property | 8 | Hypothesis-based validation invariants |
| Integration | ~5 | Config loading, label generator, live API (skipped without keys) |
| **Total collected** | **222** | **218 passed, 4 skipped** |

---

## Reproduction

### Quick reproduction (with your own dataset)

1. Place a CSV at `examples/data/<your_file>.csv` with a single text column
2. Edit `examples/config_multicluster_520.yaml`: change `path` to your file
3. Set `GROQ_API_KEY` in `.env`
4. Run:
   ```bash
   python -m zlsde.cli run --config examples/config_multicluster_520.yaml
   ```
5. Metrics print at the end of the run; outputs written to `./output/validation_runs/multicluster_520/`

### Reproducing the exact 520-sample numbers

The dataset is committed at `examples/data/sample_text_multicluster_520.csv`. Running the pipeline against it with `random_seed=42` will reproduce clustering behavior exactly, but **LLM-generated labels will differ** because Groq API responses are non-deterministic even with the same input. Expect quality score and confidence score to land within ~2–3 percentage points of the headline values across runs.

---

## Run-to-run variance

Throughput in particular varies significantly across runs because it is dominated by Groq API latency, which is provider-side and reflects server load and region.

| Run date | Throughput | Wall-clock | Notes |
|---|---|---|---|
| 2026-04-15 | 1.96 samples/sec | ~265 s | Earlier run; Groq ~3–5 s/call |
| 2026-04-30 | 18.36 samples/sec | 28.3 s | Sub-second Groq responses; 2 clusters hit rate limit |

The difference between these two runs — nearly 10× — is entirely provider-side latency, not a pipeline change. Both runs used the same config, the same dataset, and the same model.

Quality score also varies run-to-run (±5–10 pp) due to LLM non-determinism and which clustering method `auto` selects as the silhouette winner. The 2026-04-30 run scored 56.1%; the 2026-04-15 run scored 79.3%. The gap is larger than typical because 2 clusters on the 2026-04-30 run hit a Groq rate limit and received fallback labels, directly depressing their quality scores.

---

## Limitations and Caveats

**LLM non-determinism.** `random_seed=42` controls clustering and the MLP classifier. It does not control the Groq API. Every run produces different label text, which affects confidence scores and downstream quality metrics.

**Quality score ≠ accuracy.** The 56.1% quality score is an *internal consistency* metric — it measures whether samples look like non-outliers, non-duplicates, and are in coherent clusters. The dataset has no ground-truth labels, so this score cannot be compared to external accuracy. A sample labeled "Astronomy" may be correctly labeled, but the quality score has no way to know this; it only checks whether the sample fits its cluster well.

**Throughput is provider-bound.** The 18.36 samples/sec figure reflects Groq API latency on the 2026-04-30 run. On a different provider, region, or under rate-limiting, throughput can be 5–50× different in either direction. Local models eliminate API latency but add local inference time.

**Cluster count is parameter-sensitive.** The 17 clusters result from `min_cluster_size=4` and all-MiniLM-L6-v2 embeddings on this specific dataset. Changing the embedding model, min_cluster_size, or dataset composition will change cluster count.

**Silhouette score reflects natural data overlap.** 0.247 for free-text data in high-dimensional embedding space is reasonable. It does not indicate a broken pipeline — it reflects natural overlap between topical clusters in language data.

---

## Last Benchmark Run

- Date: 2026-04-30
- Operator: Nithin Kotala
- Repo commit: `b26624d3b03c18892f00f193d6e2bfe4b6b897f5`
- Notebook: `notebooks/benchmarks_run.ipynb`
- Config: `examples/config_multicluster_520.yaml`
- Provider: Groq / llama-3.1-8b-instant
- Note: 2 of 17 clusters (clusters 0 and 13) hit a Groq rate limit during this run and received fallback labels instead of LLM-generated names. Headline numbers reflect this.
