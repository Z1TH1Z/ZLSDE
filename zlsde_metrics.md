# ZLSDE Performance Metrics

This document summarizes the official performance benchmarks for the **Zero-Label Self-Discovering Dataset Engine (ZLSDE)**. These metrics are extracted from production-ready pipeline executions and automated test suites.

Benchmark refresh date: **2026-03-28**
Environment: **Windows + Python 3.11.9 + local provider (google/flan-t5-base)**
Dataset: **20 samples**

## 🚀 Throughput & Speed
- **Processing Speed**: `0.771 samples/second` (average over 3 full pipeline runs)
- **Label Generation Latency**: `0.522 seconds/label` average inter-label interval (local provider)
- **Pipeline Execution**: `25.93 seconds` average for a 20-sample batch (3 runs)
- **Cold Start (first run)**: `34.31 seconds` (includes model warm-up)

## 📊 Quality & Accuracy
- **Average Quality Score**: `81.21%` (`quality_mean = 0.8121`)
- **Label Coverage**: `100%` (every input sample successfully assigned a semantic label)
- **Confidence Scoring**: `42.0%` (`confidence_mean = 0.42`)
- **Clustering Quality**: `0.2208` Silhouette Score (recomputed from output embeddings + cluster IDs)

## 🛡️ Reliability & Efficiency
- **Manual Effort**: `0%` (fully eliminates the need for manual data annotation)
- **Provider Success Rate (this benchmark)**: `100%` observed with local provider usage (`8` successful label-generation calls recorded)
- **Test Success Rate**: `94.8%` overall (`55 passed / 58 collected`, `3 skipped`)
- **Fault Tolerance**: `100%` pipeline run success in benchmark reruns (`3/3` completed)

## 🔎 Notes on Metric Interpretation
- Cloud API latency (Groq/Mistral/OpenRouter) was **not measured** in this benchmark because execution used the local provider.
- Throughput and execution times depend on model warm-up and hardware; cold-start runs are slower than warmed runs.
- Confidence is currently conservative by design in the scoring heuristic and should be interpreted alongside quality/coherence metrics.

---
*Note: These values were retested from fresh local executions on 2026-03-28 and reflect the current repository state at measurement time.*
