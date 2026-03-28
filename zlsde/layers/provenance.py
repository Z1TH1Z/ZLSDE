"""Feature 2: Label Provenance & Explainability Engine.

Records the full decision trail for every label: which provider generated it,
the exact prompt/response, representative samples used, confidence breakdown,
and optional natural-language explanations.

Patent-relevant novelty:
- End-to-end audit trail for autonomous labelling decisions
- Per-label explainability with confidence decomposition
- Cross-iteration provenance tracking (label drift history)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from zlsde.models.data_models import (
    LabelProvenance, ProvenanceReport, RawDataItem, Label, PipelineConfig
)

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """Track and store provenance records for every label decision."""

    def __init__(self, config: PipelineConfig):
        self.enabled = config.enable_provenance
        self.explain = config.enable_explanations
        self._records: Dict[int, LabelProvenance] = {}  # cluster_id -> record
        self._provider_usage: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording API (called by label generation layer)
    # ------------------------------------------------------------------

    def record_label(
        self,
        cluster_id: int,
        label_text: str,
        provider_used: str = "unknown",
        prompt_sent: str = "",
        raw_response: str = "",
        representative_samples: Optional[List[str]] = None,
        confidence: float = 0.0,
        confidence_breakdown: Optional[Dict[str, float]] = None,
    ) -> LabelProvenance:
        """Create a provenance record for a single label decision."""
        if not self.enabled:
            return LabelProvenance(cluster_id=cluster_id, label_text=label_text)

        record = LabelProvenance(
            cluster_id=cluster_id,
            label_text=label_text,
            provider_used=provider_used,
            prompt_sent=prompt_sent,
            raw_response=raw_response,
            representative_samples=representative_samples or [],
            confidence=confidence,
            confidence_breakdown=confidence_breakdown or {},
            timestamp=datetime.now().isoformat(),
        )

        # Track iteration history if label changed
        existing = self._records.get(cluster_id)
        if existing and existing.label_text != label_text:
            record.iteration_history = existing.iteration_history + [
                {
                    "prev_label": existing.label_text,
                    "new_label": label_text,
                    "timestamp": record.timestamp,
                    "provider": provider_used,
                }
            ]

        self._records[cluster_id] = record

        # Provider usage counting
        self._provider_usage[provider_used] = (
            self._provider_usage.get(provider_used, 0) + 1
        )

        return record

    def add_explanation(self, cluster_id: int, explanation: str) -> None:
        """Attach a natural-language explanation to an existing record."""
        if cluster_id in self._records:
            self._records[cluster_id].explanation = explanation

    # ------------------------------------------------------------------
    # Batch helpers (wrap the label generator transparently)
    # ------------------------------------------------------------------

    def wrap_label_generation(
        self,
        clusters: Dict[int, List[RawDataItem]],
        label_generator,
        n_representatives: int = 5,
    ) -> Dict[int, Label]:
        """Generate labels while automatically recording provenance.

        Delegates actual generation to *label_generator* but intercepts
        the prompt, response, and representative samples for auditing.

        Returns:
            Same Dict[int, Label] as PseudoLabelGenerator.generate_labels.
        """
        labels: Dict[int, Label] = {}

        for cluster_id, items in clusters.items():
            if cluster_id == -1:
                labels[cluster_id] = Label(text="noise", confidence=0.0)
                self.record_label(cluster_id, "noise", provider_used="rule")
                continue

            # Select representatives (reuse generator logic)
            representatives = label_generator._select_representatives(items, k=n_representatives)
            rep_texts = [str(r.content)[:200] for r in representatives]

            # Build prompt
            prompt = label_generator._create_prompt(representatives)

            # Generate via provider chain
            try:
                raw_response = label_generator.provider_manager.generate_label(prompt, max_tokens=20)
                label_text = label_generator._validate_label(raw_response.strip())
                if label_text in ["unlabeled", "unknown"]:
                    label_text = label_generator._infer_rule_based_label(items)
                provider_name = self._last_successful_provider(label_generator)
            except Exception as e:
                logger.warning(f"Provenance-wrapped generation failed for cluster {cluster_id}: {e}")
                label_text = f"cluster_{cluster_id}"
                raw_response = ""
                provider_name = "fallback"

            confidence = label_generator._compute_confidence(label_text, items)

            # Confidence breakdown
            cluster_size = len(items)
            size_conf = min(1.0, cluster_size / 50.0)
            label_quality = 0.9 if len(label_text.split()) <= 3 else 0.6
            breakdown = {"size_confidence": size_conf, "label_quality": label_quality}

            labels[cluster_id] = Label(text=label_text, confidence=confidence)

            self.record_label(
                cluster_id=cluster_id,
                label_text=label_text,
                provider_used=provider_name,
                prompt_sent=prompt,
                raw_response=raw_response,
                representative_samples=rep_texts,
                confidence=confidence,
                confidence_breakdown=breakdown,
            )

        return labels

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> ProvenanceReport:
        """Compile all records into a single provenance report."""
        records = list(self._records.values())
        confidences = [r.confidence for r in records] if records else [0.0]
        explained = sum(1 for r in records if r.explanation)

        return ProvenanceReport(
            provenance_records=records,
            provider_usage=dict(self._provider_usage),
            total_labels_generated=len(records),
            total_labels_explained=explained,
            avg_confidence=float(np.mean(confidences)),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_successful_provider(label_generator) -> str:
        """Best-effort extraction of which provider last succeeded."""
        try:
            stats = label_generator.provider_manager.get_statistics()
            for name, s in stats.items():
                if s.successful_calls > 0:
                    return name
        except Exception:
            pass
        return "unknown"
