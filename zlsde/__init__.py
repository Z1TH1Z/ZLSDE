"""
ZLSDE (Zero-Label Self-Discovering Dataset Engine)

An autonomous end-to-end pipeline that transforms raw unlabeled multimodal data
into structured, high-quality labeled datasets without human annotation.
"""

__version__ = "0.1.0"

from .models.data_models import DataSource, PipelineConfig
from .orchestrator import PipelineOrchestrator

__all__ = ["PipelineConfig", "DataSource", "PipelineOrchestrator", "__version__"]
