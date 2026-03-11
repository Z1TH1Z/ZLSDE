"""Layer 4: Pseudo-Label Generator - Generate semantic labels using zero-shot LLMs."""

import logging
import numpy as np
from typing import Dict, List, Optional
from zlsde.models.data_models import RawDataItem, Label
from zlsde.providers.fallback_chain import FallbackChainManager
from zlsde.providers.exceptions import AllProvidersFailedError

logger = logging.getLogger(__name__)


class PseudoLabelGenerator:
    """Generate semantic labels for clusters using zero-shot language models."""
    
    def __init__(self, provider_manager: FallbackChainManager):
        """Initialize with provider manager.
        
        Args:
            provider_manager: FallbackChainManager instance for label generation
        """
        self.provider_manager = provider_manager
    
    def generate_labels(self, clusters: Dict[int, List[RawDataItem]]) -> Dict[int, Label]:
        """
        Generate label for each cluster.
        
        Args:
            clusters: Dictionary mapping cluster_id to list of items
            
        Returns:
            Dictionary mapping cluster_id to Label object
        """
        labels = {}
        
        for cluster_id, items in clusters.items():
            # Handle noise cluster
            if cluster_id == -1:
                labels[cluster_id] = Label(text="noise", confidence=0.0)
                continue
            
            try:
                # Select representative samples
                representatives = self._select_representatives(items, k=5)
                
                # Create prompt
                prompt = self._create_prompt(representatives)
                
                # Generate label using provider manager
                label_text = self.provider_manager.generate_label(prompt, max_tokens=20)
                
                # Compute confidence score
                confidence = self._compute_confidence(label_text, items)
                
                labels[cluster_id] = Label(text=label_text, confidence=confidence)
                logger.debug(f"Cluster {cluster_id}: '{label_text}' (confidence: {confidence:.2f})")
                
            except AllProvidersFailedError as e:
                logger.error(f"Failed to generate label for cluster {cluster_id}: {e}")
                # Fallback to default label
                labels[cluster_id] = Label(text=f"cluster_{cluster_id}", confidence=0.5)
            except Exception as e:
                logger.warning(f"Unexpected error generating label for cluster {cluster_id}: {e}")
                # Fallback to default label
                labels[cluster_id] = Label(text=f"cluster_{cluster_id}", confidence=0.5)
        
        return labels
    
    def _select_representatives(self, cluster_items: List[RawDataItem], k: int = 5) -> List[RawDataItem]:
        """
        Select centroid-nearest samples as representatives.
        
        Args:
            cluster_items: List of items in the cluster
            k: Number of representatives to select
            
        Returns:
            List of representative items (closest to centroid)
        """
        if not cluster_items:
            return []
        
        # Extract embeddings
        embeddings = np.array([item.embedding for item in cluster_items])
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Compute distances to centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        
        # Select k closest items
        k_actual = min(k, len(cluster_items))
        closest_indices = np.argsort(distances)[:k_actual]
        
        representatives = [cluster_items[i] for i in closest_indices]
        return representatives
    
    def _create_prompt(self, representatives: List[RawDataItem]) -> str:
        """
        Create prompt for LLM to identify common theme.
        
        Args:
            representatives: List of representative samples
            
        Returns:
            Formatted prompt string
        """
        if not representatives:
            return "Analyze these samples and identify their common theme or category. Provide a concise label (1-3 words)."
        
        # Create prompt with representative samples
        samples_text = "\n".join([
            f"{i+1}. {str(item.content)[:200]}"  # Truncate long content
            for i, item in enumerate(representatives)
        ])
        
        prompt = f"""Analyze these samples and identify their common theme or category.
Provide a concise label (1-3 words) that best describes what these samples have in common.

Samples:
{samples_text}

Common category:"""
        
        return prompt
    
    def _compute_confidence(self, label: str, cluster_items: List[RawDataItem]) -> float:
        """
        Compute label confidence score.
        
        Args:
            label: Generated label text
            cluster_items: All items in the cluster
            
        Returns:
            Confidence score in [0, 1]
        """
        # Simple heuristic: confidence based on cluster size and label quality
        cluster_size = len(cluster_items)
        
        # Base confidence from cluster size (larger clusters = higher confidence)
        size_confidence = min(1.0, cluster_size / 50.0)
        
        # Label quality score
        if label in ["unlabeled", "unknown", "noise"] or label.startswith("cluster_"):
            label_quality = 0.3
        elif len(label.split()) <= 3:  # Good concise label
            label_quality = 0.9
        else:
            label_quality = 0.6
        
        # Combine scores
        confidence = 0.6 * size_confidence + 0.4 * label_quality
        
        return float(np.clip(confidence, 0.0, 1.0))



def create_label_generator(config, device: str = "cpu") -> PseudoLabelGenerator:
    """Create label generator with configured providers.
    
    This is a helper function that creates the provider chain and initializes
    the label generator with the appropriate configuration.
    
    Args:
        config: PipelineConfig or ProviderConfig object with provider settings
        device: Device for local model ("cpu", "cuda", "mps")
        
    Returns:
        Initialized PseudoLabelGenerator instance
        
    Raises:
        ValueError: If no providers are available
    """
    from zlsde.providers.factory import ProviderFactory
    from zlsde.providers.fallback_chain import FallbackChainManager
    from zlsde.models.data_models import PipelineConfig
    
    # Extract provider config from pipeline config if needed
    if isinstance(config, PipelineConfig):
        provider_config = config.provider_config
    else:
        provider_config = config
    
    # Create providers using factory
    providers = ProviderFactory.create_providers(provider_config, device)
    
    # Create fallback chain manager
    provider_manager = FallbackChainManager(providers)
    
    # Create label generator
    label_generator = PseudoLabelGenerator(provider_manager)
    
    logger.info(f"Label generator initialized with {len(providers)} providers")
    
    return label_generator
