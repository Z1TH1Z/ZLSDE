"""Configuration loading and validation."""

import yaml
import json
import platform
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from zlsde.models.data_models import PipelineConfig, DataSource, ProviderConfig


# Configuration version for tracking compatibility
CONFIG_VERSION = "1.0"


class ConfigLoader:
    """Handles loading and validation of pipeline configurations."""
    
    @staticmethod
    def from_yaml(path: str) -> PipelineConfig:
        """
        Load pipeline configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
        
        Returns:
            Validated PipelineConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If configuration is invalid
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise ValueError(f"Empty configuration file: {path}")
        
        return ConfigLoader.from_dict(config_dict)
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> PipelineConfig:
        """
        Load pipeline configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            Validated PipelineConfig instance
            
        Raises:
            ValueError: If configuration is invalid
            KeyError: If required fields are missing
        """
        # Parse data sources
        data_sources = []
        data_config = config_dict.get('data', {})
        
        if not data_config.get('sources'):
            raise ValueError("Configuration must include 'data.sources'")
        
        for source in data_config.get('sources', []):
            if 'type' not in source or 'path' not in source:
                raise ValueError("Each data source must have 'type' and 'path' fields")
            
            data_sources.append(DataSource(
                type=source['type'],
                path=source['path'],
                metadata=source.get('metadata')
            ))
        
        # Parse provider configuration
        labeling_config = config_dict.get('labeling', {})
        provider_config_dict = labeling_config.get('provider_config', {})
        
        # Handle backward compatibility: if llm_model exists but provider_config doesn't
        if 'llm_model' in labeling_config and not provider_config_dict:
            provider_config = ProviderConfig(
                provider_type='local',
                local_model=labeling_config['llm_model']
            )
        else:
            provider_config = ProviderConfig(
                provider_type=provider_config_dict.get('provider_type', 'local'),
                api_providers=provider_config_dict.get('api_providers', ['groq', 'mistral', 'openrouter']),
                groq_model=provider_config_dict.get('groq_model', 'mixtral-8x7b-32768'),
                mistral_model=provider_config_dict.get('mistral_model', 'mistral-small-latest'),
                openrouter_model=provider_config_dict.get('openrouter_model', 'mistralai/mixtral-8x7b-instruct'),
                local_model=provider_config_dict.get('local_model', 'google/flan-t5-base'),
                timeout=provider_config_dict.get('timeout', 30)
            )
        
        # Validate provider configuration
        provider_config.validate()
        
        # Build config
        config = PipelineConfig(
            data_sources=data_sources,
            modality=data_config.get('modality', 'text'),
            
            # Embedding
            embedding_model=config_dict.get('embedding', {}).get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
            use_dimensionality_reduction=config_dict.get('embedding', {}).get('use_dimensionality_reduction', False),
            n_components=config_dict.get('embedding', {}).get('n_components', 50),
            
            # Clustering
            clustering_method=config_dict.get('clustering', {}).get('method', 'auto'),
            min_cluster_size=config_dict.get('clustering', {}).get('min_cluster_size', 10),
            
            # Labeling
            provider_config=provider_config,
            llm_model=labeling_config.get('llm_model'),  # Keep for backward compatibility
            use_llm=labeling_config.get('use_llm', True),
            n_representatives=labeling_config.get('n_representatives', 5),
            
            # Quality
            quality_threshold=config_dict.get('quality', {}).get('threshold', 0.7),
            anomaly_contamination=config_dict.get('quality', {}).get('anomaly_contamination', 0.1),
            duplicate_threshold=config_dict.get('quality', {}).get('duplicate_threshold', 0.95),
            
            # Training
            max_iterations=config_dict.get('training', {}).get('max_iterations', 3),
            convergence_threshold=config_dict.get('training', {}).get('convergence_threshold', 0.02),
            confidence_threshold=config_dict.get('training', {}).get('confidence_threshold', 0.8),
            
            # Output
            output_format=config_dict.get('output', {}).get('format', 'csv'),
            output_path=config_dict.get('output', {}).get('path', './output'),
            
            # System
            device=config_dict.get('system', {}).get('device', 'cpu'),
            batch_size=config_dict.get('embedding', {}).get('batch_size', 32),
            random_seed=config_dict.get('system', {}).get('random_seed', 42),
            log_level=config_dict.get('system', {}).get('log_level', 'INFO'),
        )
        
        # Validate
        ConfigLoader.validate(config)
        
        return config
    
    @staticmethod
    def validate(config: PipelineConfig) -> None:
        """
        Perform comprehensive validation of configuration.
        
        Args:
            config: PipelineConfig instance to validate
            
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Use the built-in validate method
        config.validate()
        
        # Additional cross-field validations
        if config.use_dimensionality_reduction and config.n_components >= 512:
            raise ValueError(
                f"n_components ({config.n_components}) should be less than typical embedding "
                "dimensions (512-768) when using dimensionality reduction"
            )
        
        if config.min_cluster_size > 100:
            import warnings
            warnings.warn(
                f"min_cluster_size ({config.min_cluster_size}) is quite large. "
                "This may result in many noise points for small datasets."
            )
    
    @staticmethod
    def create_version_snapshot(
        config: PipelineConfig,
        zlsde_version: str = "0.1.0",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a versioned snapshot of configuration for reproducibility.
        
        Args:
            config: PipelineConfig instance
            zlsde_version: Version of ZLSDE package
            additional_metadata: Optional additional metadata to include
            
        Returns:
            Dictionary containing versioned configuration snapshot
        """
        snapshot = {
            "config_version": CONFIG_VERSION,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "zlsde_version": zlsde_version,
            "config": {
                "data_sources": [
                    {
                        "type": ds.type,
                        "path": ds.path,
                        "metadata": ds.metadata
                    }
                    for ds in config.data_sources
                ],
                "modality": config.modality,
                "embedding_model": config.embedding_model,
                "use_dimensionality_reduction": config.use_dimensionality_reduction,
                "n_components": config.n_components,
                "clustering_method": config.clustering_method,
                "min_cluster_size": config.min_cluster_size,
                "llm_model": config.llm_model,
                "use_llm": config.use_llm,
                "n_representatives": config.n_representatives,
                "quality_threshold": config.quality_threshold,
                "anomaly_contamination": config.anomaly_contamination,
                "duplicate_threshold": config.duplicate_threshold,
                "max_iterations": config.max_iterations,
                "convergence_threshold": config.convergence_threshold,
                "confidence_threshold": config.confidence_threshold,
                "output_format": config.output_format,
                "output_path": config.output_path,
                "device": config.device,
                "batch_size": config.batch_size,
                "random_seed": config.random_seed,
                "log_level": config.log_level,
            },
            "model_versions": {
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model if config.use_llm else None,
            },
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.platform(),
                "device": config.device,
            }
        }
        
        if additional_metadata:
            snapshot["additional_metadata"] = additional_metadata
        
        return snapshot
    
    @staticmethod
    def save_version_snapshot(
        snapshot: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Save configuration snapshot to JSON file.
        
        Args:
            snapshot: Configuration snapshot dictionary
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2)
    
    @staticmethod
    def load_version_snapshot(snapshot_path: str) -> Dict[str, Any]:
        """
        Load configuration snapshot from JSON file.
        
        Args:
            snapshot_path: Path to snapshot JSON file
            
        Returns:
            Configuration snapshot dictionary
            
        Raises:
            FileNotFoundError: If snapshot file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        snapshot_file = Path(snapshot_path)
        if not snapshot_file.exists():
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
        
        with open(snapshot_file, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
        
        # Validate snapshot has required fields
        required_fields = ["config_version", "timestamp", "config"]
        for field in required_fields:
            if field not in snapshot:
                raise ValueError(f"Invalid snapshot: missing required field '{field}'")
        
        return snapshot


# Backward compatibility: keep standalone functions
def load_config_from_yaml(path: str) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.
    
    Deprecated: Use ConfigLoader.from_yaml() instead.
    
    Args:
        path: Path to YAML configuration file
    
    Returns:
        Validated PipelineConfig instance
    """
    return ConfigLoader.from_yaml(path)


def load_config_from_dict(config_dict: Dict[str, Any]) -> PipelineConfig:
    """
    Load pipeline configuration from dictionary.
    
    Deprecated: Use ConfigLoader.from_dict() instead.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        Validated PipelineConfig instance
    """
    return ConfigLoader.from_dict(config_dict)
