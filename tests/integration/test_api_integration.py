"""Integration tests for API provider integration."""

import pytest
import os
from unittest.mock import Mock, patch
from zlsde.models.data_models import PipelineConfig, DataSource, ProviderConfig
from zlsde.layers.label_generation import create_label_generator, PseudoLabelGenerator
from zlsde.config.config_loader import ConfigLoader


class TestLabelGeneratorIntegration:
    """Test label generator with provider integration."""
    
    def test_create_label_generator_with_local_config(self):
        """Test creating label generator with local provider configuration."""
        config = ProviderConfig(
            provider_type="local",
            local_model="google/flan-t5-base"
        )
        
        with patch('zlsde.providers.factory.LocalProvider') as mock_local:
            mock_instance = Mock()
            mock_instance.is_available.return_value = True
            mock_instance.get_provider_name.return_value = "Local (flan-t5-base)"
            mock_local.return_value = mock_instance
            
            generator = create_label_generator(config, device="cpu")
            
            assert isinstance(generator, PseudoLabelGenerator)
            assert generator.provider_manager is not None
    
    def test_create_label_generator_with_pipeline_config(self):
        """Test creating label generator from full pipeline config."""
        config = PipelineConfig(
            data_sources=[DataSource(type="csv", path="test.csv")],
            modality="text",
            provider_config=ProviderConfig(
                provider_type="local",
                local_model="google/flan-t5-base"
            )
        )
        
        with patch('zlsde.providers.factory.LocalProvider') as mock_local:
            mock_instance = Mock()
            mock_instance.is_available.return_value = True
            mock_instance.get_provider_name.return_value = "Local (flan-t5-base)"
            mock_local.return_value = mock_instance
            
            generator = create_label_generator(config, device="cpu")
            
            assert isinstance(generator, PseudoLabelGenerator)


class TestConfigLoaderIntegration:
    """Test configuration loading with provider config."""
    
    def test_load_config_with_provider_config(self):
        """Test loading configuration with provider_config."""
        config_dict = {
            'data': {
                'sources': [{'type': 'csv', 'path': 'test.csv'}],
                'modality': 'text'
            },
            'labeling': {
                'provider_config': {
                    'provider_type': 'api',
                    'api_providers': ['groq', 'mistral'],
                    'groq_model': 'mixtral-8x7b-32768',
                    'mistral_model': 'mistral-small-latest',
                    'local_model': 'google/flan-t5-base',
                    'timeout': 30
                },
                'use_llm': True,
                'n_representatives': 5
            }
        }
        
        config = ConfigLoader.from_dict(config_dict)
        
        assert config.provider_config.provider_type == 'api'
        assert config.provider_config.api_providers == ['groq', 'mistral']
        assert config.provider_config.groq_model == 'mixtral-8x7b-32768'
        assert config.provider_config.timeout == 30
    
    def test_load_config_backward_compatibility(self):
        """Test backward compatibility with old llm_model configuration."""
        config_dict = {
            'data': {
                'sources': [{'type': 'csv', 'path': 'test.csv'}],
                'modality': 'text'
            },
            'labeling': {
                'llm_model': 'google/flan-t5-large',
                'use_llm': True
            }
        }
        
        config = ConfigLoader.from_dict(config_dict)
        
        # Should create provider_config with local type
        assert config.provider_config.provider_type == 'local'
        assert config.provider_config.local_model == 'google/flan-t5-large'
        # Old field should still be set for backward compatibility
        assert config.llm_model == 'google/flan-t5-large'
    
    def test_load_config_default_provider_config(self):
        """Test loading configuration without provider_config uses defaults."""
        config_dict = {
            'data': {
                'sources': [{'type': 'csv', 'path': 'test.csv'}],
                'modality': 'text'
            },
            'labeling': {
                'use_llm': True
            }
        }
        
        config = ConfigLoader.from_dict(config_dict)
        
        # Should use default provider_config
        assert config.provider_config.provider_type == 'local'
        assert config.provider_config.local_model == 'google/flan-t5-base'


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.skipif(
        not os.getenv('GROQ_API_KEY'),
        reason="GROQ_API_KEY not set - skipping live API test"
    )
    def test_live_api_call_groq(self):
        """Test actual API call to Groq (requires API key)."""
        from zlsde.providers.api_providers import GroqProvider
        
        api_key = os.getenv('GROQ_API_KEY')
        provider = GroqProvider(api_key=api_key, model="mixtral-8x7b-32768", timeout=30)
        
        prompt = "What is the common theme of these items: apple, banana, orange?"
        result = provider.generate_label(prompt, max_tokens=10)
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Generated label: {result}")
    
    @pytest.mark.skipif(
        not os.getenv('MISTRAL_API_KEY'),
        reason="MISTRAL_API_KEY not set - skipping live API test"
    )
    def test_live_api_call_mistral(self):
        """Test actual API call to Mistral (requires API key)."""
        from zlsde.providers.api_providers import MistralProvider
        
        api_key = os.getenv('MISTRAL_API_KEY')
        provider = MistralProvider(api_key=api_key, model="mistral-small-latest", timeout=30)
        
        prompt = "What is the common theme of these items: apple, banana, orange?"
        result = provider.generate_label(prompt, max_tokens=10)
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Generated label: {result}")
    
    @pytest.mark.skipif(
        not os.getenv('OPENROUTER_API_KEY'),
        reason="OPENROUTER_API_KEY not set - skipping live API test"
    )
    def test_live_api_call_openrouter(self):
        """Test actual API call to OpenRouter (requires API key)."""
        from zlsde.providers.api_providers import OpenRouterProvider
        
        api_key = os.getenv('OPENROUTER_API_KEY')
        provider = OpenRouterProvider(
            api_key=api_key, 
            model="mistralai/mixtral-8x7b-instruct", 
            timeout=30
        )
        
        prompt = "What is the common theme of these items: apple, banana, orange?"
        result = provider.generate_label(prompt, max_tokens=10)
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Generated label: {result}")
