"""Unit tests for LLM providers."""

import pytest
from unittest.mock import Mock, patch
from zlsde.providers.base import LLMProvider
from zlsde.providers.api_providers import GroqProvider, MistralProvider, OpenRouterProvider
from zlsde.providers.local_provider import LocalProvider
from zlsde.providers.fallback_chain import FallbackChainManager, ProviderStats
from zlsde.providers.factory import ProviderFactory
from zlsde.providers.exceptions import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    InvalidResponseError,
    AllProvidersFailedError,
)
from zlsde.models.data_models import ProviderConfig


class TestProviderBase:
    """Test base provider interface."""
    
    def test_provider_is_abstract(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()


class TestAPIProviders:
    """Test API provider implementations."""
    
    def test_groq_provider_initialization(self):
        """Test GroqProvider initialization."""
        provider = GroqProvider(api_key="test_key", model="mixtral-8x7b-32768")
        assert provider.get_provider_name() == "Groq"
        assert provider.is_available() == True
        assert provider.endpoint == "https://api.groq.com/openai/v1/chat/completions"
    
    def test_mistral_provider_initialization(self):
        """Test MistralProvider initialization."""
        provider = MistralProvider(api_key="test_key", model="mistral-small-latest")
        assert provider.get_provider_name() == "Mistral"
        assert provider.is_available() == True
        assert provider.endpoint == "https://api.mistral.ai/v1/chat/completions"
    
    def test_openrouter_provider_initialization(self):
        """Test OpenRouterProvider initialization."""
        provider = OpenRouterProvider(api_key="test_key", model="mistralai/mixtral-8x7b-instruct")
        assert provider.get_provider_name() == "OpenRouter"
        assert provider.is_available() == True
        assert provider.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    
    def test_provider_unavailable_without_key(self):
        """Test that providers are unavailable without API key."""
        provider = GroqProvider(api_key="", model="mixtral-8x7b-32768")
        assert provider.is_available() == False


class TestLocalProvider:
    """Test local provider implementation."""
    
    def test_local_provider_initialization(self):
        """Test LocalProvider initialization."""
        # Mock the model loading to avoid downloading models in tests
        with patch('zlsde.providers.local_provider.AutoTokenizer'), \
             patch('zlsde.providers.local_provider.AutoModelForSeq2SeqLM'):
            provider = LocalProvider(model_name="google/flan-t5-base", device="cpu")
            assert "Local" in provider.get_provider_name()
            assert "flan-t5-base" in provider.get_provider_name()


class TestFallbackChainManager:
    """Test fallback chain manager."""
    
    def test_initialization_with_available_providers(self):
        """Test FallbackChainManager initialization with available providers."""
        mock_provider1 = Mock(spec=LLMProvider)
        mock_provider1.is_available.return_value = True
        mock_provider1.get_provider_name.return_value = "Provider1"
        
        mock_provider2 = Mock(spec=LLMProvider)
        mock_provider2.is_available.return_value = True
        mock_provider2.get_provider_name.return_value = "Provider2"
        
        manager = FallbackChainManager([mock_provider1, mock_provider2])
        assert len(manager.providers) == 2
        assert "Provider1" in manager.stats
        assert "Provider2" in manager.stats
    
    def test_initialization_filters_unavailable_providers(self):
        """Test that unavailable providers are filtered out."""
        mock_provider1 = Mock(spec=LLMProvider)
        mock_provider1.is_available.return_value = True
        mock_provider1.get_provider_name.return_value = "Provider1"
        
        mock_provider2 = Mock(spec=LLMProvider)
        mock_provider2.is_available.return_value = False
        mock_provider2.get_provider_name.return_value = "Provider2"
        
        manager = FallbackChainManager([mock_provider1, mock_provider2])
        assert len(manager.providers) == 1
        assert manager.providers[0].get_provider_name() == "Provider1"
    
    def test_initialization_fails_with_no_providers(self):
        """Test that initialization fails when no providers are available."""
        with pytest.raises(ValueError, match="No providers available"):
            FallbackChainManager([])
    
    def test_generate_label_success_first_provider(self):
        """Test successful label generation with first provider."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_provider_name.return_value = "TestProvider"
        mock_provider.generate_label.return_value = "test label"
        
        manager = FallbackChainManager([mock_provider])
        result = manager.generate_label("test prompt")
        
        assert result == "test label"
        assert manager.stats["TestProvider"].successful_calls == 1
        assert manager.stats["TestProvider"].failed_calls == 0
    
    def test_generate_label_fallback_on_failure(self):
        """Test fallback to second provider when first fails."""
        mock_provider1 = Mock(spec=LLMProvider)
        mock_provider1.is_available.return_value = True
        mock_provider1.get_provider_name.return_value = "Provider1"
        mock_provider1.generate_label.side_effect = ProviderError("Provider1 failed")
        
        mock_provider2 = Mock(spec=LLMProvider)
        mock_provider2.is_available.return_value = True
        mock_provider2.get_provider_name.return_value = "Provider2"
        mock_provider2.generate_label.return_value = "fallback label"
        
        manager = FallbackChainManager([mock_provider1, mock_provider2])
        result = manager.generate_label("test prompt")
        
        assert result == "fallback label"
        assert manager.stats["Provider1"].failed_calls == 1
        assert manager.stats["Provider2"].successful_calls == 1
    
    def test_generate_label_all_providers_fail(self):
        """Test that AllProvidersFailedError is raised when all providers fail."""
        mock_provider1 = Mock(spec=LLMProvider)
        mock_provider1.is_available.return_value = True
        mock_provider1.get_provider_name.return_value = "Provider1"
        mock_provider1.generate_label.side_effect = ProviderError("Provider1 failed")
        
        mock_provider2 = Mock(spec=LLMProvider)
        mock_provider2.is_available.return_value = True
        mock_provider2.get_provider_name.return_value = "Provider2"
        mock_provider2.generate_label.side_effect = ProviderError("Provider2 failed")
        
        manager = FallbackChainManager([mock_provider1, mock_provider2])
        
        with pytest.raises(AllProvidersFailedError) as exc_info:
            manager.generate_label("test prompt")
        
        assert "All providers failed" in str(exc_info.value)
        assert "Provider1" in str(exc_info.value)
        assert "Provider2" in str(exc_info.value)


class TestProviderFactory:
    """Test provider factory."""
    
    def test_create_providers_local_only(self):
        """Test creating local provider only."""
        config = ProviderConfig(
            provider_type="local",
            local_model="google/flan-t5-base"
        )
        
        with patch('zlsde.providers.factory.LocalProvider') as mock_local:
            mock_instance = Mock()
            mock_instance.is_available.return_value = True
            mock_local.return_value = mock_instance
            
            providers = ProviderFactory.create_providers(config, device="cpu")
            
            assert len(providers) == 1
            mock_local.assert_called_once_with("google/flan-t5-base", "cpu")
    
    def test_create_providers_api_with_keys(self):
        """Test creating API providers when keys are available."""
        config = ProviderConfig(
            provider_type="api",
            api_providers=["groq", "mistral"],
            groq_model="mixtral-8x7b-32768",
            mistral_model="mistral-small-latest",
            timeout=30
        )
        
        with patch.dict('os.environ', {
            'GROQ_API_KEY': 'test_groq_key',
            'MISTRAL_API_KEY': 'test_mistral_key'
        }), \
        patch('zlsde.providers.factory.load_dotenv'), \
        patch('zlsde.providers.factory.GroqProvider') as mock_groq, \
        patch('zlsde.providers.factory.MistralProvider') as mock_mistral, \
        patch('zlsde.providers.factory.LocalProvider') as mock_local:
            
            # Setup mocks
            for mock_provider in [mock_groq, mock_mistral, mock_local]:
                mock_instance = Mock()
                mock_instance.is_available.return_value = True
                mock_provider.return_value = mock_instance
            
            providers = ProviderFactory.create_providers(config, device="cpu")
            
            # Should create groq, mistral, and local as fallback
            assert len(providers) == 3
            mock_groq.assert_called_once()
            mock_mistral.assert_called_once()
            mock_local.assert_called_once()


class TestProviderConfig:
    """Test ProviderConfig data model."""
    
    def test_provider_config_validation_valid(self):
        """Test validation with valid configuration."""
        config = ProviderConfig(
            provider_type="api",
            api_providers=["groq", "mistral"],
            timeout=30
        )
        config.validate()  # Should not raise
    
    def test_provider_config_validation_invalid_type(self):
        """Test validation fails with invalid provider type."""
        config = ProviderConfig(provider_type="invalid")
        with pytest.raises(ValueError, match="provider_type must be"):
            config.validate()
    
    def test_provider_config_validation_invalid_provider(self):
        """Test validation fails with invalid API provider."""
        config = ProviderConfig(
            provider_type="api",
            api_providers=["invalid_provider"]
        )
        with pytest.raises(ValueError, match="Invalid API provider"):
            config.validate()
    
    def test_provider_config_validation_invalid_timeout(self):
        """Test validation fails with invalid timeout."""
        config = ProviderConfig(timeout=0)
        with pytest.raises(ValueError, match="timeout must be"):
            config.validate()
