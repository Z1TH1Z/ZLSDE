"""Factory for creating LLM providers based on configuration."""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

from zlsde.providers.base import LLMProvider
from zlsde.providers.api_providers import GroqProvider, MistralProvider, OpenRouterProvider
from zlsde.providers.local_provider import LocalProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating LLM providers.
    
    This factory creates and initializes providers based on configuration,
    loading API keys from environment variables and filtering out providers
    with missing credentials.
    """
    
    @staticmethod
    def create_providers(config, device: str = "cpu") -> List[LLMProvider]:
        """Create providers based on configuration.
        
        Loads environment variables from .env file and creates providers
        according to the configuration. Only providers with valid credentials
        are included in the returned list.
        
        Args:
            config: ProviderConfig object with provider settings
            device: Device for local model ("cpu", "cuda", "mps")
            
        Returns:
            List of initialized providers in fallback order
            
        Raises:
            ValueError: If no providers are available after filtering
        """
        # Load environment variables from .env file
        load_dotenv()
        
        providers = []
        
        if config.provider_type == "api":
            # Create API providers in order
            for provider_name in config.api_providers:
                provider = ProviderFactory._create_api_provider(
                    provider_name, config
                )
                if provider and provider.is_available():
                    providers.append(provider)
                    logger.info(f"Added {provider_name} provider to fallback chain")
                else:
                    logger.warning(f"Skipping {provider_name}: API key not found or invalid")
            
            # Add local as final fallback
            try:
                local_provider = LocalProvider(config.local_model, device)
                if local_provider.is_available():
                    providers.append(local_provider)
                    logger.info(f"Added local provider ({config.local_model}) as final fallback")
            except Exception as e:
                logger.warning(f"Failed to load local provider: {e}")
        
        elif config.provider_type == "local":
            # Only use local provider
            try:
                local_provider = LocalProvider(config.local_model, device)
                if local_provider.is_available():
                    providers.append(local_provider)
                    logger.info(f"Using local provider only: {config.local_model}")
            except Exception as e:
                logger.error(f"Failed to load local provider: {e}")
        
        if not providers:
            raise ValueError(
                "No providers available. Check configuration and API keys. "
                "For API providers, ensure API keys are set in .env file or environment variables."
            )
        
        return providers
    
    @staticmethod
    def _create_api_provider(name: str, config) -> Optional[LLMProvider]:
        """Create a single API provider.
        
        Loads the API key from environment variables and creates the
        appropriate provider instance.
        
        Args:
            name: Provider name ("groq", "mistral", or "openrouter")
            config: ProviderConfig object with provider settings
            
        Returns:
            Initialized provider instance, or None if API key is missing
        """
        if name == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                return GroqProvider(api_key, config.groq_model, config.timeout)
            else:
                logger.debug("GROQ_API_KEY not found in environment")
        
        elif name == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key:
                return MistralProvider(api_key, config.mistral_model, config.timeout)
            else:
                logger.debug("MISTRAL_API_KEY not found in environment")
        
        elif name == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                return OpenRouterProvider(api_key, config.openrouter_model, config.timeout)
            else:
                logger.debug("OPENROUTER_API_KEY not found in environment")
        
        return None
