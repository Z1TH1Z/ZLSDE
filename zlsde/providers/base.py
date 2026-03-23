"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM providers (API-based and local) must implement this interface
    to ensure consistent behavior across the pipeline.
    """
    
    @abstractmethod
    def generate_label(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate label text from prompt.
        
        Args:
            prompt: The input prompt for label generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated label text (cleaned and normalized)
            
        Raises:
            ProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name for logging and error messages.
        
        Returns:
            Human-readable provider name (e.g., "Groq", "Local (flan-t5-base)")
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available.
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass
