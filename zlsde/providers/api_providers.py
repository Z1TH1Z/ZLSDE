"""API provider implementations for cloud-based LLM services."""

import logging
from typing import Dict

import requests

from zlsde.providers.base import LLMProvider
from zlsde.providers.exceptions import (
    AuthenticationError,
    InvalidResponseError,
    RateLimitError,
    TimeoutError,
    ProviderError,
)

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq API provider implementation.
    
    Uses the Groq API for fast LLM inference with models like Mixtral and Llama.
    """
    
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768", 
                 timeout: int = 30):
        """Initialize Groq provider.
        
        Args:
            api_key: Groq API key for authentication
            model: Model name to use (default: mixtral-8x7b-32768)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
    
    def generate_label(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate label using Groq API.
        
        Args:
            prompt: Input prompt for label generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated label text
            
        Raises:
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
            TimeoutError: If request times out
            InvalidResponseError: If response format is invalid
            ProviderError: For other API errors
        """
        payload = self._create_request_payload(prompt, max_tokens)
        headers = self._create_headers()
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check for HTTP errors
            if response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded for {self.get_provider_name()}")
            elif response.status_code in [401, 403]:
                raise AuthenticationError(f"Authentication failed for {self.get_provider_name()}")
            
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            label_text = self._parse_response(response_json)
            
            return label_text
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timeout after {self.timeout}s for {self.get_provider_name()}")
        except (RateLimitError, AuthenticationError, TimeoutError, InvalidResponseError):
            # Re-raise our custom exceptions
            raise
        except requests.exceptions.RequestException as e:
            raise ProviderError(f"{self.get_provider_name()} request failed: {e}")
    
    def _parse_response(self, response_json: dict) -> str:
        """Parse API response to extract label text.
        
        Args:
            response_json: JSON response from API
            
        Returns:
            Extracted and cleaned label text
            
        Raises:
            InvalidResponseError: If response format is invalid
        """
        try:
            label_text = response_json["choices"][0]["message"]["content"]
            label_text = label_text.strip().lower()
            
            # Clean up label (remove extra whitespace, limit length)
            label_text = " ".join(label_text.split())
            if len(label_text) > 50:
                label_text = label_text[:50].rsplit(' ', 1)[0]  # Truncate at word boundary
            
            if not label_text:
                raise InvalidResponseError("Empty label text in response")
            
            return label_text
            
        except (KeyError, IndexError) as e:
            raise InvalidResponseError(f"Invalid response format from {self.get_provider_name()}: {e}")
    
    def _create_request_payload(self, prompt: str, max_tokens: int) -> Dict:
        """Create request payload for Groq API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            Request payload dictionary
        """
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
    
    def _create_headers(self) -> Dict:
        """Create HTTP headers for Groq API.
        
        Returns:
            Headers dictionary with authentication
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_provider_name(self) -> str:
        """Return provider name."""
        return "Groq"
    
    def is_available(self) -> bool:
        """Check if provider is available (has API key)."""
        return bool(self.api_key)


class MistralProvider(LLMProvider):
    """Mistral AI API provider implementation.
    
    Uses the Mistral AI API for LLM inference with Mistral models.
    """
    
    def __init__(self, api_key: str, model: str = "mistral-small-latest",
                 timeout: int = 30):
        """Initialize Mistral provider.
        
        Args:
            api_key: Mistral AI API key for authentication
            model: Model name to use (default: mistral-small-latest)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.endpoint = "https://api.mistral.ai/v1/chat/completions"
    
    def generate_label(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate label using Mistral AI API.
        
        Args:
            prompt: Input prompt for label generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated label text
            
        Raises:
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
            TimeoutError: If request times out
            InvalidResponseError: If response format is invalid
            ProviderError: For other API errors
        """
        payload = self._create_request_payload(prompt, max_tokens)
        headers = self._create_headers()
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check for HTTP errors
            if response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded for {self.get_provider_name()}")
            elif response.status_code in [401, 403]:
                raise AuthenticationError(f"Authentication failed for {self.get_provider_name()}")
            
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            label_text = self._parse_response(response_json)
            
            return label_text
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timeout after {self.timeout}s for {self.get_provider_name()}")
        except (RateLimitError, AuthenticationError, TimeoutError, InvalidResponseError):
            # Re-raise our custom exceptions
            raise
        except requests.exceptions.RequestException as e:
            raise ProviderError(f"{self.get_provider_name()} request failed: {e}")
    
    def _parse_response(self, response_json: dict) -> str:
        """Parse API response to extract label text.
        
        Args:
            response_json: JSON response from API
            
        Returns:
            Extracted and cleaned label text
            
        Raises:
            InvalidResponseError: If response format is invalid
        """
        try:
            label_text = response_json["choices"][0]["message"]["content"]
            label_text = label_text.strip().lower()
            
            # Clean up label (remove extra whitespace, limit length)
            label_text = " ".join(label_text.split())
            if len(label_text) > 50:
                label_text = label_text[:50].rsplit(' ', 1)[0]  # Truncate at word boundary
            
            if not label_text:
                raise InvalidResponseError("Empty label text in response")
            
            return label_text
            
        except (KeyError, IndexError) as e:
            raise InvalidResponseError(f"Invalid response format from {self.get_provider_name()}: {e}")
    
    def _create_request_payload(self, prompt: str, max_tokens: int) -> Dict:
        """Create request payload for Mistral AI API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            Request payload dictionary
        """
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
    
    def _create_headers(self) -> Dict:
        """Create HTTP headers for Mistral AI API.
        
        Returns:
            Headers dictionary with authentication
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_provider_name(self) -> str:
        """Return provider name."""
        return "Mistral"
    
    def is_available(self) -> bool:
        """Check if provider is available (has API key)."""
        return bool(self.api_key)


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider implementation.
    
    Uses the OpenRouter API which provides access to multiple LLM providers.
    """
    
    def __init__(self, api_key: str, 
                 model: str = "mistralai/mixtral-8x7b-instruct",
                 timeout: int = 30):
        """Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key for authentication
            model: Model name to use (default: mistralai/mixtral-8x7b-instruct)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate_label(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate label using OpenRouter API.
        
        Args:
            prompt: Input prompt for label generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated label text
            
        Raises:
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
            TimeoutError: If request times out
            InvalidResponseError: If response format is invalid
            ProviderError: For other API errors
        """
        payload = self._create_request_payload(prompt, max_tokens)
        headers = self._create_headers()
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check for HTTP errors
            if response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded for {self.get_provider_name()}")
            elif response.status_code in [401, 403]:
                raise AuthenticationError(f"Authentication failed for {self.get_provider_name()}")
            
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            label_text = self._parse_response(response_json)
            
            return label_text
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timeout after {self.timeout}s for {self.get_provider_name()}")
        except (RateLimitError, AuthenticationError, TimeoutError, InvalidResponseError):
            # Re-raise our custom exceptions
            raise
        except requests.exceptions.RequestException as e:
            raise ProviderError(f"{self.get_provider_name()} request failed: {e}")
    
    def _parse_response(self, response_json: dict) -> str:
        """Parse API response to extract label text.
        
        Args:
            response_json: JSON response from API
            
        Returns:
            Extracted and cleaned label text
            
        Raises:
            InvalidResponseError: If response format is invalid
        """
        try:
            label_text = response_json["choices"][0]["message"]["content"]
            label_text = label_text.strip().lower()
            
            # Clean up label (remove extra whitespace, limit length)
            label_text = " ".join(label_text.split())
            if len(label_text) > 50:
                label_text = label_text[:50].rsplit(' ', 1)[0]  # Truncate at word boundary
            
            if not label_text:
                raise InvalidResponseError("Empty label text in response")
            
            return label_text
            
        except (KeyError, IndexError) as e:
            raise InvalidResponseError(f"Invalid response format from {self.get_provider_name()}: {e}")
    
    def _create_request_payload(self, prompt: str, max_tokens: int) -> Dict:
        """Create request payload for OpenRouter API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            Request payload dictionary
        """
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
    
    def _create_headers(self) -> Dict:
        """Create HTTP headers for OpenRouter API.
        
        Returns:
            Headers dictionary with authentication
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_provider_name(self) -> str:
        """Return provider name."""
        return "OpenRouter"
    
    def is_available(self) -> bool:
        """Check if provider is available (has API key)."""
        return bool(self.api_key)
