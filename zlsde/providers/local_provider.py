"""Local model provider using transformers library."""

import logging
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from zlsde.providers.base import LLMProvider
from zlsde.providers.exceptions import ProviderError

logger = logging.getLogger(__name__)


class LocalProvider(LLMProvider):
    """Local model provider using transformers library.

    This provider wraps the existing local model implementation to conform
    to the LLMProvider interface, enabling it to work within the fallback chain.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cpu"):
        """Initialize local provider.

        Args:
            model_name: HuggingFace model name (default: google/flan-t5-base)
            device: Device to run model on ("cpu", "cuda", or "mps")
        """
        self.model_name = model_name
        self.device = device
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._load_model()

    def _load_model(self):
        """Load the local LLM model and tokenizer.

        This method attempts to load the model from HuggingFace.
        If loading fails, the provider will be marked as unavailable.
        """
        try:
            logger.info(f"Loading local model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Local model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.model = None
            self.tokenizer = None

    def generate_label(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate label using local model.

        Args:
            prompt: Input prompt for label generation
            max_tokens: Maximum tokens in response

        Returns:
            Generated label text

        Raises:
            ProviderError: If generation fails or model is not available
        """
        if not self.is_available():
            raise ProviderError(f"{self.get_provider_name()} is not available")

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_length=max_tokens, num_beams=3, temperature=0.7, do_sample=False
                )

            # Decode response
            label_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Import the extraction helper (defined in api_providers module)
            from zlsde.providers.api_providers import _extract_label_from_response

            # Extract clean label from response (removes prompt echo)
            label_text = _extract_label_from_response(label_text)
            label_text = label_text.strip().lower()

            # Final validation and cleanup
            label_text = " ".join(label_text.split())
            if len(label_text) > 50:
                label_text = label_text[:50].rsplit(" ", 1)[0]

            if not label_text:
                label_text = "unlabeled"

            return label_text

        except Exception as e:
            raise ProviderError(f"{self.get_provider_name()} generation failed: {e}")

    def get_provider_name(self) -> str:
        """Return provider name with model info."""
        return f"Local ({self.model_name})"

    def is_available(self) -> bool:
        """Check if provider is available (model loaded successfully)."""
        return self.model is not None and self.tokenizer is not None
