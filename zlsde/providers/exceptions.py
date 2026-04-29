"""Custom exceptions for LLM provider errors."""


class ProviderError(Exception):
    """Base exception for provider errors.

    All provider-specific exceptions inherit from this base class.
    """

    pass


class RateLimitError(ProviderError):
    """Raised when API rate limit is exceeded.

    This typically occurs when too many requests are made in a short time period.
    The provider should be skipped and the next provider in the fallback chain tried.
    """

    pass


class AuthenticationError(ProviderError):
    """Raised when API authentication fails.

    This typically indicates an invalid or missing API key.
    The provider should be skipped and the next provider in the fallback chain tried.
    """

    pass


class TimeoutError(ProviderError):
    """Raised when API request times out.

    This occurs when the API doesn't respond within the configured timeout period.
    The provider should be skipped and the next provider in the fallback chain tried.
    """

    pass


class InvalidResponseError(ProviderError):
    """Raised when API response is invalid or malformed.

    This occurs when the API returns a response that doesn't match the expected format
    or is missing required fields.
    The provider should be skipped and the next provider in the fallback chain tried.
    """

    pass


class AllProvidersFailedError(ProviderError):
    """Raised when all providers in fallback chain fail.

    This is the final error raised when no provider was able to successfully
    generate a label. The error message includes details about all attempted
    providers and their failure reasons.
    """

    pass
