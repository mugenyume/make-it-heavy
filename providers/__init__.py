"""
Provider factory and registry for multi-provider AI support.
"""

from .base_provider import BaseProvider
from .openrouter_provider import OpenRouterProvider
from .mistral_provider import MistralProvider
from .sambanova_provider import SambaNovaProvider
from .cerebras_provider import CerebrasProvider
from .ollama_provider import OllamaProvider
from .groq_provider import GroqProvider

class ProviderFactory:
    """Factory class for creating AI providers based on configuration."""
    
    _providers = {
        'openrouter': OpenRouterProvider,
        'mistralai': MistralProvider,
        'sambanova': SambaNovaProvider,
        'cerebras': CerebrasProvider,
        'ollama': OllamaProvider,
        'groq': GroqProvider
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, config: dict) -> BaseProvider:
        """
        Create a provider instance based on the provider name.
        
        Args:
            provider_name: Name of the provider ('openrouter', 'mistralai', 'sambanova', 'cerebras', 'ollama', 'groq')
            config: Configuration dictionary containing provider-specific settings
            
        Returns:
            BaseProvider instance
            
        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported providers: {list(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider_name]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of all available provider names."""
        return list(cls._providers.keys())
    
    @classmethod
    def get_provider_info(cls, provider_name: str) -> dict:
        """Get information about a specific provider."""
        if provider_name not in cls._providers:
            return {}
        
        provider_class = cls._providers[provider_name]
        return {
            'name': provider_name,
            'display_name': provider_class.DISPLAY_NAME,
            'description': provider_class.DESCRIPTION,
            'default_model': provider_class.DEFAULT_MODEL
        }