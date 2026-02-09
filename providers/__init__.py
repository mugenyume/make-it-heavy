"""
Provider factory and registry for multi-provider AI support.
"""

import importlib
from typing import Dict, Type

from .base_provider import BaseProvider

class ProviderFactory:
    """Factory class for creating AI providers based on configuration."""
    
    _provider_paths: Dict[str, str] = {
        'openrouter': 'providers.openrouter_provider:OpenRouterProvider',
        'mistralai': 'providers.mistral_provider:MistralProvider',
        'sambanova': 'providers.sambanova_provider:SambaNovaProvider',
        'cerebras': 'providers.cerebras_provider:CerebrasProvider',
        'ollama': 'providers.ollama_provider:OllamaProvider',
        'groq': 'providers.groq_provider:GroqProvider',
        'nvidia': 'providers.nvidia_provider:NvidiaProvider'
    }

    _provider_metadata: Dict[str, Dict[str, str]] = {
        'openrouter': {
            'display_name': 'OpenRouter',
            'description': 'OpenRouter - Unified access to 100+ AI models',
            'default_model': 'moonshotai/kimi-k2:free'
        },
        'mistralai': {
            'display_name': 'MistralAI',
            'description': 'MistralAI - Advanced language models with function calling',
            'default_model': 'mistral-large-latest'
        },
        'sambanova': {
            'display_name': 'SambaNova',
            'description': 'SambaNova - Enterprise-grade AI inference platform',
            'default_model': 'Meta-Llama-3.3-70B-Instruct'
        },
        'cerebras': {
            'display_name': 'Cerebras',
            'description': 'Cerebras - Ultra-fast AI inference with wafer-scale processors',
            'default_model': 'llama-3.3-70b'
        },
        'ollama': {
            'display_name': 'Ollama',
            'description': 'Ollama - Local AI model inference server',
            'default_model': 'llama3.1:8b'
        },
        'groq': {
            'display_name': 'Groq',
            'description': 'Groq - Ultra-fast LLM inference with specialized hardware',
            'default_model': 'llama-3.3-70b-versatile'
        },
        'nvidia': {
            'display_name': 'NVIDIA NIM',
            'description': 'NVIDIA NIM - GPU-accelerated AI model inference',
            'default_model': 'minimaxai/minimax-m2.1'
        }
    }

    _provider_classes: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def _load_provider_class(cls, provider_name: str) -> Type[BaseProvider]:
        """Load provider class lazily to avoid import-time dependency failures."""
        if provider_name in cls._provider_classes:
            return cls._provider_classes[provider_name]

        provider_path = cls._provider_paths.get(provider_name)
        if not provider_path:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported providers: {list(cls._provider_paths.keys())}"
            )

        module_name, class_name = provider_path.split(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Failed to import dependencies for provider '{provider_name}'. "
                "Install required packages with: pip install -r requirements.txt"
            ) from exc

        provider_class = getattr(module, class_name, None)
        if provider_class is None:
            raise ValueError(f"Provider class '{class_name}' not found in module '{module_name}'")

        cls._provider_classes[provider_name] = provider_class
        return provider_class
    
    @classmethod
    def create_provider(cls, provider_name: str, config: dict) -> BaseProvider:
        """
        Create a provider instance based on the provider name.
        
        Args:
            provider_name: Name of the provider ('openrouter', 'mistralai', 'sambanova', 'cerebras', 'ollama', 'groq', 'nvidia')
            config: Configuration dictionary containing provider-specific settings
            
        Returns:
            BaseProvider instance
            
        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name not in cls._provider_paths:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported providers: {list(cls._provider_paths.keys())}"
            )
        
        provider_class = cls._load_provider_class(provider_name)
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of all available provider names."""
        return list(cls._provider_paths.keys())
    
    @classmethod
    def get_provider_info(cls, provider_name: str) -> dict:
        """Get information about a specific provider."""
        if provider_name not in cls._provider_paths:
            return {}

        metadata = cls._provider_metadata.get(provider_name, {})
        return {'name': provider_name, **metadata}
