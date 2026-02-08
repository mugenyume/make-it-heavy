"""
Base provider interface for all AI providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class BaseProvider(ABC):
    """Abstract base class for all AI providers."""
    
    DISPLAY_NAME = "Base Provider"
    DESCRIPTION = "Base provider interface"
    DEFAULT_MODEL = None
    
    def __init__(self, config: dict):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self):
        """Validate the provider configuration."""
        pass
    
    @abstractmethod
    def create_chat_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create a chat completion using the provider's API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool definitions for function calling
            
        Returns:
            Dictionary containing the completion response
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the current model name being used."""
        pass
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about this provider."""
        return {
            'name': self.__class__.__name__.lower().replace('provider', ''),
            'display_name': self.DISPLAY_NAME,
            'description': self.DESCRIPTION,
            'model': self.get_model_name()
        }