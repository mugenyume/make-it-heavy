"""
OpenRouter provider implementation.
"""

from openai import OpenAI
from typing import Dict, List, Any
from .base_provider import BaseProvider

class OpenRouterProvider(BaseProvider):
    """OpenRouter provider using OpenAI-compatible API."""
    
    DISPLAY_NAME = "OpenRouter"
    DESCRIPTION = "OpenRouter - Unified access to 100+ AI models"
    DEFAULT_MODEL = "moonshotai/kimi-k2:free"
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = OpenAI(
            base_url=self.config['base_url'],
            api_key=self.config['api_key']
        )
    
    def _validate_config(self):
        """Validate OpenRouter configuration."""
        required_keys = ['api_key', 'base_url']
        for key in required_keys:
            if key not in self.config or not self.config.get(key):
                raise ValueError(f"Missing required OpenRouter config: {key}")

        if str(self.config.get('api_key', '')).strip() == "API_KEY_HERE":
            raise ValueError("OpenRouter API key is not configured. Update openrouter.api_key in config.yaml.")

        if 'model' not in self.config or not self.config['model']:
            self.config['model'] = self.DEFAULT_MODEL
    
    def create_chat_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a chat completion using OpenRouter API.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            
        Returns:
            Dictionary containing the completion response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                tools=tools
            )
            
            # Handle None tool_calls by ensuring it's always a list
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls is None:
                tool_calls = []
            
            # Convert response to dict format
            return {
                'choices': [{
                    'message': {
                        'content': response.choices[0].message.content or "",
                        'tool_calls': tool_calls
                    }
                }],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                }
            }
        except Exception as e:
            raise Exception(f"OpenRouter API call failed: {str(e)}")
    
    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.config.get('model', self.DEFAULT_MODEL)
