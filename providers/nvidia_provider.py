"""
NVIDIA NIM provider implementation.
"""

from openai import OpenAI
from typing import Dict, List, Any
from .base_provider import BaseProvider

class NvidiaProvider(BaseProvider):
    """NVIDIA NIM provider using OpenAI-compatible API."""
    
    DISPLAY_NAME = "NVIDIA NIM"
    DESCRIPTION = "NVIDIA NIM - GPU-accelerated AI model inference"
    DEFAULT_MODEL = "minimaxai/minimax-m2.1"
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = OpenAI(
            base_url=self.config['base_url'],
            api_key=self.config['api_key']
        )
    
    def _validate_config(self):
        """Validate NVIDIA NIM configuration."""
        required_keys = ['api_key', 'base_url']
        for key in required_keys:
            if key not in self.config or not self.config.get(key):
                raise ValueError(f"Missing required NVIDIA NIM config: {key}")

        if str(self.config.get('api_key', '')).strip() == "API_KEY_HERE":
            raise ValueError("NVIDIA API key is not configured. Update nvidia.api_key in config.yaml.")

        if 'model' not in self.config or not self.config['model']:
            self.config['model'] = self.DEFAULT_MODEL
    
    def create_chat_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a chat completion using NVIDIA NIM API.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            
        Returns:
            Dictionary containing the completion response
        """
        try:
            kwargs = {
                'model': self.config['model'],
                'messages': messages,
                'max_tokens': self.config.get('max_tokens', 16384),
                'temperature': self.config.get('temperature', 1.0),
                'top_p': self.config.get('top_p', 1.0),
            }
            
            if tools:
                kwargs['tools'] = tools
            
            response = self.client.chat.completions.create(**kwargs)
            
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
            raise Exception(f"NVIDIA NIM API call failed: {str(e)}")
    
    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.config.get('model', self.DEFAULT_MODEL)
