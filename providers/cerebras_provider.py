"""
Cerebras provider implementation using the official Cerebras Cloud SDK.
"""

import logging
from typing import Dict, List, Any, Optional

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    Cerebras = None

from .base_provider import BaseProvider

# Configure logging
logger = logging.getLogger(__name__)

class CerebrasProvider(BaseProvider):
    """Cerebras provider using the official Cerebras Cloud SDK."""
    
    DISPLAY_NAME = "Cerebras"
    DESCRIPTION = "Cerebras - Ultra-fast AI inference with wafer-scale processors"
    DEFAULT_MODEL = "llama3.1-70b"
    
    def __init__(self, config: dict):
        if not CEREBRAS_AVAILABLE:
            raise ImportError(
                "Cerebras Cloud SDK not available. Install with: pip install cerebras_cloud_sdk"
            )
        
        super().__init__(config)
        self.client = Cerebras(api_key=self.config['api_key'])
    
    def _validate_config(self):
        """Validate Cerebras configuration."""
        required_keys = ['api_key']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required Cerebras config: {key}")
        
        # Set default model if not specified
        if 'model' not in self.config or not self.config['model']:
            self.config['model'] = self.DEFAULT_MODEL
    
    def _safe_get_nested_value(self, data: Any, path: List[str], default: Any = None) -> Any:
        """
        Safely navigate nested data structures with null safety.
        
        Args:
            data: The data structure to navigate (dict or object)
            path: List of keys/attributes/indices to traverse
            default: Default value if path is invalid
            
        Returns:
            Value at the specified path or default
        """
        try:
            current = data
            for key in path:
                if current is None:
                    return default
                if isinstance(key, int) and isinstance(current, (list, tuple)):
                    if 0 <= key < len(current):
                        current = current[key]
                    else:
                        return default
                elif isinstance(current, dict):
                    current = current.get(key, default)
                elif hasattr(current, key):
                    current = getattr(current, key)
                else:
                    return default
            return current
        except (TypeError, IndexError, AttributeError):
            return default
    
    def create_chat_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a chat completion using Cerebras API.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            
        Returns:
            Dictionary containing the completion response
        """
        try:
            logger.info("Creating Cerebras chat completion")
            
            # Prepare request parameters
            request_params = {
                'model': self.config['model'],
                'messages': messages
            }
            
            # Add tools if provided
            if tools:
                request_params['tools'] = tools
            
            response = self.client.chat.completions.create(**request_params)
            
            # Extract values safely with null handling
            try:
                # Handle both object and dict access
                choices = response.choices
                if choices and len(choices) > 0:
                    first_choice = choices[0]
                    message = first_choice.message
                    content = message.content if message.content is not None else ""
                    tool_calls = message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else []
                else:
                    content = ""
                    tool_calls = []
                
                # Handle usage information
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0)
                
                return {
                    'choices': [{
                        'message': {
                            'content': content,
                            'tool_calls': tool_calls
                        }
                    }],
                    'usage': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens
                    }
                }
                
            except Exception as extract_error:
                logger.error(f"Error extracting response data: {str(extract_error)}")
                raise Exception(f"Failed to extract response data: {str(extract_error)}")
                
        except Exception as e:
            logger.error(f"Cerebras API call failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Provide a safe fallback response
            fallback_response = {
                'choices': [{
                    'message': {
                        'content': f"Error: Cerebras API call failed - {str(e)}",
                        'tool_calls': []
                    }
                }],
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }
            
            logger.warning("Returning fallback response due to API failure")
            return fallback_response
    
    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.config.get('model', self.DEFAULT_MODEL)
    
    def test_connection(self) -> bool:
        """
        Test the connection to Cerebras API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = self.create_chat_completion(test_messages)
            
            # Check if we got a valid response
            choices = self._safe_get_nested_value(response, ['choices'])
            content = self._safe_get_nested_value(response, ['choices', 0, 'message', 'content'])
            is_valid = choices is not None and len(choices) > 0
            is_not_error = content is not None and not str(content).startswith("Error:")
            return is_valid and is_not_error
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False