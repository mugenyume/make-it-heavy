"""
SambaNova provider implementation with comprehensive null safety and error handling.
"""

import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from .base_provider import BaseProvider

# Configure logging
logger = logging.getLogger(__name__)

class SambaNovaProvider(BaseProvider):
    """SambaNova provider using OpenAI-compatible API with robust error handling."""
    
    DISPLAY_NAME = "SambaNova"
    DESCRIPTION = "SambaNova - Enterprise-grade AI inference platform"
    DEFAULT_MODEL = "Meta-Llama-3.3-70B-Instruct"
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = OpenAI(
            base_url=self.config['base_url'],
            api_key=self.config['api_key']
        )
    
    def _validate_config(self):
        """Validate SambaNova configuration."""
        required_keys = ['api_key', 'base_url']
        for key in required_keys:
            if key not in self.config or not self.config.get(key):
                raise ValueError(f"Missing required SambaNova config: {key}")

        if str(self.config.get('api_key', '')).strip() == "API_KEY_HERE":
            raise ValueError("SambaNova API key is not configured. Update sambanova.api_key in config.yaml.")
        
        # Set default model if not specified
        if 'model' not in self.config or not self.config['model']:
            self.config['model'] = self.DEFAULT_MODEL
    
    def _safe_get_nested_value(self, data: Any, path: List[str], default: Any = None) -> Any:
        """
        Safely navigate nested data structures with null safety, handling both dicts and objects.
        
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
    
    def _validate_response_structure(self, response: Any) -> bool:
        """
        Validate that the response has the expected structure with null safety.
        
        Args:
            response: The API response to validate
            
        Returns:
            True if response structure is valid, False otherwise
        """
        if response is None:
            logger.error("Response is None")
            return False
        
        try:
            # Check choices - handle both dict and object access
            choices = self._safe_get_nested_value(response, ['choices'])
            if not choices or not isinstance(choices, (list, tuple)) or len(choices) == 0:
                logger.error("Invalid or missing choices in response")
                return False
            
            # Check first choice
            first_choice = choices[0]
            if first_choice is None:
                logger.error("First choice is None")
                return False
            
            # Check message exists
            message = self._safe_get_nested_value(first_choice, ['message'])
            if message is None:
                logger.error("Message is None in first choice")
                return False
            
            # RELAXED VALIDATION: Allow None content - this is valid for tool-only responses
            # We'll handle None content gracefully in the extraction phase
            return True
            
        except Exception as e:
            logger.error(f"Error validating response structure: {str(e)}")
            return False
    
    def create_chat_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a chat completion using SambaNova API with comprehensive error handling.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            
        Returns:
            Dictionary containing the completion response
        """
        try:
            logger.info("Creating SambaNova chat completion")
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                tools=tools
            )
            
            # Validate response structure
            if not self._validate_response_structure(response):
                raise Exception("Invalid response structure from SambaNova API")
            
            # Extract values safely with null handling
            try:
                # Handle both object and dict access
                choices = response.choices
                if choices and len(choices) > 0:
                    first_choice = choices[0]
                    message = first_choice.message
                    content = message.content if message.content is not None else ""
                    tool_calls = message.tool_calls if hasattr(message, 'tool_calls') else []
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
            logger.error(f"SambaNova API call failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Provide a safe fallback response
            fallback_response = {
                'choices': [{
                    'message': {
                        'content': f"Error: SambaNova API call failed - {str(e)}",
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
        Test the connection to SambaNova API.
        
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
