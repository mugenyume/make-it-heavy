"""
Ollama provider implementation for local AI model inference.
"""

import logging
import requests
import json
from typing import Dict, List, Any, Optional
from .base_provider import BaseProvider

# Configure logging
logger = logging.getLogger(__name__)

class OllamaProvider(BaseProvider):
    """Ollama provider for local AI model inference."""
    
    DISPLAY_NAME = "Ollama"
    DESCRIPTION = "Ollama - Local AI model inference server"
    DEFAULT_MODEL = "llama3.1:8b"
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Set default base URL if not provided
        if 'base_url' not in self.config:
            self.config['base_url'] = "http://localhost:11434"
        
        self.base_url = self.config['base_url'].rstrip('/')
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.models_endpoint = f"{self.base_url}/api/tags"
    
    def _validate_config(self):
        """Validate Ollama configuration."""
        # Set default model if not specified
        if 'model' not in self.config or not self.config['model']:
            self.config['model'] = self.DEFAULT_MODEL
        
        # Validate base_url format
        if 'base_url' in self.config:
            base_url = self.config['base_url']
            if not base_url.startswith(('http://', 'https://')):
                raise ValueError("Ollama base_url must start with http:// or https://")
    
    def _clean_response_content(self, content: str) -> str:
        """
        Clean response content by removing thinking tags and unwanted elements.
        
        Args:
            content: Raw response content
            
        Returns:
            Cleaned content
        """
        if not content:
            return content
        
        import re
        
        # Remove thinking tags and their content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        
        # Remove empty thinking tags
        content = re.sub(r'<think>\s*</think>', '', content)
        content = re.sub(r'<thinking>\s*</thinking>', '', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines to double
        content = content.strip()
        
        return content
    
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
    
    def _convert_tools_to_ollama_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tools to Ollama format if needed.
        
        Args:
            tools: List of tool definitions in OpenAI format
            
        Returns:
            List of tools in Ollama-compatible format
        """
        if not tools:
            return []
        
        # For now, return tools as-is since Ollama supports OpenAI-compatible format
        # This can be extended if Ollama requires specific formatting
        return tools

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare messages for Ollama by preserving roles/content and normalizing payload shape.
        """
        prepared_messages: List[Dict[str, Any]] = []
        allowed_roles = {'system', 'user', 'assistant', 'tool'}

        for message in messages:
            role = message.get('role')
            if role not in allowed_roles:
                continue

            content = message.get('content', '')
            if content is None:
                content = ''
            elif isinstance(content, (dict, list)):
                content = json.dumps(content, default=str)
            elif not isinstance(content, str):
                content = str(content)

            normalized_message: Dict[str, Any] = {
                'role': role,
                'content': content
            }

            if role == 'assistant' and message.get('tool_calls'):
                normalized_message['tool_calls'] = message['tool_calls']

            if role == 'tool':
                tool_call_id = message.get('tool_call_id')
                name = message.get('name')
                if tool_call_id:
                    normalized_message['tool_call_id'] = tool_call_id
                if name:
                    normalized_message['name'] = name

            prepared_messages.append(normalized_message)

        if self.config.get('use_nothink', True):
            prepared_messages.insert(0, {
                'role': 'system',
                'content': 'Do not use <think> or <thinking> tags. Respond directly and clearly.'
            })

        return prepared_messages

    def _normalize_ollama_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize Ollama-native tool call payloads into OpenAI-compatible dict tool calls.
        """
        normalized_tool_calls: List[Dict[str, Any]] = []
        if not tool_calls:
            return normalized_tool_calls

        for index, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, dict):
                function_data = tool_call.get('function', {}) or {}
                name = function_data.get('name')
                arguments = function_data.get('arguments', {})
                call_id = tool_call.get('id') or f"call_ollama_{index}"
            else:
                function_data = getattr(tool_call, 'function', None)
                name = getattr(function_data, 'name', None) if function_data else None
                arguments = getattr(function_data, 'arguments', {})
                call_id = getattr(tool_call, 'id', None) or f"call_ollama_{index}"

            if not name:
                continue

            if isinstance(arguments, str):
                arguments_text = arguments
            else:
                arguments_text = json.dumps(arguments, default=str)

            normalized_tool_calls.append({
                'id': call_id,
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': arguments_text
                }
            })

        return normalized_tool_calls
    
    def _simulate_tool_calls_from_content(self, content: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simulate tool calls based on content analysis.
        Since Ollama doesn't support function calling well, we analyze the response
        and simulate appropriate tool calls.
        
        Args:
            content: The response content to analyze
            tools: Available tools
            
        Returns:
            List of simulated tool calls
        """
        tool_calls = []
        
        if not tools or not content:
            return tool_calls
        
        content_lower = content.lower()
        
        # Create a mapping of available tools
        available_tools = {}
        for tool in tools:
            tool_name = tool.get('function', {}).get('name', '')
            if tool_name:
                available_tools[tool_name] = tool
        
        # 1. Check for calculation requests
        if 'calculate' in available_tools:
            import re
            # Look for mathematical expressions or calculation keywords
            calc_patterns = [
                r'(\d+\s*[+\-*/]\s*\d+(?:\s*[+\-*/]\s*\d+)*)',  # Basic math expressions
                r'calculate\s+([^.!?]+)',  # "calculate X"
                r'what\s+is\s+(\d+[+\-*/]\d+)',  # "what is X+Y"
            ]
            
            for pattern in calc_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.strip():
                        tool_calls.append({
                            'id': f'call_calc_{len(tool_calls)}',
                            'type': 'function',
                            'function': {
                                'name': 'calculate',
                                'arguments': json.dumps({'expression': match.strip()})
                            }
                        })
                        break  # Only add one calculation per response
        
        # 2. Check for search requests
        if 'search_web' in available_tools:
            search_keywords = [
                'search for', 'look up', 'find information', 'research',
                'what is', 'who is', 'where is', 'when did', 'how to'
            ]
            
            for keyword in search_keywords:
                if keyword in content_lower:
                    # Extract search query from context
                    search_query = content[:200]  # Use first part of content as query
                    tool_calls.append({
                        'id': f'call_search_{len(tool_calls)}',
                        'type': 'function',
                        'function': {
                            'name': 'search_web',
                            'arguments': json.dumps({'query': search_query.strip()})
                        }
                    })
                    break  # Only add one search per response
        
        # 3. Check for task completion
        if 'mark_task_complete' in available_tools:
            completion_keywords = [
                'task completed', 'finished', 'done', 'complete',
                'that concludes', 'in conclusion', 'to summarize'
            ]
            
            if any(keyword in content_lower for keyword in completion_keywords):
                tool_calls.append({
                    'id': f'call_complete_{len(tool_calls)}',
                    'type': 'function',
                    'function': {
                        'name': 'mark_task_complete',
                        'arguments': json.dumps({
                            'task_summary': 'Task completed by Ollama',
                            'completion_message': content[:500]  # Use content as completion message
                        })
                    }
                })
        
        return tool_calls
    
    def create_chat_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a chat completion using Ollama API with optimizations.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            
        Returns:
            Dictionary containing the completion response
        """
        try:
            logger.info(f"Creating Ollama chat completion with model: {self.config['model']}")

            # Prepare request payload
            # Get configurable options with sensible defaults
            options = {
                'temperature': self.config.get('temperature', 0.7),
                'top_p': self.config.get('top_p', 0.9),
                'num_ctx': self.config.get('num_ctx', 8192),
                'num_predict': self.config.get('num_predict', -1),
                'stop': self.config.get('stop', []),
            }
            
            # Remove any None values to avoid API issues
            options = {k: v for k, v in options.items() if v is not None}
            
            prepared_messages = self._prepare_messages(messages)
            payload = {
                'model': self.config['model'],
                'messages': prepared_messages,
                'stream': False,
                'options': options
            }

            native_tools_enabled = self.config.get('native_tools', True)
            if native_tools_enabled and tools:
                converted_tools = self._convert_tools_to_ollama_format(tools)
                if converted_tools:
                    payload['tools'] = converted_tools

            connect_timeout = self.config.get('connect_timeout', 15)
            read_timeout = self.config.get('timeout_seconds', 300)

            # Make request to Ollama with configurable timeout
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=(connect_timeout, read_timeout)
            )
            
            if response.status_code != 200:
                error_text = response.text[:200] if response.text else "Unknown error"
                raise Exception(f"Ollama API returned status {response.status_code}: {error_text}")
            
            response_data = response.json()
            
            # Extract message content
            message_content = self._safe_get_nested_value(response_data, ['message', 'content'], "")
            
            # Clean up thinking tags and unwanted content
            message_content = self._clean_response_content(message_content)
            
            # Prefer native tool calls and fallback to simulated calls if explicitly enabled.
            raw_tool_calls = self._safe_get_nested_value(response_data, ['message', 'tool_calls'], [])
            tool_calls = self._normalize_ollama_tool_calls(raw_tool_calls)
            simulate_tool_calls = self.config.get('simulate_tool_calls', False)
            if not tool_calls and simulate_tool_calls and tools and message_content:
                tool_calls = self._simulate_tool_calls_from_content(message_content, tools)
            
            # Handle usage information (Ollama might not provide this)
            prompt_tokens = self._safe_get_nested_value(response_data, ['prompt_eval_count'], 0)
            completion_tokens = self._safe_get_nested_value(response_data, ['eval_count'], 0)
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                'choices': [{
                    'message': {
                        'content': message_content,
                        'tool_calls': tool_calls
                    }
                }],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                }
            }
                
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama server")
            fallback_response = {
                'choices': [{
                    'message': {
                        'content': f"Error: Cannot connect to Ollama server at {self.base_url}. Make sure Ollama is running.",
                        'tool_calls': []
                    }
                }],
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }
            return fallback_response
            
        except requests.exceptions.ReadTimeout:
            logger.error(f"Ollama request timed out for model {self.config['model']}")
            fallback_response = {
                'choices': [{
                    'message': {
                        'content': (
                            f"The model {self.config['model']} is taking too long to respond. "
                            "Try using a smaller/faster model like 'llama3.1:8b' or 'phi3:mini', "
                            "or increase ollama.timeout_seconds in config.yaml."
                        ),
                        'tool_calls': []
                    }
                }],
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }
            return fallback_response
            
        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Provide a safe fallback response with helpful suggestions
            error_msg = str(e)
            if "400" in error_msg and "tool_calls" in error_msg:
                content = f"Ollama doesn't fully support function calling. The request has been simplified but may have limited tool functionality."
            elif "timeout" in error_msg.lower():
                content = f"Request timed out. Try using a smaller model like 'llama3.1:8b' instead of '{self.config['model']}'."
            else:
                content = f"Ollama API error: {error_msg[:200]}..."
            
            fallback_response = {
                'choices': [{
                    'message': {
                        'content': content,
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
        Test the connection to Ollama server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # First, try to get available models
            response = requests.get(self.models_endpoint, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
            
            # Then try a simple chat completion
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
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama server.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(self.models_endpoint, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                return [model.get('name', '') for model in models if model.get('name')]
            else:
                logger.error(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []
