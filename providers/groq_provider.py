"""
Groq provider implementation using the official Groq SDK.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class GroqAPIError(Exception):
    """Raised when Groq API returns an error."""
    pass


class GroqProvider(BaseProvider):
    """Groq provider using the official Groq SDK for fast LLM inference."""
    
    DISPLAY_NAME = "Groq"
    DESCRIPTION = "Groq - Ultra-fast LLM inference with specialized hardware"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    
    def __init__(self, config: dict):
        if not GROQ_AVAILABLE or Groq is None:
            raise ImportError(
                "Groq SDK not available. Install with: pip install groq"
            )
        
        super().__init__(config)
        
        self.client = Groq(api_key=self.config['api_key'])
    
    def _validate_config(self):
        """Validate Groq configuration."""
        required_keys = ['api_key']
        for key in required_keys:
            if key not in self.config or not self.config.get(key):
                raise ValueError(f"Missing required Groq config: {key}")

        if str(self.config.get('api_key', '')).strip() == "API_KEY_HERE":
            raise ValueError("Groq API key is not configured. Update groq.api_key in config.yaml.")
        
        if 'model' not in self.config or not self.config['model']:
            self.config['model'] = self.DEFAULT_MODEL
        elif '/' in self.config['model']:
            logger.warning(
                "Groq model '%s' looks invalid for Groq API. Falling back to default model '%s'.",
                self.config['model'],
                self.DEFAULT_MODEL
            )
            self.config['model'] = self.DEFAULT_MODEL
    
    def _serialize_tool_call(self, tool_call) -> Optional[Dict[str, Any]]:
        """
        Serialize a Groq SDK tool call object to a plain dict.
        
        Groq returns ChatCompletionMessageToolCall objects that need to be
        converted to dicts for the agent to process them correctly.
        """
        if tool_call is None:
            return None
        
        if isinstance(tool_call, dict):
            return tool_call
        
        tool_call_id = getattr(tool_call, 'id', None) or f"call_{id(tool_call)}"
        tool_type = getattr(tool_call, 'type', 'function') or 'function'
        
        function_obj = getattr(tool_call, 'function', None)
        if function_obj is None:
            return None
        
        function_name = getattr(function_obj, 'name', None)
        if not function_name:
            return None
        
        function_args = getattr(function_obj, 'arguments', '{}')
        if function_args is None:
            function_args = '{}'
        elif not isinstance(function_args, str):
            function_args = str(function_args)
        
        return {
            "id": tool_call_id,
            "type": tool_type,
            "function": {
                "name": function_name,
                "arguments": function_args
            }
        }
    
    def _serialize_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """Serialize a list of Groq SDK tool call objects to plain dicts."""
        if not tool_calls:
            return []
        
        serialized = []
        for tool_call in tool_calls:
            serialized_call = self._serialize_tool_call(tool_call)
            if serialized_call:
                serialized.append(serialized_call)
        return serialized
    
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Validate message structure before sending to API."""
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            if 'role' not in msg:
                raise ValueError(f"Message {i} missing 'role' field")
            if msg['role'] not in ('system', 'user', 'assistant', 'tool'):
                raise ValueError(f"Message {i} has invalid role: {msg['role']}")
    
    def _validate_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Validate and sanitize tool definitions."""
        if not tools:
            return None
        
        validated_tools = []
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                logger.warning(f"Tool {i} is not a dict, skipping")
                continue
            if 'type' not in tool:
                tool['type'] = 'function'
            if 'function' not in tool:
                logger.warning(f"Tool {i} missing 'function' field, skipping")
                continue
            
            func_def = tool.get('function', {})
            if not func_def.get('name'):
                logger.warning(f"Tool {i} missing function name, skipping")
                continue
            
            if 'parameters' not in func_def:
                func_def['parameters'] = {'type': 'object', 'properties': {}}
            
            validated_tools.append(tool)
        
        return validated_tools if validated_tools else None
    
    def _parse_text_format_tool_call(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse text-format tool calls like <function=name{"arg": "value"}></function>
        
        Groq sometimes generates tool calls in text format instead of structured format,
        especially when the model is confused about tool calling. This method attempts
        to extract and parse them.
        """
        tool_calls = []
        
        pattern = r'<function\s*=\s*([a-zA-Z_][\w\-]*)\s*>?\s*(\{.*?\})\s*</function>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for i, (func_name, args_str) in enumerate(matches):
            args_str = args_str.strip()
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                # Fallback for malformed escaping found in failed_generation payloads.
                try:
                    args = json.loads(args_str.replace("\\'", "'"))
                except json.JSONDecodeError:
                    args = {}
            
            tool_calls.append({
                "id": f"call_text_{i}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args)
                }
            })
        
        if tool_calls:
            logger.info(f"Parsed {len(tool_calls)} text-format tool call(s) from response")
        
        return tool_calls

    def _extract_text_without_function_calls(self, text: str) -> str:
        """Strip inline <function=...>...</function> blocks and return residual content."""
        if not text:
            return ""
        cleaned = re.sub(
            r'<function\s*=\s*[a-zA-Z_][\w\-]*\s*>?\s*\{.*?\}\s*</function>',
            '',
            text,
            flags=re.DOTALL
        )
        return cleaned.strip()
    
    def _extract_failed_generation(self, error_msg: str) -> Optional[str]:
        """Extract the failed_generation field from Groq error message."""
        match = re.search(r"'failed_generation':\s*'(.*?)'(?:\}|,)", error_msg, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r'"failed_generation":\s*"(.*?)"(?:\}|,)', error_msg, re.DOTALL)
        if match:
            return match.group(1)
        return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def create_chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a chat completion using Groq API.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            
        Returns:
            Dictionary containing the completion response
            
        Raises:
            GroqAPIError: When the API call fails
            ValueError: When input validation fails
        """
        self._validate_messages(messages)
        validated_tools = self._validate_tools(tools)
        
        request_params = {
            'model': self.config['model'],
            'messages': messages,
            'temperature': self.config.get('temperature', 0.7),
            'top_p': self.config.get('top_p', 1.0),
            'stream': False
        }

        max_tokens = self.config.get('max_tokens')
        if max_tokens is not None:
            request_params['max_tokens'] = max_tokens
        
        if validated_tools:
            request_params['tools'] = validated_tools
        
        logger.info(f"Creating Groq chat completion with model: {self.config['model']}")
        logger.debug(f"Request params: model={self.config['model']}, tools_count={len(validated_tools) if validated_tools else 0}")
        
        try:
            response = self.client.chat.completions.create(**request_params)
        except Exception as api_error:
            error_msg = str(api_error)
            error_type = type(api_error).__name__
            error_lower = error_msg.lower()
            
            if 'tool_use_failed' in error_lower or 'failed_generation' in error_lower:
                logger.debug(
                    "Groq returned tool_use_failed; attempting to recover from failed_generation payload."
                )
                failed_gen = self._extract_failed_generation(error_msg)
                if failed_gen:
                    logger.info("Attempting to parse text-format tool calls from failed_generation")
                    parsed_tool_calls = self._parse_text_format_tool_call(failed_gen)
                    if parsed_tool_calls:
                        logger.debug("Recovered %d tool call(s) from failed_generation.", len(parsed_tool_calls))
                        recovered_content = self._extract_text_without_function_calls(failed_gen)
                        return {
                            'choices': [{
                                'message': {
                                    'content': recovered_content,
                                    'tool_calls': parsed_tool_calls
                                }
                            }],
                            'usage': {
                                'prompt_tokens': 0,
                                'completion_tokens': 0,
                                'total_tokens': 0
                            }
                        }
                raise GroqAPIError(
                    f"Groq tool call format error. The model generated invalid tool calls. "
                    f"Try simplifying your request or using a different model. Details: {error_msg}"
                ) from api_error

            logger.error(f"Groq API call failed: {error_type}: {error_msg}")

            if 'rate_limit' in error_lower or 'rate limit' in error_lower:
                raise GroqAPIError(f"Groq rate limit exceeded. Please wait and try again: {error_msg}") from api_error
            elif 'invalid_api_key' in error_lower or 'authentication' in error_lower:
                raise GroqAPIError(f"Groq authentication failed. Check your API key: {error_msg}") from api_error
            elif 'context_length' in error_lower or 'token' in error_lower:
                raise GroqAPIError(f"Groq context length exceeded: {error_msg}") from api_error
            elif 'model' in error_lower and ('not found' in error_lower or 'unavailable' in error_lower):
                raise GroqAPIError(f"Groq model not available: {error_msg}") from api_error
            else:
                raise GroqAPIError(f"Groq API call failed ({error_type}): {error_msg}") from api_error
        
        try:
            choices = response.choices
            if not choices or len(choices) == 0:
                raise GroqAPIError("Groq returned empty choices array")
            
            first_choice = choices[0]
            message = first_choice.message
            
            content = message.content if message.content is not None else ""
            
            raw_tool_calls = message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else []
            tool_calls = self._serialize_tool_calls(raw_tool_calls)
            
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
                total_tokens = getattr(usage, 'total_tokens', 0) or 0
            
            result = {
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
            
            logger.debug(f"Groq response: content_len={len(content)}, tool_calls={len(tool_calls)}")
            return result
            
        except GroqAPIError:
            raise
        except Exception as extract_error:
            error_msg = str(extract_error)
            logger.error(f"Failed to extract Groq response data: {error_msg}")
            raise GroqAPIError(f"Failed to parse Groq response: {error_msg}") from extract_error
    
    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.config.get('model', self.DEFAULT_MODEL)
    
    def test_connection(self) -> bool:
        """
        Test the connection to Groq API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = self.create_chat_completion(test_messages)
            
            choices = response.get('choices', [])
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            is_valid = len(choices) > 0
            is_not_error = not str(content).startswith("Error:")
            return is_valid and is_not_error
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Groq.
        
        Returns:
            List of model names
        """
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile", 
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "gemma2-9b-it"
        ]
