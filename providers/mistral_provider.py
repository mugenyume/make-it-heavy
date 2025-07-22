"""
MistralAI provider implementation.
"""

from openai import OpenAI
from typing import Dict, List, Any
from .base_provider import BaseProvider

class MistralProvider(BaseProvider):
    """MistralAI provider using OpenAI-compatible API."""
    
    DISPLAY_NAME = "MistralAI"
    DESCRIPTION = "MistralAI - Advanced language models with function calling"
    DEFAULT_MODEL = "mistral-large-latest"
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = OpenAI(
            base_url=self.config['base_url'],
            api_key=self.config['api_key']
        )
    
    def _validate_config(self):
        """Validate MistralAI configuration."""
        required_keys = ['api_key', 'base_url', 'model']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required MistralAI config: {key}")
    
    def create_chat_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a chat completion using MistralAI API.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            
        Returns:
            Dictionary containing the completion response
        """
        try:
            # Fix message ordering for MistralAI compatibility
            # Mistral requires last message to be user or tool (not assistant)
            fixed_messages = self._fix_message_ordering(messages)
            
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=fixed_messages,
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
            raise Exception(f"MistralAI API call failed: {str(e)}")
    
    def _fix_message_ordering(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fix message ordering for MistralAI API compatibility.
        
        MistralAI requires that the last message in the conversation
        must be from the user or a tool, not from the assistant.
        
        Args:
            messages: Original list of messages
            
        Returns:
            Fixed list of messages with proper ordering
        """
        if not messages:
            return messages
            
        # Check if last message is from assistant
        last_message = messages[-1]
        if last_message.get('role') == 'assistant':
            # Create a copy of messages and add a dummy user message
            fixed_messages = messages.copy()
            fixed_messages.append({
                "role": "user",
                "content": "Continue."
            })
            return fixed_messages
        
        return messages
    
    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.config.get('model', self.DEFAULT_MODEL)