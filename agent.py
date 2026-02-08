import json
import yaml
from typing import Any, Dict, List, Optional
from providers import ProviderFactory
from tools import discover_tools

class AIAgent:
    """AI Agent that works with any provider through the provider abstraction."""
    
    def __init__(self, config_path="config.yaml", provider_name=None, silent=False):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Silent mode for orchestrator (suppresses debug output)
        self.silent = silent
        
        # Determine provider name
        if provider_name is None:
            provider_name = self.config.get('provider', {}).get('name', 'openrouter')
        
        # Get provider-specific configuration
        provider_config = self.config.get(provider_name, {})
        if not provider_config:
            raise ValueError(f"No configuration found for provider: {provider_name}")
        
        # Initialize provider using factory
        self.provider = ProviderFactory.create_provider(provider_name, provider_config)
        
        # Discover tools dynamically
        self.discovered_tools = discover_tools(self.config, silent=self.silent)
        
        # Build tools array for the provider
        self.tools = [tool.to_openrouter_schema() for tool in self.discovered_tools.values()]
        
        # Build tool mapping
        self.tool_mapping = {name: tool.execute for name, tool in self.discovered_tools.items()}
        
        # Store provider info for display
        self.provider_info = self.provider.get_provider_info()
        
        if not self.silent:
            print(f"🤖 AI Agent initialized with {self.provider_info['display_name']} ({self.provider_info['model']})")

    def _normalize_tool_call(self, tool_call: Any) -> Optional[Dict[str, Any]]:
        """Normalize provider-specific tool call formats into a plain dict."""
        if tool_call is None:
            return None

        if isinstance(tool_call, dict):
            call_id = tool_call.get('id')
            function_data = tool_call.get('function', {}) or {}
            name = function_data.get('name')
            arguments = function_data.get('arguments', "{}")
        else:
            call_id = getattr(tool_call, 'id', None)
            function_obj = getattr(tool_call, 'function', None)
            name = getattr(function_obj, 'name', None) if function_obj else None
            arguments = getattr(function_obj, 'arguments', "{}") if function_obj else "{}"

        if not name:
            return None

        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments)
        elif arguments is None:
            arguments = "{}"
        elif not isinstance(arguments, str):
            arguments = str(arguments)

        return {
            "id": call_id or f"call_{id(tool_call)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments
            }
        }

    def _normalize_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Normalize a list of tool calls and drop malformed entries."""
        if not tool_calls:
            return []

        normalized = []
        for tool_call in tool_calls:
            normalized_call = self._normalize_tool_call(tool_call)
            if normalized_call:
                normalized.append(normalized_call)
        return normalized

    def _parse_tool_arguments(self, raw_arguments: Any) -> Dict[str, Any]:
        """Parse tool call arguments safely and always return a dict."""
        if raw_arguments is None:
            return {}

        if isinstance(raw_arguments, dict):
            return raw_arguments

        if isinstance(raw_arguments, str):
            raw_arguments = raw_arguments.strip()
            if not raw_arguments:
                return {}
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON arguments: {exc}") from exc

            if isinstance(parsed, dict):
                return parsed
            raise ValueError("Tool arguments JSON must decode to an object")

        raise ValueError("Unsupported tool arguments format")

    def call_llm(self, messages):
        """Make API call to the configured provider with tools"""
        try:
            response = self.provider.create_chat_completion(
                messages=messages,
                tools=self.tools if self.tools else None
            )
            return response
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def handle_tool_call(self, tool_call):
        """Handle a tool call and return the result message"""
        try:
            normalized_tool_call = self._normalize_tool_call(tool_call)
            if not normalized_tool_call:
                raise ValueError("Malformed tool call payload")

            tool_name = normalized_tool_call['function']['name']
            tool_args = self._parse_tool_arguments(normalized_tool_call['function'].get('arguments'))
            tool_call_id = normalized_tool_call['id']
            
            # Call appropriate tool from tool_mapping
            if tool_name in self.tool_mapping:
                tool_result = self.tool_mapping[tool_name](**tool_args)
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            
            # Return tool result message
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": json.dumps(tool_result, default=str)
            }
        
        except Exception as e:
            # Handle error case with null-safe extraction
            try:
                normalized_tool_call = self._normalize_tool_call(tool_call) if tool_call else None
                tool_call_id = normalized_tool_call.get('id', 'unknown') if normalized_tool_call else 'unknown'
                tool_name = normalized_tool_call.get('function', {}).get('name', 'unknown') if normalized_tool_call else 'unknown'
            except Exception:
                tool_call_id = 'unknown'
                tool_name = 'unknown'
            
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": json.dumps({"error": f"Tool execution failed: {str(e)}"})
            }
    
    def run(self, user_input: str):
        """Run the agent with user input and return FULL conversation content"""
        # Initialize messages with system prompt and user input
        messages = [
            {
                "role": "system",
                "content": self.config['system_prompt']
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        # Track all assistant responses for full content capture
        full_response_content = []
        
        # Implement agentic loop
        max_iterations = self.config.get('agent', {}).get('max_iterations', 10)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            if not self.silent:
                print(f"🔄 Agent iteration {iteration}/{max_iterations}")
            
            # Call LLM
            response = self.call_llm(messages)
            
            # Add the response to messages
            assistant_message = (
                response.get('choices', [{}])[0].get('message', {})
                if isinstance(response, dict) else {}
            )

            raw_content = assistant_message.get('content', '')
            if raw_content is None:
                content = ''
            elif isinstance(raw_content, (list, dict)):
                content = json.dumps(raw_content, default=str)
            elif isinstance(raw_content, str):
                content = raw_content
            else:
                content = str(raw_content)

            tool_calls = self._normalize_tool_calls(assistant_message.get('tool_calls'))

            assistant_message_payload = {"role": "assistant"}
            # Keep content present for assistant messages to maximize provider compatibility.
            assistant_message_payload["content"] = content
            if tool_calls:
                assistant_message_payload["tool_calls"] = tool_calls

            messages.append(assistant_message_payload)
            
            # Capture assistant content for full response
            # Only add non-empty content
            if content and content.strip():
                full_response_content.append(content.strip())
            
            # Check if there are tool calls
            if tool_calls:
                if not self.silent:
                    print(f"🔧 Agent making {len(tool_calls)} tool call(s)")
                
                # Check if this is a direct completion tool call
                task_completed = False
                for tool_call in tool_calls:
                    tool_name = tool_call['function']['name']
                    
                    if not self.silent:
                        print(f"   📞 Calling tool: {tool_name}")
                    
                    # Special handling for mark_task_complete to extract completion message
                    if tool_name == "mark_task_complete":
                        task_completed = True
                        try:
                            tool_args = self._parse_tool_arguments(tool_call['function'].get('arguments'))
                        except Exception:
                            tool_args = {}
                        completion_message = tool_args.get('completion_message', 'Task completed successfully.')
                        if not isinstance(completion_message, str):
                            completion_message = json.dumps(completion_message, default=str)
                        
                        if not self.silent:
                            print("✅ Task completion tool called - exiting loop")
                        
                        # Return the actual completion message instead of generic text
                        return completion_message
                    
                    # Handle other tools normally
                    tool_result = self.handle_tool_call(tool_call)
                    messages.append(tool_result)
                
                # If task was completed, we already returned above
                if task_completed:
                    return "Task completed successfully."
            else:
                # IMPROVED EMPTY RESPONSE HANDLING
                # If no tool calls, check if we have meaningful content
                if content and content.strip():
                    # We have content, this is likely a final response
                    return content.strip()
                elif full_response_content:
                    # We have accumulated content from previous iterations
                    string_content = [str(item) if not isinstance(item, str) else item for item in full_response_content]
                    return "\n\n".join(string_content)
                else:
                    # No content and no tool calls - this could be an empty response
                    # Instead of continuing indefinitely, provide a meaningful fallback
                    if not self.silent:
                        print("⚠️ Agent provided empty response without tool calls - ending loop")
                    
                    # Return a helpful message instead of getting stuck
                    return "I apologize, but I wasn't able to generate a meaningful response to your request. Please try rephrasing your question or providing more specific details."
        
        # If max iterations reached, return whatever content we gathered
        if full_response_content:
            string_content = [str(item) if not isinstance(item, str) else item for item in full_response_content]
            return "\n\n".join(string_content)
        else:
            return "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
