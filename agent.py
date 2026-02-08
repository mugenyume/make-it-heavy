import json
import yaml
from difflib import SequenceMatcher
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

    def _finalize_response_content(self, content_blocks: List[str]) -> str:
        """Deduplicate highly similar blocks and return a clean final response."""
        if not content_blocks:
            return ""

        deduplicated_blocks: List[str] = []
        normalized_blocks: List[str] = []

        for block in content_blocks:
            stripped = block.strip()
            if not stripped:
                continue

            normalized = " ".join(stripped.lower().split())
            is_duplicate = False

            for existing_normalized in normalized_blocks:
                if normalized == existing_normalized:
                    is_duplicate = True
                    break

                if len(normalized) >= 100 and len(existing_normalized) >= 100:
                    similarity = SequenceMatcher(None, normalized, existing_normalized).ratio()
                    if similarity >= 0.94:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated_blocks.append(stripped)
                normalized_blocks.append(normalized)

        return "\n\n".join(deduplicated_blocks)

    def call_llm(self, messages, include_tools: bool = True):
        """Make API call to the configured provider, optionally with tools."""
        try:
            response = self.provider.create_chat_completion(
                messages=messages,
                tools=self.tools if include_tools and self.tools else None
            )
            return response
        except ImportError as e:
            raise Exception(f"Provider dependency missing: {str(e)}. Install required packages.")
        except ValueError as e:
            raise Exception(f"Invalid request: {str(e)}")
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            raise Exception(f"LLM call failed ({error_type}): {error_msg}")
    
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
        full_response_content: List[str] = []
        non_completion_tool_calls = 0
        no_tool_streak = 0
        completion_message_fallback: Optional[str] = None
        finalize_after_no_tool_streak = int(
            self.config.get('agent', {}).get('finalize_after_no_tool_streak', 2)
        )
        
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
                no_tool_streak = 0
                if not self.silent:
                    print(f"🔧 Agent making {len(tool_calls)} tool call(s)")
                
                task_completed = False
                for tool_call in tool_calls:
                    tool_name = tool_call['function']['name']
                    
                    if not self.silent:
                        print(f"   📞 Calling tool: {tool_name}")
                    
                    # Special handling for mark_task_complete to extract completion message
                    if tool_name == "mark_task_complete":
                        # Only allow task completion after some useful work happened.
                        has_meaningful_content = len(full_response_content) > 0 or non_completion_tool_calls > 0
                        
                        if not has_meaningful_content:
                            if not self.silent:
                                print("⚠️ Task completion called too early - continuing work")
                            continue
                        
                        # Execute completion tool and end loop with accumulated assistant content.
                        tool_result = self.handle_tool_call(tool_call)
                        messages.append(tool_result)
                        try:
                            tool_args = self._parse_tool_arguments(tool_call['function'].get('arguments'))
                        except Exception:
                            tool_args = {}
                        completion_message = tool_args.get('completion_message')
                        if completion_message is not None:
                            if not isinstance(completion_message, str):
                                completion_message = json.dumps(completion_message, default=str)
                            completion_message = completion_message.strip()
                            if completion_message:
                                completion_message_fallback = completion_message
                        task_completed = True
                        
                        if not self.silent:
                            print("✅ Task completion tool called - exiting loop")
                        continue
                    
                    # Handle other tools normally
                    tool_result = self.handle_tool_call(tool_call)
                    messages.append(tool_result)
                    non_completion_tool_calls += 1
                
                if task_completed:
                    finalized = self._finalize_response_content(full_response_content)
                    if finalized:
                        return finalized
                    if completion_message_fallback:
                        return completion_message_fallback
                    return "Task completed successfully."
            else:
                no_tool_streak += 1
                if not self.silent:
                    print("💭 Agent responded without tool calls - continuing loop")

                # If the model repeatedly responds directly without tools, finalize gracefully.
                if no_tool_streak >= max(1, finalize_after_no_tool_streak):
                    finalized = self._finalize_response_content(full_response_content)
                    if finalized:
                        return finalized
        
        # If max iterations reached, return whatever content we gathered
        if full_response_content:
            string_content = [str(item) if not isinstance(item, str) else item for item in full_response_content]
            return self._finalize_response_content(string_content)
        elif completion_message_fallback:
            return completion_message_fallback
        else:
            return "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
