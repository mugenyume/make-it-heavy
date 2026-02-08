import json
import yaml
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Dict, Any
from agent import AIAgent

class TaskOrchestrator:
    def __init__(self, config_path="config.yaml", provider_name=None, silent=False):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_agents = self.config['orchestrator']['parallel_agents']
        self.task_timeout = self.config['orchestrator']['task_timeout']
        self.aggregation_strategy = self.config['orchestrator']['aggregation_strategy']
        self.silent = silent
        self.provider_name = provider_name
        
        # Initialize provider using factory
        if provider_name is None:
            provider_name = self.config.get('provider', {}).get('name', 'openrouter')
        
        # Get provider-specific configuration
        provider_config = self.config.get(provider_name, {})
        if not provider_config:
            raise ValueError(f"No configuration found for provider: {provider_name}")
        
        from providers import ProviderFactory
        self.provider = ProviderFactory.create_provider(provider_name, provider_config)
        
        # Track agent progress
        self.agent_progress = {}
        self.agent_results = {}
        self.progress_lock = threading.Lock()

    def _build_fallback_subtasks(self, user_input: str, num_agents: int) -> List[str]:
        """Create deterministic fallback research questions for any agent count."""
        templates = [
            "What are the most important background facts about {user_input}?",
            "What are the key insights and implications related to {user_input}?",
            "What alternative perspectives or counterarguments exist about {user_input}?",
            "Which claims about {user_input} should be verified and what evidence supports them?",
            "What practical takeaways or recommendations can be derived from {user_input}?",
            "What risks, limitations, or uncertainties are associated with {user_input}?"
        ]

        questions = []
        for index in range(num_agents):
            template = templates[index % len(templates)]
            if index >= len(templates):
                template = template.replace("?", f" (alternative angle {index + 1})?")
            questions.append(template.format(user_input=user_input))

        return questions

    def _normalize_generated_subtasks(
        self,
        generated: Any,
        user_input: str,
        num_agents: int
    ) -> List[str]:
        """Coerce generated subtasks into a clean list with the exact desired count."""
        if not isinstance(generated, list):
            return self._build_fallback_subtasks(user_input, num_agents)

        normalized = []
        for item in generated:
            task_text = str(item).strip()
            if task_text:
                normalized.append(task_text)
            if len(normalized) >= num_agents:
                return normalized[:num_agents]

        if len(normalized) < num_agents:
            fallback = self._build_fallback_subtasks(user_input, num_agents)
            for fallback_task in fallback:
                if len(normalized) >= num_agents:
                    break
                normalized.append(fallback_task)

        return normalized[:num_agents]
    
    def decompose_task(self, user_input: str, num_agents: int) -> List[str]:
        """Use AI to dynamically generate different questions based on user input"""
        
        # Create question generation agent
        question_agent = AIAgent(silent=True, provider_name=self.provider_name)
        
        # Get question generation prompt from config
        prompt_template = self.config['orchestrator']['question_generation_prompt']
        generation_prompt = prompt_template.format(
            user_input=user_input,
            num_agents=num_agents
        )
        
        # Keep original behavior: only disable completion signaling for this stage.
        question_agent.tools = [
            tool for tool in question_agent.tools
            if tool.get('function', {}).get('name') != 'mark_task_complete'
        ]
        question_agent.tool_mapping = {
            name: func for name, func in question_agent.tool_mapping.items()
            if name != 'mark_task_complete'
        }
        
        try:
            # Get AI-generated questions
            response = question_agent.run(generation_prompt)
            
            # Parse JSON response (also handle wrapped markdown blocks)
            cleaned_response = response.strip()
            try:
                questions = json.loads(cleaned_response)
            except json.JSONDecodeError:
                start = cleaned_response.find('[')
                end = cleaned_response.rfind(']')
                if start == -1 or end == -1 or start >= end:
                    raise
                questions = json.loads(cleaned_response[start:end + 1])
            
            return self._normalize_generated_subtasks(questions, user_input, num_agents)
            
        except Exception:
            # Fallback: create simple variations if AI fails
            return self._build_fallback_subtasks(user_input, num_agents)
    
    def update_agent_progress(self, agent_id: int, status: str, result: str = None):
        """Thread-safe progress tracking"""
        with self.progress_lock:
            self.agent_progress[agent_id] = status
            if result is not None:
                self.agent_results[agent_id] = result
    
    def run_agent_parallel(self, agent_id: int, subtask: str) -> Dict[str, Any]:
        """
        Run a single agent with the given subtask.
        Returns result dictionary with agent_id, status, and response.
        """
        try:
            self.update_agent_progress(agent_id, "PROCESSING...")
            
            # Use AIAgent with specified provider
            agent = AIAgent(silent=True, provider_name=self.provider_name)
            
            start_time = time.time()
            response = agent.run(subtask)
            execution_time = time.time() - start_time
            
            self.update_agent_progress(agent_id, "COMPLETED", response)
            
            return {
                "agent_id": agent_id,
                "status": "success", 
                "response": response,
                "execution_time": execution_time
            }
            
        except Exception as e:
            self.update_agent_progress(agent_id, f"FAILED: {str(e)}")
            # Simple error handling
            return {
                "agent_id": agent_id,
                "status": "error",
                "response": f"Error: {str(e)}",
                "execution_time": 0
            }
    
    def aggregate_results(self, agent_results: List[Dict[str, Any]]) -> str:
        """
        Combine results from all agents into a comprehensive final answer.
        Uses the configured aggregation strategy.
        """
        # LESS AGGRESSIVE FILTERING - Accept more responses as valid
        successful_results = []
        for r in agent_results:
            if r["status"] == "success" and r.get("response"):
                response = str(r["response"]).strip()
                # Only filter out obvious errors and completely empty responses
                if response and not response.startswith("Error:") and len(response) > 10:
                    successful_results.append(r)
        
        if not successful_results:
            # Check if any agents provided any response at all (even short ones)
            fallback_results = []
            for r in agent_results:
                if r["status"] == "success" and r.get("response"):
                    response = str(r["response"]).strip()
                    if response and not response.startswith("Error:"):
                        fallback_results.append(r)
            
            if fallback_results:
                # Use fallback results if we have any non-error responses
                successful_results = fallback_results
            else:
                return "All agents failed to provide meaningful results. Please try again with a different question."
        
        # Extract responses for aggregation
        responses = []
        for r in successful_results:
            response = str(r["response"]).strip()
            if response:
                responses.append(response)
        
        if not responses:
            return "The agents processed the request but did not provide meaningful responses."
        
        if self.aggregation_strategy == "consensus":
            return self._aggregate_consensus(responses, successful_results)
        else:
            # Default to consensus
            return self._aggregate_consensus(responses, successful_results)
    
    def _aggregate_consensus(self, responses: List[str], _results: List[Dict[str, Any]]) -> str:
        """
        Use one final AI call to synthesize all agent responses into a coherent answer.
        """
        # Filter out empty responses
        valid_responses = [r.strip() for r in responses if r and r.strip()]
        
        if not valid_responses:
            return "The agents processed the request but did not provide meaningful responses."
        
        if len(valid_responses) == 1:
            return valid_responses[0]
        
        # Create synthesis agent to combine all responses
        synthesis_agent = AIAgent(silent=True, provider_name=self.provider_name)
        
        # Build agent responses section
        agent_responses_text = ""
        for i, response in enumerate(valid_responses, 1):
            agent_responses_text += f"=== AGENT {i} RESPONSE ===\n{response}\n\n"
        
        # Get synthesis prompt from config and format it
        synthesis_prompt_template = self.config['orchestrator']['synthesis_prompt']
        synthesis_prompt = synthesis_prompt_template.format(
            num_responses=len(valid_responses),
            agent_responses=agent_responses_text
        )
        
        # Completely remove all tools from synthesis agent to force direct response
        synthesis_agent.tools = []
        synthesis_agent.tool_mapping = {}
        
        # Get the synthesized response
        try:
            final_answer = synthesis_agent.run(synthesis_prompt)
            if final_answer and final_answer.strip():
                return final_answer.strip()
            else:
                # If synthesis returns empty, fallback to concatenation
                combined = []
                for i, response in enumerate(valid_responses, 1):
                    combined.append(f"=== Agent {i} Response ===")
                    combined.append(response)
                    combined.append("")
                return "\n".join(combined).strip()
        except Exception as e:
            # Log the error for debugging
            print(f"\n🚨 SYNTHESIS FAILED: {str(e)}")
            print("📋 Falling back to concatenated responses\n")
            # Fallback: if synthesis fails, concatenate responses
            combined = []
            for i, response in enumerate(valid_responses, 1):
                combined.append(f"=== Agent {i} Response ===")
                combined.append(response)
                combined.append("")
            return "\n".join(combined).strip()
    
    def get_progress_status(self) -> Dict[int, str]:
        """Get current progress status for all agents"""
        with self.progress_lock:
            return self.agent_progress.copy()
    
    def orchestrate(self, user_input: str):
        """
        Main orchestration method.
        Takes user input, delegates to parallel agents, and returns aggregated result.
        """
        
        # Reset progress tracking
        self.agent_progress = {}
        self.agent_results = {}
        
        # Decompose task into subtasks
        subtasks = self.decompose_task(user_input, self.num_agents)
        
        # Initialize progress tracking
        for i in range(self.num_agents):
            self.agent_progress[i] = "QUEUED"
        
        # Execute agents in parallel
        agent_results = []
        
        executor = ThreadPoolExecutor(max_workers=self.num_agents)
        try:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(self.run_agent_parallel, i, subtasks[i]): i 
                for i in range(self.num_agents)
            }

            # Wait for completion up to global task timeout.
            done, not_done = wait(future_to_agent.keys(), timeout=self.task_timeout)

            # Collect completed futures.
            for future in done:
                try:
                    result = future.result()
                    agent_results.append(result)
                except Exception as e:
                    agent_id = future_to_agent[future]
                    self.update_agent_progress(agent_id, f"FAILED: {str(e)}")
                    agent_results.append({
                        "agent_id": agent_id,
                        "status": "error",
                        "response": f"Agent {agent_id + 1} failed: {str(e)}",
                        "execution_time": 0
                    })

            # Mark timed-out futures without failing the entire orchestration.
            for future in not_done:
                agent_id = future_to_agent[future]
                future.cancel()
                self.update_agent_progress(agent_id, "TIMEOUT")
                agent_results.append({
                    "agent_id": agent_id,
                    "status": "timeout",
                    "response": f"Agent {agent_id + 1} timed out after {self.task_timeout} seconds.",
                    "execution_time": self.task_timeout
                })
        finally:
            # Do not block on pending worker threads after timeout.
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python 3.8 compatibility (cancel_futures unavailable).
                executor.shutdown(wait=False)
        
        # Sort results by agent_id for consistent output
        agent_results.sort(key=lambda x: x["agent_id"])
        
        # Aggregate results
        final_result = self.aggregate_results(agent_results)
        
        return final_result
