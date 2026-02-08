import orchestrator as orchestrator_module
import threading


def test_fallback_subtasks_match_requested_count_and_question_style():
    orchestrator = orchestrator_module.TaskOrchestrator.__new__(orchestrator_module.TaskOrchestrator)

    for agent_count in (1, 4, 6):
        questions = orchestrator._build_fallback_subtasks("AI policy", agent_count)
        assert len(questions) == agent_count
        assert all(question.strip().endswith("?") for question in questions)


class FakeQuestionAgent:
    instances = []

    def __init__(self, silent=True, provider_name=None):
        self.tools = [
            {"function": {"name": "mark_task_complete"}},
            {"function": {"name": "search_web"}},
        ]
        self.tool_mapping = {
            "mark_task_complete": lambda **kwargs: {"ok": True},
            "search_web": lambda **kwargs: {"ok": True},
        }
        FakeQuestionAgent.instances.append(self)

    def run(self, prompt):
        return '["Question A?", "Question B?"]'


def test_question_generation_disables_only_completion_tool(monkeypatch):
    FakeQuestionAgent.instances = []
    monkeypatch.setattr(orchestrator_module, "AIAgent", FakeQuestionAgent)

    orchestrator = orchestrator_module.TaskOrchestrator.__new__(orchestrator_module.TaskOrchestrator)
    orchestrator.provider_name = "openrouter"
    orchestrator.config = {
        "orchestrator": {
            "question_generation_prompt": "Prompt for {user_input} and {num_agents}",
        }
    }

    questions = orchestrator.decompose_task("topic", 2)

    assert questions == ["Question A?", "Question B?"]
    question_agent = FakeQuestionAgent.instances[0]
    tool_names = [tool.get("function", {}).get("name") for tool in question_agent.tools]
    assert "mark_task_complete" not in tool_names
    assert "search_web" in tool_names
    assert "mark_task_complete" not in question_agent.tool_mapping
    assert "search_web" in question_agent.tool_mapping


class FlakyAgent:
    calls = 0

    def __init__(self, silent=True, provider_name=None):
        pass

    def run(self, subtask):
        FlakyAgent.calls += 1
        if FlakyAgent.calls == 1:
            raise RuntimeError("rate limit exceeded")
        return f"Recovered response for: {subtask}"


def test_run_agent_parallel_retries_transient_failures(monkeypatch):
    FlakyAgent.calls = 0
    monkeypatch.setattr(orchestrator_module, "AIAgent", FlakyAgent)

    orchestrator = orchestrator_module.TaskOrchestrator.__new__(orchestrator_module.TaskOrchestrator)
    orchestrator.provider_name = "groq"
    orchestrator.agent_retry_attempts = 2
    orchestrator.agent_retry_backoff_seconds = 0
    orchestrator.progress_lock = threading.Lock()
    orchestrator.agent_progress = {}
    orchestrator.agent_results = {}

    result = orchestrator.run_agent_parallel(0, "subtask")

    assert result["status"] == "success"
    assert "Recovered response" in result["response"]
    assert FlakyAgent.calls == 2


def test_aggregate_results_shows_failure_reasons():
    orchestrator = orchestrator_module.TaskOrchestrator.__new__(orchestrator_module.TaskOrchestrator)
    orchestrator.aggregation_strategy = "consensus"
    orchestrator.provider_name = "openrouter"
    orchestrator.config = {"provider": {"name": "openrouter"}}

    results = [
        {"agent_id": 0, "status": "error", "response": "Error: rate limit exceeded", "execution_time": 0},
        {"agent_id": 1, "status": "error", "response": "Error: connection timeout", "execution_time": 0},
    ]

    message = orchestrator.aggregate_results(results)
    assert "All agents failed to provide meaningful results." in message
    assert "Top failure reasons" in message


def test_aggregate_results_auth_failure_message_for_groq():
    orchestrator = orchestrator_module.TaskOrchestrator.__new__(orchestrator_module.TaskOrchestrator)
    orchestrator.aggregation_strategy = "consensus"
    orchestrator.provider_name = "groq"
    orchestrator.config = {"provider": {"name": "groq"}}

    results = [
        {
            "agent_id": 0,
            "status": "error",
            "response": "Error: LLM call failed (GroqAPIError): Groq authentication failed. Check your API key: Error code: 401 - {'error': {'message': 'Invalid API Key', 'code': 'invalid_api_key'}}",
            "execution_time": 0,
        }
    ]

    message = orchestrator.aggregate_results(results)
    assert "Authentication failed for provider 'groq'" in message
    assert "starts with 'gsk_'" in message
