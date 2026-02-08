import orchestrator as orchestrator_module


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
