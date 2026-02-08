import json

import agent as agent_module


class FakeTool:
    def __init__(self, name: str):
        self.name = name

    def to_openrouter_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, **kwargs):
        return {"ok": True, "tool": self.name, "args": kwargs}


class SequencedProvider:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def create_chat_completion(self, messages, tools=None):
        index = self.calls if self.calls < len(self.responses) else len(self.responses) - 1
        self.calls += 1
        return self.responses[index]

    def get_provider_info(self):
        return {"display_name": "Fake", "model": "fake-model"}


def assistant_response(content="", tool_calls=None):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": tool_calls or [],
                }
            }
        ]
    }


def completion_tool_call(call_id="call_complete"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "mark_task_complete",
            "arguments": json.dumps(
                {
                    "task_summary": "done",
                    "completion_message": "short completion payload",
                }
            ),
        },
    }


def build_agent(monkeypatch, responses):
    provider = SequencedProvider(responses)

    monkeypatch.setattr(
        agent_module.ProviderFactory,
        "create_provider",
        classmethod(lambda cls, provider_name, config: provider),
    )

    monkeypatch.setattr(
        agent_module,
        "discover_tools",
        lambda config, silent=False: {
            "mark_task_complete": FakeTool("mark_task_complete"),
            "search_web": FakeTool("search_web"),
        },
    )

    return agent_module.AIAgent(provider_name="openrouter", silent=True), provider


def test_mark_task_complete_returns_accumulated_content(monkeypatch):
    responses = [
        assistant_response("First research finding."),
        assistant_response("Second research finding.", [completion_tool_call()]),
    ]

    ai_agent, provider = build_agent(monkeypatch, responses)
    result = ai_agent.run("research topic")

    assert result == "First research finding.\n\nSecond research finding."
    assert provider.calls == 2


def test_premature_completion_is_ignored(monkeypatch):
    responses = [
        assistant_response("", [completion_tool_call("call_early")]),
        assistant_response("Useful analysis chunk."),
        assistant_response("", [completion_tool_call("call_final")]),
    ]

    ai_agent, provider = build_agent(monkeypatch, responses)
    result = ai_agent.run("research topic")

    assert result == "Useful analysis chunk."
    assert provider.calls == 3


def test_no_tool_responses_finalize_after_no_tool_streak(monkeypatch):
    responses = [
        assistant_response("Draft analysis part one."),
        assistant_response("Draft analysis part two."),
        assistant_response("", [completion_tool_call()]),
    ]

    ai_agent, provider = build_agent(monkeypatch, responses)
    result = ai_agent.run("research topic")

    assert result == "Draft analysis part one.\n\nDraft analysis part two."
    assert provider.calls == 2
