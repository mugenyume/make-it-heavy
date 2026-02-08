from providers.groq_provider import GroqProvider


def test_parse_text_format_tool_call_supports_angle_bracket_variant():
    provider = GroqProvider.__new__(GroqProvider)
    payload = (
        "Context before tool call.\n"
        "<function=mark_task_complete>{\"task_summary\":\"done\",\"completion_message\":\"ok\"}</function>"
    )

    tool_calls = provider._parse_text_format_tool_call(payload)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "mark_task_complete"
    assert '"task_summary": "done"' in tool_calls[0]["function"]["arguments"] or '"task_summary":"done"' in tool_calls[0]["function"]["arguments"]


def test_extract_text_without_function_calls_preserves_plain_answer():
    provider = GroqProvider.__new__(GroqProvider)
    payload = (
        "Long answer body with useful details.\n\n"
        "<function=mark_task_complete>{\"task_summary\":\"done\",\"completion_message\":\"ok\"}</function>"
    )

    cleaned = provider._extract_text_without_function_calls(payload)
    assert "Long answer body with useful details." in cleaned
    assert "<function=" not in cleaned
