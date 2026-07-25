"""Regression coverage for multimodal interim assistant content."""

from run_agent import AIAgent


def _agent() -> AIAgent:
    return object.__new__(AIAgent)


def test_interim_visible_text_accepts_openai_multimodal_content_blocks():
    agent = _agent()
    message = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I inspected the screenshot."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "output_text", "text": "The layout is editorial."},
        ],
    }

    assert agent._interim_assistant_visible_text(message) == (
        "I inspected the screenshot.\nThe layout is editorial."
    )


def test_interim_visible_text_accepts_raw_strings_and_nested_text_values():
    agent = _agent()
    message = {
        "role": "assistant",
        "content": [
            "First",
            {"type": "text", "text": {"value": "Second"}},
            None,
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
        ],
    }

    assert agent._interim_assistant_visible_text(message) == "First\nSecond"


def test_interim_visible_text_ignores_image_only_and_malformed_blocks():
    agent = _agent()
    message = {
        "role": "assistant",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            {"type": "text", "text": ["not", "text"]},
            17,
        ],
    }

    assert agent._interim_assistant_visible_text(message) == ""


def test_interim_visible_text_still_strips_think_blocks_from_text_parts():
    agent = _agent()
    message = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "<think>private</think>Visible"},
        ],
    }

    assert agent._interim_assistant_visible_text(message) == "Visible"
