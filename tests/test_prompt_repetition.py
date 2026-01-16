import copy
import json

import httpx

from autocapture.agents.llm_client import AgentLLMClient
from autocapture.config import AppConfig
from autocapture.llm.prompt_repetition import apply_prompt_repetition


def test_prompt_repetition_text_only_keeps_system_once() -> None:
    messages = [
        {"role": "system", "content": "sys-1"},
        {"role": "system", "content": "sys-2"},
        {"role": "user", "content": "question"},
        {"role": "user", "content": "context"},
    ]
    original = copy.deepcopy(messages)

    repeated = apply_prompt_repetition(messages, enabled=True)

    assert messages == original
    assert repeated[:2] == original[:2]
    assert repeated[2:] == original[2:] + original[2:]
    assert len(repeated) == 6


def test_prompt_repetition_openai_segments_drops_images() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "more"},
            ],
        },
    ]
    original = copy.deepcopy(messages)

    repeated = apply_prompt_repetition(messages, enabled=True)

    assert messages == original
    assert len(repeated) == 3
    assert any(segment.get("type") == "image_url" for segment in repeated[1]["content"])
    assert all(segment.get("type") != "image_url" for segment in repeated[2]["content"])
    assert [segment["text"] for segment in repeated[2]["content"]] == ["look", "more"]


def test_prompt_repetition_ollama_drops_images_in_repeat() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "describe", "images": ["img1"]},
    ]
    original = copy.deepcopy(messages)

    repeated = apply_prompt_repetition(messages, enabled=True)

    assert messages == original
    assert len(repeated) == 3
    assert "images" in repeated[1]
    assert "images" not in repeated[2]
    assert repeated[2]["content"] == "describe"


def test_agent_llm_client_applies_prompt_repetition_openai_compatible() -> None:
    config = AppConfig()
    config.llm.provider = "openai_compatible"
    config.llm.openai_compatible_base_url = "http://testserver"
    config.llm.prompt_repetition = True

    captured: dict[str, list[dict[str, str]]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured["messages"] = payload["messages"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://testserver") as client:
        llm_client = AgentLLMClient(config, http_client=client)
        response = llm_client.generate_text("sys", "user", "ctx")

    messages = captured["messages"]
    assert response.text == "ok"
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "user",
        "user",
    ]
    assert [message["content"] for message in messages[1:]] == [
        "user",
        "ctx",
        "user",
        "ctx",
    ]
