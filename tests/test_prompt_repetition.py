import json

import httpx

from autocapture.agents.llm_client import AgentLLMClient
from autocapture.config import AppConfig
from autocapture.llm.prompt_strategy import (
    PromptStrategy,
    PromptStrategySettings,
    apply_prompt_strategy,
)


def _settings() -> PromptStrategySettings:
    config = AppConfig()
    config.llm.strategy_auto_mode = False
    config.llm.prompt_strategy_default = "repeat_2x"
    return PromptStrategySettings.from_llm_config(config.llm, data_dir=None)


def test_prompt_strategy_repeats_last_user_only() -> None:
    messages = [
        {"role": "system", "content": "sys-1"},
        {"role": "system", "content": "sys-2"},
        {"role": "user", "content": "question"},
        {"role": "user", "content": "context"},
    ]
    result = apply_prompt_strategy(
        messages,
        _settings(),
        override_strategy=PromptStrategy.REPEAT_2X,
        task_type="test",
    )

    assert [msg["role"] for msg in result.messages] == ["system", "system", "user", "user"]
    assert result.messages[-1]["content"] == "context\n\n---\n\ncontext"
    assert result.messages[2]["content"] == "question"


def test_prompt_strategy_appends_text_only_for_images() -> None:
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
    result = apply_prompt_strategy(
        messages,
        _settings(),
        override_strategy=PromptStrategy.REPEAT_2X,
        task_type="test",
    )

    repeated = result.messages[-1]["content"]
    assert any(segment.get("type") == "image_url" for segment in repeated)
    assert repeated[-1]["text"] == "\n\n---\n\nlookmore"


def test_prompt_strategy_inserts_step_by_step_phrase() -> None:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q"}]
    config = AppConfig()
    config.llm.enable_step_by_step = True
    config.llm.strategy_auto_mode = False
    settings = PromptStrategySettings.from_llm_config(config.llm, data_dir=None)
    result = apply_prompt_strategy(
        messages,
        settings,
        override_strategy=PromptStrategy.STEP_BY_STEP,
        step_by_step_requested=True,
        task_type="test",
    )
    assert result.messages[-1]["content"].endswith("Let's think step by step.")


def test_prompt_strategy_degrades_when_prompt_too_large() -> None:
    config = AppConfig()
    config.llm.strategy_auto_mode = False
    config.llm.prompt_strategy_default = "repeat_3x"
    config.llm.max_prompt_chars_for_repetition = 10
    settings = PromptStrategySettings.from_llm_config(config.llm, data_dir=None)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "long text"}]
    result = apply_prompt_strategy(
        messages,
        settings,
        override_strategy=PromptStrategy.REPEAT_3X,
        task_type="test",
    )
    assert result.metadata.safe_mode_degraded is True
    assert result.metadata.strategy == PromptStrategy.BASELINE


def test_agent_llm_client_applies_prompt_strategy_openai_compatible() -> None:
    config = AppConfig()
    config.llm.provider = "openai_compatible"
    config.llm.openai_compatible_base_url = "http://testserver"
    config.llm.strategy_auto_mode = False
    config.llm.prompt_strategy_default = "repeat_2x"

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
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"].startswith("user\n\nctx")
    assert messages[-1]["content"].endswith("\n\n---\n\nuser\n\nctx")
