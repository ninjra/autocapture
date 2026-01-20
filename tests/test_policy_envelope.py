from __future__ import annotations

import pytest

from autocapture.config import AppConfig
from autocapture.llm.providers import LLMProvider
from autocapture.policy import PolicyEnvelope


class DummyProvider(LLMProvider):
    def __init__(self, base_url: str | None) -> None:
        self._base_url = base_url

    async def generate_answer(
        self,
        system_prompt: str,
        query: str,
        context_pack_text: str,
        *,
        temperature: float | None = None,
        priority: str = "foreground",
    ) -> str:
        _ = (system_prompt, query, context_pack_text, temperature, priority)
        return "ok"


@pytest.mark.anyio
async def test_policy_envelope_allows_local_when_offline() -> None:
    config = AppConfig()
    config.offline = True
    config.privacy.cloud_enabled = False
    policy = PolicyEnvelope(config)
    provider = DummyProvider("http://127.0.0.1:8000")
    result = await policy.execute_stage(
        stage=None,
        provider=provider,
        decision=None,
        system_prompt="s",
        user_prompt="u",
        context_pack_text="c",
    )
    assert result == "ok"


@pytest.mark.anyio
async def test_policy_envelope_blocks_cloud_when_offline() -> None:
    config = AppConfig()
    config.offline = True
    config.privacy.cloud_enabled = True
    policy = PolicyEnvelope(config)
    provider = DummyProvider("https://example.com")
    with pytest.raises(RuntimeError):
        await policy.execute_stage(
            stage=None,
            provider=provider,
            decision=None,
            system_prompt="s",
            user_prompt="u",
            context_pack_text="c",
        )


@pytest.mark.anyio
async def test_policy_envelope_blocks_cloud_without_privacy_opt_in() -> None:
    config = AppConfig()
    config.offline = False
    config.privacy.cloud_enabled = False
    policy = PolicyEnvelope(config)
    provider = DummyProvider("https://example.com")
    with pytest.raises(RuntimeError):
        await policy.execute_stage(
            stage=None,
            provider=provider,
            decision=None,
            system_prompt="s",
            user_prompt="u",
            context_pack_text="c",
        )


def test_policy_envelope_blocks_cloud_images_without_opt_in() -> None:
    config = AppConfig()
    config.offline = False
    config.privacy.cloud_enabled = True
    config.privacy.allow_cloud_images = False
    policy = PolicyEnvelope(config)
    with pytest.raises(RuntimeError):
        policy.execute_vision_sync(
            stage=None,
            call=lambda: "ok",
            cloud=True,
        )
