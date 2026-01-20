from __future__ import annotations

import pytest

from autocapture.config import LLMConfig, PrivacyConfig, ProviderRoutingConfig
from autocapture.memory.router import ProviderRouter


def test_openai_blocks_when_offline() -> None:
    routing = ProviderRoutingConfig(llm="openai")
    llm_config = LLMConfig(openai_api_key="test")
    privacy = PrivacyConfig(cloud_enabled=True)
    router = ProviderRouter(routing, llm_config, offline=True, privacy=privacy)
    with pytest.raises(RuntimeError):
        router.select_llm()


def test_openai_compatible_blocks_cloud_when_disabled() -> None:
    routing = ProviderRoutingConfig(llm="openai_compatible")
    llm_config = LLMConfig(
        openai_compatible_base_url="https://example.com",
        openai_compatible_api_key="test",
    )
    privacy = PrivacyConfig(cloud_enabled=False)
    router = ProviderRouter(routing, llm_config, offline=False, privacy=privacy)
    with pytest.raises(RuntimeError):
        router.select_llm()


def test_openai_compatible_allows_local_when_offline() -> None:
    routing = ProviderRoutingConfig(llm="openai_compatible")
    llm_config = LLMConfig(openai_compatible_base_url="http://127.0.0.1:8000")
    privacy = PrivacyConfig(cloud_enabled=False)
    router = ProviderRouter(routing, llm_config, offline=True, privacy=privacy)
    provider, decision = router.select_llm()
    assert provider is not None
    assert decision.llm_provider == "openai_compatible"
