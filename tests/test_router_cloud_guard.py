from __future__ import annotations

import pytest

from autocapture.config import LLMConfig, PrivacyConfig, ProviderRoutingConfig
from autocapture.memory.router import ProviderRouter


def test_router_blocks_openai_without_cloud_enabled() -> None:
    routing = ProviderRoutingConfig(llm="openai")
    llm_config = LLMConfig(openai_api_key="test-key")
    privacy = PrivacyConfig(cloud_enabled=False)
    router = ProviderRouter(routing, llm_config, offline=False, privacy=privacy)
    with pytest.raises(RuntimeError):
        router.select_llm()
