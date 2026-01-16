from __future__ import annotations

import pytest

from autocapture.config import AppConfig, LLMConfig, PrivacyConfig, ProviderRoutingConfig
from autocapture.memory.router import ProviderRouter
from autocapture.model_ops import StageRouter


def test_router_blocks_openai_without_cloud_enabled() -> None:
    routing = ProviderRoutingConfig(llm="openai")
    llm_config = LLMConfig(openai_api_key="test-key")
    privacy = PrivacyConfig(cloud_enabled=False)
    router = ProviderRouter(routing, llm_config, offline=False, privacy=privacy)
    with pytest.raises(RuntimeError):
        router.select_llm()


def test_stage_router_blocks_cloud_without_opt_in() -> None:
    config = AppConfig()
    config.model_stages.final_answer.provider = "openai"
    config.model_stages.final_answer.allow_cloud = False
    config.llm.openai_api_key = "test-key"
    config.offline = False
    config.privacy.cloud_enabled = True
    router = StageRouter(config)
    with pytest.raises(RuntimeError):
        router.select_llm("final_answer")


def test_stage_router_blocks_remote_openai_compatible_without_opt_in() -> None:
    config = AppConfig()
    config.model_stages.final_answer.provider = "openai_compatible"
    config.model_stages.final_answer.base_url = "https://example.com"
    config.model_stages.final_answer.allow_cloud = False
    config.offline = False
    config.privacy.cloud_enabled = True
    router = StageRouter(config)
    with pytest.raises(RuntimeError):
        router.select_llm("final_answer")
