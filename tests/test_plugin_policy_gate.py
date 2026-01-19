from __future__ import annotations

import pytest

from autocapture.config import AppConfig, ModelStageConfig
from autocapture.plugins.errors import PluginPolicyError
from autocapture.plugins.policy import PolicyGate


def test_guard_cloud_text_blocks_offline() -> None:
    config = AppConfig()
    config.offline = True
    config.privacy.cloud_enabled = True
    stage = ModelStageConfig(allow_cloud=True)
    gate = PolicyGate(config)
    with pytest.raises(PluginPolicyError):
        gate.guard_cloud_text(
            stage="final_answer",
            stage_config=stage,
            provider="openai",
            base_url=None,
            cloud=True,
        )


def test_guard_cloud_images_blocks_without_permission() -> None:
    config = AppConfig()
    config.offline = False
    config.privacy.cloud_enabled = True
    config.privacy.allow_cloud_images = False
    gate = PolicyGate(config)
    with pytest.raises(PluginPolicyError):
        gate.guard_cloud_images(
            provider="openai",
            base_url="https://api.openai.com",
            allow_cloud=True,
        )


def test_guard_cloud_images_allows_local_endpoint() -> None:
    config = AppConfig()
    gate = PolicyGate(config)
    gate.guard_cloud_images(
        provider="ollama",
        base_url="http://127.0.0.1:11434",
        allow_cloud=False,
    )
