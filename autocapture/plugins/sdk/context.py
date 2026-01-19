"""Plugin SDK context objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import AppConfig, ModelStageConfig
from ...llm.governor import LLMGovernor
from ...llm.prompt_strategy import PromptStrategySettings
from ...runtime_governor import RuntimeGovernor
from ...security.secret_store import SecretStore
from ..policy import PolicyGate


@dataclass(frozen=True)
class LLMProviderInfo:
    provider_id: str
    model: str
    base_url: str | None
    cloud: bool


@dataclass(frozen=True)
class PluginContext:
    config: AppConfig
    plugin_id: str
    plugin_settings: dict[str, Any]
    policy: PolicyGate
    data_dir: Path


@dataclass(frozen=True)
class LLMProviderContext(PluginContext):
    stage: str
    stage_config: ModelStageConfig
    routing_override: str | None
    prompt_strategy: PromptStrategySettings
    governor: LLMGovernor | None
    secrets: SecretStore


@dataclass(frozen=True)
class VisionExtractorContext(PluginContext):
    runtime_governor: RuntimeGovernor | None = None


@dataclass(frozen=True)
class RetrievalContext(PluginContext):
    db: Any | None = None
    embedder: Any | None = None
    vector_index: Any | None = None
    reranker: Any | None = None
    spans_index: Any | None = None
    runtime_governor: RuntimeGovernor | None = None
