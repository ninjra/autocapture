"""Stage-based model routing for LLM workflows."""

from __future__ import annotations

from dataclasses import dataclass
from ..config import AppConfig, ModelStageConfig
from ..llm.providers import LLMProvider
from ..llm.governor import get_global_governor
from ..llm.prompt_strategy import PromptStrategySettings
from ..logging_utils import get_logger
from ..plugins import PluginManager
from ..plugins.sdk import LLMProviderInfo


@dataclass(frozen=True)
class StageDecision:
    stage: str
    provider: str
    model: str
    base_url: str | None
    cloud: bool
    temperature: float


class StageRouter:
    def __init__(self, config: AppConfig, *, plugin_manager: PluginManager | None = None) -> None:
        self._config = config
        self._log = get_logger("model_router")
        self._prompt_strategy = PromptStrategySettings.from_llm_config(
            config.llm, data_dir=config.capture.data_dir
        )
        self._governor = get_global_governor(config)
        self._plugins = plugin_manager or PluginManager(config)

    def select_llm(
        self, stage: str, *, routing_override: str | None = None
    ) -> tuple[LLMProvider, StageDecision]:
        stage_config = _resolve_stage_config(self._config, stage)
        if not stage_config.enabled:
            raise RuntimeError(f"Stage '{stage}' is disabled")
        provider = _resolve_provider(self._config, stage_config, routing_override)
        resolved = self._plugins.resolve_extension(
            "llm.provider",
            provider,
            stage=stage,
            stage_config=stage_config,
            factory_kwargs={
                "stage": stage,
                "stage_config": stage_config,
                "prompt_strategy": self._prompt_strategy,
                "governor": self._governor,
                "routing_override": routing_override,
            },
        )
        if not isinstance(resolved, tuple) or len(resolved) != 2:
            raise RuntimeError("LLM provider factory must return (provider, info)")
        client, info = resolved
        info = _normalize_llm_info(info, provider)

        decision = StageDecision(
            stage=stage,
            provider=info.provider_id,
            model=info.model,
            base_url=info.base_url,
            cloud=info.cloud,
            temperature=stage_config.temperature,
        )
        self._log.info(
            "Stage {} routed to {} (cloud={}, model={})",
            stage,
            info.provider_id,
            info.cloud,
            info.model,
        )
        return client, decision


def _resolve_stage_config(config: AppConfig, stage: str) -> ModelStageConfig:
    stages = config.model_stages
    mapping = {
        "query_refine": stages.query_refine,
        "draft_generate": stages.draft_generate,
        "final_answer": stages.final_answer,
        "tool_transform": stages.tool_transform,
    }
    if stage not in mapping:
        raise RuntimeError(f"Unknown stage '{stage}'")
    return mapping[stage]


def _resolve_provider(
    config: AppConfig, stage_config: ModelStageConfig, routing_override: str | None
) -> str:
    candidates = [
        routing_override,
        stage_config.provider,
        config.routing.llm,
        config.llm.provider,
    ]
    for candidate in candidates:
        if candidate:
            return candidate.strip().lower()
    return "ollama"


def _normalize_llm_info(info: object, fallback_provider: str) -> LLMProviderInfo:
    if isinstance(info, LLMProviderInfo):
        return info
    if isinstance(info, dict):
        return LLMProviderInfo(
            provider_id=str(info.get("provider_id") or fallback_provider),
            model=str(info.get("model") or ""),
            base_url=info.get("base_url"),
            cloud=bool(info.get("cloud")),
        )
    raise RuntimeError("Invalid LLM provider info payload")
