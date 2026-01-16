"""Stage-based model routing for LLM workflows."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from ..config import AppConfig, LLMConfig, ModelStageConfig, is_loopback_host
from ..llm.providers import LLMProvider, OllamaProvider, OpenAICompatibleProvider, OpenAIProvider
from ..logging_utils import get_logger


@dataclass(frozen=True)
class StageDecision:
    stage: str
    provider: str
    model: str
    base_url: str | None
    cloud: bool
    temperature: float


class StageRouter:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._log = get_logger("model_router")

    def select_llm(
        self, stage: str, *, routing_override: str | None = None
    ) -> tuple[LLMProvider, StageDecision]:
        stage_config = _resolve_stage_config(self._config, stage)
        if not stage_config.enabled:
            raise RuntimeError(f"Stage '{stage}' is disabled")
        provider = _resolve_provider(self._config, stage_config, routing_override)
        model = stage_config.model or _default_model(provider, self._config.llm)

        base_url = stage_config.base_url
        api_key = stage_config.api_key
        if provider == "ollama":
            base_url = base_url or self._config.llm.ollama_url
            cloud = False
            client = OllamaProvider(
                base_url,
                model,
                timeout_s=self._config.llm.timeout_s,
                retries=self._config.llm.retries,
            )
        elif provider == "openai_compatible":
            base_url = base_url or self._config.llm.openai_compatible_base_url
            api_key = api_key or self._config.llm.openai_compatible_api_key
            if not base_url:
                raise RuntimeError(f"Stage '{stage}' requires openai_compatible base_url")
            cloud = _is_cloud_endpoint(base_url)
            _guard_cloud(self._config, stage_config, stage, provider, base_url, cloud)
            client = OpenAICompatibleProvider(
                base_url,
                model,
                api_key=api_key,
                timeout_s=self._config.llm.timeout_s,
                retries=self._config.llm.retries,
            )
        elif provider == "openai":
            api_key = api_key or self._config.llm.openai_api_key
            if not api_key:
                raise RuntimeError(f"Stage '{stage}' requires OpenAI API key")
            cloud = True
            _guard_cloud(self._config, stage_config, stage, provider, None, cloud)
            client = OpenAIProvider(
                api_key,
                model,
                timeout_s=self._config.llm.timeout_s,
                retries=self._config.llm.retries,
            )
        else:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")

        decision = StageDecision(
            stage=stage,
            provider=provider,
            model=model,
            base_url=base_url,
            cloud=cloud,
            temperature=stage_config.temperature,
        )
        self._log.info(
            "Stage {} routed to {} (cloud={}, model={})",
            stage,
            provider,
            cloud,
            model,
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


def _default_model(provider: str, llm_config: LLMConfig) -> str:
    if provider == "openai":
        return llm_config.openai_model
    if provider == "openai_compatible":
        return llm_config.openai_compatible_model
    return llm_config.ollama_model


def _is_cloud_endpoint(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return bool(host) and not is_loopback_host(host)


def _guard_cloud(
    config: AppConfig,
    stage_config: ModelStageConfig,
    stage: str,
    provider: str,
    base_url: str | None,
    cloud: bool,
) -> None:
    if not cloud:
        return
    if not stage_config.allow_cloud:
        raise RuntimeError(
            f"Cloud provider blocked for stage '{stage}'. "
            f"Set model_stages.{stage}.allow_cloud=true to allow."
        )
    if config.offline:
        raise RuntimeError(
            f"Cloud provider blocked for stage '{stage}' because offline=true."
        )
    if not config.privacy.cloud_enabled:
        raise RuntimeError(
            f"Cloud provider blocked for stage '{stage}' because privacy.cloud_enabled=false."
        )
