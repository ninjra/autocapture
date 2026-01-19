"""Stage-based model routing for LLM workflows."""

from __future__ import annotations

from dataclasses import dataclass
from ..config import (
    AppConfig,
    ModelStageConfig,
    StagePolicy,
    StageRequirementsConfig,
    StageSamplingConfig,
)
from ..llm.providers import LLMProvider
from ..llm.governor import get_global_governor
from ..llm.prompt_strategy import PromptStrategySettings
from ..logging_utils import get_logger
from ..plugins import PluginManager
from ..plugins.sdk import LLMProviderInfo
from .registry import ModelRegistry, RegistryError
from ..resilience import CircuitBreaker


@dataclass(frozen=True)
class StageDecision:
    stage: str
    provider: str
    provider_type: str
    model: str
    base_url: str | None
    cloud: bool
    temperature: float
    model_id: str | None = None
    policy: StagePolicy | None = None
    requirements: StageRequirementsConfig | None = None
    sampling: StageSamplingConfig | None = None


class StageRouter:
    def __init__(self, config: AppConfig, *, plugin_manager: PluginManager | None = None) -> None:
        self._config = config
        self._log = get_logger("model_router")
        self._prompt_strategy = PromptStrategySettings.from_llm_config(
            config.llm, data_dir=config.capture.data_dir
        )
        self._governor = get_global_governor(config)
        self._plugins = plugin_manager or PluginManager(config)
        self._registry = (
            ModelRegistry(config.model_registry)
            if config.model_registry.enabled and config.model_registry.stages
            else None
        )
        self._provider_breakers: dict[str, CircuitBreaker] = {}

    def select_llm(
        self, stage: str, *, routing_override: str | None = None
    ) -> tuple[LLMProvider, StageDecision]:
        if self._registry is not None and self._registry.enabled:
            return self._select_registry(stage, routing_override=routing_override)
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
            provider_type=info.provider_id,
            model=info.model,
            base_url=info.base_url,
            cloud=info.cloud,
            temperature=stage_config.temperature,
            model_id=None,
            policy=None,
            requirements=None,
            sampling=StageSamplingConfig(temperature=stage_config.temperature),
        )
        self._log.info(
            "Stage {} routed to {} (cloud={}, model={})",
            stage,
            info.provider_id,
            info.cloud,
            info.model,
        )
        return client, decision

    def _select_registry(
        self, stage: str, *, routing_override: str | None
    ) -> tuple[LLMProvider, StageDecision]:
        assert self._registry is not None
        try:
            candidates = self._registry.stage_candidates(stage)
        except RegistryError as exc:
            raise RuntimeError(str(exc)) from exc
        if routing_override:
            override = routing_override.strip().lower()
            candidates = [item for item in candidates if item.provider.id == override]
            if not candidates:
                raise RuntimeError(f"Routing override '{routing_override}' not found in registry")
        last_exc: Exception | None = None
        for candidate in candidates:
            policy = candidate.stage
            model = candidate.model
            provider_spec = candidate.provider
            breaker = self._provider_breakers.get(provider_spec.id)
            if breaker is None:
                breaker = CircuitBreaker(
                    failure_threshold=provider_spec.circuit_breaker.failure_threshold,
                    reset_timeout_s=provider_spec.circuit_breaker.reset_timeout_s,
                )
                self._provider_breakers[provider_spec.id] = breaker
            if not breaker.allow():
                continue
            stage_config = _stage_config_from_policy(policy, model, provider_spec)
            try:
                resolved = self._plugins.resolve_extension(
                    "llm.provider",
                    provider_spec.type,
                    stage=stage,
                    stage_config=stage_config,
                    factory_kwargs={
                        "stage": stage,
                        "stage_config": stage_config,
                        "prompt_strategy": self._prompt_strategy,
                        "governor": self._governor,
                        "routing_override": routing_override,
                        "provider_alias": provider_spec.id,
                    },
                )
                if not isinstance(resolved, tuple) or len(resolved) != 2:
                    raise RuntimeError("LLM provider factory must return (provider, info)")
                client, info = resolved
                info = _normalize_llm_info(info, provider_spec.type)
                cloud = info.cloud
                decision = StageDecision(
                    stage=stage,
                    provider=provider_spec.id,
                    provider_type=provider_spec.type,
                    model=model.upstream_model_name,
                    base_url=provider_spec.base_url,
                    cloud=cloud,
                    temperature=policy.sampling.temperature,
                    model_id=model.id,
                    policy=policy,
                    requirements=policy.requirements,
                    sampling=policy.sampling,
                )
                self._log.info(
                    "Stage {} routed to {} (type={}, model={}, policy={})",
                    stage,
                    provider_spec.id,
                    provider_spec.type,
                    model.id,
                    policy.id,
                )
                breaker.record_success()
                return client, decision
            except Exception as exc:
                breaker.record_failure(exc)
                last_exc = exc
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Stage '{stage}' has no eligible providers")


def _resolve_stage_config(config: AppConfig, stage: str) -> ModelStageConfig:
    stages = config.model_stages
    mapping = {
        "query_refine": stages.query_refine,
        "draft_generate": stages.draft_generate,
        "final_answer": stages.final_answer,
        "tool_transform": stages.tool_transform,
        "entailment_judge": stages.entailment_judge,
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


def _stage_config_from_policy(
    policy: StagePolicy, model, provider_spec
) -> ModelStageConfig:
    allow_cloud = bool(policy.allow_cloud and provider_spec.allow_cloud)
    return ModelStageConfig(
        provider=provider_spec.type,
        model=model.upstream_model_name,
        base_url=provider_spec.base_url,
        api_key=provider_spec.api_key,
        allow_cloud=allow_cloud,
        enabled=True,
        temperature=policy.sampling.temperature,
    )

