"""Model registry + stage policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..config import ModelRegistryConfig, ModelSpec, ProviderSpec, StagePolicy


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageModel:
    stage: StagePolicy
    model: ModelSpec
    provider: ProviderSpec


class ModelRegistry:
    def __init__(self, config: ModelRegistryConfig) -> None:
        self._config = config
        self._providers = {provider.id: provider for provider in config.providers}
        self._models = {model.id: model for model in config.models}
        self._stages = {stage.id: stage for stage in config.stages}

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled and self._stages)

    def stage(self, stage_id: str) -> StagePolicy | None:
        return self._stages.get(stage_id)

    def provider(self, provider_id: str) -> ProviderSpec:
        provider = self._providers.get(provider_id)
        if provider is None:
            raise RegistryError(f"Unknown provider_id '{provider_id}'")
        return provider

    def model(self, model_id: str) -> ModelSpec:
        model = self._models.get(model_id)
        if model is None:
            raise RegistryError(f"Unknown model_id '{model_id}'")
        return model

    def stage_candidates(self, stage_id: str) -> list[StageModel]:
        stage = self.stage(stage_id)
        if stage is None:
            raise RegistryError(f"Unknown stage '{stage_id}'")
        model_ids = [stage.primary_model_id] + list(stage.fallback_model_ids)
        seen: set[str] = set()
        ordered: list[StageModel] = []
        for model_id in model_ids:
            if model_id in seen:
                continue
            seen.add(model_id)
            model = self.model(model_id)
            provider = self.provider(model.provider_id)
            ordered.append(StageModel(stage=stage, model=model, provider=provider))
        return ordered

    def model_ids(self) -> set[str]:
        return set(self._models)

    def models(self) -> Iterable[ModelSpec]:
        return self._models.values()

    def direct_candidate(self, model_id: str) -> StageModel:
        model = self.model(model_id)
        provider = self.provider(model.provider_id)
        stage = StagePolicy(
            id=f"direct:{model_id}",
            primary_model_id=model_id,
        )
        return StageModel(stage=stage, model=model, provider=provider)

    def providers(self) -> Iterable[ProviderSpec]:
        return self._providers.values()

    def decode_backend(self, stage: StagePolicy) -> ProviderSpec | None:
        backend_id = stage.decode.backend_provider_id
        if not backend_id:
            return None
        return self.provider(backend_id)


__all__ = [
    "RegistryError",
    "ModelRegistry",
    "StageModel",
]
