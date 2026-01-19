from __future__ import annotations

import pytest

from autocapture.config import (
    DecodeConfig,
    ModelRegistryConfig,
    ModelSpec,
    ProviderSpec,
    StagePolicy,
)
from autocapture.model_ops.registry import ModelRegistry


def _base_registry() -> ModelRegistryConfig:
    providers = [
        ProviderSpec(
            id="local",
            type="openai_compatible",
            base_url="http://localhost:8001",
        )
    ]
    models = [ModelSpec(id="m1", provider_id="local", upstream_model_name="model-1")]
    stages = [StagePolicy(id="final_answer", primary_model_id="m1")]
    return ModelRegistryConfig(enabled=True, providers=providers, models=models, stages=stages)


def test_registry_candidates_resolve() -> None:
    config = _base_registry()
    registry = ModelRegistry(config)
    candidates = registry.stage_candidates("final_answer")
    assert [candidate.model.id for candidate in candidates] == ["m1"]


def test_registry_requires_known_provider() -> None:
    with pytest.raises(ValueError):
        ModelRegistryConfig(
            enabled=True,
            providers=[],
            models=[ModelSpec(id="m1", provider_id="missing", upstream_model_name="model")],
            stages=[StagePolicy(id="final", primary_model_id="m1")],
        )


def test_registry_requires_decode_backend_when_nonstandard() -> None:
    with pytest.raises(ValueError):
        ModelRegistryConfig(
            enabled=True,
            providers=[
                ProviderSpec(id="p1", type="openai_compatible", base_url="http://localhost")
            ],
            models=[ModelSpec(id="m1", provider_id="p1", upstream_model_name="model")],
            stages=[
                StagePolicy(
                    id="final",
                    primary_model_id="m1",
                    decode=DecodeConfig(strategy="swift"),
                )
            ],
        )
