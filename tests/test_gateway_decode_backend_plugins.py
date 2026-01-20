from __future__ import annotations

from autocapture.config import (
    AppConfig,
    DecodeConfig,
    ModelRegistryConfig,
    ModelSpec,
    ProviderSpec,
    StagePolicy,
)
from autocapture.gateway.service import GatewayRouter


class _StubPlugins:
    def __init__(self, backend):
        self.backend = backend
        self.calls: list[tuple[str, str]] = []

    def resolve_extension(self, kind: str, extension_id: str, **_kwargs):
        self.calls.append((kind, extension_id))
        return self.backend


class _FailingPlugins:
    def resolve_extension(self, kind: str, extension_id: str, **_kwargs):
        _ = kind, extension_id
        raise RuntimeError("boom")


def _config_with_decode_backend() -> ModelRegistryConfig:
    provider = ProviderSpec(id="p1", type="openai_compatible", base_url="http://localhost")
    decode_provider = ProviderSpec(
        id="decode",
        type="openai_compatible",
        base_url="http://localhost",
    )
    model = ModelSpec(id="m1", provider_id="p1", upstream_model_name="model-1")
    stage = StagePolicy(
        id="final_answer",
        primary_model_id="m1",
        decode=DecodeConfig(strategy="medusa", backend_provider_id="decode"),
    )
    return ModelRegistryConfig(
        enabled=True,
        providers=[provider, decode_provider],
        models=[model],
        stages=[stage],
    )


def test_decode_backend_prefers_plugin() -> None:
    registry = _config_with_decode_backend()
    config = AppConfig(model_registry=registry)
    plugin_backend = ProviderSpec(
        id="decode",
        type="openai_compatible",
        base_url="http://localhost",
    )
    plugins = _StubPlugins(plugin_backend)
    router = GatewayRouter(config, registry=None, plugin_manager=plugins)
    stage = registry.stages[0]
    backend = router._resolve_decode_backend(stage)
    assert backend is plugin_backend
    assert plugins.calls == [("decode.backend", "decode")]


def test_decode_backend_falls_back_to_registry() -> None:
    registry = _config_with_decode_backend()
    config = AppConfig(model_registry=registry)
    router = GatewayRouter(config, registry=None, plugin_manager=_FailingPlugins())
    stage = registry.stages[0]
    backend = router._resolve_decode_backend(stage)
    assert backend.id == "decode"
