"""Decode backend helpers for gateway routing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..config import CircuitBreakerConfig, ProviderSpec


class DecodeBackendSettings(BaseModel):
    base_url: str
    provider_type: str = Field(
        "openai_compatible",
        description="Provider type for decode backend (openai_compatible|openai|gateway).",
    )
    api_key_env: str | None = None
    api_key: str | None = None
    timeout_s: float = Field(60.0, gt=0.0)
    retries: int = Field(3, ge=0, le=10)
    headers: dict[str, str] = Field(default_factory=dict)
    allow_cloud: bool = False
    max_concurrency: int = Field(2, ge=1)
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()


def decode_backend_from_settings(backend_id: str, settings: dict[str, Any]) -> ProviderSpec:
    try:
        parsed = DecodeBackendSettings.model_validate(settings)
    except ValidationError as exc:
        raise RuntimeError("decode_backend_settings_invalid") from exc
    return ProviderSpec(
        id=backend_id,
        type=parsed.provider_type,
        base_url=parsed.base_url,
        api_key_env=parsed.api_key_env,
        api_key=parsed.api_key,
        timeout_s=parsed.timeout_s,
        retries=parsed.retries,
        headers=parsed.headers,
        allow_cloud=parsed.allow_cloud,
        circuit_breaker=parsed.circuit_breaker,
        max_concurrency=parsed.max_concurrency,
    )


def extract_backend_settings(
    plugin_settings: dict[str, Any] | None,
    backend_id: str,
) -> dict[str, Any]:
    settings = plugin_settings or {}
    if isinstance(settings, dict):
        backends = settings.get("backends")
        if isinstance(backends, dict):
            override = backends.get(backend_id)
            if isinstance(override, dict):
                return override
    return settings if isinstance(settings, dict) else {}


__all__ = ["DecodeBackendSettings", "decode_backend_from_settings", "extract_backend_settings"]
