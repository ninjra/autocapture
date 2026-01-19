"""Plugin manifest models."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import EXTENSION_KINDS, FACTORY_TYPES
from .errors import PluginManifestError

_PLUGIN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,63}$")
_EXTENSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,63}$")


class CompatibilityRange(BaseModel):
    app_min: str | None = Field(None, description="Minimum compatible app version.")
    app_max: str | None = Field(None, description="Maximum compatible app version.")
    python: str | None = Field(None, description="Python version specifier.")


class DataHandlingPillar(BaseModel):
    cloud: Literal["none", "optional", "required"] = "none"
    cloud_images: Literal["none", "optional", "required"] = "none"
    supports_local: bool = True
    pii: bool | None = None


class ExtensionPillars(BaseModel):
    data_handling: DataHandlingPillar | None = None
    data_types: list[str] | None = None
    determinism: str | None = None
    citations: str | None = None
    gpu: dict[str, Any] | None = None


class UIBadge(BaseModel):
    text: str
    tone: Literal["neutral", "info", "warning", "danger"] = "neutral"


class UIInfo(BaseModel):
    description: str | None = None
    badge: UIBadge | None = None
    icon: str | None = None
    category: str | None = None


class FactoryDescriptor(BaseModel):
    type: Literal["python", "bundle", "file"]
    entrypoint: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def _validate_factory(self) -> "FactoryDescriptor":
        if self.type not in FACTORY_TYPES:
            raise ValueError(f"Unsupported factory type: {self.type}")
        if self.type == "python" and not self.entrypoint:
            raise ValueError("python factory requires entrypoint")
        if self.type in {"bundle", "file"} and not self.path:
            raise ValueError(f"{self.type} factory requires path")
        return self


class ExtensionManifestV1(BaseModel):
    kind: str
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    pillars: ExtensionPillars | None = None
    factory: FactoryDescriptor
    config_schema: dict[str, Any] | None = None
    ui: UIInfo | None = None

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        if value not in EXTENSION_KINDS:
            raise ValueError(f"Unsupported extension kind: {value}")
        return value

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not _EXTENSION_ID_RE.match(value):
            raise ValueError("Extension id must match [a-z0-9_.-] and be <=64 chars")
        return value

    @field_validator("aliases")
    @classmethod
    def _validate_aliases(cls, value: list[str]) -> list[str]:
        for alias in value:
            if not _EXTENSION_ID_RE.match(alias):
                raise ValueError(f"Invalid alias '{alias}'")
        return value


class PluginManifestV1(BaseModel):
    schema_version: int = 1
    plugin_id: str
    name: str
    version: str
    description: str | None = None
    author: str | None = None
    homepage: str | None = None
    enabled_by_default: bool = True
    compatibility: CompatibilityRange | None = None
    permissions: dict[str, Any] | None = None
    extensions: list[ExtensionManifestV1]

    @field_validator("plugin_id")
    @classmethod
    def _validate_plugin_id(cls, value: str) -> str:
        if not _PLUGIN_ID_RE.match(value):
            raise ValueError("plugin_id must match [a-z0-9_.-] and be 3-64 chars")
        return value

    @model_validator(mode="after")
    def _validate_extensions(self) -> "PluginManifestV1":
        seen: set[tuple[str, str]] = set()
        for ext in self.extensions:
            key = (ext.kind, ext.id)
            if key in seen:
                raise ValueError(f"Duplicate extension entry: {ext.kind}:{ext.id}")
            seen.add(key)
        return self


def parse_manifest(payload: dict[str, Any]) -> PluginManifestV1:
    if not isinstance(payload, dict):
        raise PluginManifestError("Manifest payload must be a mapping")
    schema = payload.get("schema_version", 1)
    if schema != 1:
        raise PluginManifestError(f"Unsupported schema_version: {schema}")
    try:
        return PluginManifestV1.model_validate(payload)
    except Exception as exc:  # pragma: no cover - guarded by tests
        raise PluginManifestError(str(exc)) from exc
