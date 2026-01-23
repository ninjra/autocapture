"""Pydantic models for plugin config APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import DiffEntry


class PluginExtensionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extension_id: str
    kind: str
    name: str
    config_schema: dict[str, Any] | None = None
    ui: dict[str, Any] | None = None
    pillars: dict[str, Any] | None = None


class PluginSchemaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin_id: str
    name: str
    version: str
    extensions: list[PluginExtensionSchema] = Field(default_factory=list)


class PluginsSchemaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugins: list[PluginSchemaResponse] = Field(default_factory=list)


class PluginDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin_id: str
    name: str
    version: str
    description: str | None = None
    source: str
    enabled: bool
    blocked: bool
    reason: str | None = None
    lock_status: str
    lock_manifest: str | None = None
    lock_code: str | None = None
    manifest_sha256: str
    code_sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)
    manifest: dict[str, Any]
    config: dict[str, Any]


class PluginConfigPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate: dict[str, Any]


class PluginConfigPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_id: str
    diff: list[DiffEntry]
    warnings: list[str] = Field(default_factory=list)
    impacts: list[str] = Field(default_factory=list)
    candidate: dict[str, Any]


class PluginConfigApplyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate: dict[str, Any]
    preview_id: str
    confirm: bool = False
    confirm_phrase: str | None = None


class PluginConfigApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    applied_at_utc: str
    config: dict[str, Any]
