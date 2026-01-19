"""Plugin settings stored in settings.json."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..settings_store import read_settings, update_settings

PLUGIN_SETTINGS_KEY = "plugins"


class PluginLock(BaseModel):
    manifest_sha256: str
    code_sha256: str
    accepted_at_utc: str
    source: str | None = None


class PluginSettings(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)
    extension_overrides: dict[str, str] = Field(default_factory=dict)
    locks: dict[str, PluginLock] = Field(default_factory=dict)
    configs: dict[str, dict[str, Any]] = Field(default_factory=dict)


def load_plugin_settings(settings_path: Path) -> PluginSettings:
    raw = read_settings(settings_path).get(PLUGIN_SETTINGS_KEY)
    if not isinstance(raw, dict):
        return PluginSettings()
    try:
        return PluginSettings.model_validate(raw)
    except Exception:
        return PluginSettings()


def write_plugin_settings(settings_path: Path, settings: PluginSettings) -> dict[str, Any]:
    def _update(current: dict[str, Any]) -> dict[str, Any]:
        merged = dict(current)
        merged[PLUGIN_SETTINGS_KEY] = settings.model_dump(mode="json")
        return merged

    return update_settings(settings_path, _update)


def update_plugin_settings(
    settings_path: Path,
    updater: Callable[[PluginSettings], PluginSettings | None],
) -> PluginSettings:
    def _update(current: dict[str, Any]) -> dict[str, Any]:
        existing = current.get(PLUGIN_SETTINGS_KEY)
        if isinstance(existing, dict):
            try:
                settings = PluginSettings.model_validate(existing)
            except Exception:
                settings = PluginSettings()
        else:
            settings = PluginSettings()
        updated = updater(settings)
        if updated is None:
            updated = settings
        merged = dict(current)
        merged[PLUGIN_SETTINGS_KEY] = updated.model_dump(mode="json")
        return merged

    updated_raw = update_settings(settings_path, _update)
    updated_section = updated_raw.get(PLUGIN_SETTINGS_KEY, {})
    if isinstance(updated_section, dict):
        try:
            return PluginSettings.model_validate(updated_section)
        except Exception:
            return PluginSettings()
    return PluginSettings()


def record_lock(
    settings: PluginSettings,
    *,
    plugin_id: str,
    manifest_sha256: str,
    code_sha256: str,
    source: str | None = None,
) -> PluginSettings:
    payload = PluginLock(
        manifest_sha256=manifest_sha256,
        code_sha256=code_sha256,
        accepted_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source=source,
    )
    settings.locks[plugin_id] = payload
    return settings
