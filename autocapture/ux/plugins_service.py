"""Plugin config preview/apply and schema helpers."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from .diff import diff_values
from .preview import PreviewTokenManager, hash_payload
from .plugins_models import (
    PluginConfigApplyRequest,
    PluginConfigApplyResponse,
    PluginConfigPreviewRequest,
    PluginConfigPreviewResponse,
    PluginDetailResponse,
    PluginExtensionSchema,
    PluginsSchemaResponse,
    PluginSchemaResponse,
)
from ..config import AppConfig
from ..plugins.manager import PluginManager, PluginStatus
from ..plugins.settings import load_plugin_settings, update_plugin_settings


class PluginsService:
    def __init__(self, config: AppConfig, plugins: PluginManager) -> None:
        self._config = config
        self._plugins = plugins
        self._settings_path = Path(config.capture.data_dir) / "settings.json"
        self._preview = PreviewTokenManager(Path(config.capture.data_dir))

    def schemas(self) -> PluginsSchemaResponse:
        statuses = self._catalog()
        return PluginsSchemaResponse(
            plugins=[self._schema_from_status(status) for status in statuses if status],
        )

    def schema(self, plugin_id: str) -> PluginSchemaResponse:
        status = self._status(plugin_id)
        return self._schema_from_status(status)

    def detail(self, plugin_id: str) -> PluginDetailResponse:
        status = self._status(plugin_id)
        settings = load_plugin_settings(self._settings_path)
        config = settings.configs.get(plugin_id, {})
        manifest = status.plugin.manifest.model_dump(mode="json")
        return PluginDetailResponse(
            plugin_id=status.plugin.plugin_id,
            name=status.plugin.manifest.name,
            version=status.plugin.manifest.version,
            description=status.plugin.manifest.description,
            source=status.plugin.source.source_type.value,
            enabled=status.enabled,
            blocked=status.blocked,
            reason=status.reason,
            lock_status=status.lock_status,
            lock_manifest=status.lock_manifest,
            lock_code=status.lock_code,
            manifest_sha256=status.manifest_sha256,
            code_sha256=status.code_sha256 or None,
            warnings=list(status.plugin.warnings or []),
            manifest=manifest,
            config=config,
        )

    def preview_config(
        self,
        plugin_id: str,
        request: PluginConfigPreviewRequest,
    ) -> PluginConfigPreviewResponse:
        status = self._status(plugin_id)
        settings = load_plugin_settings(self._settings_path)
        current = settings.configs.get(plugin_id, {})
        candidate = request.candidate or {}
        merged = _merge(current, candidate)
        diff = diff_values(current, merged)
        preview_id = self._preview.issue(
            kind="plugins",
            version=hash_payload(current),
            payload_hash=hash_payload(candidate),
        )
        warnings = _warnings_for_status(status)
        return PluginConfigPreviewResponse(
            preview_id=preview_id,
            diff=diff,
            warnings=warnings,
            impacts=[],
            candidate=merged,
        )

    def apply_config(
        self,
        plugin_id: str,
        request: PluginConfigApplyRequest,
    ) -> PluginConfigApplyResponse:
        status = self._status(plugin_id)
        settings = load_plugin_settings(self._settings_path)
        current = settings.configs.get(plugin_id, {})
        candidate = request.candidate or {}
        self._preview.validate(
            request.preview_id,
            kind="plugins",
            version=hash_payload(current),
            payload_hash=hash_payload(candidate),
        )
        merged = _merge(current, candidate)

        def _apply(existing):
            current_config = existing.configs.get(plugin_id, {})
            existing.configs[plugin_id] = _merge(current_config, candidate)
            return existing

        update_plugin_settings(self._settings_path, _apply)
        self._plugins.refresh()
        _ = status
        return PluginConfigApplyResponse(
            status="ok",
            applied_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            config=merged,
        )

    def _catalog(self) -> list[PluginStatus]:
        self._plugins.refresh()
        return [status for status in self._plugins.catalog() if status.plugin.plugin_id != "__discovery__"]

    def _status(self, plugin_id: str) -> PluginStatus:
        self._plugins.refresh()
        for status in self._plugins.catalog():
            if status.plugin.plugin_id == plugin_id:
                return status
        raise ValueError(f"Unknown plugin_id: {plugin_id}")

    def _schema_from_status(self, status: PluginStatus) -> PluginSchemaResponse:
        manifest = status.plugin.manifest
        extensions = [
            PluginExtensionSchema(
                extension_id=ext.id,
                kind=ext.kind,
                name=ext.name,
                config_schema=ext.config_schema,
                ui=ext.ui.model_dump(mode="json") if ext.ui else None,
                pillars=ext.pillars.model_dump(mode="json") if ext.pillars else None,
            )
            for ext in manifest.extensions
        ]
        return PluginSchemaResponse(
            plugin_id=manifest.plugin_id,
            name=manifest.name,
            version=manifest.version,
            extensions=extensions,
        )


def _warnings_for_status(status: PluginStatus) -> list[str]:
    warnings: list[str] = []
    if status.blocked:
        warnings.append(status.reason or "blocked")
    if not status.enabled:
        warnings.append("plugin_disabled")
    return warnings


def _merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base or {})
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result
