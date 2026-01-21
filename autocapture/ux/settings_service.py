"""Settings preview/apply logic shared by API + CLI."""

from __future__ import annotations

import datetime as dt
from copy import deepcopy
from pathlib import Path
from typing import Any

from .diff import diff_values
from .models import (
    SettingsApplyRequest,
    SettingsApplyResponse,
    SettingsEffectiveResponse,
    SettingsPreviewRequest,
    SettingsPreviewResponse,
)
from .preview import PreviewTokenManager, hash_payload
from .redaction import redact_payload
from .settings_schema import build_settings_schema
from ..config import AppConfig, apply_settings_overrides
from ..settings_store import read_settings, update_settings


class SettingsService:
    def __init__(self, config: AppConfig, *, plugins: object | None = None) -> None:
        self._config = config
        self._plugins = plugins
        self._schema = build_settings_schema()
        self._preview = PreviewTokenManager(Path(config.capture.data_dir))

    def schema(self):
        return self._schema

    def effective(self) -> SettingsEffectiveResponse:
        settings = self._with_defaults(self._load_raw())
        effective = self._effective_view(self._apply_overrides(settings))
        return SettingsEffectiveResponse(
            schema_version=self._schema.schema_version,
            settings=redact_payload(settings),
            effective=redact_payload(effective),
            redacted=True,
        )

    def preview(self, request: SettingsPreviewRequest) -> SettingsPreviewResponse:
        current = self._with_defaults(self._load_raw())
        candidate = self._merge(current, request.candidate)
        diff = diff_values(current, candidate)
        effective_before = self._effective_view(self._apply_overrides(current))
        effective_after = self._effective_view(self._apply_overrides(candidate))
        effective_diff = diff_values(effective_before, effective_after)
        impacts, warnings = self._preview_impacts(diff)
        preview_id = self._preview.issue(
            kind="settings",
            version=hash_payload(current),
            payload_hash=hash_payload(request.candidate),
        )
        return SettingsPreviewResponse(
            schema_version=self._schema.schema_version,
            preview_id=preview_id,
            diff=diff,
            effective_diff=effective_diff,
            impacts=impacts,
            warnings=warnings,
            effective_preview=redact_payload(effective_after),
        )

    def apply(self, request: SettingsApplyRequest) -> SettingsApplyResponse:
        current = self._with_defaults(self._load_raw())
        current_version = hash_payload(current)
        candidate_hash = hash_payload(request.candidate)
        self._preview.validate(
            request.preview_id,
            kind="settings",
            version=current_version,
            payload_hash=candidate_hash,
        )
        tier = (request.tier or "").strip() or "guided"
        self._validate_tier_confirm(tier, request.confirm, request.confirm_phrase)
        merged = self._merge(current, request.candidate)
        settings_path = self._settings_path()
        update_settings(settings_path, lambda existing: self._merge(existing, request.candidate))
        apply_settings_overrides(self._config)
        if self._plugins is not None:
            try:
                getattr(self._plugins, "refresh")()
            except Exception:
                pass
        effective = self._effective_view(self._apply_overrides(merged))
        return SettingsApplyResponse(
            status="ok",
            applied_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            effective=redact_payload(effective),
        )

    def _settings_path(self) -> Path:
        return Path(self._config.capture.data_dir) / "settings.json"

    def _load_raw(self) -> dict[str, Any]:
        return read_settings(self._settings_path())

    def _with_defaults(self, settings: dict[str, Any]) -> dict[str, Any]:
        data = dict(settings or {})
        if "routing" not in data:
            data["routing"] = self._model_dump(self._config.routing)
        if "privacy" not in data:
            data["privacy"] = {
                "paused": self._config.privacy.paused,
                "snooze_until_utc": self._to_iso(self._config.privacy.snooze_until_utc),
                "sanitize_default": self._config.privacy.sanitize_default,
                "extractive_only_default": self._config.privacy.extractive_only_default,
                "cloud_enabled": self._config.privacy.cloud_enabled,
                "allow_cloud_images": self._config.privacy.allow_cloud_images,
                "allow_token_vault_decrypt": self._config.privacy.allow_token_vault_decrypt,
                "exclude_monitors": list(self._config.privacy.exclude_monitors),
                "exclude_processes": list(self._config.privacy.exclude_processes),
                "exclude_window_title_regex": list(self._config.privacy.exclude_window_title_regex),
                "exclude_regions": list(self._config.privacy.exclude_regions),
                "mask_regions": list(self._config.privacy.mask_regions),
            }
        if "active_preset" not in data:
            data["active_preset"] = self._config.presets.active_preset
        if "backup" not in data:
            data["backup"] = {"last_export_at_utc": None}
        if "llm" not in data:
            data["llm"] = self._model_dump(self._config.llm)
        if "tracking" not in data:
            data["tracking"] = {
                "enabled": self._config.tracking.enabled,
                "track_mouse_movement": self._config.tracking.track_mouse_movement,
                "enable_clipboard": self._config.tracking.enable_clipboard,
                "retention_days": self._config.tracking.retention_days,
            }
        if "plugins" not in data:
            data["plugins"] = {
                "enabled": [],
                "disabled": [],
                "extension_overrides": {},
                "locks": {},
                "configs": {},
            }
        return data

    def _apply_overrides(self, settings: dict[str, Any]) -> AppConfig:
        config_copy = deepcopy(self._config)
        apply_settings_overrides(config_copy, settings)
        return config_copy

    def _effective_view(self, config: AppConfig) -> dict[str, Any]:
        return {
            "active_preset": config.presets.active_preset,
            "routing": self._model_dump(config.routing),
            "privacy": {
                "paused": config.privacy.paused,
                "snooze_until_utc": self._to_iso(config.privacy.snooze_until_utc),
                "sanitize_default": config.privacy.sanitize_default,
                "extractive_only_default": config.privacy.extractive_only_default,
                "cloud_enabled": config.privacy.cloud_enabled,
                "allow_cloud_images": config.privacy.allow_cloud_images,
                "allow_token_vault_decrypt": config.privacy.allow_token_vault_decrypt,
                "exclude_monitors": list(config.privacy.exclude_monitors),
                "exclude_processes": list(config.privacy.exclude_processes),
                "exclude_window_title_regex": list(config.privacy.exclude_window_title_regex),
                "exclude_regions": list(config.privacy.exclude_regions),
                "mask_regions": list(config.privacy.mask_regions),
            },
            "llm": self._model_dump(config.llm),
            "tracking": {
                "enabled": config.tracking.enabled,
                "track_mouse_movement": config.tracking.track_mouse_movement,
                "enable_clipboard": config.tracking.enable_clipboard,
                "retention_days": config.tracking.retention_days,
            },
        }

    def _preview_impacts(self, diff: list) -> tuple[list[str], list[str]]:
        impacts: list[str] = []
        warnings: list[str] = []
        for entry in diff:
            path = entry.path
            if path == "privacy.cloud_enabled" and entry.after is True:
                impacts.append("Cloud processing enabled")
                warnings.append("Cloud providers may send data off-device")
            if path == "privacy.allow_cloud_images" and entry.after is True:
                impacts.append("Cloud image processing enabled")
                warnings.append("Images may be sent to remote providers")
            if path == "privacy.allow_token_vault_decrypt" and entry.after is True:
                impacts.append("Token vault decryption enabled")
                warnings.append("Decrypted tokens can be requested via API")
            if path.startswith("routing."):
                impacts.append("Routing changes apply immediately to new requests")
            if path == "privacy.paused" and entry.after is True:
                impacts.append("Capture paused")
            if path == "tracking.enabled" and entry.after is False:
                impacts.append("Host event tracking disabled")
            if path == "tracking.enable_clipboard" and entry.after is True:
                warnings.append("Clipboard tracking may capture sensitive data")
        impacts = sorted(set(impacts))
        warnings = sorted(set(warnings))
        return impacts, warnings

    def _validate_tier_confirm(self, tier: str, confirm: bool, phrase: str | None) -> None:
        if tier != "expert":
            return
        tier_info = next((item for item in self._schema.tiers if item.tier_id == tier), None)
        if not tier_info or not tier_info.requires_confirm:
            return
        if not confirm:
            raise ValueError("Expert tier changes require confirm=true")
        expected = (tier_info.confirm_phrase or "").strip()
        if expected and (phrase or "").strip() != expected:
            raise ValueError("Expert tier confirm phrase mismatch")

    @staticmethod
    def _model_dump(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "dict"):
            return value.dict()
        return dict(value)

    @staticmethod
    def _merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        result = dict(base or {})
        for key, value in (updates or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = SettingsService._merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _to_iso(value: dt.datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.isoformat()
