"""Plugin manager orchestration."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import sys
from time import monotonic
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .catalog import DiscoveredPlugin, PluginSourceType, discover_plugins
from .errors import PluginLockError, PluginPolicyError, PluginResolutionError
from .hash import hash_directory, hash_distribution_files, hash_traversable
from .policy import PolicyGate
from .registry import ExtensionRecord, ExtensionRegistry
from .settings import PluginSettings, load_plugin_settings, record_lock, update_plugin_settings
from .sdk.context import PluginContext
from ..logging_utils import get_logger
from ..observability.metrics import (
    plugin_healthcheck_latency_ms,
    plugin_load_failures_total,
    plugins_discovered_total,
    plugins_enabled_total,
)

try:  # optional
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except Exception:  # pragma: no cover - optional
    SpecifierSet = None  # type: ignore
    Version = None  # type: ignore


@dataclass(frozen=True)
class PluginStatus:
    plugin: DiscoveredPlugin
    enabled: bool
    blocked: bool
    reason: str | None
    lock_status: str
    lock_manifest: str | None
    lock_code: str | None
    manifest_sha256: str
    code_sha256: str


class PluginManager:
    def __init__(
        self,
        config,
        *,
        settings_path: Path | None = None,
        plugin_dir: Path | None = None,
        entry_points: Iterable | None = None,
    ) -> None:
        self._config = config
        self._settings_path = settings_path or Path(config.capture.data_dir) / "settings.json"
        config_dir = getattr(config, "plugins", None)
        directory_override = getattr(config_dir, "directory", None) if config_dir else None
        safe_mode_override = getattr(config_dir, "safe_mode", None) if config_dir else None
        self._plugin_dir = (
            plugin_dir or directory_override or Path(config.capture.data_dir) / "plugins"
        )
        self._entry_points = entry_points
        self._log = get_logger("plugins")
        self._policy = PolicyGate(config)
        self._registry: ExtensionRegistry | None = None
        self._catalog: list[DiscoveredPlugin] = []
        self._statuses: list[PluginStatus] = []
        self._settings: PluginSettings = PluginSettings()
        self._instance_cache: dict[tuple[str, str, str], Any] = {}
        env_safe = os.environ.get("AUTOCAPTURE_SAFE_MODE")
        self._safe_mode = (
            _parse_bool(env_safe) if env_safe is not None else bool(safe_mode_override)
        )
        self.refresh()

    def refresh(self) -> None:
        discovery = discover_plugins(plugin_dir=self._plugin_dir, entry_points=self._entry_points)
        self._catalog = discovery.plugins
        self._settings = load_plugin_settings(self._settings_path)
        self._statuses = _build_statuses(
            self._catalog,
            self._settings,
            safe_mode=self._safe_mode,
            errors=discovery.errors,
        )
        registry = ExtensionRegistry()
        for status in self._statuses:
            if not status.enabled or status.blocked:
                continue
            plugin = status.plugin
            for extension in plugin.manifest.extensions:
                registry.register_extension(
                    kind=extension.kind,
                    extension_id=extension.id,
                    plugin_id=plugin.plugin_id,
                    name=extension.name,
                    manifest=extension,
                    source=plugin.source.source_type.value,
                )
        self._registry = registry
        self._instance_cache.clear()
        plugins_discovered_total.inc(len(self._catalog))
        enabled_count = sum(1 for status in self._statuses if status.enabled and not status.blocked)
        plugins_enabled_total.set(enabled_count)

    def catalog(self) -> list[PluginStatus]:
        return list(self._statuses)

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode

    def registry(self) -> ExtensionRegistry:
        if self._registry is None:
            self.refresh()
        assert self._registry is not None
        return self._registry

    def list_extensions(self, kind: str) -> list[ExtensionRecord]:
        return self.registry().list_extensions(kind)

    def resolve_record(self, kind: str, extension_id: str) -> ExtensionRecord:
        return self.registry().resolve(
            kind, extension_id, overrides=self._settings.extension_overrides
        )

    def prompt_bundles(self) -> list[Path]:
        bundles: list[Path] = []
        for record in self.registry().list_extensions("prompt.bundle"):
            try:
                path = self.resolve_extension(
                    "prompt.bundle",
                    record.extension_id,
                    use_cache=False,
                )
            except Exception:
                continue
            if isinstance(path, Path) and path.exists():
                bundles.append(path)
        return bundles

    def resolve_extension(
        self,
        kind: str,
        extension_id: str,
        *,
        overrides: dict[str, str] | None = None,
        stage: str | None = None,
        stage_config: Any | None = None,
        factory_kwargs: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> Any:
        if overrides is None:
            overrides = self._settings.extension_overrides
        record = self.registry().resolve(kind, extension_id, overrides=overrides)
        cache_key = (kind, record.extension_id, record.plugin_id)
        if use_cache and cache_key in self._instance_cache:
            return self._instance_cache[cache_key]
        decision = self._policy.enforce_extension_policy(
            stage=stage, stage_config=stage_config, extension=record.manifest
        )
        if not decision.ok:
            raise PluginPolicyError(decision.reason or "policy_denied")
        try:
            instance = self._instantiate(record, factory_kwargs or {})
        except Exception:
            plugin_load_failures_total.labels(record.plugin_id).inc()
            raise
        if use_cache:
            self._instance_cache[cache_key] = instance
        return instance

    def enable_plugin(self, plugin_id: str, *, accept_hashes: bool) -> PluginStatus:
        self.refresh()
        plugin = _find_plugin(self._catalog, plugin_id)
        if plugin is None:
            raise PluginResolutionError(f"Unknown plugin_id: {plugin_id}")
        status = _status_for(self._statuses, plugin_id)
        if status and status.enabled and not status.blocked:
            return status
        manifest_hash, code_hash = _compute_hashes(plugin)
        if not accept_hashes:
            raise PluginLockError(
                f"Plugin '{plugin_id}' requires acceptance",
            )

        def _apply(settings: PluginSettings) -> PluginSettings:
            if plugin_id not in settings.enabled:
                settings.enabled.append(plugin_id)
            if plugin_id in settings.disabled:
                settings.disabled.remove(plugin_id)
            record_lock(
                settings,
                plugin_id=plugin_id,
                manifest_sha256=manifest_hash,
                code_sha256=code_hash,
                source=plugin.source.source_type.value,
            )
            return settings

        update_plugin_settings(self._settings_path, _apply)
        self.refresh()
        status = _status_for(self._statuses, plugin_id)
        if status is None:
            raise PluginResolutionError(f"Failed to enable plugin {plugin_id}")
        return status

    def disable_plugin(self, plugin_id: str) -> PluginStatus:
        def _apply(settings: PluginSettings) -> PluginSettings:
            if plugin_id in settings.enabled:
                settings.enabled.remove(plugin_id)
            if plugin_id not in settings.disabled:
                settings.disabled.append(plugin_id)
            return settings

        update_plugin_settings(self._settings_path, _apply)
        self.refresh()
        status = _status_for(self._statuses, plugin_id)
        if status is None:
            raise PluginResolutionError(f"Unknown plugin_id: {plugin_id}")
        return status

    def lock_plugin(self, plugin_id: str) -> PluginStatus:
        plugin = _find_plugin(self._catalog, plugin_id)
        if plugin is None:
            raise PluginResolutionError(f"Unknown plugin_id: {plugin_id}")
        manifest_hash, code_hash = _compute_hashes(plugin)

        def _apply(settings: PluginSettings) -> PluginSettings:
            record_lock(
                settings,
                plugin_id=plugin_id,
                manifest_sha256=manifest_hash,
                code_sha256=code_hash,
                source=plugin.source.source_type.value,
            )
            return settings

        update_plugin_settings(self._settings_path, _apply)
        self.refresh()
        status = _status_for(self._statuses, plugin_id)
        if status is None:
            raise PluginResolutionError(f"Unknown plugin_id: {plugin_id}")
        return status

    def run_healthchecks(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for status in self._statuses:
            if not status.enabled or status.blocked:
                continue
            plugin = status.plugin
            entries: list[dict[str, Any]] = []
            start = monotonic()
            context = PluginContext(
                config=self._config,
                plugin_id=plugin.plugin_id,
                plugin_settings=self._settings.configs.get(plugin.plugin_id, {}),
                policy=self._policy,
                data_dir=Path(self._config.capture.data_dir),
            )
            for ext in plugin.manifest.extensions:
                if ext.factory.type != "python":
                    continue
                try:
                    factory = _load_entrypoint(ext.factory.entrypoint)
                    health_fn = getattr(factory, "health", None)
                    health = {"ok": True, "detail": "skipped"}
                    if callable(health_fn):
                        try:
                            health = health_fn()
                        except TypeError:
                            try:
                                health = health_fn(context)
                            except TypeError:
                                health = {"ok": True, "detail": "skipped"}
                    entries.append({"extension": f"{ext.kind}:{ext.id}", "health": health})
                except Exception as exc:
                    entries.append(
                        {
                            "extension": f"{ext.kind}:{ext.id}",
                            "health": {"ok": False, "detail": str(exc)},
                        }
                    )
            elapsed_ms = (monotonic() - start) * 1000
            plugin_healthcheck_latency_ms.labels(plugin.plugin_id).observe(elapsed_ms)
            results[plugin.plugin_id] = {
                "ok": (
                    all(item.get("health", {}).get("ok", False) for item in entries)
                    if entries
                    else True
                ),
                "extensions": entries,
                "latency_ms": round(elapsed_ms, 2),
            }
        return results

    def _instantiate(self, record: ExtensionRecord, factory_kwargs: dict[str, Any]) -> Any:
        plugin = _find_plugin(self._catalog, record.plugin_id)
        if plugin is None:
            raise PluginResolutionError(f"Plugin not found for {record.plugin_id}")
        factory = record.manifest.factory
        if factory.type == "python":
            func = _load_entrypoint(factory.entrypoint)
            context = PluginContext(
                config=self._config,
                plugin_id=record.plugin_id,
                plugin_settings=self._settings.configs.get(record.plugin_id, {}),
                policy=self._policy,
                data_dir=Path(self._config.capture.data_dir),
            )
            return func(context, **factory_kwargs)
        if factory.type in {"bundle", "file"}:
            root = plugin.source.root_path
            if root is None:
                raise PluginResolutionError("Plugin root path unavailable for bundle")
            path = Path(root) / factory.path
            return path
        raise PluginResolutionError(f"Unsupported factory type: {factory.type}")


def _build_statuses(
    plugins: list[DiscoveredPlugin],
    settings: PluginSettings,
    *,
    safe_mode: bool,
    errors: list[str],
) -> list[PluginStatus]:
    statuses: list[PluginStatus] = []
    seen: dict[str, list[DiscoveredPlugin]] = {}
    for plugin in plugins:
        seen.setdefault(plugin.plugin_id, []).append(plugin)
    for plugin_id, variants in seen.items():
        if len(variants) > 1:
            for plugin in variants:
                statuses.append(
                    PluginStatus(
                        plugin=plugin,
                        enabled=False,
                        blocked=True,
                        reason="duplicate_plugin_id",
                        lock_status="invalid",
                        lock_manifest=None,
                        lock_code=None,
                        manifest_sha256=plugin.manifest_sha256,
                        code_sha256=plugin.code_sha256 or "",
                    )
                )
            continue
        plugin = variants[0]
        manifest_hash, code_hash = _compute_hashes(plugin)
        enabled = _is_enabled(plugin, settings, safe_mode)
        lock = settings.locks.get(plugin.plugin_id)
        lock_status = "missing" if lock is None else "ok"
        blocked = False
        reason = None
        if enabled and plugin.source.source_type != PluginSourceType.BUILTIN:
            if lock is None:
                blocked = True
                reason = "lock_missing"
            else:
                if lock.manifest_sha256 != manifest_hash or lock.code_sha256 != code_hash:
                    blocked = True
                    reason = "lock_mismatch"
                    lock_status = "mismatch"
        if not _compat_ok(plugin):
            enabled = False
            blocked = True
            reason = "incompatible"
        statuses.append(
            PluginStatus(
                plugin=plugin,
                enabled=enabled,
                blocked=blocked,
                reason=reason,
                lock_status=lock_status,
                lock_manifest=lock.manifest_sha256 if lock else None,
                lock_code=lock.code_sha256 if lock else None,
                manifest_sha256=manifest_hash,
                code_sha256=code_hash,
            )
        )
    for error in errors:
        plugin = DiscoveredPlugin(
            plugin_id="__discovery__",
            manifest=_empty_manifest(),
            source=_empty_source(),
            manifest_bytes=b"",
            manifest_sha256="",
            warnings=[error],
        )
        statuses.append(
            PluginStatus(
                plugin=plugin,
                enabled=False,
                blocked=True,
                reason="discovery_error",
                lock_status="invalid",
                lock_manifest=None,
                lock_code=None,
                manifest_sha256="",
                code_sha256="",
            )
        )
    return statuses


def _is_enabled(plugin: DiscoveredPlugin, settings: PluginSettings, safe_mode: bool) -> bool:
    if safe_mode and plugin.source.source_type != PluginSourceType.BUILTIN:
        return False
    if plugin.plugin_id in settings.disabled:
        return False
    if plugin.plugin_id in settings.enabled:
        return True
    return bool(plugin.manifest.enabled_by_default)


def _compute_hashes(plugin: DiscoveredPlugin) -> tuple[str, str]:
    manifest_hash = plugin.manifest_sha256
    if plugin.source.source_type == PluginSourceType.ENTRYPOINT:
        dist = plugin.source.distribution
        if dist is None:
            return manifest_hash, ""
        return manifest_hash, hash_distribution_files(dist)
    root = plugin.source.root_path
    if root is None:
        return manifest_hash, ""
    if hasattr(root, "iterdir") and not isinstance(root, Path):
        return manifest_hash, hash_traversable(root)
    return manifest_hash, hash_directory(Path(root))


def _load_entrypoint(entrypoint: str) -> Any:
    if not entrypoint:
        raise PluginResolutionError("Missing factory entrypoint")
    module_name, _, attr = entrypoint.partition(":")
    if not module_name or not attr:
        raise PluginResolutionError(f"Invalid entrypoint '{entrypoint}'")
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    return value


def _find_plugin(plugins: list[DiscoveredPlugin], plugin_id: str) -> DiscoveredPlugin | None:
    for plugin in plugins:
        if plugin.plugin_id == plugin_id:
            return plugin
    return None


def _status_for(statuses: list[PluginStatus], plugin_id: str) -> PluginStatus | None:
    for status in statuses:
        if status.plugin.plugin_id == plugin_id:
            return status
    return None


def _compat_ok(plugin: DiscoveredPlugin) -> bool:
    compat = plugin.manifest.compatibility
    if compat is None:
        return True
    if compat.python and SpecifierSet and Version:
        try:
            if Version(".".join(map(str, sys.version_info[:3]))) not in SpecifierSet(compat.python):
                return False
        except Exception:
            return True
    if compat.app_min or compat.app_max:
        try:
            app_version = metadata.version("autocapture")
        except Exception:
            return True
        if SpecifierSet and Version:
            parts = []
            if compat.app_min:
                parts.append(f">={compat.app_min}")
            if compat.app_max:
                parts.append(f"<={compat.app_max}")
            spec = SpecifierSet(",".join(parts))
            try:
                return Version(app_version) in spec
            except Exception:
                return True
    return True


def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    return True


def _empty_manifest() -> Any:
    from .manifest import PluginManifestV1

    return PluginManifestV1(
        plugin_id="__discovery__",
        name="Discovery Error",
        version="0",
        extensions=[],
    )


def _empty_source() -> Any:
    from .catalog import PluginSource, PluginSourceType

    return PluginSource(
        source_type=PluginSourceType.DIRECTORY,
        location="",
        manifest_path="",
    )
