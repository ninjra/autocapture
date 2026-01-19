"""Plugin discovery utilities."""

from __future__ import annotations

import importlib.metadata as metadata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import yaml

from .errors import PluginDiscoveryError
from .hash import sha256_bytes
from .manifest import PluginManifestV1, parse_manifest

try:
    import importlib.resources as resources
except Exception:  # pragma: no cover - fallback
    from importlib import resources  # type: ignore


class PluginSourceType(str, Enum):
    BUILTIN = "builtin"
    DIRECTORY = "directory"
    ENTRYPOINT = "entrypoint"


@dataclass(frozen=True)
class PluginSource:
    source_type: PluginSourceType
    location: str
    manifest_path: str
    root_path: Path | object | None = None
    assets_path: Path | None = None
    entry_point: metadata.EntryPoint | None = None
    distribution: metadata.Distribution | None = None


@dataclass(frozen=True)
class DiscoveredPlugin:
    plugin_id: str
    manifest: PluginManifestV1
    source: PluginSource
    manifest_bytes: bytes
    manifest_sha256: str
    code_sha256: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DiscoveryResult:
    plugins: list[DiscoveredPlugin]
    errors: list[str]


def discover_plugins(
    *,
    plugin_dir: Path,
    entry_points: Iterable[metadata.EntryPoint] | None = None,
) -> DiscoveryResult:
    plugins: list[DiscoveredPlugin] = []
    errors: list[str] = []
    for plugin in _discover_builtin():
        plugins.append(plugin)
    for plugin in _discover_directory(plugin_dir, errors):
        plugins.append(plugin)
    for plugin in _discover_entrypoints(entry_points, errors):
        plugins.append(plugin)
    priority = {
        PluginSourceType.BUILTIN: 0,
        PluginSourceType.DIRECTORY: 1,
        PluginSourceType.ENTRYPOINT: 2,
    }
    plugins.sort(key=lambda item: (priority.get(item.source.source_type, 99), item.plugin_id))
    return DiscoveryResult(plugins=plugins, errors=errors)


def _discover_builtin() -> Iterable[DiscoveredPlugin]:
    package = "autocapture.plugins.builtin"
    root = resources.files(package)
    for entry in _iter_traversable(root):
        if entry.name != "plugin.yaml":
            continue
        try:
            manifest_bytes = entry.read_bytes()
            manifest = _load_manifest(manifest_bytes)
            assets = entry.parent / "assets"
            assets_path = Path(assets) if assets.is_dir() else None
            source = PluginSource(
                source_type=PluginSourceType.BUILTIN,
                location=package,
                manifest_path=str(entry),
                root_path=entry.parent if hasattr(entry, "parent") else None,
                assets_path=assets_path,
            )
            yield DiscoveredPlugin(
                plugin_id=manifest.plugin_id,
                manifest=manifest,
                source=source,
                manifest_bytes=manifest_bytes,
                manifest_sha256=sha256_bytes(manifest_bytes),
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise PluginDiscoveryError(f"Failed to load builtin manifest: {exc}") from exc


def _discover_directory(plugin_dir: Path, errors: list[str]) -> Iterable[DiscoveredPlugin]:
    if not plugin_dir.exists():
        return []
    plugins: list[DiscoveredPlugin] = []
    for child in sorted(plugin_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        manifest_path = child / "plugin.yaml"
        if not manifest_path.exists():
            continue
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = _load_manifest(manifest_bytes)
            assets_dir = child / "assets"
            source = PluginSource(
                source_type=PluginSourceType.DIRECTORY,
                location=str(child),
                manifest_path=str(manifest_path),
                root_path=child,
                assets_path=assets_dir if assets_dir.exists() else None,
            )
            plugins.append(
                DiscoveredPlugin(
                    plugin_id=manifest.plugin_id,
                    manifest=manifest,
                    source=source,
                    manifest_bytes=manifest_bytes,
                    manifest_sha256=sha256_bytes(manifest_bytes),
                )
            )
        except Exception as exc:
            errors.append(f"Directory plugin {child.name} failed: {exc}")
    return plugins


def _discover_entrypoints(
    entry_points: Iterable[metadata.EntryPoint] | None,
    errors: list[str],
) -> Iterable[DiscoveredPlugin]:
    plugins: list[DiscoveredPlugin] = []
    if entry_points is None:
        entry_points = metadata.entry_points(group="autocapture.plugins")
    for entry in sorted(entry_points, key=lambda ep: ep.name):
        try:
            dist = entry.dist
            if dist is None:
                raise PluginDiscoveryError("Entry point missing distribution")
            manifest_rel = Path("autocapture_plugins") / f"{entry.name}.yaml"
            manifest_path = dist.locate_file(manifest_rel)
            if not manifest_path.exists():
                raise PluginDiscoveryError(f"Manifest not found: {manifest_rel}")
            manifest_bytes = Path(manifest_path).read_bytes()
            manifest = _load_manifest(manifest_bytes)
            assets_root = dist.locate_file(Path("autocapture_plugins") / entry.name / "assets")
            assets_path = Path(assets_root) if Path(assets_root).exists() else None
            root_path = dist.locate_file(Path("autocapture_plugins") / entry.name)
            source = PluginSource(
                source_type=PluginSourceType.ENTRYPOINT,
                location=dist.metadata.get("Name", ""),
                manifest_path=str(manifest_path),
                root_path=Path(root_path) if Path(root_path).exists() else None,
                assets_path=assets_path,
                entry_point=entry,
                distribution=dist,
            )
            plugins.append(
                DiscoveredPlugin(
                    plugin_id=manifest.plugin_id,
                    manifest=manifest,
                    source=source,
                    manifest_bytes=manifest_bytes,
                    manifest_sha256=sha256_bytes(manifest_bytes),
                )
            )
        except Exception as exc:
            errors.append(f"Entrypoint plugin {entry.name} failed: {exc}")
    return plugins


def _load_manifest(manifest_bytes: bytes) -> PluginManifestV1:
    payload = yaml.safe_load(manifest_bytes) or {}
    return parse_manifest(payload)


def _iter_traversable(root) -> Iterable:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
            for entry in entries:
                if entry.is_dir():
                    stack.append(entry)
                else:
                    yield entry
        except Exception:
            continue
