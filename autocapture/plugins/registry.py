"""Extension registry and resolution."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PluginResolutionError
from ..observability.metrics import extension_resolution_conflicts_total
from .manifest import ExtensionManifestV1


@dataclass(frozen=True)
class ExtensionRecord:
    kind: str
    extension_id: str
    plugin_id: str
    name: str
    manifest: ExtensionManifestV1
    source: str


class ExtensionRegistry:
    def __init__(self) -> None:
        self._extensions: dict[str, dict[str, list[ExtensionRecord]]] = {}
        self._aliases: dict[str, dict[str, set[str]]] = {}

    def register_extension(
        self,
        *,
        kind: str,
        extension_id: str,
        plugin_id: str,
        name: str,
        manifest: ExtensionManifestV1,
        source: str,
    ) -> None:
        bucket = self._extensions.setdefault(kind, {})
        records = bucket.setdefault(extension_id, [])
        records.append(
            ExtensionRecord(
                kind=kind,
                extension_id=extension_id,
                plugin_id=plugin_id,
                name=name,
                manifest=manifest,
                source=source,
            )
        )
        alias_bucket = self._aliases.setdefault(kind, {})
        for alias in manifest.aliases:
            alias_bucket.setdefault(alias, set()).add(extension_id)

    def resolve(
        self,
        kind: str,
        extension_id: str,
        *,
        overrides: dict[str, str] | None = None,
    ) -> ExtensionRecord:
        overrides = overrides or {}
        canonical_id = self._resolve_alias(kind, extension_id)
        records = self._extensions.get(kind, {}).get(canonical_id, [])
        if not records:
            raise PluginResolutionError(f"Extension not available: {kind}:{extension_id}")
        if len(records) == 1:
            return records[0]
        override_key = f"{kind}:{extension_id}"
        override = overrides.get(override_key) or overrides.get(f"{kind}:{canonical_id}")
        if override:
            for record in records:
                if record.plugin_id == override:
                    return record
            raise PluginResolutionError(
                f"Override plugin '{override}' not found for {kind}:{canonical_id}"
            )
        extension_resolution_conflicts_total.labels(kind, canonical_id).inc()
        raise PluginResolutionError(
            f"Multiple providers for {kind}:{canonical_id}; set plugins.extension_overrides"
        )

    def list_extensions(self, kind: str) -> list[ExtensionRecord]:
        bucket = self._extensions.get(kind, {})
        results: list[ExtensionRecord] = []
        for extension_id in sorted(bucket.keys()):
            results.extend(bucket[extension_id])
        return results

    def list_kinds(self) -> list[str]:
        return sorted(self._extensions.keys())

    def _resolve_alias(self, kind: str, extension_id: str) -> str:
        alias_bucket = self._aliases.get(kind, {})
        candidates = alias_bucket.get(extension_id)
        if not candidates:
            return extension_id
        if len(candidates) > 1:
            extension_resolution_conflicts_total.labels(kind, extension_id).inc()
            raise PluginResolutionError(
                f"Alias '{extension_id}' is ambiguous for kind {kind}: {sorted(candidates)}"
            )
        return next(iter(candidates))
