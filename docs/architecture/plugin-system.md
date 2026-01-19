# Plugin System Architecture (SPEC-1)

## Overview
Autocapture uses a plugin-first module system to wrap providers and pipeline stages behind
deterministic extension IDs. Plugins are discovered without importing code, must be explicitly
enabled, and are locked by hash to prevent silent changes.

## Layout
- Core implementation: `autocapture/plugins/`
- Built-in manifests: `autocapture/plugins/builtin/**/plugin.yaml`
- SDK surface for authors: `autocapture/plugins/sdk/`
- Settings storage: `settings.json` under `capture.data_dir`

## Manifest v1
Each plugin ships a `plugin.yaml`:
- `plugin_id`, `name`, `version`, `enabled_by_default`
- `extensions[]` with `kind`, `id`, `name`, `aliases`, `pillars`, `factory`
- `factory.type` supports `python`, `bundle`, and `file`

Factories are resolved only when the extension is used. Disabled plugins are not imported.

## Discovery
Plugins are discovered from three sources (deterministic ordering):
1. Built-in manifests packaged with the app.
2. Directory plugins under `${capture.data_dir}/plugins`.
3. Installed packages via entry points group `autocapture.plugins`.

Entry-point plugins must include `autocapture_plugins/<plugin_id>.yaml` in their distribution.

## Enablement + Locks
External plugins require explicit enablement and hash acceptance:
- `settings.json` section `plugins.enabled`
- `plugins.locks[plugin_id]` stores `manifest_sha256` + `code_sha256`
- If hashes change, the plugin is blocked until re-approved.

Safe mode (`AUTOCAPTURE_SAFE_MODE=1` or `plugins.safe_mode=true`) loads built-ins only.

## Resolution + Overrides
Extensions are resolved by `(kind, extension_id)`:
- Single match: used directly.
- Multiple matches: fail closed unless an override exists in
  `plugins.extension_overrides`, e.g. `{"llm.provider:openai": "vendor.plugin_id"}`.

## Policy Gate
Core-owned guards enforce privacy and offline rules at resolution time:
- Text-to-cloud requires `privacy.cloud_enabled=true`, `offline=false`,
  and `model_stages.<stage>.allow_cloud=true`.
- Image-to-cloud additionally requires `privacy.allow_cloud_images=true`.

Manifests can declare data-handling pillars for UI warnings, but cannot bypass core gates.

## Hashing
Hashing is deterministic:
- Manifest hash: SHA-256 of `plugin.yaml` bytes.
- Code hash: all files under plugin root (excluding caches), or distribution files for entry points.

Hashes are stored on enable and verified at startup.
