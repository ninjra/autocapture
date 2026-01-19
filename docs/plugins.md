# Plugins

Autocapture uses a plugin-first system for providers and pipeline stages. Plugins are discovered
without importing code and must be explicitly enabled before their extensions are usable.

## Terminology
- **Plugin**: a package or folder with `plugin.yaml`.
- **Extension**: a specific capability (e.g., `llm.provider:openai`).

## Discovery
Plugins are discovered from:
1. Built-ins bundled with Autocapture.
2. Folder plugins at `${capture.data_dir}/plugins`.
3. Entry-point plugins under the `autocapture.plugins` group.

Discovery does **not** import plugin code.

## Enable / Disable
External plugins require explicit enablement and hash acceptance.

CLI:
```bash
poetry run autocapture plugins list
poetry run autocapture plugins enable <plugin_id> --accept-hashes
poetry run autocapture plugins disable <plugin_id>
poetry run autocapture plugins lock <plugin_id>   # re-approve hashes
poetry run autocapture plugins doctor
```

API:
- `GET /api/plugins/catalog`
- `GET /api/plugins/extensions?kind=llm.provider`
- `POST /api/plugins/enable`
- `POST /api/plugins/disable`
- `POST /api/plugins/lock`
- `GET /api/plugins/health`

UI:
Open the **Plugins** tab to enable/disable plugins and review hashes.

## Settings Structure
Plugin state is stored in `settings.json`:
```json
{
  "plugins": {
    "enabled": ["vendor.plugin_id"],
    "disabled": [],
    "extension_overrides": {
      "llm.provider:openai": "vendor.plugin_id"
    },
    "locks": {
      "vendor.plugin_id": {
        "manifest_sha256": "...",
        "code_sha256": "...",
        "accepted_at_utc": "2026-01-19T00:00:00Z"
      }
    },
    "configs": {
      "vendor.plugin_id": {
        "custom_key": "value"
      }
    }
  }
}
```

## Safety Rules
Core-owned policy gates enforce:
- `privacy.cloud_enabled=true` and `offline=false` for cloud text providers.
- `privacy.allow_cloud_images=true` for cloud image providers.
- Stage-level `allow_cloud` flags when a stage context is present.

Plugins cannot bypass these rules.

## Deterministic Resolution
If two enabled plugins provide the same extension ID, resolution fails unless
`plugins.extension_overrides` specifies the winning plugin ID.

## Safe Mode
Set `AUTOCAPTURE_SAFE_MODE=1` or `plugins.safe_mode=true` to load built-ins only.
