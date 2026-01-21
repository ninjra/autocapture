# Security

## Phase 1 (Local mode)
- API binds to `127.0.0.1` by default.
- If you bind the API to a non-loopback address, an API key is required (fail closed).
- HTTPS is required when binding to non-loopback hosts.
- Cloud calls are disabled unless `privacy.cloud_enabled` is set.
- Screenshots are pruned after `retention.screenshot_ttl_days`.
- The bundled dashboard enforces a strict Content Security Policy and renders API output as plain text.

## Phase 2 (Remote mode)
- Requires a private overlay network (Tailscale or WireGuard).
- Set `mode.mode=remote` and `mode.overlay_interface` to the overlay interface.
- In remote mode, `api.bind_host` is automatically derived from `mode.overlay_interface`.
- HTTPS must be enabled with `mode.https_enabled=true` and TLS cert/key paths.
- Google OIDC is required when remote mode is enabled, with `mode.google_allowed_emails` restricts access.
- Offline guard is disabled in remote mode (OIDC/JWKS fetch requires outbound HTTPS).
- Remote mode fails closed if TLS or OIDC settings are missing.

## Secrets
- Local pseudonymization key stored under `data/secrets/pseudonym.key`.
- On Windows, DPAPI is used when available.
- On POSIX systems, secret files are tightened to `0600` permissions (best effort).
- Avoid logging sensitive data.

## Key portability (export/import)
- Export an encrypted bundle: `autocapture keys export --out keys.json --password "passphrase"`.
- Import on a new machine: `autocapture keys import keys.json --password "passphrase"`.
- Bundles use scrypt-derived keys + AES-256-GCM; losing the password means the keys are unrecoverable.
- Import writes keys into the configured secure storage (DPAPI-backed files on Windows, `0600` files on POSIX).
- If a key provider is `env:*`, set the environment variable manually after import.

## Database encryption at rest
- SQLite can be encrypted with SQLCipher by enabling `database.encryption_enabled`.
- SQLCipher keys can be stored via DPAPI-protected files (`dpapi_file`), plain files, or env vars.
- Use `autocapture db encrypt` to migrate a plaintext SQLite DB to encrypted form safely.
- Host tracking DB supports optional SQLCipher encryption via `tracking.encryption_enabled`.
- Raw input timelines (if enabled) are stored in the tracking DB and pruned via
  `tracking.raw_event_retention_days`.

## Native extensions
- Native SQLite extensions must be allowlisted in `CHECKSUMS.json`.
- `security.secure_mode=true` fails closed on unknown or mismatched checksums.
- Run `python tools/verify_checksums.py` to validate the allowlist offline.

## Encryption in motion
- Postgres URLs must include `sslmode=require`/`verify-*` when `database.require_tls_for_remote=true`.
- Qdrant URLs must be HTTPS when `qdrant.require_tls_for_remote=true`.

## Overlay tracker sensitive fields
- `overlay_items.last_window_title_raw`
- `overlay_items.last_browser_url_raw`
- `overlay_events.raw_window_title`
- `overlay_events.raw_browser_url`
- `overlay_items.display_name`

These fields are tagged for redaction by the cleaner/secret redaction helpers. The overlay
tracker itself does not mask them at collection time.

## Token vault (reversible pseudonyms)
- When `privacy.token_vault_enabled=true`, sensitive tokens map to encrypted originals in `token_vault`.
- Decrypting tokens requires local authorization (`privacy.allow_token_vault_decrypt=true`) and API key when remote.
- Token vault plaintext is never sent to cloud LLMs.
