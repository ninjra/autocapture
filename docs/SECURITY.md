# Security

## Phase 1 (Local mode)
- API binds to `127.0.0.1` by default.
- If you bind the API to a non-loopback address, an API key is required (fail closed).
- Cloud calls are disabled unless `privacy.cloud_enabled` is set.
- Screenshots are pruned after `retention.screenshot_ttl_days`.

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
