# Security

## Phase 1 (Local mode)
- API binds to `127.0.0.1` by default.
- Cloud calls are disabled unless `privacy.cloud_enabled` is set.
- Screenshots are pruned after `retention.screenshot_ttl_days`.

## Phase 2 (Remote mode)
- Requires a private overlay network (Tailscale or WireGuard).
- Set `mode.mode=remote` and `mode.overlay_interface` to the overlay interface.
- HTTPS must be enabled with `mode.https_enabled=true` and TLS cert/key paths.
- Google OIDC is required when remote mode is enabled, with `mode.google_allowed_emails` restricts access.
- Remote mode fails closed if TLS or OIDC settings are missing.

## Secrets
- Local pseudonymization key stored under `data/secrets/pseudonym.key`.
- On Windows, DPAPI is used when available.
- Avoid logging sensitive data.
