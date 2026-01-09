# Mobile / PWA Usage (Phase 2)

## Requirements
- Private overlay network (Tailscale/WireGuard).
- Remote mode enabled (`mode.mode=remote`).
- `mode.overlay_interface` set to your overlay adapter (e.g. `tailscale0`, `wg0`).
- HTTPS enabled with TLS certificate and key.
- Google OIDC configured with allowed emails.

## Steps
1. Join the overlay network from your phone.
2. Start Autocapture with remote mode settings.
3. Open the overlay IP in your phone browser (HTTPS).
4. Log in using Google OIDC.
5. Install the PWA from the browser prompt.

## Notes
- Remote mode fails closed if TLS or OIDC config is missing.
- In remote mode, `api.bind_host` is automatically derived from `mode.overlay_interface`
  (you can still override it manually if needed).
- Offline guard is intentionally disabled in remote mode (Google OIDC/JWKS fetch
  requires outbound HTTPS).
- Avoid public port forwarding.
