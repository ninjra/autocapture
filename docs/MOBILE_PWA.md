# Mobile / PWA Usage (Phase 2)

## Requirements
- Private overlay network (Tailscale/WireGuard).
- Remote mode enabled (`mode.mode=remote`).
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
- Avoid public port forwarding.
