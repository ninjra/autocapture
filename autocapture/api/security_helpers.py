"""Security helper utilities shared by API routes and middleware."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from ..config import AppConfig, is_loopback_host
from ..security.session import SecuritySessionManager, is_test_mode


def build_security_manager(config: AppConfig) -> SecuritySessionManager | None:
    if (
        config.security.local_unlock_enabled
        and config.mode.mode != "remote"
        and is_loopback_host(config.api.bind_host)
        and config.security.provider != "disabled"
    ):
        return SecuritySessionManager(
            ttl_seconds=config.security.session_ttl_seconds,
            provider=config.security.provider,
            test_mode_bypass=is_test_mode(),
        )
    return None


def _needs_unlock(request: Request, prefixes: tuple[str, ...]) -> bool:
    path = request.url.path
    if not path.startswith("/api/"):
        return False
    return any(path.startswith(prefix) for prefix in prefixes)


def _extract_unlock_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1]
    return request.query_params.get("unlock")


def _bridge_token_valid(request: Request, config: AppConfig) -> bool:
    token = request.headers.get("X-Bridge-Token")
    expected = config.api.bridge_token
    if not token or not expected:
        return False
    return hmac.compare_digest(token, expected)


def _require_api_key(request: Request, config: AppConfig) -> None:
    if not config.api.require_api_key:
        return
    key = request.headers.get("X-API-Key")
    auth = request.headers.get("Authorization", "")
    if not key and auth.startswith("Bearer "):
        key = auth.split(" ", 1)[1]
    if not key or not config.api.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not hmac.compare_digest(key, config.api.api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")
