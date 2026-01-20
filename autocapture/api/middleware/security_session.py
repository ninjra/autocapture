"""Unlock gating middleware."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from ...config import AppConfig
from ...security.session import SecuritySessionManager
from ..security_helpers import _bridge_token_valid, _extract_unlock_token, _needs_unlock


def install_security_session_middleware(
    app,
    *,
    config: AppConfig,
    security_manager: SecuritySessionManager | None,
    protected_prefixes: tuple[str, ...],
) -> None:
    @app.middleware("http")
    async def security_session_middleware(request: Request, call_next):  # type: ignore[no-redef]
        if security_manager and _needs_unlock(request, protected_prefixes):
            if request.url.path == "/api/events/ingest" and _bridge_token_valid(request, config):
                return await call_next(request)
            token = _extract_unlock_token(request)
            if not security_manager.is_unlocked(token):
                return JSONResponse(status_code=401, content={"detail": "Unlock required"})
        return await call_next(request)
