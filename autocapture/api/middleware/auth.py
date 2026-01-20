"""Authentication middleware."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from ...config import AppConfig
from ...logging_utils import get_logger
from ...security.oidc import GoogleOIDCVerifier
from ..security_helpers import _bridge_token_valid, _require_api_key


def install_auth_middleware(
    app,
    *,
    config: AppConfig,
    oidc_verifier: GoogleOIDCVerifier | None,
) -> None:
    log = get_logger("api")
    if config.mode.mode == "remote":

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):  # type: ignore[no-redef]
            if request.url.path == "/api/events/ingest" and _bridge_token_valid(request, config):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or not oidc_verifier:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            token = auth.split(" ", 1)[1]
            try:
                oidc_verifier.verify(token)
            except Exception as exc:
                log.warning("OIDC verification failed: {}", exc)
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            return await call_next(request)

    else:

        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next):  # type: ignore[no-redef]
            if request.url.path.startswith("/api/"):
                if request.url.path == "/api/events/ingest" and _bridge_token_valid(request, config):
                    return await call_next(request)
                _require_api_key(request, config)
            return await call_next(request)
