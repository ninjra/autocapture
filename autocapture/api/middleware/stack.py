"""Middleware stack installation."""

from __future__ import annotations

from ...config import AppConfig
from ...security.oidc import GoogleOIDCVerifier
from ...security.session import SecuritySessionManager
from .auth import install_auth_middleware
from .rate_limit import RateLimiter, install_rate_limit_middleware
from .security_headers import install_security_headers
from .security_session import install_security_session_middleware

RATE_LIMITED_PATHS = {
    "/api/answer",
    "/api/retrieve",
    "/api/context-pack",
}

PROTECTED_PREFIXES = (
    "/api/answer",
    "/api/retrieve",
    "/api/context-pack",
    "/api/context_pack",
    "/api/screenshot/",
    "/api/context_pack/",
    "/api/context-pack/",
    "/api/event/",
    "/api/highlights",
    "/api/privacy/resolve_tokens",
    "/api/delete_range",
    "/api/delete_all",
    "/api/delete/",
    "/api/state",
    "/api/settings",
    "/api/doctor",
    "/api/audit",
    "/api/events/ingest",
    "/api/storage",
    "/api/plugins",
    "/api/tracking",
)


def install_middleware(
    app,
    *,
    config: AppConfig,
    security_manager: SecuritySessionManager | None,
) -> None:
    # Order is intentional: headers -> auth -> rate limit -> unlock gating.
    install_security_headers(app)

    oidc_verifier: GoogleOIDCVerifier | None = None
    if config.mode.mode == "remote":
        oidc_verifier = GoogleOIDCVerifier(
            config.mode.google_oauth_client_id or "",
            config.mode.google_allowed_emails,
        )
    install_auth_middleware(
        app,
        config=config,
        oidc_verifier=oidc_verifier,
    )

    rate_limiter = RateLimiter(config.api.rate_limit_rps, config.api.rate_limit_burst)
    install_rate_limit_middleware(
        app,
        rate_limiter=rate_limiter,
        rate_limited_paths=RATE_LIMITED_PATHS,
    )

    install_security_session_middleware(
        app,
        config=config,
        security_manager=security_manager,
        protected_prefixes=PROTECTED_PREFIXES,
    )
