"""Security headers middleware."""

from __future__ import annotations

from fastapi import Request


def install_security_headers(app) -> None:
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):  # type: ignore[no-redef]
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        headers.setdefault("X-Robots-Tag", "noindex, nofollow")
        content_type = headers.get("content-type", "").lower()
        if content_type.startswith("text/html"):
            headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; "
                "img-src 'self' data:; "
                "script-src 'self'; "
                "style-src 'self'; "
                "object-src 'none'; "
                "base-uri 'none'; "
                "frame-ancestors 'none'",
            )
            headers.setdefault("Cache-Control", "no-store")
        if request.url.path.startswith("/api/"):
            headers.setdefault("Cache-Control", "no-store")
        return response
