"""Google OIDC verification helpers."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

from ..logging_utils import get_logger


@dataclass
class TokenClaims:
    email: str
    exp: int


class GoogleOIDCVerifier:
    def __init__(self, client_id: str, allowed_emails: list[str]) -> None:
        self._client_id = client_id
        self._allowed_emails = {email.lower() for email in allowed_emails}
        self._log = get_logger("security.oidc")
        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at: dt.datetime | None = None

    def verify(self, token: str) -> TokenClaims:
        jwks = self._get_jwks()
        headers = jwt.get_unverified_header(token)
        kid = headers.get("kid")
        key = next((jwk for jwk in jwks["keys"] if jwk["kid"] == kid), None)
        if not key:
            raise ValueError("Unknown signing key")
        claims = jwt.decode(
            token,
            key=jwt.algorithms.RSAAlgorithm.from_jwk(key),
            algorithms=["RS256"],
            audience=self._client_id,
        )
        email = claims.get("email")
        if not email or email.lower() not in self._allowed_emails:
            raise ValueError("Email not allowed")
        return TokenClaims(email=email, exp=int(claims["exp"]))

    def _get_jwks(self) -> dict[str, Any]:
        if self._jwks and self._jwks_fetched_at:
            if (dt.datetime.now(dt.timezone.utc) - self._jwks_fetched_at).total_seconds() < 3600:
                return self._jwks
        response = httpx.get("https://www.googleapis.com/oauth2/v3/certs", timeout=10.0)
        response.raise_for_status()
        self._jwks = response.json()
        self._jwks_fetched_at = dt.datetime.now(dt.timezone.utc)
        return self._jwks
