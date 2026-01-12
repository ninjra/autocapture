from __future__ import annotations

import datetime as dt

import json

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from autocapture.security.oidc import GoogleOIDCVerifier


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def _make_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())
    return private_key, public_jwk


def _make_token(private_key, kid: str, claims: dict) -> str:
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def test_email_verified_required(monkeypatch: pytest.MonkeyPatch) -> None:
    private_key, public_jwk = _make_keypair()
    kid = "kid-1"
    jwk_dict = json.loads(public_jwk)
    jwk_dict["kid"] = kid
    jwks = {"keys": [jwk_dict]}

    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: DummyResponse(jwks),
    )

    verifier = GoogleOIDCVerifier("client", ["user@example.com"])
    claims = {
        "iss": "https://accounts.google.com",
        "aud": "client",
        "exp": int((dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5)).timestamp()),
        "email": "user@example.com",
        "email_verified": False,
    }
    token = _make_token(private_key, kid, claims)
    with pytest.raises(ValueError, match="Email not verified"):
        verifier.verify(token)


def test_leeway_allows_clock_skew(monkeypatch: pytest.MonkeyPatch) -> None:
    private_key, public_jwk = _make_keypair()
    kid = "kid-2"
    jwk_dict = json.loads(public_jwk)
    jwk_dict["kid"] = kid
    jwks = {"keys": [jwk_dict]}
    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: DummyResponse(jwks),
    )

    verifier = GoogleOIDCVerifier("client", ["user@example.com"])
    expired = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=30)
    claims = {
        "iss": "accounts.google.com",
        "aud": "client",
        "exp": int(expired.timestamp()),
        "email": "user@example.com",
        "email_verified": True,
    }
    token = _make_token(private_key, kid, claims)
    verifier.verify(token)


def test_kid_miss_triggers_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    private_key, public_jwk = _make_keypair()
    kid = "kid-refresh"
    jwks_initial = {"keys": []}
    jwk_dict = json.loads(public_jwk)
    jwk_dict["kid"] = kid
    jwks_updated = {"keys": [jwk_dict]}
    calls = {"count": 0}

    def _fake_get(*_args, **_kwargs):
        calls["count"] += 1
        payload = jwks_initial if calls["count"] == 1 else jwks_updated
        return DummyResponse(payload)

    monkeypatch.setattr("httpx.get", _fake_get)

    verifier = GoogleOIDCVerifier("client", ["user@example.com"])
    claims = {
        "iss": "https://accounts.google.com",
        "aud": "client",
        "exp": int((dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5)).timestamp()),
        "email": "user@example.com",
        "email_verified": True,
    }
    token = _make_token(private_key, kid, claims)
    verifier.verify(token)
    assert calls["count"] == 2
