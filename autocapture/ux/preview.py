"""Preview token issuance and validation (HMAC, local-only)."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any

from .redaction import redact_payload


class PreviewTokenError(ValueError):
    pass


def _stable_json(value: Any) -> str:
    def _normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): _normalize(val) for key, val in sorted(item.items())}
        if isinstance(item, list):
            return [_normalize(val) for val in item]
        if isinstance(item, (dt.datetime,)):
            return item.isoformat()
        return item

    normalized = _normalize(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


class PreviewTokenManager:
    def __init__(self, data_dir: Path, *, default_ttl_s: int = 600) -> None:
        self._data_dir = data_dir
        self._secret_path = data_dir / ".preview_secret"
        self._default_ttl_s = int(default_ttl_s)
        self._secret = self._load_or_create_secret()

    def issue(self, *, kind: str, version: str, payload_hash: str, ttl_s: int | None = None) -> str:
        now = dt.datetime.now(dt.timezone.utc)
        ttl = int(ttl_s or self._default_ttl_s)
        payload = {
            "kind": kind,
            "version": version,
            "payload_hash": payload_hash,
            "iat": now.isoformat(),
            "exp": (now + dt.timedelta(seconds=ttl)).isoformat(),
            "nonce": secrets.token_hex(8),
        }
        body = _stable_json(payload).encode("utf-8")
        sig = hmac.new(self._secret, body, hashlib.sha256).digest()
        return f"{_b64url(body)}.{_b64url(sig)}"

    def validate(
        self,
        token: str,
        *,
        kind: str,
        version: str,
        payload_hash: str,
        now: dt.datetime | None = None,
    ) -> dict[str, Any]:
        if "." not in token:
            raise PreviewTokenError("invalid token format")
        body_b64, sig_b64 = token.split(".", 1)
        try:
            body = _b64url_decode(body_b64)
            signature = _b64url_decode(sig_b64)
        except Exception as exc:
            raise PreviewTokenError("invalid token encoding") from exc
        expected = hmac.new(self._secret, body, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise PreviewTokenError("invalid token signature")
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise PreviewTokenError("invalid token payload") from exc
        now = now or dt.datetime.now(dt.timezone.utc)
        exp = payload.get("exp")
        if exp:
            try:
                exp_dt = dt.datetime.fromisoformat(exp)
            except ValueError as exc:
                raise PreviewTokenError("invalid token expiry") from exc
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=dt.timezone.utc)
            if exp_dt <= now:
                raise PreviewTokenError("token expired")
        if payload.get("kind") != kind:
            raise PreviewTokenError("token kind mismatch")
        if payload.get("version") != version:
            raise PreviewTokenError("token version mismatch")
        if payload.get("payload_hash") != payload_hash:
            raise PreviewTokenError("token payload mismatch")
        return payload

    def redact(self) -> dict[str, Any]:
        return redact_payload({"secret_path": str(self._secret_path)})

    def _load_or_create_secret(self) -> bytes:
        if self._secret_path.exists():
            try:
                return self._secret_path.read_bytes()
            except Exception:
                pass
        secret = secrets.token_bytes(32)
        self._secret_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._secret_path.write_bytes(secret)
            if os.name != "nt":
                os.chmod(self._secret_path, 0o600)
        except Exception:
            return secret
        return secret
