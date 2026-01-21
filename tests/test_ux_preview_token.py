from __future__ import annotations

import datetime as dt

import pytest

from autocapture.ux.preview import PreviewTokenManager, PreviewTokenError


def test_preview_token_roundtrip(tmp_path) -> None:
    manager = PreviewTokenManager(tmp_path, default_ttl_s=1)
    token = manager.issue(kind="settings", version="v1", payload_hash="h1", ttl_s=1)
    payload = manager.validate(token, kind="settings", version="v1", payload_hash="h1")
    assert payload["kind"] == "settings"


def test_preview_token_rejects_mismatch(tmp_path) -> None:
    manager = PreviewTokenManager(tmp_path)
    token = manager.issue(kind="settings", version="v1", payload_hash="h1", ttl_s=1)
    with pytest.raises(PreviewTokenError):
        manager.validate(token, kind="settings", version="v1", payload_hash="h2")


def test_preview_token_expiry(tmp_path) -> None:
    manager = PreviewTokenManager(tmp_path, default_ttl_s=1)
    token = manager.issue(kind="settings", version="v1", payload_hash="h1", ttl_s=1)
    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5)
    with pytest.raises(PreviewTokenError):
        manager.validate(token, kind="settings", version="v1", payload_hash="h1", now=future)
