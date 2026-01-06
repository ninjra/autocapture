from __future__ import annotations

import pytest

from autocapture.capture.raw_input import RawInputListener


def test_raw_input_no_hotkey_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    def registrar(_hwnd: int, _modifiers: int, _vk: int) -> bool:
        raise AssertionError("hotkey registrar should not be called")

    listener = RawInputListener(
        idle_grace_ms=1000,
        on_activity=None,
        on_hotkey=None,
        hotkey_registrar=registrar,
    )

    assert listener._register_hotkey(1234) is True
