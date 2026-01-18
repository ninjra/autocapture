from __future__ import annotations

import os
import sys

import pytest


def test_popup_default_payload_types() -> None:
    if not os.environ.get("AUTOCAPTURE_QT_TESTS"):
        pytest.skip("Qt UI tests require AUTOCAPTURE_QT_TESTS=1")
    if os.environ.get("CI"):
        pytest.skip("Qt UI tests disabled in CI")
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            pytest.skip("No display available for Qt tests")
    pytest.importorskip("PySide6")
    try:
        from PySide6 import QtWidgets
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")

    from autocapture.ui.popup import SearchPopup

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    popup = SearchPopup("http://localhost")

    assert popup._default_payload(popup.suggestions_ready) == []
    assert popup._default_payload(popup.answer_ready) == {}
