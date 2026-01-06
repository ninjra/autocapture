from __future__ import annotations

import pytest


def test_popup_default_payload_types() -> None:
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
