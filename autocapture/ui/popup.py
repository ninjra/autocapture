"""Popup search UI."""

from __future__ import annotations

import uuid
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import sys
import ctypes

import httpx
from PySide6 import QtCore, QtGui, QtWidgets

from ..config import SearchPopupConfig
from ..logging_utils import get_logger


class SearchPopup(QtWidgets.QWidget):
    suggestions_ready = QtCore.Signal(str, list)
    answer_ready = QtCore.Signal(str, dict)

    def __init__(self, api_base_url: str, config: SearchPopupConfig | None = None) -> None:
        super().__init__()
        self._api_base_url = api_base_url.rstrip("/")
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._config = config or SearchPopupConfig()
        self._suggest_timer = QtCore.QTimer(self)
        self._suggest_timer.setSingleShot(True)
        self._suggest_timer.setInterval(180)
        self._suggest_timer.timeout.connect(self._request_suggestions)
        self._current_suggest_token: str | None = None
        self._current_answer_token: str | None = None
        self._unlock_token: str | None = None
        self._unlock_expires_at: dt.datetime | None = None
        self._pinned = bool(self._config.pin_default)
        self._focusable = True
        self._previous_hwnd: int | None = None
        self._active_opacity = float(self._config.active_opacity)
        self._inactive_opacity = min(
            float(self._config.inactive_opacity), float(self._config.active_opacity)
        )
        self._log = get_logger("ui.popup")
        self._setup_ui()
        self.suggestions_ready.connect(self._apply_suggestions)
        self.answer_ready.connect(self._apply_answer)

    def _setup_ui(self) -> None:
        self.setWindowFlags(
            QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        container = QtWidgets.QFrame()
        container.setObjectName("container")
        container.setStyleSheet(
            "#container {"
            "background-color: rgba(20, 22, 33, 220);"
            "border-radius: 18px;"
            "border: 1px solid rgba(120, 180, 255, 120);"
            "}"
        )

        self._input = QtWidgets.QLineEdit()
        self._input.setPlaceholderText("Ask your timeline...")
        self._input.textChanged.connect(self._on_text_changed)
        self._input.returnPressed.connect(self._submit_query)
        self._input.setStyleSheet(
            "QLineEdit {"
            "padding: 10px;"
            "border-radius: 12px;"
            "border: 1px solid rgba(140, 160, 220, 120);"
            "background: rgba(12, 14, 20, 200);"
            "color: #f4f5f7;"
            "font-size: 15px;"
            "}"
        )

        self._time_range = QtWidgets.QComboBox()
        self._time_range.addItems(["Last 24h", "Last 7 days", "Last 30 days"])
        self._time_range.setCurrentIndex(0)
        self._time_range.setStyleSheet("color: #f4f5f7;")

        self._sanitize_toggle = QtWidgets.QCheckBox("Sanitize")
        self._sanitize_toggle.setChecked(True)
        self._sanitize_toggle.setStyleSheet("color: #d0d7ff;")

        self._extractive_toggle = QtWidgets.QCheckBox("Extractive-only")
        self._extractive_toggle.setChecked(True)
        self._extractive_toggle.setStyleSheet("color: #d0d7ff;")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._time_range)
        controls.addStretch(1)
        controls.addWidget(self._sanitize_toggle)
        controls.addWidget(self._extractive_toggle)

        self._pin_button = QtWidgets.QToolButton()
        self._pin_button.setText("Pin")
        self._pin_button.setCheckable(True)
        self._pin_button.setChecked(self._pinned)
        self._pin_button.clicked.connect(self._toggle_pin)
        self._pin_button.setStyleSheet(
            "QToolButton { color: #d0d7ff; padding: 4px 8px; }"
            "QToolButton:checked { background: rgba(90, 120, 200, 120); }"
        )

        self._opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._opacity_slider.setRange(40, 100)
        self._opacity_slider.setValue(int(self._active_opacity * 100))
        self._opacity_slider.setFixedWidth(120)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)

        opacity_label = QtWidgets.QLabel("Opacity")
        opacity_label.setStyleSheet("color: #d0d7ff; font-size: 11px;")

        popup_controls = QtWidgets.QHBoxLayout()
        popup_controls.addWidget(self._pin_button)
        popup_controls.addStretch(1)
        popup_controls.addWidget(opacity_label)
        popup_controls.addWidget(self._opacity_slider)

        self._suggestions = QtWidgets.QListWidget()
        self._suggestions.setFixedHeight(160)
        self._suggestions.setStyleSheet(
            "QListWidget {"
            "background: transparent;"
            "border: none;"
            "color: #d0d7ff;"
            "}"
            "QListWidget::item { padding: 6px; }"
            "QListWidget::item:selected { background: rgba(120, 120, 200, 80); }"
        )
        self._suggestions.itemActivated.connect(self._use_suggestion)

        self._preview = QtWidgets.QLabel("")
        self._preview.setTextFormat(QtCore.Qt.PlainText)
        self._preview.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._preview.setOpenExternalLinks(False)
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet("color: #9ad1ff; font-size: 12px;")

        self._citations = QtWidgets.QTextBrowser()
        self._citations.setOpenExternalLinks(True)
        self._citations.setStyleSheet(
            "color: #c2d6ff; font-size: 11px; background: transparent; border: none;"
        )
        self._citations.setFixedHeight(60)

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(self._input)
        layout.addLayout(controls)
        layout.addLayout(popup_controls)
        layout.addWidget(self._suggestions)
        layout.addWidget(self._preview)
        layout.addWidget(self._citations)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.addWidget(container)
        self.resize(540, 320)

    def show_popup(self) -> None:
        self._capture_previous_focus()
        self._set_focusable(True)
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            rect = screen.availableGeometry()
            self.move(
                rect.center().x() - self.width() // 2,
                rect.center().y() - self.height() // 2,
            )
        self.show()
        self.raise_()
        if self._config.focus_steal_on_show:
            self.activateWindow()
            self._input.setFocus()
        self._apply_opacity(active=True)

    def _on_text_changed(self, text: str) -> None:
        self._suggest_timer.start()
        if not text.strip():
            self._suggestions.clear()

    def _request_suggestions(self) -> None:
        query = self._input.text().strip()
        if not query:
            return
        token = str(uuid.uuid4())
        self._current_suggest_token = token
        future = self._executor.submit(self._fetch_suggestions, query)
        future.add_done_callback(
            lambda fut: self._emit_future_result(self.suggestions_ready, token, fut)
        )

    def _fetch_suggestions(self, query: str) -> list[dict[str, Any]]:
        payload = {"q": query}
        with httpx.Client(timeout=2.0) as client:
            response = client.post(f"{self._api_base_url}/api/suggest", json=payload)
            response.raise_for_status()
            return response.json()

    def _apply_suggestions(self, token: str, suggestions: list[dict[str, Any]]) -> None:
        if token != self._current_suggest_token:
            return
        self._suggestions.clear()
        for item in suggestions:
            list_item = QtWidgets.QListWidgetItem(item.get("snippet", ""))
            list_item.setData(QtCore.Qt.UserRole, item)
            self._suggestions.addItem(list_item)

    def _use_suggestion(self, item: QtWidgets.QListWidgetItem) -> None:
        data = item.data(QtCore.Qt.UserRole) or {}
        snippet = data.get("snippet")
        if snippet:
            self._input.setText(snippet)
            self._input.setCursorPosition(len(snippet))

    def _submit_query(self) -> None:
        query = self._input.text().strip()
        if not query:
            return
        token = str(uuid.uuid4())
        self._current_answer_token = token
        self._preview.setText("Thinking...")
        future = self._executor.submit(self._fetch_answer, query)
        future.add_done_callback(
            lambda fut: self._emit_future_result(self.answer_ready, token, fut)
        )
        if self._config.focus_return_on_submit and not self._pinned:
            self._restore_previous_focus()
            self._set_focusable(False)
            if self._config.fade_when_inactive:
                self._apply_opacity(active=False)

    def _fetch_answer(self, query: str) -> dict[str, Any]:
        payload = {
            "q": query,
            "sanitize": self._sanitize_toggle.isChecked(),
            "extractive_only": self._extractive_toggle.isChecked(),
            "time_range": self._selected_time_range(),
        }
        with httpx.Client(timeout=45.0) as client:
            headers = {}
            token = self._ensure_unlocked(client)
            if token:
                headers["Authorization"] = f"Bearer {token}"
            response = client.post(
                f"{self._api_base_url}/api/answer", json=payload, headers=headers
            )
            response.raise_for_status()
            return response.json()

    def _apply_answer(self, token: str, answer: dict[str, Any]) -> None:
        if token != self._current_answer_token:
            return
        text = answer.get("answer", "")
        self._preview.setText(text)
        self._apply_citations(answer)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.hide()
        event.ignore()

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self._apply_opacity(active=True)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusOutEvent(event)
        if self._config.fade_when_inactive and not self._pinned:
            self._apply_opacity(active=False)

    def _emit_future_result(self, signal, token: str, future) -> None:
        try:
            payload = future.result()
        except Exception as exc:
            self._log.warning("Popup request failed: {}", exc)
            payload = self._default_payload(signal)
        signal.emit(token, payload)

    def _default_payload(self, signal):
        if signal == self.suggestions_ready:
            return []
        if signal == self.answer_ready:
            return {}
        return {}

    def _selected_time_range(self) -> list[str]:
        now = dt.datetime.now(dt.timezone.utc)
        selection = self._time_range.currentIndex()
        if selection == 0:
            start = now - dt.timedelta(hours=24)
        elif selection == 1:
            start = now - dt.timedelta(days=7)
        else:
            start = now - dt.timedelta(days=30)
        return [start.isoformat(), now.isoformat()]

    def _ensure_unlocked(self, client: httpx.Client) -> str | None:
        if self._unlock_token and self._unlock_expires_at:
            if self._unlock_expires_at > dt.datetime.now(dt.timezone.utc):
                return self._unlock_token
        try:
            response = client.post(f"{self._api_base_url}/api/unlock")
            response.raise_for_status()
            data = response.json()
            token = data.get("token")
            expires_at = data.get("expires_at")
            if token and expires_at:
                self._unlock_token = token
                self._unlock_expires_at = dt.datetime.fromisoformat(expires_at)
            return token
        except Exception as exc:
            self._log.warning("Unlock failed: {}", exc)
            return None

    def _apply_citations(self, answer: dict[str, Any]) -> None:
        citations = answer.get("citations") or []
        pack = answer.get("used_context_pack") or {}
        evidence = pack.get("evidence") or []
        evidence_map = {
            item.get("id"): (item.get("meta") or {}).get("event_id") for item in evidence
        }
        if not citations:
            self._citations.setHtml("")
            return
        links = []
        token = self._unlock_token or ""
        for cite in citations:
            event_id = evidence_map.get(cite)
            if event_id:
                url = f"{self._api_base_url}/api/screenshot/{event_id}"
                if token:
                    url = f"{url}?unlock={token}"
                links.append(f'<a href="{url}">{cite}</a>')
            else:
                links.append(str(cite))
        html = "Citations: " + ", ".join(links)
        self._citations.setHtml(html)

    def _toggle_pin(self) -> None:
        self._pinned = self._pin_button.isChecked()
        if self._pinned:
            self._set_focusable(True)
            self._apply_opacity(active=True)
        else:
            if self._config.focus_return_on_submit:
                self._restore_previous_focus()
                self._set_focusable(False)
            if self._config.fade_when_inactive:
                self._apply_opacity(active=False)

    def _on_opacity_changed(self, value: int) -> None:
        self._active_opacity = max(0.4, min(1.0, value / 100.0))
        self._inactive_opacity = min(self._inactive_opacity, self._active_opacity)
        self._apply_opacity(active=self._pinned or self.hasFocus())

    def _apply_opacity(self, *, active: bool) -> None:
        if active or self._pinned:
            self.setWindowOpacity(self._active_opacity)
        else:
            self.setWindowOpacity(self._inactive_opacity)

    def _set_focusable(self, focusable: bool) -> None:
        if self._focusable == focusable:
            return
        self._focusable = focusable
        flag = getattr(QtCore.Qt, "WindowDoesNotAcceptFocus", None)
        if flag is not None:
            self.setWindowFlag(flag, not focusable)
            if self.isVisible():
                self.hide()
                self.show()

    def _capture_previous_focus(self) -> None:
        if sys.platform != "win32":
            self._previous_hwnd = None
            return
        try:
            self._previous_hwnd = ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            self._previous_hwnd = None

    def _restore_previous_focus(self) -> None:
        if sys.platform != "win32":
            return
        if not self._previous_hwnd:
            return
        try:
            ctypes.windll.user32.SetForegroundWindow(self._previous_hwnd)
        except Exception:
            return
