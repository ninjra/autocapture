"""Windows overlay UI for hotness tracker."""

from __future__ import annotations

import datetime as dt
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ...config import OverlayTrackerConfig
from ...logging_utils import get_logger
from ...security.redaction import redact_mapping, redact_value
from ...tracking.win_foreground import get_foreground_context, is_fullscreen_window
from ..clock import Clock
from ..core import hotness
from ..schemas import OverlayEventEvidence, OverlayItemSummary, OverlayProjectSummary
from ..store import OverlayTrackerStore


class OverlayDataWorker(QtCore.QObject):
    data_ready = QtCore.Signal(object, object, object, object)
    evidence_ready = QtCore.Signal(int, object)

    def __init__(
        self,
        store: OverlayTrackerStore,
        config: OverlayTrackerConfig,
        clock: Clock,
    ) -> None:
        super().__init__()
        self._store = store
        self._config = config
        self._clock = clock
        self._log = get_logger("overlay_tracker.ui.worker")
        self._timer: QtCore.QTimer | None = None

    @QtCore.Slot()
    def start(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self._config.ui.refresh_ms)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    @QtCore.Slot()
    def stop(self) -> None:
        if self._timer:
            self._timer.stop()

    @QtCore.Slot()
    def _refresh(self) -> None:
        try:
            now = self._clock.now()
            projects = self._store.query_projects()
            active, stale = self._store.query_items(
                now,
                stale_after_s=self._config.stale_after_hours * 3600,
            )
            self.data_ready.emit(projects, active, stale, now)
        except Exception as exc:  # pragma: no cover - defensive
            self._log.warning("Overlay UI refresh failed: {}", exc)

    @QtCore.Slot(int)
    def fetch_evidence(self, item_id: int) -> None:
        try:
            evidence = self._store.query_evidence(item_id, limit=50)
            self.evidence_ready.emit(item_id, evidence)
        except Exception as exc:  # pragma: no cover - defensive
            self._log.warning("Overlay evidence fetch failed: {}", exc)


class OverlayItemWidget(QtWidgets.QFrame):
    clicked = QtCore.Signal(int)

    def __init__(
        self,
        item: OverlayItemSummary,
        *,
        now_utc: dt.datetime,
        half_life_minutes: float,
        sanitize: bool,
    ) -> None:
        super().__init__()
        self._item_id = item.item_id
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setObjectName("overlay_item")
        if item.display_name:
            title_value = item.display_name
            title_key = "display_name"
        elif item.window_title:
            title_value = item.window_title
            title_key = "window_title"
        else:
            title_value = "(untitled)"
            title_key = None
        title = _clean(title_value, sanitize, key=title_key)
        process = _clean(item.process_name, sanitize)
        age = _format_age(now_utc, item.last_activity_at_utc)
        state = item.state.upper() if item.state else ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("overlay_item_title")
        meta_label = QtWidgets.QLabel(f"{process} - {age}")
        meta_label.setObjectName("overlay_item_meta")
        layout.addWidget(title_label)
        layout.addWidget(meta_label)
        if state and state != "IDLE":
            badge = QtWidgets.QLabel(state)
            badge.setObjectName("overlay_item_badge")
            layout.addWidget(badge)

        half_life_s = max(1.0, half_life_minutes * 60.0)
        alpha = 0.35 + 0.65 * hotness(item.last_activity_at_utc, now_utc, half_life_s)
        effect = QtWidgets.QGraphicsOpacityEffect(self)
        effect.setOpacity(max(0.2, min(1.0, alpha)))
        self.setGraphicsEffect(effect)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self._item_id)
        super().mousePressEvent(event)


class OverlayEvidenceDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlay Evidence")
        self.setMinimumWidth(520)
        layout = QtWidgets.QVBoxLayout(self)
        self._text = QtWidgets.QTextBrowser()
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    def update_evidence(self, evidence: list[OverlayEventEvidence], sanitize: bool) -> None:
        lines: list[str] = []
        for entry in evidence:
            local_ts = entry.ts_utc.astimezone().isoformat(timespec="seconds")
            utc_ts = entry.ts_utc.isoformat(timespec="seconds")
            process = _clean(entry.process_name, sanitize)
            title = _clean(entry.raw_window_title or "", sanitize, key="raw_window_title")
            url = _clean(entry.raw_browser_url or "", sanitize, key="raw_browser_url")
            lines.append(
                f"[{local_ts} / {utc_ts}] {entry.event_type} {process}"
            )
            if title:
                lines.append(f"  title: {title}")
            if url:
                lines.append(f"  url: {url}")
            lines.append(f"  collector: {entry.collector}")
            lines.append(f"  schema: {entry.schema_version} app: {entry.app_version}")
            payload = _clean_payload(entry.payload, sanitize)
            if payload:
                lines.append(f"  payload: {payload}")
            lines.append("")
        self._text.setPlainText("\n".join(lines))


class OverlayWindow(QtWidgets.QWidget):
    evidence_requested = QtCore.Signal(int)

    def __init__(
        self,
        config: OverlayTrackerConfig,
        clock: Clock,
        *,
        sanitize: bool,
    ) -> None:
        super().__init__()
        self._config = config
        self._clock = clock
        self._sanitize = sanitize
        self._log = get_logger("overlay_tracker.ui")
        self._evidence_dialog: OverlayEvidenceDialog | None = None
        self._interactive_timer = QtCore.QTimer(self)
        self._interactive_timer.setSingleShot(True)
        self._interactive_timer.timeout.connect(self._disable_interactive)
        self._fullscreen_timer = QtCore.QTimer(self)
        self._fullscreen_timer.setInterval(1000)
        self._fullscreen_timer.timeout.connect(self._check_fullscreen)
        self._fullscreen_hidden = False
        self._user_hidden = False

        self._build_ui()
        self._apply_window_flags(click_through=config.ui.click_through_default)
        self._position_window()
        self._fullscreen_timer.start()

    def _build_ui(self) -> None:
        self.setObjectName("overlay_root")
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Hotness Overlay")
        title.setObjectName("overlay_title")
        layout.addWidget(title)

        self._projects_label = QtWidgets.QLabel("Projects")
        self._projects_label.setObjectName("overlay_section")
        layout.addWidget(self._projects_label)
        self._projects_container = QtWidgets.QVBoxLayout()
        layout.addLayout(self._projects_container)

        self._active_label = QtWidgets.QLabel("Active")
        self._active_label.setObjectName("overlay_section")
        layout.addWidget(self._active_label)
        self._active_container = QtWidgets.QVBoxLayout()
        layout.addLayout(self._active_container)

        self._stale_toggle = QtWidgets.QToolButton()
        self._stale_toggle.setText("Stale")
        self._stale_toggle.setCheckable(True)
        self._stale_toggle.setChecked(False)
        self._stale_toggle.toggled.connect(self._toggle_stale)
        layout.addWidget(self._stale_toggle)

        self._stale_wrapper = QtWidgets.QWidget()
        self._stale_container = QtWidgets.QVBoxLayout(self._stale_wrapper)
        self._stale_wrapper.setVisible(False)
        layout.addWidget(self._stale_wrapper)

        self.setStyleSheet(
            """
            #overlay_root { background: rgba(20, 20, 24, 0.65); border-radius: 12px; }
            #overlay_title { color: #f0f0f5; font-size: 14px; font-weight: 600; }
            #overlay_section { color: #cfd2dc; font-size: 11px; text-transform: uppercase; }
            #overlay_item_title { color: #f5f6fa; font-size: 12px; }
            #overlay_item_meta { color: #a7adbd; font-size: 10px; }
            #overlay_item_badge { color: #ffd166; font-size: 9px; }
            """
        )

    def _apply_window_flags(self, *, click_through: bool) -> None:
        flags = (
            QtCore.Qt.Tool
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.WindowDoesNotAcceptFocus
        )
        if hasattr(QtCore.Qt, "WindowTransparentForInput") and click_through:
            flags |= QtCore.Qt.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, click_through)
        if self.isVisible():
            self.show()

    def _position_window(self) -> None:
        screen = QtGui.QGuiApplication.primaryScreen()
        if not screen:
            return
        geometry = screen.availableGeometry()
        width = self._config.ui.width_px
        if self._config.ui.dock == "left":
            x = geometry.x() + 8
        else:
            x = geometry.x() + geometry.width() - width - 8
        y = geometry.y() + 8
        height = geometry.height() - 16
        self.setGeometry(x, y, width, height)

    def set_interactive(self, enabled: bool) -> None:
        self._apply_window_flags(click_through=not enabled)
        if enabled:
            self._interactive_timer.start(self._config.ui.interactive_timeout_seconds * 1000)
        else:
            self._interactive_timer.stop()

    def toggle_visible(self) -> None:
        if self.isVisible():
            self.hide()
            self._user_hidden = True
        else:
            self.show()
            self._user_hidden = False

    def update_data(
        self,
        projects: list[OverlayProjectSummary],
        active: list[OverlayItemSummary],
        stale: list[OverlayItemSummary],
        now_utc: dt.datetime,
    ) -> None:
        active = sorted(active, key=lambda item: item.last_activity_at_utc, reverse=True)
        stale = sorted(stale, key=lambda item: item.last_activity_at_utc, reverse=True)
        _clear_layout(self._projects_container)
        for project in projects:
            label = QtWidgets.QLabel(_clean(project.name, self._sanitize))
            label.setObjectName("overlay_project")
            self._projects_container.addWidget(label)

        _clear_layout(self._active_container)
        for item in active:
            widget = OverlayItemWidget(
                item,
                now_utc=now_utc,
                half_life_minutes=self._config.hotness_half_life_minutes,
                sanitize=self._sanitize,
            )
            widget.clicked.connect(self.evidence_requested.emit)
            self._active_container.addWidget(widget)

        _clear_layout(self._stale_container)
        for item in stale:
            widget = OverlayItemWidget(
                item,
                now_utc=now_utc,
                half_life_minutes=self._config.hotness_half_life_minutes,
                sanitize=self._sanitize,
            )
            widget.clicked.connect(self.evidence_requested.emit)
            self._stale_container.addWidget(widget)

    def show_evidence(self, evidence: list[OverlayEventEvidence]) -> None:
        if self._evidence_dialog is None:
            self._evidence_dialog = OverlayEvidenceDialog(self)
        self._evidence_dialog.update_evidence(evidence, self._sanitize)
        self._evidence_dialog.show()
        self._evidence_dialog.raise_()

    def _toggle_stale(self, checked: bool) -> None:
        self._stale_wrapper.setVisible(checked)

    def _disable_interactive(self) -> None:
        self.set_interactive(False)

    def _check_fullscreen(self) -> None:
        if not self._config.ui.auto_hide_fullscreen:
            return
        if self._user_hidden:
            return
        ctx = get_foreground_context()
        fullscreen = bool(ctx and ctx.hwnd and is_fullscreen_window(ctx.hwnd))
        if fullscreen and self.isVisible():
            self.hide()
            self._fullscreen_hidden = True
        elif not fullscreen and self._fullscreen_hidden:
            self.show()
            self._fullscreen_hidden = False


class OverlayUiController(QtCore.QObject):
    def __init__(
        self,
        config: OverlayTrackerConfig,
        store: OverlayTrackerStore,
        clock: Clock,
        *,
        sanitize: bool,
    ) -> None:
        super().__init__()
        self._config = config
        self._store = store
        self._clock = clock
        self._sanitize = sanitize
        self._window = OverlayWindow(config, clock, sanitize=sanitize)
        self._thread = QtCore.QThread()
        self._worker = OverlayDataWorker(store, config, clock)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start)
        self._worker.data_ready.connect(self._window.update_data)
        self._worker.evidence_ready.connect(lambda _id, ev: self._window.show_evidence(ev))
        self._window.evidence_requested.connect(self._worker.fetch_evidence)

    @property
    def window(self) -> QtWidgets.QWidget:
        return self._window

    def start(self) -> None:
        self._thread.start()
        self._window.show()

    def stop(self) -> None:
        QtCore.QMetaObject.invokeMethod(self._worker, "stop", QtCore.Qt.QueuedConnection)
        self._thread.quit()
        self._thread.wait(1000)
        self._window.close()

    def toggle_visible(self) -> None:
        self._window.toggle_visible()

    def set_interactive(self, enabled: bool) -> None:
        self._window.set_interactive(enabled)

    def invoke(self, func: Callable[[], None]) -> None:
        QtCore.QTimer.singleShot(0, self._window, func)


def _clear_layout(layout: QtWidgets.QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _format_age(now_utc: dt.datetime, ts: dt.datetime) -> str:
    delta = max(0, int((now_utc - ts).total_seconds()))
    if delta < 60:
        return f"{delta}s"
    minutes = delta // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def _clean(value: str, sanitize: bool, *, key: str | None = None) -> str:
    if not sanitize:
        return value
    redacted = redact_value(value, key=key)
    if isinstance(redacted, str):
        return redacted
    return str(redacted)


def _clean_payload(payload: dict, sanitize: bool) -> dict:
    if not sanitize:
        return payload
    return redact_mapping(payload)
