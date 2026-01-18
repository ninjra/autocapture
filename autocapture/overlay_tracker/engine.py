"""Event processing loop for overlay tracker."""

from __future__ import annotations

import datetime as dt
import queue
import threading
from dataclasses import dataclass
from typing import Literal

from ..logging_utils import get_logger
from ..config import OverlayTrackerConfig
from .clock import Clock
from .core import normalize_title, resolve_identity, should_deny_process
from .schemas import OverlayCollectorContext, OverlayCollectorEvent, OverlayPersistEvent
from .store import OverlayTrackerStore

_QUEUE_MAXSIZE = 5000
_MAX_URL_LEN = 2048


@dataclass(slots=True)
class OverlayCommand:
    action: Literal["toggle_running", "rename", "snooze", "cycle_project"]
    payload: dict


@dataclass(slots=True)
class _QueueItem:
    kind: Literal["event", "command"]
    payload: OverlayCollectorEvent | OverlayCommand


class OverlayTrackerEngine:
    def __init__(
        self,
        config: OverlayTrackerConfig,
        store: OverlayTrackerStore,
        clock: Clock,
    ) -> None:
        self._config = config
        self._store = store
        self._clock = clock
        self._log = get_logger("overlay_tracker.engine")
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_context: OverlayCollectorContext | None = None
        self._last_input_by_identity: dict[str, dt.datetime] = {}
        self._current_item_id: int | None = None
        self._dropped = 0
        self._last_error: str | None = None
        self._last_retention_run: dt.datetime | None = None

    @property
    def current_item_id(self) -> int | None:
        return self._current_item_id

    def health(self) -> dict[str, object]:
        return {
            "queue_depth": self._queue.qsize(),
            "dropped_events": self._dropped,
            "last_error": self._last_error,
            "last_retention_run": self._last_retention_run.isoformat()
            if self._last_retention_run
            else None,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log.info("Overlay tracker engine started")

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._log.info("Overlay tracker engine stopped")

    def submit_event(self, event: OverlayCollectorEvent) -> None:
        try:
            self._queue.put_nowait(_QueueItem(kind="event", payload=event))
        except queue.Full:
            self._dropped += 1

    def submit_command(self, command: OverlayCommand) -> None:
        try:
            self._queue.put_nowait(_QueueItem(kind="command", payload=command))
        except queue.Full:
            self._dropped += 1

    def _run_loop(self) -> None:
        batch: list[OverlayPersistEvent] = []
        max_batch = 50
        last_retention_check = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                item = None
            if item is not None:
                items = [item]
                while len(items) < max_batch:
                    try:
                        items.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                for entry in items:
                    try:
                        if entry.kind == "event":
                            batch.extend(self._handle_event(entry.payload))
                        else:
                            self._handle_command(entry.payload)
                    except Exception as exc:  # pragma: no cover - defensive
                        self._last_error = str(exc)
                        self._log.warning("Overlay tracker handler failed: {}", exc)
                    self._queue.task_done()
            if batch:
                try:
                    item_ids = self._store.record_events(batch)
                    if item_ids:
                        self._current_item_id = item_ids[-1]
                except Exception as exc:  # pragma: no cover - defensive
                    self._last_error = str(exc)
                    self._log.warning("Overlay tracker write failed: {}", exc)
                batch.clear()
            now = self._clock.now()
            if (now - last_retention_check).total_seconds() >= 60:
                last_retention_check = now
                try:
                    self._store.retention_cleanup(
                        event_days=self._config.retention.event_days,
                        event_cap=self._config.retention.event_cap,
                        now_utc=now,
                    )
                    self._last_retention_run = now
                except Exception as exc:  # pragma: no cover - defensive
                    self._last_error = str(exc)
                    self._log.warning("Overlay tracker retention failed: {}", exc)

    def _handle_event(self, event: OverlayCollectorEvent) -> list[OverlayPersistEvent]:
        if event.event_type == "foreground":
            if should_deny_process(event.context.process_name, self._config.policy.deny_processes):
                self._last_context = None
                return []
            self._last_context = event.context
            return [self._build_persist_event(event.context, event, event.context.window_title)]
        if event.event_type == "input_activity":
            if not self._last_context:
                return []
            if should_deny_process(self._last_context.process_name, self._config.policy.deny_processes):
                return []
            return self._handle_input_activity(event)
        return []

    def _handle_input_activity(self, event: OverlayCollectorEvent) -> list[OverlayPersistEvent]:
        context = self._last_context
        if not context:
            return []
        persist = self._build_persist_event(context, event, context.window_title)
        if not persist.identity_key:
            return []
        last = self._last_input_by_identity.get(persist.identity_key)
        debounce_s = self._config.collectors.input_debounce_ms / 1000.0
        if last and (persist.ts_utc - last).total_seconds() < debounce_s:
            return []
        self._last_input_by_identity[persist.identity_key] = persist.ts_utc
        return [persist]

    def _build_persist_event(
        self,
        context: OverlayCollectorContext,
        event: OverlayCollectorEvent,
        raw_title: str | None,
    ) -> OverlayPersistEvent:
        title_limit = self._config.policy.max_window_title_length
        raw_window_title = _truncate(raw_title, title_limit)
        raw_browser_url = _truncate(context.browser_url, _MAX_URL_LEN)
        identity = resolve_identity(
            self._config,
            process_name=context.process_name,
            window_title=raw_window_title,
            browser_url=raw_browser_url,
        )
        normalized = normalize_title(raw_window_title, max_len=title_limit)
        payload = {"normalized_title": normalized}
        if event.metadata:
            payload["meta"] = event.metadata
        return OverlayPersistEvent(
            event_type=event.event_type,
            ts_utc=event.ts_utc,
            process_name=context.process_name,
            raw_window_title=raw_window_title,
            raw_browser_url=raw_browser_url,
            identity_type=identity.identity_type,
            identity_key=identity.identity_key,
            collector=event.collector,
            app_version=_app_version(),
            payload=payload,
        )

    def _handle_command(self, command: OverlayCommand) -> None:
        item_id = self._current_item_id
        if not item_id:
            return
        if command.action == "toggle_running":
            self._store.toggle_running(item_id)
            self._store.append_action_event(
                item_id,
                event_type="state_change",
                payload={"state": "toggle_running"},
                ts_utc=self._clock.now(),
                app_version=_app_version(),
            )
        elif command.action == "rename":
            name = command.payload.get("name")
            if isinstance(name, str):
                self._store.rename_item(item_id, name)
                self._store.append_action_event(
                    item_id,
                    event_type="rename",
                    payload={"name": name},
                    ts_utc=self._clock.now(),
                    app_version=_app_version(),
                )
        elif command.action == "snooze":
            until = command.payload.get("until")
            if isinstance(until, dt.datetime) or until is None:
                self._store.snooze_item(item_id, until)
                payload = {"until": until.isoformat()} if isinstance(until, dt.datetime) else {}
                self._store.append_action_event(
                    item_id,
                    event_type="snooze",
                    payload=payload,
                    ts_utc=self._clock.now(),
                    app_version=_app_version(),
                )
        elif command.action == "cycle_project":
            self._store.cycle_project(item_id)
            self._store.append_action_event(
                item_id,
                event_type="project_cycle",
                payload={},
                ts_utc=self._clock.now(),
                app_version=_app_version(),
            )


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit]


_APP_VERSION: str | None | Literal["__pending__"] = "__pending__"


def _app_version() -> str | None:
    global _APP_VERSION
    if _APP_VERSION != "__pending__":
        return _APP_VERSION
    try:
        import importlib.metadata

        _APP_VERSION = importlib.metadata.version("autocapture")
    except Exception:
        _APP_VERSION = None
    return _APP_VERSION
