"""Host vector tracker core and event sources."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol
from uuid import uuid4

from ..config import TrackingConfig
from ..logging_utils import get_logger
from .store import SqliteHostEventStore, safe_payload
from .types import (
    ClipboardChangeEvent,
    ForegroundChangeEvent,
    ForegroundContext,
    HostEventRow,
    InputVectorEvent,
)
from .win_clipboard import (
    clipboard_has_image,
    clipboard_has_text,
    get_clipboard_sequence_number,
)
from .win_foreground import get_foreground_context


EventType = InputVectorEvent | ForegroundChangeEvent | ClipboardChangeEvent


class EventSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class ForegroundPollSource:
    """Poll foreground context changes on a fixed cadence."""

    def __init__(
        self,
        interval_ms: int,
        on_event: Callable[[ForegroundChangeEvent], None],
    ) -> None:
        self._interval = interval_ms / 1000.0
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._log = get_logger("tracking.foreground")
        self._last_context: ForegroundContext | None = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._log.info("Foreground poll source started")

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._log.info("Foreground poll source stopped")

    def _run(self) -> None:
        while self._running.is_set():
            ctx = get_foreground_context()
            if ctx and (
                self._last_context is None
                or ctx.process_name != self._last_context.process_name
                or ctx.window_title != self._last_context.window_title
            ):
                event = ForegroundChangeEvent(
                    ts_ms=_now_ms(),
                    new=ctx,
                    old=self._last_context,
                )
                self._last_context = ctx
                try:
                    self._on_event(event)
                except Exception as exc:  # pragma: no cover - defensive
                    self._log.debug("Foreground event callback failed: {}", exc)
                self._log.debug(
                    "Foreground change: {} - {}",
                    ctx.process_name,
                    _truncate(ctx.window_title),
                )
            time.sleep(self._interval)


class ClipboardPollSource:
    """Poll clipboard metadata changes without reading content."""

    def __init__(
        self,
        interval_ms: int,
        on_event: Callable[[ClipboardChangeEvent], None],
    ) -> None:
        self._interval = interval_ms / 1000.0
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._log = get_logger("tracking.clipboard")
        self._last_sequence = 0

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._log.info("Clipboard poll source started")

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._log.info("Clipboard poll source stopped")

    def _run(self) -> None:
        while self._running.is_set():
            sequence = get_clipboard_sequence_number()
            if sequence and sequence != self._last_sequence:
                has_text = clipboard_has_text()
                has_image = clipboard_has_image()
                kind = _clipboard_kind(has_text, has_image)
                event = ClipboardChangeEvent(
                    ts_ms=_now_ms(),
                    has_text=has_text,
                    has_image=has_image,
                    kind=kind,
                )
                self._last_sequence = sequence
                try:
                    self._on_event(event)
                except Exception as exc:  # pragma: no cover - defensive
                    self._log.debug("Clipboard event callback failed: {}", exc)
            time.sleep(self._interval)


@dataclass
class _InputBucket:
    ts_start_ms: int
    ts_end_ms: int
    context: ForegroundContext | None
    keyboard_events: int = 0
    mouse_left_clicks: int = 0
    mouse_right_clicks: int = 0
    mouse_middle_clicks: int = 0
    mouse_wheel_events: int = 0
    mouse_wheel_delta: int = 0
    mouse_move_dx: int = 0
    mouse_move_dy: int = 0


class HostEventAggregator:
    """Aggregates raw input events into storage-friendly buckets."""

    def __init__(
        self,
        flush_interval_ms: int,
        idle_grace_ms: int,
        track_mouse_movement: bool,
    ) -> None:
        self._flush_interval_ms = flush_interval_ms
        self._idle_grace_ms = idle_grace_ms
        self._track_mouse_movement = track_mouse_movement
        self._bucket: _InputBucket | None = None
        self._last_input_ms: int | None = None
        self._session_id: str | None = None
        self._foreground_ctx: ForegroundContext | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def handle_event(self, event: EventType, now_ms: int) -> list[HostEventRow]:
        rows: list[HostEventRow] = []
        if isinstance(event, InputVectorEvent):
            rows.extend(self._handle_input(event, now_ms))
            if (
                event.device == "mouse"
                and self._session_id
                and self._bucket_has_input()
            ):
                rows.extend(self._flush_bucket(now_ms))
        elif isinstance(event, ForegroundChangeEvent):
            rows.extend(self._handle_foreground(event, now_ms))
        elif isinstance(event, ClipboardChangeEvent):
            rows.append(self._clipboard_row(event))
        return rows

    def handle_tick(self, now_ms: int) -> list[HostEventRow]:
        rows: list[HostEventRow] = []
        rows.extend(self._maybe_flush(now_ms))
        if (
            self._session_id
            and self._last_input_ms is not None
            and now_ms - self._last_input_ms >= self._idle_grace_ms
        ):
            rows.extend(self._flush_bucket(now_ms))
            rows.append(self._session_row(now_ms, "session_end"))
            self._session_id = None
            self._last_input_ms = None
        return rows

    def flush_all(self, now_ms: int) -> list[HostEventRow]:
        rows = self._flush_bucket(now_ms)
        if self._session_id:
            rows.append(self._session_row(now_ms, "session_end"))
            self._session_id = None
        return rows

    def _handle_input(self, event: InputVectorEvent, now_ms: int) -> list[HostEventRow]:
        rows: list[HostEventRow] = []
        if self._session_id is None:
            self._session_id = str(uuid4())
            rows.append(self._session_row(now_ms, "session_start"))
            self._bucket = None
        self._last_input_ms = now_ms
        bucket = self._bucket
        if bucket is None:
            bucket = _InputBucket(
                ts_start_ms=now_ms,
                ts_end_ms=now_ms,
                context=self._foreground_ctx,
            )
            self._bucket = bucket
        bucket.ts_end_ms = now_ms
        if event.device == "keyboard":
            count = 1
            if event.mouse:
                count = max(1, event.mouse.get("events", 1))
            bucket.keyboard_events += count
        else:
            data = event.mouse or {}
            bucket.mouse_left_clicks += data.get("left_clicks", 0)
            bucket.mouse_right_clicks += data.get("right_clicks", 0)
            bucket.mouse_middle_clicks += data.get("middle_clicks", 0)
            bucket.mouse_wheel_events += data.get("wheel_events", 0)
            bucket.mouse_wheel_delta += data.get("wheel_delta", 0)
            if self._track_mouse_movement:
                bucket.mouse_move_dx += data.get("move_dx", 0)
                bucket.mouse_move_dy += data.get("move_dy", 0)
        rows.extend(self._maybe_flush(now_ms))
        return rows

    def _handle_foreground(
        self, event: ForegroundChangeEvent, now_ms: int
    ) -> list[HostEventRow]:
        rows: list[HostEventRow] = []
        rows.extend(self._flush_bucket(now_ms))
        self._foreground_ctx = event.new
        rows.append(self._foreground_row(event))
        return rows

    def _maybe_flush(self, now_ms: int) -> list[HostEventRow]:
        if not self._bucket:
            return []
        if now_ms - self._bucket.ts_start_ms >= self._flush_interval_ms:
            return self._flush_bucket(now_ms)
        return []

    def _bucket_has_input(self) -> bool:
        if not self._bucket:
            return False
        bucket = self._bucket
        if bucket.keyboard_events:
            return True
        if (
            bucket.mouse_left_clicks
            or bucket.mouse_right_clicks
            or bucket.mouse_middle_clicks
            or bucket.mouse_wheel_events
            or bucket.mouse_wheel_delta
        ):
            return True
        if self._track_mouse_movement and (
            bucket.mouse_move_dx or bucket.mouse_move_dy
        ):
            return True
        return False

    def _flush_bucket(self, now_ms: int) -> list[HostEventRow]:
        if not self._bucket:
            return []
        bucket = self._bucket
        self._bucket = None
        payload = {
            "keyboard_events": bucket.keyboard_events,
            "mouse_left_clicks": bucket.mouse_left_clicks,
            "mouse_right_clicks": bucket.mouse_right_clicks,
            "mouse_middle_clicks": bucket.mouse_middle_clicks,
            "mouse_wheel_events": bucket.mouse_wheel_events,
            "mouse_wheel_delta": bucket.mouse_wheel_delta,
        }
        if self._track_mouse_movement:
            payload["mouse_move_dx"] = bucket.mouse_move_dx
            payload["mouse_move_dy"] = bucket.mouse_move_dy
        row = HostEventRow(
            id=str(uuid4()),
            ts_start_ms=bucket.ts_start_ms,
            ts_end_ms=max(bucket.ts_end_ms, now_ms),
            kind="input_bucket",
            session_id=self._session_id,
            app_name=bucket.context.process_name if bucket.context else None,
            window_title=bucket.context.window_title if bucket.context else None,
            payload_json=safe_payload(payload),
        )
        return [row]

    def _foreground_row(self, event: ForegroundChangeEvent) -> HostEventRow:
        payload = {
            "old_app": event.old.process_name if event.old else None,
            "old_title": event.old.window_title if event.old else None,
            "new_app": event.new.process_name,
            "new_title": event.new.window_title,
        }
        return HostEventRow(
            id=str(uuid4()),
            ts_start_ms=event.ts_ms,
            ts_end_ms=event.ts_ms,
            kind="foreground_change",
            session_id=self._session_id,
            app_name=event.new.process_name,
            window_title=event.new.window_title,
            payload_json=safe_payload(payload),
        )

    def _clipboard_row(self, event: ClipboardChangeEvent) -> HostEventRow:
        bucket_ms = (event.ts_ms // 5000) * 5000
        payload = {
            "clipboard_changed": True,
            "clipboard_kind": event.kind,
        }
        return HostEventRow(
            id=str(uuid4()),
            ts_start_ms=bucket_ms,
            ts_end_ms=bucket_ms,
            kind="clipboard_change",
            session_id=self._session_id,
            app_name=self._foreground_ctx.process_name
            if self._foreground_ctx
            else None,
            window_title=self._foreground_ctx.window_title
            if self._foreground_ctx
            else None,
            payload_json=safe_payload(payload),
        )

    def _session_row(
        self, ts_ms: int, kind: Literal["session_start", "session_end"]
    ) -> HostEventRow:
        payload = {"event": kind}
        return HostEventRow(
            id=str(uuid4()),
            ts_start_ms=ts_ms,
            ts_end_ms=ts_ms,
            kind=kind,
            session_id=self._session_id,
            app_name=self._foreground_ctx.process_name
            if self._foreground_ctx
            else None,
            window_title=self._foreground_ctx.window_title
            if self._foreground_ctx
            else None,
            payload_json=safe_payload(payload),
        )


class HostVectorTracker:
    """Local-first tracker for host interaction vectors."""

    def __init__(
        self,
        config: TrackingConfig,
        data_dir: str | Path | None,
        idle_grace_ms: int,
    ) -> None:
        self._config = config
        self._data_dir = data_dir
        self._idle_grace_ms = idle_grace_ms
        self._log = get_logger("tracking")
        self._queue: queue.Queue[EventType] = queue.Queue(maxsize=config.queue_maxsize)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dropped = 0
        self._last_drop_log = 0.0
        self._sources: list[EventSource] = []
        self._rows_written = 0
        self._db_path = self._resolve_db_path()

    def _resolve_db_path(self) -> str:
        db_path = self._config.db_path
        if not db_path.is_absolute():
            if not self._data_dir:
                return str(db_path)
            return str((Path(self._data_dir) / db_path).resolve())
        return str(db_path)

    def start(self) -> None:
        if not self._config.enabled:
            self._log.info("Host vector tracking disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        db_path = Path(self._db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._log.info("Starting host vector tracker (db={})", db_path)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._sources = [
            ForegroundPollSource(
                interval_ms=self._config.foreground_poll_ms,
                on_event=self._enqueue_event,
            )
        ]
        if self._config.enable_clipboard:
            self._log.warning(
                "Clipboard tracking enabled: storing clipboard change metadata only."
            )
            self._sources.append(
                ClipboardPollSource(
                    interval_ms=self._config.clipboard_poll_ms,
                    on_event=self._enqueue_event,
                )
            )
        for source in self._sources:
            source.start()

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        for source in self._sources:
            source.stop()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._log.info(
            "Host vector tracker stopped (rows={}, dropped={})",
            self._rows_written,
            self._dropped,
        )

    def ingest_input_event(self, event: InputVectorEvent) -> None:
        if not self._config.enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log > 5:
                self._last_drop_log = now
                self._log.warning("Dropping input events (total={})", self._dropped)

    def _enqueue_event(self, event: EventType) -> None:
        if not self._config.enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1

    def _run_loop(self) -> None:
        store = SqliteHostEventStore(Path(self._db_path))
        store.init_schema()
        aggregator = HostEventAggregator(
            flush_interval_ms=self._config.flush_interval_ms,
            idle_grace_ms=self._idle_grace_ms,
            track_mouse_movement=self._config.track_mouse_movement,
        )
        last_prune_check = 0.0
        while not self._stop.is_set() or not self._queue.empty():
            now_ms = _now_ms()
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                event = None
            rows: list[HostEventRow] = []
            if event is not None:
                rows.extend(aggregator.handle_event(event, now_ms))
                self._queue.task_done()
            rows.extend(aggregator.handle_tick(now_ms))
            if rows:
                store.insert_many(rows)
                self._rows_written += len(rows)
            if self._config.retention_days is not None:
                now = time.monotonic()
                if now - last_prune_check > 60:
                    last_prune_check = now
                    cutoff_ms = _now_ms() - int(
                        self._config.retention_days * 86400 * 1000
                    )
                    store.prune_older_than(cutoff_ms)
        rows = aggregator.flush_all(_now_ms())
        if rows:
            store.insert_many(rows)
            self._rows_written += len(rows)
        store.close()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _truncate(value: str, limit: int = 80) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _clipboard_kind(has_text: bool, has_image: bool) -> str:
    if has_text:
        return "text"
    if has_image:
        return "image"
    return "unknown"
