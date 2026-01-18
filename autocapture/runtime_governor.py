"""Runtime governor for fullscreen pause and QoS mode selection."""

from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .config import RuntimeConfig, RuntimeQosProfile
from .logging_utils import get_logger
from .observability.metrics import (
    runtime_mode_changes_total,
    runtime_mode_state,
    runtime_pause_reason_total,
)
from .storage.database import DatabaseManager
from .storage.models import RuntimeStateRecord
from .tracking.win_foreground import get_foreground_context, is_fullscreen_window
from .gpu_lease import GpuLease, get_global_gpu_lease
from .runtime_env import ProfileName
from .runtime_pause import PauseController
from .runtime_profile import ProfileScheduler


class RuntimeMode(str, Enum):
    FULLSCREEN_HARD_PAUSE = "FULLSCREEN_HARD_PAUSE"
    ACTIVE_INTERACTIVE = "ACTIVE_INTERACTIVE"
    IDLE_DRAIN = "IDLE_DRAIN"


@dataclass(frozen=True, slots=True)
class FullscreenState:
    is_fullscreen: bool
    hwnd: int | None
    process_name: str | None
    window_title: str | None


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    mode: RuntimeMode
    reason: str | None
    since_ts: dt.datetime
    last_fullscreen: FullscreenState


@dataclass(frozen=True, slots=True)
class RuntimeQosBudget:
    sleep_ms: int
    max_batch: int | None
    max_concurrency: int | None
    gpu_policy: str


class WindowMonitor:
    """Poll foreground window state to detect fullscreen usage."""

    def __init__(self) -> None:
        self._last_state = FullscreenState(False, None, None, None)

    def sample(self) -> FullscreenState:
        ctx = get_foreground_context()
        if not ctx or not ctx.hwnd:
            self._last_state = FullscreenState(False, None, None, None)
            return self._last_state
        fullscreen = bool(is_fullscreen_window(ctx.hwnd))
        state = FullscreenState(
            fullscreen,
            ctx.hwnd,
            ctx.process_name,
            ctx.window_title,
        )
        self._last_state = state
        return state

    @property
    def last_state(self) -> FullscreenState:
        return self._last_state


class RuntimeGovernor:
    """Single source of truth for runtime mode, worker QoS, and fullscreen pause."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        db_manager: DatabaseManager | None = None,
        raw_input: object | None = None,
        window_monitor: WindowMonitor | None = None,
        gpu_lease: GpuLease | None = None,
        pause_controller: PauseController | None = None,
        profile_override: ProfileName | None = None,
        profile_scheduler: ProfileScheduler | None = None,
    ) -> None:
        self._config = config
        self._db = db_manager
        self._raw_input = raw_input
        self._monitor = window_monitor or WindowMonitor()
        self._gpu_lease = gpu_lease or get_global_gpu_lease()
        self._pause = pause_controller
        self._profile_override = profile_override
        self._profile_scheduler = profile_scheduler
        self._log = get_logger("runtime.governor")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[RuntimeMode], None]] = []
        self._current_mode = RuntimeMode.ACTIVE_INTERACTIVE
        self._since_ts = dt.datetime.now(dt.timezone.utc)
        self._pause_reason: str | None = None
        self._last_fullscreen: FullscreenState = FullscreenState(False, None, None, None)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log.info("Runtime governor started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._log.info("Runtime governor stopped")

    def subscribe(self, callback: Callable[[RuntimeMode], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[RuntimeMode], None]) -> None:
        with self._lock:
            self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    @property
    def current_mode(self) -> RuntimeMode:
        with self._lock:
            return self._current_mode

    @property
    def pause_reason(self) -> str | None:
        with self._lock:
            return self._pause_reason

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return RuntimeSnapshot(
                mode=self._current_mode,
                reason=self._pause_reason,
                since_ts=self._since_ts,
                last_fullscreen=self._last_fullscreen,
            )

    def allow_workers(self) -> bool:
        if not self._auto_pause_enabled():
            return True
        return self.current_mode != RuntimeMode.FULLSCREEN_HARD_PAUSE

    def qos_budget(self, mode: RuntimeMode | None = None) -> RuntimeQosBudget:
        mode = mode or self.current_mode
        profile = self.qos_profile(mode)
        if not self._config.qos.enabled:
            sleep_ms = int(max(0.01, self.poll_interval_s(1.0)) * 1000)
            return RuntimeQosBudget(
                sleep_ms=sleep_ms,
                max_batch=None,
                max_concurrency=None,
                gpu_policy="allow_gpu",
            )
        sleep_ms = int(max(0.01, self.poll_interval_s(1.0)) * 1000)
        if getattr(profile, "sleep_ms", None) is not None:
            sleep_ms = int(max(0, int(profile.sleep_ms)))
        return RuntimeQosBudget(
            sleep_ms=sleep_ms,
            max_batch=getattr(profile, "max_batch", None),
            max_concurrency=getattr(profile, "max_concurrency", None),
            gpu_policy=str(getattr(profile, "gpu_policy", "allow_gpu") or "allow_gpu"),
        )

    def qos_profile(self, mode: RuntimeMode | None = None) -> RuntimeQosProfile:
        mode = mode or self.current_mode
        if self._profile_scheduler:
            if self._profile_override == ProfileName.IDLE:
                return self._profile_scheduler.qos_profile(ProfileName.IDLE)
            if self._profile_override == ProfileName.FOREGROUND:
                return self._profile_scheduler.qos_profile(ProfileName.FOREGROUND)
            name = ProfileName.IDLE if mode == RuntimeMode.IDLE_DRAIN else ProfileName.FOREGROUND
            return self._profile_scheduler.qos_profile(name)
        if mode == RuntimeMode.IDLE_DRAIN:
            return self._config.qos.profile_idle
        return self._config.qos.profile_active

    def poll_interval_s(self, default: float) -> float:
        if not self._profile_scheduler:
            return default
        name = ProfileName.FOREGROUND
        if self._profile_override == ProfileName.IDLE:
            name = ProfileName.IDLE
        elif self._profile_override == ProfileName.FOREGROUND:
            name = ProfileName.FOREGROUND
        elif self.current_mode == RuntimeMode.IDLE_DRAIN:
            name = ProfileName.IDLE
        return self._profile_scheduler.profile(name).poll_interval_s

    def allow_vision_extract(self) -> bool:
        profile = self.qos_profile()
        return bool(profile.vision_extract)

    def allow_ui_grounding(self) -> bool:
        profile = self.qos_profile()
        return bool(profile.ui_grounding)

    def is_fullscreen_pause(self) -> bool:
        return self.current_mode == RuntimeMode.FULLSCREEN_HARD_PAUSE

    def tick(self) -> None:
        state = self._monitor.sample()
        mode, reason = self._determine_mode(state)
        self._maybe_update(mode, reason, state)

    def _run_loop(self) -> None:
        interval = max(0.01, 1.0 / max(self._config.auto_pause.poll_hz, 0.1))
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # pragma: no cover - defensive
                self._log.debug("Runtime governor loop failed: {}", exc)
            self._stop.wait(interval)

    def _determine_mode(self, state: FullscreenState) -> tuple[RuntimeMode, str | None]:
        if self._pause and self._pause.is_paused():
            pause_state = self._pause.get_state()
            reason = pause_state.reason or "pause_latch"
            return RuntimeMode.FULLSCREEN_HARD_PAUSE, reason
        if self._auto_pause_enabled() and state.is_fullscreen:
            if self._config.auto_pause.fullscreen_hard_pause_enabled:
                return RuntimeMode.FULLSCREEN_HARD_PAUSE, "fullscreen"
            return RuntimeMode.ACTIVE_INTERACTIVE, "fullscreen_soft"
        now_ms = int(time.monotonic() * 1000)
        last_input_ms = self._get_last_input_ms()
        idle_grace_ms = max(0, int(self._config.qos.idle_grace_ms))
        if last_input_ms is None:
            return RuntimeMode.IDLE_DRAIN, None
        if now_ms - last_input_ms <= idle_grace_ms:
            return RuntimeMode.ACTIVE_INTERACTIVE, None
        return RuntimeMode.IDLE_DRAIN, None

    def _get_last_input_ms(self) -> int | None:
        raw = self._raw_input
        if raw is None:
            return None
        value = getattr(raw, "last_input_ts", None)
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def _auto_pause_enabled(self) -> bool:
        auto_pause = self._config.auto_pause
        enabled = getattr(auto_pause, "enabled", None)
        if enabled is None:
            return bool(getattr(auto_pause, "on_fullscreen", False))
        return bool(enabled)

    def _maybe_update(self, mode: RuntimeMode, reason: str | None, state: FullscreenState) -> None:
        notify = False
        with self._lock:
            if mode != self._current_mode:
                self._current_mode = mode
                self._since_ts = dt.datetime.now(dt.timezone.utc)
                self._pause_reason = reason
                notify = True
            self._last_fullscreen = state if state.is_fullscreen else self._last_fullscreen

        if notify:
            runtime_mode_changes_total.labels(mode.value).inc()
            _set_runtime_mode_metrics(mode)
            if reason:
                runtime_pause_reason_total.labels(reason).inc()
            if mode == RuntimeMode.FULLSCREEN_HARD_PAUSE and self._config.auto_pause.release_gpu:
                self._gpu_lease.release_all("fullscreen")
            self._log.info(
                "Runtime mode change: mode={} reason={} since_ts={}",
                mode.value,
                reason or "",
                self._since_ts.isoformat(),
            )
            self._persist_state(mode, reason, state)
            self._notify_callbacks(mode)

    def _notify_callbacks(self, mode: RuntimeMode) -> None:
        callbacks: list[Callable[[RuntimeMode], None]] = []
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(mode)
            except Exception as exc:  # pragma: no cover - defensive
                self._log.debug("Runtime callback failed: {}", exc)

    def _persist_state(self, mode: RuntimeMode, reason: str | None, state: FullscreenState) -> None:
        if self._db is None:
            return

        def _update(session) -> None:
            record = session.get(RuntimeStateRecord, 1)
            if record is None:
                record = RuntimeStateRecord(id=1)
                session.add(record)
            record.current_mode = mode.value
            record.pause_reason = reason
            record.since_ts = self._since_ts
            record.last_fullscreen_hwnd = str(state.hwnd) if state.hwnd else None
            record.last_fullscreen_process = state.process_name
            record.last_fullscreen_title = state.window_title

        try:
            self._db.transaction(_update)
        except Exception as exc:  # pragma: no cover - defensive
            self._log.debug("Failed to persist runtime state: {}", exc)


def _set_runtime_mode_metrics(mode: RuntimeMode) -> None:
    for value in RuntimeMode:
        runtime_mode_state.labels(value.value).set(1.0 if value == mode else 0.0)
