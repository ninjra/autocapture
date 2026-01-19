"""Windows fullscreen watchdog that toggles the runtime pause latch."""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path
from typing import Any

import win32api
import win32con
import win32gui
import win32process


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def main() -> int:
    runtime_dir = _runtime_dir()
    pause_flag = runtime_dir / "pause.flag"
    reason_path = runtime_dir / "pause_reason.json"
    log_path = runtime_dir / "watchdog.log"

    hz = _parse_float(os.environ.get("WATCHDOG_HZ"), default=8.0)
    tol_px = int(_parse_float(os.environ.get("WATCHDOG_TOL_PX"), default=2.0))
    include_titles = _parse_bool(os.environ.get("WATCHDOG_INCLUDE_TITLES"), default=False)
    interval = max(0.05, 1.0 / max(hz, 0.1))

    was_fullscreen = False
    last_error_log = 0.0
    _log(log_path, f"watchdog start (hz={hz}, tol={tol_px}, titles={include_titles})")

    while True:
        try:
            hwnd = win32gui.GetForegroundWindow()
            fullscreen = _is_fullscreen(hwnd, tol_px)
            if fullscreen:
                if not was_fullscreen:
                    _log(log_path, "fullscreen detected; pausing")
                _write_reason(
                    reason_path,
                    _build_reason_payload(
                        hwnd,
                        include_titles=include_titles,
                    ),
                )
                _touch_flag(pause_flag)
            else:
                if was_fullscreen:
                    _log(log_path, "fullscreen cleared; resuming")
                _clear_flag(pause_flag)
                _clear_reason_if_owned(reason_path)
            was_fullscreen = fullscreen
        except Exception as exc:
            now = time.time()
            if now - last_error_log > 5.0:
                last_error_log = now
                _log(log_path, f"watchdog error: {exc.__class__.__name__}: {exc}")
            _clear_flag(pause_flag)
            _write_reason(reason_path, _build_error_payload(exc))
        time.sleep(interval)


def _runtime_dir() -> Path:
    raw = os.environ.get("AUTOCAPTURE_RUNTIME_DIR") or "C:\\autocapture_runtime"
    raw = _normalize_runtime_dir(raw)
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_runtime_dir(raw: str) -> str:
    value = raw.strip()
    if value.lower().startswith("/mnt/") and len(value) > 6:
        drive = value[5:6].upper()
        rest = value[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return value


def _is_fullscreen(hwnd: int, tol_px: int) -> bool:
    if not hwnd:
        return False
    if win32gui.IsIconic(hwnd):
        return False
    try:
        rect = win32gui.GetWindowRect(hwnd)
        monitor = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        info = win32api.GetMonitorInfo(monitor)
        mon_rect = info.get("Monitor")
    except Exception:
        return False
    if not mon_rect:
        return False
    left, top, right, bottom = rect
    m_left, m_top, m_right, m_bottom = mon_rect
    return (
        abs(left - m_left) <= tol_px
        and abs(top - m_top) <= tol_px
        and abs(right - m_right) <= tol_px
        and abs(bottom - m_bottom) <= tol_px
    )


def _build_reason_payload(hwnd: int, *, include_titles: bool) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    payload: dict[str, Any] = {
        "reason": "fullscreen",
        "source": "win_watchdog",
        "ts_ms": now_ms,
        "title_redacted": not include_titles,
    }
    app_exe = _get_process_exe(hwnd)
    if app_exe:
        payload["app_exe"] = app_exe
    if include_titles:
        title = _get_window_title(hwnd)
        if title:
            payload["window_title"] = title[:256]
            payload["title_redacted"] = False
    return payload


def _build_error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "reason": "watchdog_error",
        "source": "win_watchdog",
        "ts_ms": int(time.time() * 1000),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc)[:256],
        "title_redacted": True,
    }


def _get_window_title(hwnd: int) -> str | None:
    try:
        return win32gui.GetWindowText(hwnd)
    except Exception:
        return None


def _get_process_exe(hwnd: int) -> str | None:
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
    except Exception:
        return None
    if not pid:
        return None
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_ulong(len(buf))
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    return None


def _write_reason(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, sort_keys=True)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp_path, path)


def _touch_flag(path: Path) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return
    except OSError:
        try:
            path.touch(exist_ok=True)
        except OSError:
            return


def _clear_flag(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _clear_reason_if_owned(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("source") != "win_watchdog":
            return
    except FileNotFoundError:
        return
    except Exception:
        return
    _clear_flag(path)


def _log(path: Path, message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_float(raw: str | None, *, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
