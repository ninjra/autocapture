"""Windows fullscreen watchdog that toggles pause latch."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


MONITOR_DEFAULTTONEAREST = 2
DWMWA_CLOAKED = 14


@dataclass(frozen=True)
class ForegroundInfo:
    hwnd: int
    rect: tuple[int, int, int, int]
    monitor: tuple[int, int, int, int]
    exe_name: str | None
    title: str | None
    cloaked: bool
    iconic: bool


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", ctypes.c_ulong),
    ]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fullscreen pause watchdog")
    parser.add_argument("--poll-hz", type=float, default=8.0)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--manual-flag", default="manual_pause.flag")
    parser.add_argument("--redact-titles", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    runtime_dir = Path(
        args.runtime_dir or os.environ.get("AUTOCAPTURE_RUNTIME_DIR") or "C:/autocapture_runtime"
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)

    redact_titles = args.redact_titles or _parse_bool(
        os.environ.get("AUTOCAPTURE_REDACT_WINDOW_TITLES"), True
    )
    pause_flag = runtime_dir / "pause.flag"
    reason_path = runtime_dir / "pause_reason.json"
    manual_flag = runtime_dir / args.manual_flag
    poll_interval = max(0.05, 1.0 / max(args.poll_hz, 0.1))

    last_error_log = 0.0
    while True:
        paused = False
        reason = None
        try:
            if manual_flag.exists():
                paused = True
                reason = {"reason": "manual", "state": "manual"}
            else:
                info = _get_foreground_info()
                if info and _is_fullscreen(info, args.tolerance):
                    paused = True
                    reason = {
                        "reason": "fullscreen",
                        "state": "fullscreen",
                        "app": info.exe_name,
                        "title": info.title,
                    }
        except Exception as exc:
            paused = False
            reason = None
            now = time.time()
            if now - last_error_log > 5.0:
                last_error_log = now
                print(f"Watchdog error: {exc}", file=sys.stderr)

        if paused:
            _touch_flag(pause_flag)
            payload = _build_reason_payload(reason or {"reason": "paused"}, redact_titles)
            _atomic_write_json(reason_path, payload)
        else:
            _clear_flag(pause_flag)
            _clear_flag(reason_path)

        time.sleep(poll_interval)


def _get_foreground_info() -> ForegroundInfo | None:
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    iconic = bool(user32.IsIconic(hwnd))
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    mi = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    if not user32.GetMonitorInfoW(monitor, ctypes.byref(mi)):
        return None
    exe_name = _get_process_name(hwnd)
    title = _get_window_title(hwnd)
    cloaked = _is_cloaked(hwnd)
    return ForegroundInfo(
        hwnd=hwnd,
        rect=(rect.left, rect.top, rect.right, rect.bottom),
        monitor=(mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom),
        exe_name=exe_name,
        title=title,
        cloaked=cloaked,
        iconic=iconic,
    )


def _is_fullscreen(info: ForegroundInfo, tolerance: int) -> bool:
    if info.iconic or info.cloaked:
        return False
    left, top, right, bottom = info.rect
    m_left, m_top, m_right, m_bottom = info.monitor
    return (
        abs(left - m_left) <= tolerance
        and abs(top - m_top) <= tolerance
        and abs(right - m_right) <= tolerance
        and abs(bottom - m_bottom) <= tolerance
    )


def _get_process_name(hwnd: int) -> str | None:
    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)
    if not handle:
        return None
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = ctypes.c_ulong(len(buf))
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return Path(buf.value).name
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    return None


def _get_window_title(hwnd: int) -> str | None:
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return None
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _is_cloaked(hwnd: int) -> bool:
    try:
        dwmapi = ctypes.windll.dwmapi
    except Exception:
        return False
    cloaked = ctypes.c_int(0)
    try:
        res = dwmapi.DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED,
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
        return res == 0 and bool(cloaked.value)
    except Exception:
        return False


def _build_reason_payload(reason: dict, redact_titles: bool) -> dict[str, object]:
    payload = {
        "reason": str(reason.get("reason") or "paused"),
        "source": "watchdog",
        "state": reason.get("state"),
        "app": reason.get("app"),
        "title": reason.get("title"),
        "ts_ms": int(time.time() * 1000),
    }
    if redact_titles and payload.get("title"):
        payload["title"] = "<redacted>"
    return {k: v for k, v in payload.items() if v is not None}


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


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


if __name__ == "__main__":
    raise SystemExit(main())
