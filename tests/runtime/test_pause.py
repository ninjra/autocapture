from __future__ import annotations

import json

from autocapture.runtime_env import load_runtime_env
from autocapture.runtime_pause import PauseController
import autocapture.runtime_pause as runtime_pause


def test_pause_latch_create_clear(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOCAPTURE_RUNTIME_DIR", str(tmp_path))
    runtime_env = load_runtime_env()
    pause = PauseController(runtime_env)

    pause.write_pause({"reason": "manual"}, "unit")
    assert runtime_env.pause_latch_path.exists() is True
    assert pause.is_paused() is True

    pause.clear_pause("unit")
    assert runtime_env.pause_latch_path.exists() is False


def test_pause_reason_atomic_write(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOCAPTURE_RUNTIME_DIR", str(tmp_path))
    runtime_env = load_runtime_env()
    pause = PauseController(runtime_env)

    calls: list[tuple[str, str]] = []
    original_replace = runtime_pause.os.replace

    def _wrapped(src: str, dst: str) -> None:
        calls.append((str(src), str(dst)))
        original_replace(src, dst)

    monkeypatch.setattr(runtime_pause.os, "replace", _wrapped)

    pause.write_pause({"reason": "manual"}, "unit")
    assert calls
    assert runtime_env.pause_reason_path.exists() is True
    payload = json.loads(runtime_env.pause_reason_path.read_text(encoding="utf-8"))
    assert payload.get("reason") == "manual"
    assert payload.get("source") == "unit"
    assert isinstance(payload.get("ts_ms"), int)
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())
