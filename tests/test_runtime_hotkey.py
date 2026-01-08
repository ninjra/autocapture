from __future__ import annotations

from autocapture.config import AppConfig, DatabaseConfig
import autocapture.runtime as runtime_module


class FakeRawInput:
    def __init__(self, *args, **kwargs) -> None:
        self.callback = None

    def set_hotkey_callback(self, callback) -> None:
        self.callback = callback

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class FakeOrchestrator:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def pause(self) -> None:
        return None

    def resume(self) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    @property
    def is_paused(self) -> bool:
        return False


class FakeSupervisor:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def flush(self) -> None:
        return None

    def notify_ocr_observation(self, _observation_id: str) -> None:
        return None


def test_runtime_sets_hotkey_callback(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "RawInputListener", FakeRawInput)
    monkeypatch.setattr(runtime_module, "CaptureOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(runtime_module, "WorkerSupervisor", FakeSupervisor)
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )
    runtime = runtime_module.AppRuntime(config)

    def handler():
        return None

    runtime.set_hotkey_callback(handler)
    assert runtime._raw_input.callback == handler
