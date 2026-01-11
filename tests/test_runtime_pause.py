from __future__ import annotations

from autocapture.config import AppConfig, DatabaseConfig
import autocapture.runtime as runtime_module


class FakeRawInput:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def set_hotkey_callback(self, callback) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class FakeOrchestrator:
    def __init__(self, *args, **kwargs) -> None:
        self.pause_called = False

    def pause(self) -> None:
        self.pause_called = True

    def resume(self) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    @property
    def is_paused(self) -> bool:
        return self.pause_called


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


class FakeRetentionScheduler:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class FakeMetricsServer:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def test_runtime_starts_paused_when_configured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runtime_module, "RawInputListener", FakeRawInput)
    monkeypatch.setattr(runtime_module, "CaptureOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(runtime_module, "WorkerSupervisor", FakeSupervisor)
    monkeypatch.setattr(runtime_module, "RetentionScheduler", FakeRetentionScheduler)
    monkeypatch.setattr(runtime_module, "MetricsServer", FakeMetricsServer)

    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
        privacy={"paused": True},
    )
    runtime = runtime_module.AppRuntime(config)
    runtime.start()
    assert runtime._orchestrator.pause_called is True
    runtime.stop()
