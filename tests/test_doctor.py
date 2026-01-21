from __future__ import annotations

from autocapture.config import AppConfig, DatabaseConfig
from autocapture import doctor
from autocapture.doctor import DoctorCheckResult, run_doctor


def test_doctor_reports_failure_and_nonzero() -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )

    def failing_check(_config: AppConfig) -> DoctorCheckResult:
        return DoctorCheckResult(name="forced_failure", ok=False, detail="boom")

    code, report = run_doctor(config, checks=[failing_check])
    assert code == 2
    assert report.ok is False


def test_check_port_permission_error_is_skipped(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise PermissionError("blocked")

    monkeypatch.setattr(doctor.socket, "socket", _raise)

    result = doctor._check_port("api_port", "127.0.0.1", 1234)
    assert result.ok is True
    assert "permission denied" in result.detail.lower()


def test_doctor_handles_check_exceptions() -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )

    def exploding(_config: AppConfig) -> DoctorCheckResult:
        raise RuntimeError("boom")

    code, report = run_doctor(config, checks=[exploding])
    assert code == 2
    assert report.ok is False
    assert report.results[0].ok is False
    assert "boom" in report.results[0].detail


def test_ffmpeg_required_when_video_enabled() -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
        capture={"record_video": True},
        ffmpeg={"enabled": False},
    )

    result = doctor._check_ffmpeg(config)
    assert result.ok is False
    assert "ffmpeg disabled" in result.detail.lower()
