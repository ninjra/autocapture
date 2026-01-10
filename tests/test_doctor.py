from __future__ import annotations

from autocapture.config import AppConfig, DatabaseConfig
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
