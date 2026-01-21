"""Doctor service wrapper for UX surfaces."""

from __future__ import annotations

import datetime as dt

from .models import DoctorCheck, DoctorReport
from .redaction import redact_payload
from ..config import AppConfig
from ..doctor import run_doctor


class DoctorService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def run(self, *, verbose: bool = False) -> DoctorReport:
        _exit_code, report = run_doctor(self._config)
        results: list[DoctorCheck] = []
        for result in report.results:
            detail = result.detail if verbose else result.detail
            detail = redact_payload(detail)
            severity = "info" if result.ok else "critical"
            results.append(
                DoctorCheck(
                    name=result.name,
                    ok=result.ok,
                    detail=str(detail),
                    severity=severity,
                )
            )
        return DoctorReport(
            ok=report.ok,
            generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            results=results,
        )
