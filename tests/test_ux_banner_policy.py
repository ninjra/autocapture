from __future__ import annotations

from autocapture.ux.banners import BannerPolicy
from autocapture.ux.models import EvidenceSummary


def test_banner_no_evidence() -> None:
    summary = EvidenceSummary(total=0, citable=0, redacted=0, injection_risk_max=0.0)
    banner = BannerPolicy.evaluate(locked=False, evidence=summary)
    assert banner.level == "no_evidence"


def test_banner_locked() -> None:
    summary = EvidenceSummary(total=3, citable=3, redacted=0, injection_risk_max=0.0)
    banner = BannerPolicy.evaluate(locked=True, evidence=summary)
    assert banner.level == "locked"


def test_banner_degraded() -> None:
    summary = EvidenceSummary(total=2, citable=2, redacted=0, injection_risk_max=0.1)
    banner = BannerPolicy.evaluate(locked=False, evidence=summary, degraded_reasons=["stage_timeout"])
    assert banner.level == "degraded"
