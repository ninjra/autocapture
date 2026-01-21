"""Evidence-first banner policy."""

from __future__ import annotations

from typing import Any, Iterable

from .models import AnswerBanner, BannerAction, EvidenceSummary


def build_evidence_summary(evidence: Iterable[Any], time_range: tuple | None = None) -> EvidenceSummary:
    total = 0
    citable = 0
    redacted = 0
    max_risk = 0.0
    for item in evidence or []:
        total += 1
        non_citable = False
        redacted_text = None
        risk = 0.0
        if isinstance(item, dict):
            non_citable = bool(item.get("non_citable"))
            redacted_text = item.get("redacted_text")
            risk = float(item.get("injection_risk", 0.0) or 0.0)
        else:
            non_citable = bool(getattr(item, "non_citable", False))
            redacted_text = getattr(item, "redacted_text", None)
            risk = float(getattr(item, "injection_risk", 0.0) or 0.0)
        if not non_citable:
            citable += 1
        if redacted_text:
            redacted += 1
        if risk > max_risk:
            max_risk = risk
    time_range_iso = None
    if time_range and len(time_range) == 2:
        time_range_iso = (str(time_range[0]), str(time_range[1]))
    return EvidenceSummary(
        total=total,
        citable=citable,
        redacted=redacted,
        injection_risk_max=max_risk,
        time_range=time_range_iso,
    )


class BannerPolicy:
    @staticmethod
    def evaluate(
        *,
        locked: bool,
        evidence: EvidenceSummary,
        degraded_reasons: Iterable[str] | None = None,
        mode: str | None = None,
    ) -> AnswerBanner:
        reasons = [reason for reason in (degraded_reasons or []) if reason]
        if locked:
            return AnswerBanner(
                level="locked",
                title="Unlock required",
                message="Unlock the local session to continue.",
                reasons=["unlock_required"],
                actions=[BannerAction(label="Unlock", type="unlock")],
            )
        if evidence.total <= 0:
            return AnswerBanner(
                level="no_evidence",
                title="No evidence found",
                message="Try a different time range or add more context.",
                reasons=["no_evidence"],
                actions=[
                    BannerAction(label="Expand time range", type="time_range", value="30d"),
                    BannerAction(label="Refine query", type="refine_query"),
                ],
            )
        degraded = False
        if reasons:
            degraded = True
        if mode in {"CONFLICT", "BLOCKED"}:
            degraded = True
            reasons.append(mode.lower())
        if degraded:
            return AnswerBanner(
                level="degraded",
                title="Degraded answer",
                message="Answer quality may be reduced; see reasons for details.",
                reasons=sorted(set(reasons)),
            )
        return AnswerBanner(level="none", title="", message="", reasons=[])
