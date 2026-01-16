"""PromptOps gating and candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import PromptOpsConfig
from ..evals import EvalMetrics


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    improved_metrics: list[str]
    regressions: list[str]
    threshold_violations: list[str]
    details: dict[str, Any]


def aggregate_metrics(metrics_list: list[EvalMetrics], mode: str) -> EvalMetrics:
    if not metrics_list:
        raise ValueError("aggregate_metrics requires at least one metrics entry")
    if mode not in {"worst_case", "mean"}:
        raise ValueError("aggregate_metrics mode must be 'worst_case' or 'mean'")
    if mode == "mean":
        count = len(metrics_list)
        return EvalMetrics(
            citation_coverage=sum(m.citation_coverage for m in metrics_list) / count,
            verifier_pass_rate=sum(m.verifier_pass_rate for m in metrics_list) / count,
            refusal_rate=sum(m.refusal_rate for m in metrics_list) / count,
            mean_latency_ms=sum(m.mean_latency_ms for m in metrics_list) / count,
        )
    return EvalMetrics(
        citation_coverage=min(m.citation_coverage for m in metrics_list),
        verifier_pass_rate=min(m.verifier_pass_rate for m in metrics_list),
        refusal_rate=max(m.refusal_rate for m in metrics_list),
        mean_latency_ms=max(m.mean_latency_ms for m in metrics_list),
    )


def evaluate_candidate(
    promptops: PromptOpsConfig, baseline: EvalMetrics, candidate: EvalMetrics
) -> GateDecision:
    threshold_violations: list[str] = []
    regressions: list[str] = []
    improved_metrics: list[str] = []

    if candidate.verifier_pass_rate < promptops.min_verifier_pass_rate:
        threshold_violations.append("verifier_pass_rate_below_floor")
    if candidate.citation_coverage < promptops.min_citation_coverage:
        threshold_violations.append("citation_coverage_below_floor")
    if candidate.refusal_rate > promptops.max_refusal_rate:
        threshold_violations.append("refusal_rate_above_ceiling")
    if (
        promptops.max_mean_latency_ms is not None
        and candidate.mean_latency_ms > promptops.max_mean_latency_ms
    ):
        threshold_violations.append("latency_above_ceiling")

    if candidate.verifier_pass_rate < baseline.verifier_pass_rate - promptops.acceptance_tolerance:
        regressions.append("verifier_pass_rate_regressed")
    if (
        candidate.citation_coverage
        < baseline.citation_coverage - promptops.tolerance_citation_coverage
    ):
        regressions.append("citation_coverage_regressed")
    if candidate.refusal_rate > baseline.refusal_rate + promptops.tolerance_refusal_rate:
        regressions.append("refusal_rate_regressed")
    if (
        promptops.tolerance_mean_latency_ms is not None
        and candidate.mean_latency_ms
        > baseline.mean_latency_ms + promptops.tolerance_mean_latency_ms
    ):
        regressions.append("latency_regressed")

    if (
        candidate.verifier_pass_rate
        >= baseline.verifier_pass_rate + promptops.min_improve_verifier_pass_rate
    ):
        improved_metrics.append("verifier_pass_rate")
    if (
        candidate.citation_coverage
        >= baseline.citation_coverage + promptops.min_improve_citation_coverage
    ):
        improved_metrics.append("citation_coverage")
    if candidate.refusal_rate <= baseline.refusal_rate - promptops.min_improve_refusal_rate:
        improved_metrics.append("refusal_rate")
    if (
        candidate.mean_latency_ms
        <= baseline.mean_latency_ms - promptops.min_improve_mean_latency_ms
    ):
        improved_metrics.append("mean_latency_ms")

    improvements_ok = True
    if promptops.require_improvement:
        improvements_ok = bool(improved_metrics)

    deltas = {
        "verifier_pass_rate": candidate.verifier_pass_rate - baseline.verifier_pass_rate,
        "citation_coverage": candidate.citation_coverage - baseline.citation_coverage,
        "refusal_rate": baseline.refusal_rate - candidate.refusal_rate,
        "mean_latency_ms": baseline.mean_latency_ms - candidate.mean_latency_ms,
    }
    passed = not threshold_violations and not regressions and improvements_ok
    return GateDecision(
        passed=passed,
        improved_metrics=improved_metrics,
        regressions=regressions,
        threshold_violations=threshold_violations,
        details={
            "baseline": baseline.to_dict(),
            "candidate": candidate.to_dict(),
            "deltas": deltas,
            "tolerances": {
                "acceptance_tolerance": promptops.acceptance_tolerance,
                "tolerance_citation_coverage": promptops.tolerance_citation_coverage,
                "tolerance_refusal_rate": promptops.tolerance_refusal_rate,
                "tolerance_mean_latency_ms": promptops.tolerance_mean_latency_ms,
            },
            "thresholds": {
                "min_verifier_pass_rate": promptops.min_verifier_pass_rate,
                "min_citation_coverage": promptops.min_citation_coverage,
                "max_refusal_rate": promptops.max_refusal_rate,
                "max_mean_latency_ms": promptops.max_mean_latency_ms,
            },
            "min_improvements": {
                "min_improve_verifier_pass_rate": promptops.min_improve_verifier_pass_rate,
                "min_improve_citation_coverage": promptops.min_improve_citation_coverage,
                "min_improve_refusal_rate": promptops.min_improve_refusal_rate,
                "min_improve_mean_latency_ms": promptops.min_improve_mean_latency_ms,
            },
            "require_improvement": promptops.require_improvement,
        },
    )


def is_candidate_better(a: EvalMetrics, b: EvalMetrics) -> bool:
    return (
        a.verifier_pass_rate,
        a.citation_coverage,
        -a.refusal_rate,
        -a.mean_latency_ms,
    ) > (
        b.verifier_pass_rate,
        b.citation_coverage,
        -b.refusal_rate,
        -b.mean_latency_ms,
    )

