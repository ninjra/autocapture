"""PromptOps eval aggregation and acceptance gate."""

from __future__ import annotations

from typing import Any

from ..config import PromptOpsConfig
from ..evals import EvalMetrics


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


def compute_deltas(baseline: EvalMetrics, proposed: EvalMetrics) -> dict[str, float]:
    return {
        "verifier_pass_rate": proposed.verifier_pass_rate - baseline.verifier_pass_rate,
        "citation_coverage": proposed.citation_coverage - baseline.citation_coverage,
        "refusal_rate": baseline.refusal_rate - proposed.refusal_rate,
        "mean_latency_ms": baseline.mean_latency_ms - proposed.mean_latency_ms,
    }


def gate_decision(
    config: PromptOpsConfig, baseline: EvalMetrics, proposed: EvalMetrics
) -> dict[str, Any]:
    reasons: list[str] = []
    deltas = compute_deltas(baseline, proposed)
    thresholds = {
        "tolerance_verifier_pass_rate": config.acceptance_tolerance,
        "tolerance_citation_coverage": config.tolerance_citation_coverage,
        "tolerance_refusal_rate": config.tolerance_refusal_rate,
        "tolerance_latency_ms": config.tolerance_latency_ms,
        "min_verifier_pass_rate": config.min_verifier_pass_rate,
        "min_citation_coverage": config.min_citation_coverage,
        "max_refusal_rate": config.max_refusal_rate,
        "max_mean_latency_ms": config.max_mean_latency_ms,
        "min_delta_verifier_pass_rate": config.min_delta_verifier_pass_rate,
        "min_delta_citation_coverage": config.min_delta_citation_coverage,
        "min_delta_refusal_rate": config.min_delta_refusal_rate,
        "min_delta_latency_ms": config.min_delta_latency_ms,
        "require_improvement": config.require_improvement,
    }

    if proposed.verifier_pass_rate < baseline.verifier_pass_rate - config.acceptance_tolerance:
        reasons.append("verifier_pass_rate_regressed")
    if proposed.citation_coverage < baseline.citation_coverage - config.tolerance_citation_coverage:
        reasons.append("citation_coverage_regressed")
    if proposed.refusal_rate > baseline.refusal_rate + config.tolerance_refusal_rate:
        reasons.append("refusal_rate_regressed")
    if proposed.mean_latency_ms > baseline.mean_latency_ms + config.tolerance_latency_ms:
        reasons.append("latency_regressed")

    if proposed.verifier_pass_rate < config.min_verifier_pass_rate:
        reasons.append("verifier_pass_rate_below_floor")
    if proposed.citation_coverage < config.min_citation_coverage:
        reasons.append("citation_coverage_below_floor")
    if proposed.refusal_rate > config.max_refusal_rate:
        reasons.append("refusal_rate_above_ceiling")
    if proposed.mean_latency_ms > config.max_mean_latency_ms:
        reasons.append("latency_above_ceiling")

    if config.require_improvement:
        improvement = (
            deltas["verifier_pass_rate"] >= config.min_delta_verifier_pass_rate
            or deltas["citation_coverage"] >= config.min_delta_citation_coverage
            or deltas["refusal_rate"] >= config.min_delta_refusal_rate
            or deltas["mean_latency_ms"] >= config.min_delta_latency_ms
        )
        if not improvement:
            reasons.append("no_metric_improved")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "deltas": deltas,
        "thresholds": thresholds,
    }
