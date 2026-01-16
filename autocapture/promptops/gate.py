"""PromptOps eval aggregation and acceptance gate (legacy wrapper)."""

from __future__ import annotations

from typing import Any

from ..config import PromptOpsConfig
from ..evals import EvalMetrics
from .gates import GateDecision, aggregate_metrics, evaluate_candidate


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
    decision: GateDecision = evaluate_candidate(config, baseline, proposed)
    reasons = decision.threshold_violations + decision.regressions
    if config.require_improvement and not decision.improved_metrics:
        reasons.append("no_metric_improved")
    return {
        "accepted": decision.passed,
        "reasons": reasons,
        "deltas": decision.details.get("deltas", {}),
        "thresholds": decision.details,
    }
