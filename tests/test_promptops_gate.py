from __future__ import annotations

from autocapture.config import PromptOpsConfig
from autocapture.evals import EvalMetrics
from autocapture.promptops.gate import aggregate_metrics, gate_decision


def test_aggregate_metrics_worst_case() -> None:
    metrics = [
        EvalMetrics(
            citation_coverage=0.7,
            verifier_pass_rate=0.8,
            refusal_rate=0.1,
            mean_latency_ms=1000.0,
        ),
        EvalMetrics(
            citation_coverage=0.9,
            verifier_pass_rate=0.6,
            refusal_rate=0.2,
            mean_latency_ms=1200.0,
        ),
    ]
    aggregated = aggregate_metrics(metrics, "worst_case")
    assert aggregated.verifier_pass_rate == 0.6
    assert aggregated.citation_coverage == 0.7
    assert aggregated.refusal_rate == 0.2
    assert aggregated.mean_latency_ms == 1200.0


def test_gate_rejects_no_improvement() -> None:
    config = PromptOpsConfig()
    baseline = EvalMetrics(0.7, 0.7, 0.1, 1000.0)
    proposed = EvalMetrics(0.7, 0.7, 0.1, 1000.0)
    decision = gate_decision(config, baseline, proposed)
    assert not decision["accepted"]
    assert "no_metric_improved" in decision["reasons"]


def test_gate_accepts_with_min_delta() -> None:
    config = PromptOpsConfig()
    baseline = EvalMetrics(0.7, 0.6, 0.1, 1000.0)
    proposed = EvalMetrics(0.7, 0.63, 0.1, 1000.0)
    decision = gate_decision(config, baseline, proposed)
    assert decision["accepted"]


def test_gate_rejects_floor_violations() -> None:
    config = PromptOpsConfig()
    baseline = EvalMetrics(0.7, 0.6, 0.1, 1000.0)
    proposed = EvalMetrics(0.4, 0.65, 0.1, 1000.0)
    decision = gate_decision(config, baseline, proposed)
    assert not decision["accepted"]
    assert "citation_coverage_below_floor" in decision["reasons"]


def test_gate_rejects_regressions() -> None:
    config = PromptOpsConfig()
    baseline = EvalMetrics(0.8, 0.7, 0.1, 1000.0)
    proposed = EvalMetrics(0.7, 0.7, 0.1, 1000.0)
    decision = gate_decision(config, baseline, proposed)
    assert not decision["accepted"]
    assert "citation_coverage_regressed" in decision["reasons"]
