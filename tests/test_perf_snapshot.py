import pytest
from prometheus_client import CollectorRegistry, Histogram

from autocapture.observability.perf_snapshot import histogram_quantiles


def test_histogram_quantiles_basic() -> None:
    registry = CollectorRegistry()
    hist = Histogram(
        "test_latency_ms",
        "Test latency",
        buckets=(10, 20, 30),
        registry=registry,
    )
    for value in (5, 7, 12, 18, 25, 29):
        hist.observe(value)

    stats = histogram_quantiles("test_latency_ms", registry=registry)

    assert stats is not None
    assert stats["count"] == 6
    assert stats["p50"] == pytest.approx(15.0, rel=0.05)
    assert stats["p95"] == pytest.approx(28.5, rel=0.05)
