"""OpenTelemetry helpers with strict attribute allowlist."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

from ..logging_utils import get_logger


_ALLOWED_ATTRS = {
    "stage_name",
    "success",
    "provider_id",
    "error_type",
    "count",
    "mode",
    "queue_depth",
}


def _otel_available() -> bool:
    try:
        import opentelemetry  # noqa: F401

        return True
    except Exception:
        return False


def safe_attributes(attrs: dict[str, Any] | None) -> dict[str, Any]:
    if not attrs:
        return {}
    safe: dict[str, Any] = {}
    for key, value in attrs.items():
        if key not in _ALLOWED_ATTRS:
            continue
        if isinstance(value, (int, float, bool)):
            safe[key] = value
        elif isinstance(value, str):
            safe[key] = value[:128]
    return safe


@dataclass
class OTelState:
    enabled: bool
    tracer: Any | None
    meter: Any | None
    histograms: dict[str, Any]
    counters: dict[str, Any]
    gauge_values: dict[str, float]
    gauge_attrs: dict[str, dict[str, Any]]


_STATE = OTelState(
    enabled=False,
    tracer=None,
    meter=None,
    histograms={},
    counters={},
    gauge_values={},
    gauge_attrs={},
)


def init_otel(enabled: bool, *, test_mode: bool = False) -> None:
    log = get_logger("otel")
    if not enabled or not _otel_available():
        _STATE.enabled = False
        return
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        resource = Resource.create({"service.name": "autocapture"})
        provider = TracerProvider(resource=resource)
        if test_mode:
            try:
                from opentelemetry.sdk.trace.export import InMemorySpanExporter
            except Exception:
                from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                    InMemorySpanExporter,
                )

            exporter = InMemorySpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            _STATE.histograms["__test_exporter__"] = exporter
        trace.set_tracer_provider(provider)
        _STATE.tracer = trace.get_tracer("autocapture")

        from opentelemetry.sdk.metrics import MeterProvider

        metrics.set_meter_provider(MeterProvider(resource=resource))
        _STATE.meter = metrics.get_meter("autocapture")
        _STATE.enabled = True
        log.info("OpenTelemetry enabled")
    except Exception as exc:
        _STATE.enabled = False
        log.warning("OpenTelemetry unavailable: {}", exc)


def otel_enabled() -> bool:
    return bool(_STATE.enabled and _STATE.tracer)


@contextmanager
def otel_span(name: str, attrs: dict[str, Any] | None = None):
    if not otel_enabled():
        yield None
        return
    safe = safe_attributes(attrs)
    with _STATE.tracer.start_as_current_span(name, attributes=safe) as span:
        yield span


def record_histogram(name: str, value: float, attrs: dict[str, Any] | None = None) -> None:
    if not otel_enabled() or _STATE.meter is None:
        return
    if name not in _STATE.histograms:
        _STATE.histograms[name] = _STATE.meter.create_histogram(name)
    _STATE.histograms[name].record(value, attributes=safe_attributes(attrs))


def increment_counter(name: str, value: int = 1, attrs: dict[str, Any] | None = None) -> None:
    if not otel_enabled() or _STATE.meter is None:
        return
    if name not in _STATE.counters:
        _STATE.counters[name] = _STATE.meter.create_counter(name)
    _STATE.counters[name].add(value, attributes=safe_attributes(attrs))


def set_gauge(name: str, value: float, attrs: dict[str, Any] | None = None) -> None:
    if not otel_enabled() or _STATE.meter is None:
        return
    if name not in _STATE.gauge_values:
        _STATE.gauge_values[name] = value
        _STATE.gauge_attrs[name] = safe_attributes(attrs)

        def _callback(_options):
            from opentelemetry.metrics import Observation

            return [Observation(_STATE.gauge_values[name], _STATE.gauge_attrs[name])]

        _STATE.meter.create_observable_gauge(name, callbacks=[_callback])
        return
    _STATE.gauge_values[name] = value
    _STATE.gauge_attrs[name] = safe_attributes(attrs)


def exported_spans() -> Iterable[Any]:
    exporter = _STATE.histograms.get("__test_exporter__")
    if exporter is None:
        return []
    try:
        return exporter.get_finished_spans()
    except Exception:
        return []
