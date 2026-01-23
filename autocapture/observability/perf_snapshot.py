"""Performance snapshot helpers for runtime/UI visibility."""

from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import psutil
from prometheus_client import REGISTRY
from prometheus_client.core import CollectorRegistry

from .metrics import get_gpu_snapshot


PERF_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass
class ProcessSampler:
    process: psutil.Process

    @classmethod
    def create(cls) -> "ProcessSampler":
        process = psutil.Process()
        # Prime CPU percent to avoid the first read returning 0.0 indefinitely.
        try:
            process.cpu_percent(interval=None)
        except Exception:
            pass
        return cls(process=process)

    def snapshot(self) -> dict[str, float | None]:
        cpu = None
        cpu_raw = None
        cpu_count = None
        rss = None
        try:
            cpu_raw = float(self.process.cpu_percent(interval=None))
            cpu_count = psutil.cpu_count() or 1
            if cpu_count > 0:
                cpu = max(0.0, min(cpu_raw / cpu_count, 100.0))
            else:
                cpu = cpu_raw
        except Exception:
            cpu = None
        try:
            rss = float(self.process.memory_info().rss) / (1024**2)
        except Exception:
            rss = None
        return {
            "cpu_percent": cpu,
            "cpu_percent_raw": cpu_raw,
            "cpu_count": cpu_count,
            "rss_mb": rss,
        }


def collect_samples(
    registry: CollectorRegistry | None = None,
) -> dict[str, list]:
    registry = registry or REGISTRY
    samples: dict[str, list] = {}
    for metric in registry.collect():
        samples[metric.name] = list(metric.samples)
    return samples


def sum_samples(samples: Iterable, name: str) -> float | None:
    total = 0.0
    found = False
    for sample in samples:
        if sample.name != name:
            continue
        try:
            total += float(sample.value or 0.0)
        except (TypeError, ValueError):
            continue
        found = True
    return total if found else None


def histogram_quantiles(
    metric_name: str,
    *,
    registry: CollectorRegistry | None = None,
    samples: dict[str, list] | None = None,
    quantiles: tuple[float, ...] = (0.5, 0.95),
) -> dict[str, Any] | None:
    if samples is None:
        samples = collect_samples(registry)
    metric_samples = samples.get(metric_name)
    if not metric_samples:
        return None
    buckets: dict[float, float] = {}
    count = 0.0
    total = 0.0
    count_seen = False
    for sample in metric_samples:
        if sample.name == f"{metric_name}_bucket":
            le_raw = sample.labels.get("le") if hasattr(sample, "labels") else None
            if le_raw is None:
                continue
            if le_raw == "+Inf":
                le = math.inf
            else:
                try:
                    le = float(le_raw)
                except (TypeError, ValueError):
                    continue
            try:
                value = float(sample.value or 0.0)
            except (TypeError, ValueError):
                continue
            buckets[le] = buckets.get(le, 0.0) + value
        elif sample.name == f"{metric_name}_count":
            count_seen = True
            try:
                count += float(sample.value or 0.0)
            except (TypeError, ValueError):
                pass
        elif sample.name == f"{metric_name}_sum":
            try:
                total += float(sample.value or 0.0)
            except (TypeError, ValueError):
                pass
    if not buckets:
        return None
    ordered = sorted(buckets.items(), key=lambda item: item[0])
    if not count_seen:
        count = ordered[-1][1]
    quantile_values = {}
    for q in quantiles:
        quantile_values[f"p{int(q * 100)}"] = _histogram_quantile(q, ordered)
    return {
        "count": int(count),
        "sum": total,
        **quantile_values,
    }


def _histogram_quantile(q: float, buckets: list[tuple[float, float]]) -> float | None:
    if not buckets or q <= 0:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_le = 0.0
    prev_count = 0.0
    for le, count in buckets:
        if count >= rank:
            if math.isinf(le):
                return prev_le
            bucket_count = count - prev_count
            if bucket_count <= 0:
                return le
            return prev_le + (rank - prev_count) / bucket_count * (le - prev_le)
        prev_le = le
        prev_count = count
    return buckets[-1][0]


class PerfSnapshotBuilder:
    def __init__(
        self,
        *,
        component: str,
        registry: CollectorRegistry | None = None,
        include_metrics: bool = True,
        include_gpu: bool = True,
    ) -> None:
        self._component = component
        self._registry = registry or REGISTRY
        self._include_metrics = include_metrics
        self._include_gpu = include_gpu
        self._process = ProcessSampler.create()
        self._last_capture_count: float | None = None
        self._last_capture_ts: float | None = None

    def snapshot(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        now = time.monotonic()
        payload: dict[str, Any] = {
            "schema_version": PERF_SNAPSHOT_SCHEMA_VERSION,
            "component": self._component,
            "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "process": self._process.snapshot(),
        }
        if self._include_metrics:
            samples = collect_samples(self._registry)
            payload["captures"] = self._capture_stats(samples, now)
            payload["queues"] = self._queue_stats(samples)
            payload["latency_ms"] = {
                "ocr": histogram_quantiles("ocr_latency_ms", samples=samples),
                "embedding": histogram_quantiles("embedding_latency_ms", samples=samples),
                "retrieval": histogram_quantiles("retrieval_latency_ms", samples=samples),
            }
        if self._include_gpu:
            payload["gpu"] = get_gpu_snapshot(refresh=False)
        if extra:
            payload.update(extra)
        return payload

    def _capture_stats(self, samples: dict[str, list], now: float) -> dict[str, Any]:
        total = sum_samples(samples.get("captures_taken_total", []), "captures_taken_total")
        dropped = sum_samples(samples.get("captures_dropped_total", []), "captures_dropped_total")
        skipped = sum_samples(
            samples.get("captures_skipped_backpressure_total", []),
            "captures_skipped_backpressure_total",
        )
        roi_full = sum_samples(samples.get("roi_queue_full_total", []), "roi_queue_full_total")
        disk_low = sum_samples(samples.get("disk_low_total", []), "disk_low_total")
        per_min = None
        delta_s = None
        if total is not None:
            if self._last_capture_count is not None and self._last_capture_ts is not None:
                delta = max(total - self._last_capture_count, 0.0)
                elapsed = max(now - self._last_capture_ts, 1e-3)
                per_min = (delta / elapsed) * 60.0
                delta_s = elapsed
            self._last_capture_count = total
            self._last_capture_ts = now
        return {
            "total": total,
            "dropped": dropped,
            "skipped_backpressure": skipped,
            "roi_queue_full": roi_full,
            "disk_low": disk_low,
            "per_min": per_min,
            "delta_s": delta_s,
        }

    @staticmethod
    def _queue_stats(samples: dict[str, list]) -> dict[str, Any]:
        return {
            "roi_queue_depth": sum_samples(
                samples.get("roi_queue_depth", []), "roi_queue_depth"
            ),
            "ocr_backlog": sum_samples(samples.get("ocr_backlog", []), "ocr_backlog"),
            "embedding_backlog": sum_samples(
                samples.get("embedding_backlog", []), "embedding_backlog"
            ),
            "enrichment_backlog": sum_samples(
                samples.get("enrichment_backlog", []), "enrichment_backlog"
            ),
            "enrichment_at_risk": sum_samples(
                samples.get("enrichment_at_risk", []), "enrichment_at_risk"
            ),
        }
