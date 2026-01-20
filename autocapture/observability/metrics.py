"""Prometheus metrics helpers."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from ..config import ObservabilityConfig
from ..logging_utils import get_logger

captures_taken_total = Counter("captures_taken_total", "Total captures taken")
captures_dropped_total = Counter("captures_dropped_total", "Total captures dropped")
captures_skipped_backpressure_total = Counter(
    "captures_skipped_backpressure_total", "Captures skipped due to backpressure"
)
roi_queue_full_total = Counter("roi_queue_full_total", "ROI queue full events")
disk_low_total = Counter("disk_low_total", "Disk low backpressure events")
roi_queue_depth = Gauge("roi_queue_depth", "ROI queue depth")
ocr_backlog = Gauge("ocr_backlog", "OCR backlog count")
ocr_backlog_gauge = Gauge("ocr_backlog_gauge", "OCR backlog gauge")
ocr_stale_processing_gauge = Gauge("ocr_stale_processing_gauge", "Stale OCR processing count")
ocr_latency_ms = Histogram("ocr_latency_ms", "OCR latency (ms)")
embedding_latency_ms = Histogram("embedding_latency_ms", "Embedding latency (ms)")
retrieval_latency_ms = Histogram("retrieval_latency_ms", "Retrieval latency (ms)")
embedding_backlog = Gauge("embedding_backlog", "Embedding backlog count")
retention_files_deleted_total = Counter("retention_files_deleted_total", "Retention deletions")
enrichment_backlog = Gauge("enrichment_backlog", "Events missing enrichment outputs")
enrichment_at_risk = Gauge(
    "enrichment_at_risk", "Events nearing retention expiry without enrichment"
)
enrichment_oldest_age_hours = Gauge(
    "enrichment_oldest_age_hours", "Age (hours) of the oldest event missing enrichment"
)
enrichment_jobs_enqueued_total = Counter(
    "enrichment_jobs_enqueued_total",
    "Enrichment jobs enqueued",
    ["stage"],
)
enrichment_failures_total = Counter("enrichment_failures_total", "Enrichment scheduler failures")
vector_search_failures_total = Counter(
    "autocapture_vector_search_failures_total", "Vector search failures"
)
video_frames_dropped_total = Counter(
    "autocapture_video_frames_dropped_total", "Video frames dropped"
)
video_backpressure_events_total = Counter(
    "autocapture_video_backpressure_events_total", "Video backpressure events"
)
video_disabled = Gauge("autocapture_video_disabled", "Video capture disabled (0/1)")
metrics_port_gauge = Gauge("autocapture_metrics_port", "Metrics server port")
_metrics_port_value: Optional[int] = None
worker_errors_total = Counter("worker_errors_total", "Worker errors", ["worker"])
worker_restarts_total = Counter("worker_restarts_total", "Worker thread restarts", ["worker_type"])
media_folder_size_gb = Gauge("media_folder_size_gb", "Media folder size (GB)")
process_cpu_percent = Gauge("process_cpu_percent", "Process CPU percent")
process_rss_mb = Gauge("process_rss_mb", "Process RSS (MB)")
gpu_utilization = Gauge("gpu_utilization_percent", "GPU utilization percent")
gpu_memory_used_mb = Gauge("gpu_memory_used_mb", "GPU memory used (MB)")
runtime_mode_state = Gauge("runtime_mode_state", "Runtime mode active (1/0)", ["mode"])
runtime_mode_changes_total = Counter("runtime_mode_changes_total", "Runtime mode changes", ["mode"])
runtime_pause_reason_total = Counter(
    "runtime_pause_reason_total", "Runtime pause reasons", ["reason"]
)
plugins_discovered_total = Counter("plugins_discovered_total", "Plugins discovered")
plugins_enabled_total = Gauge("plugins_enabled_total", "Enabled plugins count")
plugin_load_failures_total = Counter(
    "plugin_load_failures_total",
    "Plugin load failures",
    ["plugin_id"],
)
extension_resolution_conflicts_total = Counter(
    "extension_resolution_conflicts_total",
    "Extension resolution conflicts",
    ["kind", "extension_id"],
)
plugin_healthcheck_latency_ms = Histogram(
    "plugin_healthcheck_latency_ms",
    "Plugin healthcheck latency (ms)",
    ["plugin_id"],
)
gateway_requests_total = Counter(
    "gateway_requests_total",
    "Gateway requests processed",
    ["endpoint", "status"],
)
gateway_latency_ms = Histogram(
    "gateway_latency_ms",
    "Gateway request latency (ms)",
    ["endpoint"],
)
gateway_failures_total = Counter(
    "gateway_failures_total",
    "Gateway failures",
    ["endpoint", "reason"],
)
graph_requests_total = Counter(
    "graph_requests_total",
    "Graph worker requests processed",
    ["adapter", "endpoint", "status"],
)
graph_latency_ms = Histogram(
    "graph_latency_ms",
    "Graph worker request latency (ms)",
    ["adapter", "endpoint"],
)
graph_failures_total = Counter(
    "graph_failures_total",
    "Graph worker failures",
    ["adapter", "endpoint", "reason"],
)
memory_ingest_total = Counter(
    "memory_service_ingest_total",
    "Memory service ingest results",
    ["status"],
)
memory_ingest_reject_total = Counter(
    "memory_service_ingest_reject_total",
    "Memory service ingest rejects",
    ["reason"],
)
memory_query_total = Counter(
    "memory_service_query_total",
    "Memory service query results",
    ["status"],
)
memory_query_latency_ms = Histogram(
    "memory_service_query_latency_ms",
    "Memory service query latency (ms)",
)
memory_feedback_total = Counter(
    "memory_service_feedback_total",
    "Memory service feedback results",
    ["status"],
)
verification_failures_total = Counter(
    "verification_failures_total",
    "Verification failures",
    ["stage", "reason"],
)

# Folder size stats are expensive on large trees; only refresh periodically.
_FOLDER_SIZE_UPDATE_INTERVAL_S = 60.0


class MetricsServer:
    def __init__(self, config: ObservabilityConfig, data_dir: Path) -> None:
        self._config = config
        self._data_dir = data_dir
        self._log = get_logger("metrics")
        self._loop_thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._actual_port: Optional[int] = None
        self._folder_size_cache_gb: float = 0.0
        self._folder_size_last_ts: float = 0.0

    def start(self) -> None:
        if self._started.is_set():
            return
        self._started.set()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._run_server()
        self._loop_thread.start()

    def stop(self) -> None:
        self._started.clear()
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)

    def _run_server(self) -> None:
        bind_host = self._config.prometheus_bind_host
        base_port = self._config.prometheus_port
        fallbacks = self._config.prometheus_port_fallbacks
        ports = [base_port] + [base_port + i for i in range(1, fallbacks + 1)]
        for idx, port in enumerate(ports):
            try:
                start_http_server(port, addr=bind_host)
                self._actual_port = port
                global _metrics_port_value
                _metrics_port_value = port
                metrics_port_gauge.set(port)
                if idx == 0:
                    self._log.info(
                        "Prometheus metrics server running on {}:{}",
                        bind_host,
                        port,
                    )
                else:
                    self._log.warning(
                        "Prometheus port {} in use; using fallback port {}",
                        base_port,
                        port,
                    )
                return
            except OSError as exc:  # pragma: no cover - network binding
                if exc.errno not in {48, 98, 10048}:
                    self._log.warning("Failed to start metrics server: {}", exc)
                    if self._config.prometheus_fail_fast:
                        raise
                    return
                if self._config.prometheus_fail_fast:
                    self._log.error("Prometheus port {} in use; fail_fast enabled", base_port)
                    raise
        self._log.warning(
            "Failed to bind Prometheus metrics server after {} fallbacks",
            fallbacks,
        )

    def _run_loop(self) -> None:
        process = psutil.Process()
        while self._started.is_set():
            try:
                now = time.monotonic()
                if now - self._folder_size_last_ts >= _FOLDER_SIZE_UPDATE_INTERVAL_S:
                    self._folder_size_last_ts = now
                    self._folder_size_cache_gb = _folder_size_gb(self._data_dir / "media")
                media_folder_size_gb.set(self._folder_size_cache_gb)
                process_cpu_percent.set(process.cpu_percent(interval=None))
                process_rss_mb.set(process.memory_info().rss / (1024**2))
                if self._config.enable_gpu_stats:
                    _update_gpu_stats()
            except Exception as exc:  # pragma: no cover - defensive
                self._log.debug("Metrics update failed: {}", exc)
            time.sleep(5.0)

    @property
    def actual_port(self) -> Optional[int]:
        return self._actual_port


def get_metrics_port() -> Optional[int]:
    return _metrics_port_value


def _folder_size_gb(path: Path) -> float:
    return _folder_size_bytes(path) / (1024**3)


def _folder_size_bytes(path: Path) -> int:
    """Best-effort recursive folder size (bytes).

    Uses os.scandir for performance and tolerates files disappearing mid-walk.
    """

    def _walk(p: Path) -> int:
        total = 0
        try:
            with os.scandir(p) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            total += _walk(Path(entry.path))
                    except (FileNotFoundError, PermissionError, OSError):
                        continue
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return 0
        return total

    return _walk(path)


def _update_gpu_stats() -> bool:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    lines = result.stdout.strip().splitlines()
    if not lines:
        return False
    parts = lines[0].split(",")
    if len(parts) < 2:
        return False
    try:
        gpu_utilization.set(float(parts[0].strip()))
        gpu_memory_used_mb.set(float(parts[1].strip()))
    except ValueError:
        return False
    return True


_gpu_probe_lock = threading.Lock()
_last_gpu_probe = 0.0
_last_gpu_ok: bool | None = None


def get_gpu_snapshot(refresh: bool = True, min_interval_s: float = 2.0) -> dict:
    global _last_gpu_probe, _last_gpu_ok
    now = time.monotonic()
    if refresh:
        with _gpu_probe_lock:
            if now - _last_gpu_probe >= min_interval_s:
                try:
                    _last_gpu_ok = _update_gpu_stats()
                except Exception:
                    _last_gpu_ok = False
                _last_gpu_probe = now

    ok = bool(_last_gpu_ok)
    if not ok:
        return {
            "available": False,
            "utilization_percent": None,
            "memory_used_mb": None,
        }

    util = float(getattr(gpu_utilization, "_value").get())
    mem = float(getattr(gpu_memory_used_mb, "_value").get())
    return {
        "available": True,
        "utilization_percent": util,
        "memory_used_mb": mem,
    }
