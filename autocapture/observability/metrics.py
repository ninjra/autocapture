"""Prometheus metrics helpers."""

from __future__ import annotations

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
ocr_stale_processing_gauge = Gauge(
    "ocr_stale_processing_gauge", "Stale OCR processing count"
)
ocr_latency_ms = Histogram("ocr_latency_ms", "OCR latency (ms)")
embedding_latency_ms = Histogram("embedding_latency_ms", "Embedding latency (ms)")
retrieval_latency_ms = Histogram("retrieval_latency_ms", "Retrieval latency (ms)")
retention_files_deleted_total = Counter(
    "retention_files_deleted_total", "Retention deletions"
)
worker_errors_total = Counter("worker_errors_total", "Worker errors", ["worker"])
media_folder_size_gb = Gauge("media_folder_size_gb", "Media folder size (GB)")
process_cpu_percent = Gauge("process_cpu_percent", "Process CPU percent")
process_rss_mb = Gauge("process_rss_mb", "Process RSS (MB)")
gpu_utilization = Gauge("gpu_utilization_percent", "GPU utilization percent")
gpu_memory_used_mb = Gauge("gpu_memory_used_mb", "GPU memory used (MB)")


class MetricsServer:
    def __init__(self, config: ObservabilityConfig, data_dir: Path) -> None:
        self._config = config
        self._data_dir = data_dir
        self._log = get_logger("metrics")
        self._server_thread: Optional[threading.Thread] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self) -> None:
        if self._started.is_set():
            return
        self._started.set()
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._server_thread.start()
        self._loop_thread.start()

    def stop(self) -> None:
        self._started.clear()
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)
        if self._server_thread:
            self._server_thread.join(timeout=1.0)

    def _run_server(self) -> None:
        try:
            start_http_server(self._config.prometheus_port)
            self._log.info(
                "Prometheus metrics server running on {}", self._config.prometheus_port
            )
        except Exception as exc:  # pragma: no cover - network binding
            self._log.warning("Failed to start metrics server: {}", exc)

    def _run_loop(self) -> None:
        process = psutil.Process()
        while self._started.is_set():
            try:
                media_folder_size_gb.set(_folder_size_gb(self._data_dir / "media"))
                process_cpu_percent.set(process.cpu_percent(interval=None))
                process_rss_mb.set(process.memory_info().rss / (1024**2))
                if self._config.enable_gpu_stats:
                    _update_gpu_stats()
            except Exception as exc:  # pragma: no cover - defensive
                self._log.debug("Metrics update failed: {}", exc)
            time.sleep(5.0)


def _folder_size_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024**3)


def _update_gpu_stats() -> None:
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
        return
    if result.returncode != 0:
        return
    lines = result.stdout.strip().splitlines()
    if not lines:
        return
    parts = lines[0].split(",")
    if len(parts) >= 2:
        gpu_utilization.set(float(parts[0].strip()))
        gpu_memory_used_mb.set(float(parts[1].strip()))
