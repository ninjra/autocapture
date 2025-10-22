"""Prometheus metrics exporter."""

from __future__ import annotations

import threading
from prometheus_client import Counter, Gauge, start_http_server

from ..config import ObservabilityConfig
from ..logging_utils import get_logger


class MetricsServer:
    """Expose Prometheus metrics for capture throughput and latency."""

    def __init__(self, config: ObservabilityConfig) -> None:
        self._config = config
        self._log = get_logger("metrics")
        self._started = threading.Event()
        self.captures_total = Counter("autocapture_frames_total", "Total frames captured")
        self.ocr_backlog = Gauge("autocapture_ocr_backlog", "Pending OCR jobs")
        self.embedding_backlog = Gauge(
            "autocapture_embedding_backlog", "Pending spans waiting for embedding"
        )
        self.disk_usage_gb = Gauge("autocapture_disk_usage_gb", "Current NAS usage in GB")
        self.gpu_utilization = Gauge("autocapture_gpu_utilization", "Recent GPU utilization %")

    def start(self) -> None:
        if not self._started.is_set():
            start_http_server(self._config.prometheus_port)
            self._started.set()
            self._log.info("Prometheus exporter listening on %s", self._config.prometheus_port)

    def observe_gpu(self, util_percent: float) -> None:
        if self._config.enable_gpu_stats:
            self.gpu_utilization.set(util_percent)

    def observe_disk(self, usage_gb: float) -> None:
        self.disk_usage_gb.set(usage_gb)

    def increment_captures(self, count: int = 1) -> None:
        self.captures_total.inc(count)

    def set_ocr_backlog(self, count: int) -> None:
        self.ocr_backlog.set(count)

    def set_embedding_backlog(self, count: int) -> None:
        self.embedding_backlog.set(count)
