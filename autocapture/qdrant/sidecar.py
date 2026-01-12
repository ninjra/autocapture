"""Qdrant sidecar process manager."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..config import AppConfig, is_loopback_host
from ..logging_utils import get_logger
from ..paths import resolve_qdrant_path


@dataclass(frozen=True)
class QdrantEndpoint:
    host: str
    port: int


def _parse_qdrant_url(url: str) -> QdrantEndpoint | None:
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname
    port = parsed.port or 6333
    if not host:
        return None
    return QdrantEndpoint(host=host, port=port)


def should_manage_sidecar(config: AppConfig) -> bool:
    if not config.qdrant.enabled:
        return False
    endpoint = _parse_qdrant_url(config.qdrant.url)
    if not endpoint:
        return False
    return is_loopback_host(endpoint.host)


class QdrantSidecar:
    def __init__(self, config: AppConfig, data_dir: Path, log_dir: Path) -> None:
        self._config = config
        self._data_dir = Path(data_dir)
        self._log_dir = Path(log_dir)
        self._log = get_logger("qdrant.sidecar")
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: object | None = None
        self._endpoint = _parse_qdrant_url(config.qdrant.url)
        self._binary = resolve_qdrant_path(config.qdrant)
        self._warned_missing = False

    @property
    def binary_path(self) -> Path | None:
        return self._binary

    def start(self) -> None:
        if not should_manage_sidecar(self._config):
            self._log.info("Qdrant sidecar disabled (remote or disabled config).")
            return
        if not self._endpoint:
            self._log.warning("Qdrant URL invalid; sidecar disabled.")
            return
        if self._is_healthy():
            self._log.info("Qdrant already running at {}.", self._config.qdrant.url)
            return
        if not self._binary or not self._binary.exists():
            if not self._warned_missing:
                self._warned_missing = True
                self._log.warning(
                    "Qdrant binary missing; set qdrant.binary_path or bundle qdrant.exe."
                )
            return

        data_root = self._data_dir / "qdrant"
        storage_path = data_root / "storage"
        data_root.mkdir(parents=True, exist_ok=True)
        storage_path.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / "qdrant.log"
        self._log_file = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "QDRANT__SERVICE__HOST": "127.0.0.1",
                "QDRANT__SERVICE__HTTP_PORT": str(self._endpoint.port),
                "QDRANT__SERVICE__GRPC_PORT": str(self._endpoint.port + 1),
                "QDRANT__STORAGE__STORAGE_PATH": str(storage_path),
                "QDRANT__TELEMETRY_DISABLED": "true",
            }
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                [str(self._binary)],
                cwd=str(data_root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=self._log_file,
                creationflags=creationflags,
            )
        except OSError as exc:
            self._log.warning("Failed to launch Qdrant: {}", exc)
            self._close_log_file()
            self._process = None
            return

        if not self._wait_for_ready(timeout_s=20.0):
            self._log.warning("Qdrant did not become healthy; continuing without it.")

    def stop(self, timeout_s: float = 5.0) -> None:
        if not self._process:
            self._close_log_file()
            return
        self._log.info("Stopping Qdrant sidecar")
        self._process.terminate()
        try:
            self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._log.warning("Qdrant did not exit in time; killing.")
            self._process.kill()
        self._process = None
        self._close_log_file()

    def _close_log_file(self) -> None:
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _is_healthy(self) -> bool:
        if not self._endpoint:
            return False
        try:
            response = httpx.get(
                f"http://{self._endpoint.host}:{self._endpoint.port}/health",
                timeout=1.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _wait_for_ready(self, timeout_s: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                return False
            if self._is_healthy():
                return True
            time.sleep(0.5)
        return False
