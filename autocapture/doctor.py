"""Health checks for Autocapture."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import httpx
from sqlalchemy import text

from .config import AppConfig, is_loopback_host
from .capture.backends import DxCamBackend, MssBackend
from .capture.raw_input import LASTINPUTINFO, Win32Api, probe_raw_input
from .embeddings.service import EmbeddingService
from .encryption import EncryptionManager
from .logging_utils import get_logger
from .memory.entities import SecretStore
from .observability.metrics import get_metrics_port
from .paths import (
    is_wsl,
    resolve_ffmpeg_path,
    resolve_qdrant_path,
    resource_root,
    windows_to_wsl_path,
)
from .storage.database import DatabaseManager


@dataclass(frozen=True)
class DoctorCheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    results: list[DoctorCheckResult]


def run_doctor(
    config: AppConfig,
    *,
    checks: Iterable[Callable[[AppConfig], DoctorCheckResult]] | None = None,
) -> tuple[int, DoctorReport]:
    log = get_logger("doctor")
    if checks is None:
        checks = [
            _check_paths,
            _check_database,
            _check_encryption,
            _check_qdrant,
            _check_ffmpeg,
            _check_gpu,
            _check_capture_backends,
            _check_ocr,
            _check_embeddings,
            _check_vector_index,
            _check_api_port,
            _check_metrics,
            _check_raw_input,
        ]

    log.info("Doctor config summary:\n{}", _redact_config(config))
    results: list[DoctorCheckResult] = []
    for check in checks:
        try:
            results.append(check(config))
        except Exception as exc:
            name = getattr(check, "__name__", "check").replace("_check_", "")
            log.warning("Doctor check %s failed: %s", name, exc)
            results.append(DoctorCheckResult(name=name, ok=False, detail=f"check error: {exc}"))
    ok = all(result.ok for result in results)
    _print_table(results)
    return (0 if ok else 2), DoctorReport(ok=ok, results=results)


def _redact_config(config: AppConfig) -> str:
    data = config.model_dump() if hasattr(config, "model_dump") else config.dict()
    secrets = {
        "api_key",
        "openai_api_key",
        "google_oauth_client_secret",
        "github_token",
        "key_provider",
        "key_name",
    }

    def _redact(value):
        if isinstance(value, dict):
            return {
                key: ("***" if key in secrets and value.get(key) else _redact(val))
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [_redact(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return json.dumps(_redact(data), ensure_ascii=False, indent=2)


def _print_table(results: Iterable[DoctorCheckResult]) -> None:
    rows = list(results)
    header = f"{'CHECK':<28} {'RESULT':<6} DETAILS"
    divider = "-" * len(header)
    print(header)
    print(divider)
    for result in rows:
        status = "PASS" if result.ok else "FAIL"
        print(f"{result.name:<28} {status:<6} {result.detail}")


def _check_paths(config: AppConfig) -> DoctorCheckResult:
    paths = [
        ("capture.staging_dir", config.capture.staging_dir),
        ("capture.data_dir", config.capture.data_dir),
    ]
    for name, path in paths:
        try:
            raw = str(path)
            if sys.platform != "win32" and _looks_like_windows_path(raw):
                if is_wsl():
                    mapped = windows_to_wsl_path(raw)
                    path = Path(mapped)
                else:
                    detail = f"path={raw} error=windows_path_on_posix"
                    detail += "; use a POSIX path or enable WSL path mapping"
                    return DoctorCheckResult(name, False, detail)
            Path(path).mkdir(parents=True, exist_ok=True)
            probe = Path(path) / ".doctor_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as exc:
            detail = f"path={path} error={exc}"
            if sys.platform == "win32":
                missing = [
                    var
                    for var in (
                        "LOCALAPPDATA",
                        "USERPROFILE",
                        "APPDATA",
                        "HOMEDRIVE",
                        "HOMEPATH",
                    )
                    if not os.environ.get(var)
                ]
                if missing:
                    detail += f"; missing_env={','.join(missing)}"
                detail += (
                    "; set LOCALAPPDATA (recommended) or override capture.data_dir/"
                    "capture.staging_dir in config"
                )
            if is_wsl():
                detail += "; WSL detected: prefer paths under /mnt/<drive>/ or ~/.autocapture"
            return DoctorCheckResult(name, False, detail)
    return DoctorCheckResult("paths", True, "Writable")


def _check_gpu(config: AppConfig) -> DoctorCheckResult:
    if not config.observability.enable_gpu_stats:
        return DoctorCheckResult("gpu", True, "GPU stats disabled")
    expected_cuda = (
        str(config.ocr.device).strip().lower() == "cuda"
        or str(config.reranker.device).strip().lower() == "cuda"
    )
    details: list[str] = []
    available = False
    torch_cuda = False
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # type: ignore

            torch_cuda = bool(torch.cuda.is_available())
            if torch_cuda:
                name = torch.cuda.get_device_name(0)
                details.append(f"torch_cuda={name}")
                available = True
            else:
                details.append("torch_cuda_unavailable")
        except Exception as exc:
            details.append(f"torch_error={exc}")
    else:
        details.append("torch_missing")
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            result = subprocess.run(
                [nvidia, "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.splitlines()[0].strip()
                details.append(f"nvidia-smi={line}")
                available = True
            else:
                err = (result.stderr or "").strip()
                if err:
                    details.append(f"nvidia-smi_error={err}")
        except Exception as exc:
            details.append(f"nvidia-smi_error={exc}")
    else:
        details.append("nvidia-smi_missing")
    if not available and is_wsl():
        details.append("wsl_gpu_missing")
    detail = "; ".join(details) if details else "GPU not detected"
    if any("torch_missing" in item for item in details):
        detail += "; install CUDA-enabled PyTorch (see docs/operations.md)"
    if any("torch_cuda_unavailable" in item for item in details):
        detail += "; torch installed but CUDA unavailable"
    if any("nvidia-smi_missing" in item for item in details):
        detail += "; nvidia-smi missing (install NVIDIA drivers)"
    if expected_cuda and not available:
        return DoctorCheckResult("gpu", False, detail)
    return DoctorCheckResult("gpu", True, detail)


def _looks_like_windows_path(raw: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\\\/]", raw))


def _check_database(config: AppConfig) -> DoctorCheckResult:
    try:
        db = DatabaseManager(config.database)
        with db.session() as session:
            session.execute(text("SELECT 1"))
        detail = "Connected"
        try:
            _check_migrations(db)
            detail = "Connected + migrations ok"
        except Exception as exc:
            detail = f"Connected; migration check warning: {exc}"
        return DoctorCheckResult("database", True, detail)
    except Exception as exc:
        detail = str(exc)
        if "Secure mode required" in detail:
            detail += " (enable database.encryption_enabled + sqlcipher, or run autocapture setup)"
        return DoctorCheckResult("database", False, detail)


def _check_migrations(db: DatabaseManager) -> None:
    import importlib.util

    if importlib.util.find_spec("alembic") is None:
        raise RuntimeError("alembic not installed")
    from alembic.config import Config  # type: ignore
    from alembic.runtime.migration import MigrationContext  # type: ignore
    from alembic.script import ScriptDirectory  # type: ignore

    base_dir = resource_root()
    config_path = base_dir / "alembic.ini"
    alembic_cfg = Config(str(config_path))
    script_location = base_dir / "alembic"
    alembic_cfg.set_main_option("script_location", str(script_location))
    script = ScriptDirectory.from_config(alembic_cfg)
    head = script.get_current_head()
    with db.engine.connect() as connection:
        context = MigrationContext.configure(connection)
        current = context.get_current_revision()
    if head and current != head:
        raise RuntimeError(f"DB revision {current} != head {head}")


def _check_encryption(config: AppConfig) -> DoctorCheckResult:
    if not config.encryption.enabled:
        return DoctorCheckResult("encryption", True, "Disabled")
    try:
        _ = EncryptionManager(config.encryption)
        _ = SecretStore(Path(config.capture.data_dir)).get_or_create()
    except Exception as exc:
        return DoctorCheckResult("encryption", False, str(exc))
    return DoctorCheckResult("encryption", True, "Initialized")


def _available_onnx_providers() -> list[str]:
    import importlib.util

    if importlib.util.find_spec("onnxruntime") is None:
        return []
    import onnxruntime as ort  # type: ignore

    return list(ort.get_available_providers())


def _parse_qdrant_host(url: str) -> tuple[str | None, int]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname
    port = parsed.port or 6333
    return host, port


def _uses_qdrant(config: AppConfig) -> bool:
    vector_backend = (getattr(config.routing, "vector_backend", "") or "").strip().lower()
    spans_backend = (getattr(config.routing, "spans_v2_backend", "") or "").strip().lower()
    return vector_backend == "qdrant" or spans_backend == "qdrant"


def _check_qdrant(config: AppConfig) -> DoctorCheckResult:
    if not _uses_qdrant(config):
        return DoctorCheckResult("qdrant", True, "Disabled (routing)")
    host, port = _parse_qdrant_host(config.qdrant.url)
    if not host:
        return DoctorCheckResult("qdrant", False, "Invalid qdrant.url")
    binary_path = resolve_qdrant_path(config.qdrant)
    try:
        response = httpx.get(f"http://{host}:{port}/health", timeout=1.0)
        if response.status_code == 200:
            detail = f"Running at {config.qdrant.url}"
            if binary_path:
                detail += f"; binary={binary_path}"
            return DoctorCheckResult("qdrant", True, detail)
    except Exception:
        pass

    if not is_loopback_host(host):
        return DoctorCheckResult(
            "qdrant",
            True,
            f"Remote Qdrant configured at {config.qdrant.url}; not managed locally",
        )

    if binary_path:
        return DoctorCheckResult(
            "qdrant",
            True,
            f"Not running; binary found at {binary_path} (sidecar will manage).",
        )

    detail = (
        "Qdrant not running and binary missing; "
        "run tools/vendor_windows_binaries.py or set qdrant.binary_path."
    )
    ok = sys.platform != "win32"
    return DoctorCheckResult("qdrant", ok, detail)


def _check_capture_backends(config: AppConfig) -> DoctorCheckResult:
    if sys.platform != "win32":
        return DoctorCheckResult("screen_capture", True, "Non-Windows platform")
    results: list[str] = []
    errors: list[str] = []
    for label, backend_cls in (
        ("dxcam", DxCamBackend),
        ("mss", MssBackend),
    ):
        try:
            backend = backend_cls()
            frames = backend.grab_all()
            if frames:
                results.append(label)
            else:
                errors.append(f"{label}: no frames")
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    if results:
        detail = f"OK ({', '.join(results)})"
        if errors:
            detail += f"; fallback errors: {', '.join(errors)}"
        return DoctorCheckResult("screen_capture", True, detail)
    return DoctorCheckResult("screen_capture", False, "; ".join(errors) or "Unavailable")


def _check_ffmpeg(config: AppConfig) -> DoctorCheckResult:
    if not config.capture.record_video:
        return DoctorCheckResult("ffmpeg", True, "Video recording disabled")
    if not config.ffmpeg.enabled:
        return DoctorCheckResult("ffmpeg", False, "ffmpeg disabled while record_video is enabled")
    try:
        path = resolve_ffmpeg_path(config.ffmpeg)
        if path is None:
            return DoctorCheckResult(
                "ffmpeg",
                False,
                "FFmpeg missing; install/bundle ffmpeg or set ffmpeg.explicit_path",
            )
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            result = subprocess.run(
                [str(path), "-version"],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=creationflags,
            )
            if result.returncode != 0:
                return DoctorCheckResult(
                    "ffmpeg", False, f"ffmpeg -version failed: {result.stderr.strip()}"
                )
            first_line = result.stdout.splitlines()[0] if result.stdout else ""
            return DoctorCheckResult("ffmpeg", True, f"Using {path} ({first_line})")
        except Exception as exc:
            return DoctorCheckResult("ffmpeg", False, f"ffmpeg probe failed: {exc}")
    except Exception as exc:
        return DoctorCheckResult("ffmpeg", False, str(exc))


def _check_ocr(config: AppConfig) -> DoctorCheckResult:
    import importlib.util

    engine = (config.vision_extract.engine or "").lower()
    if config.routing.ocr == "disabled" or engine in {"disabled", "off"}:
        return DoctorCheckResult("ocr", True, "Disabled")
    if engine not in {"rapidocr", "rapidocr-onnxruntime"}:
        return DoctorCheckResult("ocr", True, f"Skipped (engine={engine})")
    if importlib.util.find_spec("rapidocr_onnxruntime") is None:
        return DoctorCheckResult("ocr", False, "rapidocr_onnxruntime not installed")
    from .vision.rapidocr import RapidOCRExtractor, available_onnx_providers, select_onnx_provider

    try:
        image = _build_ocr_fixture()
        providers = available_onnx_providers()
        selected, _use_cuda = select_onnx_provider(config.ocr, providers)
        selected_name = selected or "none"
        processor = RapidOCRExtractor(config.ocr)
        spans = processor.extract(image)
        text = " ".join(span[0] for span in spans if span)
        if not text.strip():
            return DoctorCheckResult("ocr", False, "Empty OCR output")
        detail = (
            f"device={config.ocr.device}; "
            f"selected_provider={selected_name}; "
            f"available_providers={providers or 'none'}"
        )
        if config.ocr.device.lower() == "cuda" and selected_name != "CUDAExecutionProvider":
            detail += " (CUDAExecutionProvider missing; install onnxruntime-gpu + CUDA/cuDNN)"
        return DoctorCheckResult("ocr", True, detail)
    except Exception as exc:
        return DoctorCheckResult("ocr", False, str(exc))


def _build_ocr_fixture():
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    image = Image.new("RGB", (200, 80), color="white")
    draw = ImageDraw.Draw(image)
    text = "TEST"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    position = ((200 - text_w) // 2, (80 - text_h) // 2)
    draw.text(position, text, fill="black", font=font)
    return np.array(image)


def _check_embeddings(config: AppConfig) -> DoctorCheckResult:
    if config.routing.embedding == "disabled" or config.embed.text_model == "disabled":
        return DoctorCheckResult("embeddings", True, "Disabled")
    if config.offline:
        return DoctorCheckResult(
            "embeddings",
            False,
            "Offline mode enabled; set offline:false to download models or pre-cache embeddings.",
        )
    try:
        embedder = EmbeddingService(config.embed)
        vectors = embedder.embed_texts(["doctor probe"])
        if not vectors or not vectors[0]:
            return DoctorCheckResult("embeddings", False, "No vectors produced")
        return DoctorCheckResult("embeddings", True, f"{embedder.model_name}")
    except Exception as exc:
        return DoctorCheckResult("embeddings", False, str(exc))


def _check_vector_index(config: AppConfig) -> DoctorCheckResult:
    if not _uses_qdrant(config):
        return DoctorCheckResult("vector_index", True, "SQLite (routing)")
    host, _port = _parse_qdrant_host(config.qdrant.url)
    if host and is_loopback_host(host) and resolve_qdrant_path(config.qdrant):
        return DoctorCheckResult(
            "vector_index",
            True,
            "Qdrant configured for sidecar; will initialize when running.",
        )
    try:
        from qdrant_client import QdrantClient  # type: ignore

        client = QdrantClient(url=config.qdrant.url, timeout=2.0)
        _ = client.get_collections()
        return DoctorCheckResult("vector_index", True, "Qdrant reachable")
    except Exception as exc:
        return DoctorCheckResult("vector_index", False, str(exc))


def _check_api_port(config: AppConfig) -> DoctorCheckResult:
    return _check_port("api_port", config.api.bind_host, config.api.port)


def _check_metrics(config: AppConfig) -> DoctorCheckResult:
    port = get_metrics_port() or config.observability.prometheus_port
    host = config.observability.prometheus_bind_host
    if _port_in_use(host, port):
        try:
            response = httpx.get(f"http://{host}:{port}/metrics", timeout=2.0)
            if response.status_code == 200:
                return DoctorCheckResult("metrics", True, "metrics endpoint responding")
            return DoctorCheckResult("metrics", False, f"unexpected status {response.status_code}")
        except Exception as exc:
            return DoctorCheckResult("metrics", False, str(exc))
    return _check_port("metrics", host, port)


def _check_port(name: str, host: str, port: int) -> DoctorCheckResult:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return DoctorCheckResult(name, True, f"Port {port} available")
    except PermissionError as exc:
        return DoctorCheckResult(
            name,
            True,
            f"Skipped port probe for {port}: permission denied ({exc})",
        )
    except Exception as exc:
        return DoctorCheckResult(name, False, str(exc))


def _port_in_use(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def _check_raw_input(config: AppConfig) -> DoctorCheckResult:
    _ = config
    if sys.platform != "win32":
        return DoctorCheckResult("raw_input", True, "Non-Windows platform")
    probe = probe_raw_input()
    if probe["available"]:
        return DoctorCheckResult("raw_input", True, "Raw input ready")
    try:
        win32 = Win32Api()
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if win32.user32.GetLastInputInfo(ctypes.byref(info)):
            return DoctorCheckResult(
                "raw_input",
                True,
                f"Fallback polling active ({probe['error']})",
            )
    except Exception:
        pass
    return DoctorCheckResult("raw_input", False, probe.get("error") or "unavailable")
