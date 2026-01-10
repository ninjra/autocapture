"""Health checks for Autocapture."""

from __future__ import annotations

import ctypes
import json
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import httpx
from sqlalchemy import text

from .config import AppConfig
from .capture.backends import DxCamBackend, MssBackend
from .capture.raw_input import LASTINPUTINFO, Win32Api, probe_raw_input
from .embeddings.service import EmbeddingService
from .encryption import EncryptionManager
from .logging_utils import get_logger
from .memory.entities import EntityResolver, SecretStore
from .observability.metrics import get_metrics_port
from .paths import resolve_ffmpeg_path
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
            _check_ffmpeg,
            _check_capture_backends,
            _check_ocr,
            _check_embeddings,
            _check_vector_index,
            _check_api_port,
            _check_metrics,
            _check_raw_input,
        ]

    log.info("Doctor config summary:\n{}", _redact_config(config))
    results = [check(config) for check in checks]
    ok = all(result.ok for result in results)
    _print_table(results)
    return (0 if ok else 2), DoctorReport(ok=ok, results=results)


def _redact_config(config: AppConfig) -> str:
    data = (
        config.model_dump() if hasattr(config, "model_dump") else config.dict()
    )
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
            Path(path).mkdir(parents=True, exist_ok=True)
            probe = Path(path) / ".doctor_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as exc:
            return DoctorCheckResult(name, False, f"{exc}")
    return DoctorCheckResult("paths", True, "Writable")


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
        return DoctorCheckResult("database", False, str(exc))


def _check_migrations(db: DatabaseManager) -> None:
    import importlib.util

    if importlib.util.find_spec("alembic") is None:
        raise RuntimeError("alembic not installed")
    from alembic.config import Config  # type: ignore
    from alembic.runtime.migration import MigrationContext  # type: ignore
    from alembic.script import ScriptDirectory  # type: ignore

    config_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    alembic_cfg = Config(str(config_path))
    script_location = Path(__file__).resolve().parents[1] / "alembic"
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


def _check_capture_backends(config: AppConfig) -> DoctorCheckResult:
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
    try:
        path = resolve_ffmpeg_path(config.ffmpeg)
        if path is None:
            return DoctorCheckResult(
                "ffmpeg",
                True,
                "FFmpeg missing; video disabled (allow_disable=true)",
            )
        return DoctorCheckResult("ffmpeg", True, f"Using {path}")
    except Exception as exc:
        return DoctorCheckResult("ffmpeg", False, str(exc))


def _check_ocr(config: AppConfig) -> DoctorCheckResult:
    try:
        from .worker.event_worker import OCRProcessor
        from PIL import Image
    except Exception as exc:
        return DoctorCheckResult("ocr", False, str(exc))
    try:
        image = _build_ocr_fixture()
        processor = OCRProcessor()
        spans = processor.run(image)
        text = " ".join(span[1] for span in spans if len(span) > 1)
        if not text.strip():
            return DoctorCheckResult("ocr", False, "Empty OCR output")
        return DoctorCheckResult("ocr", True, "OK")
    except Exception as exc:
        return DoctorCheckResult("ocr", False, str(exc))


def _build_ocr_fixture():
    from PIL import Image, ImageDraw, ImageFont

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
    return image


def _check_embeddings(config: AppConfig) -> DoctorCheckResult:
    if config.routing.embedding == "disabled" or config.embed.text_model == "disabled":
        return DoctorCheckResult("embeddings", True, "Disabled")
    try:
        embedder = EmbeddingService(config.embed)
        vectors = embedder.embed_texts(["doctor probe"])
        if not vectors or not vectors[0]:
            return DoctorCheckResult("embeddings", False, "No vectors produced")
        return DoctorCheckResult("embeddings", True, f"{embedder.model_name}")
    except Exception as exc:
        return DoctorCheckResult("embeddings", False, str(exc))


def _check_vector_index(config: AppConfig) -> DoctorCheckResult:
    if not config.qdrant.enabled:
        return DoctorCheckResult("vector_index", True, "Disabled")
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
            return DoctorCheckResult(
                "metrics", False, f"unexpected status {response.status_code}"
            )
        except Exception as exc:
            return DoctorCheckResult("metrics", False, str(exc))
    return _check_port("metrics", host, port)


def _check_port(name: str, host: str, port: int) -> DoctorCheckResult:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return DoctorCheckResult(name, True, f"Port {port} available")
    except Exception as exc:
        return DoctorCheckResult(name, False, str(exc))


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


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
