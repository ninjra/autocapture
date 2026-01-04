"""Application bootstrap / CLI.

This module is the canonical entrypoint for Autocapture.
It replaces earlier experimental bootstrap code and avoids circular imports.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from . import claim_single_instance, ensure_expected_interpreter
from .config import AppConfig, load_config
from .logging_utils import configure_logging
from .runtime import AppRuntime
from .storage.database import DatabaseManager
from .worker.event_worker import EventIngestWorker


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="autocapture")
    p.add_argument(
        "--config",
        default=os.environ.get("AUTOCAPTURE_CONFIG", "autocapture.yml"),
        help="Path to config YAML (default: autocapture.yml or AUTOCAPTURE_CONFIG).",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("run", help="Run the capture + OCR orchestrator (default).")
    sub.add_parser("app", help="Run the tray UI + full local pipeline.")
    sub.add_parser("tray", help="Alias for app.")
    sub.add_parser("api", help="Run the local API + UI server.")
    sub.add_parser("worker", help="Run the OCR ingest worker loop only.")
    sub.add_parser("doctor", help="Run quick environment/self checks and exit.")
    sub.add_parser("print-config", help="Load config and print resolved values.")

    return p.parse_args(argv)


def _doctor(config: AppConfig) -> int:
    """Lightweight sanity checks that are cheap and help during deployment."""
    ok = True

    # Paths
    cap = config.capture
    for name, p in [("staging_dir", cap.staging_dir), ("data_dir", cap.data_dir)]:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Cannot create %s=%r: %s", name, p, exc)
            ok = False

    # DB
    try:
        _ = DatabaseManager(config.database)
    except Exception as exc:
        logger.error("Database init failed: %s", exc)
        ok = False

    if config.tracking.enabled:
        tracking_dir = _resolve_tracking_dir(config)
        try:
            tracking_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Tracking DB directory: %s", tracking_dir)
        except Exception as exc:
            logger.error("Cannot create tracking DB directory %s: %s", tracking_dir, exc)
            ok = False

    # Encryption key (if enabled)
    if getattr(config, "encryption", None) and config.encryption.enabled:
        try:
            from .encryption import EncryptionManager

            _ = EncryptionManager(config.encryption)
        except Exception as exc:
            logger.error("Encryption init failed: %s", exc)
            ok = False

    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception as exc:
        logger.error("OCR dependency missing (rapidocr_onnxruntime): %s", exc)
        ok = False

    logger.info("Doctor result: %s", "OK" if ok else "FAILED")
    return 0 if ok else 2


def _run_runtime(config: AppConfig) -> int:
    runtime = AppRuntime(config)
    runtime.start()
    logger.info("Autocapture running. Press Ctrl+C to stop.")
    runtime.wait_forever()
    runtime.stop()
    return 0


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    cmd = args.cmd or "run"

    ensure_expected_interpreter()

    config_path = Path(args.config)
    config = load_config(config_path)
    configure_logging(getattr(config, "logging", None))

    if cmd == "print-config":
        # Avoid importing rich; keep it simple and predictable.
        logger.info("Resolved config loaded from %s", config_path)
        logger.info("%s", config.model_dump() if hasattr(config, "model_dump") else config.dict())
        return

    if cmd == "doctor":
        raise SystemExit(_doctor(config))

    if cmd == "api":
        from .api.server import create_app
        import uvicorn

        _validate_remote_mode(config)
        app = create_app(config)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=config.api.bind_host,
                port=config.api.port,
                log_level="info",
                ssl_certfile=(
                    str(config.mode.tls_cert_path) if config.mode.https_enabled else None
                ),
                ssl_keyfile=(
                    str(config.mode.tls_key_path) if config.mode.https_enabled else None
                ),
            )
        )
        server.run()
        return

    if cmd == "worker":
        if not claim_single_instance():
            logger.warning("Autocapture worker already active in another interpreter. Exiting.")
            raise SystemExit(0)
        worker = EventIngestWorker(config)
        logger.info("OCR ingest worker running. Press Ctrl+C to stop.")
        try:
            worker.run_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down worker")
        return

    if cmd in {"app", "tray"}:
        if not claim_single_instance():
            logger.warning(
                "Autocapture already active in another interpreter. Exiting."
            )
            raise SystemExit(0)
        from .ui.tray import run_tray

        log_dir = Path(config.capture.data_dir) / "logs"
        run_tray(config_path, log_dir)
        return

    if not claim_single_instance():
        logger.warning(
            "Autocapture orchestrator already active in another interpreter. Exiting."
        )
        raise SystemExit(0)

    try:
        raise SystemExit(_run_runtime(config))
    except KeyboardInterrupt:
        logger.info("Shutting down")
        # Let asyncio loop close cleanly.
        time.sleep(0.1)


def _validate_remote_mode(config: AppConfig) -> None:
    if config.mode.mode != "remote":
        return
    missing = []
    if not config.mode.overlay_interface:
        missing.append("mode.overlay_interface")
    if not config.mode.https_enabled:
        missing.append("mode.https_enabled")
    if not config.mode.tls_cert_path or not config.mode.tls_key_path:
        missing.append("TLS cert/key")
    if not config.mode.google_oauth_client_id or not config.mode.google_oauth_client_secret:
        missing.append("Google OIDC client")
    if not config.mode.google_allowed_emails:
        missing.append("mode.google_allowed_emails")
    if config.api.bind_host in ("0.0.0.0", "127.0.0.1"):
        missing.append("api.bind_host (overlay IP)")
    if missing:
        raise RuntimeError(
            "Remote mode misconfigured. Missing: " + ", ".join(missing)
        )


def _resolve_tracking_dir(config: AppConfig) -> Path:
    db_path = config.tracking.db_path
    if db_path.is_absolute():
        return db_path.parent
    return Path(config.capture.data_dir) / db_path.parent


if __name__ == "__main__":
    main()
