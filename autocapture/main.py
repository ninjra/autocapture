"""Application bootstrap / CLI.

This module is the canonical entrypoint for Autocapture.
It replaces earlier experimental bootstrap code and avoids circular imports.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import claim_single_instance, ensure_expected_interpreter
from .config import AppConfig, is_loopback_host, load_config, overlay_interface_ips
from .logging_utils import configure_logging, get_logger
from .doctor import run_doctor
from .paths import default_config_path, ensure_config_path
from .runtime import AppRuntime
from .security.offline_guard import apply_offline_guard
from .storage.database import DatabaseManager
from .worker.supervisor import WorkerSupervisor


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="autocapture")
    p.add_argument(
        "--config",
        default=os.environ.get("AUTOCAPTURE_CONFIG", str(default_config_path())),
        help=("Path to config YAML (default: AUTOCAPTURE_CONFIG or platform default)."),
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="Optional log directory override (defaults to platform log dir).",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("run", help="Run the capture + OCR orchestrator (default).")
    sub.add_parser("app", help="Run the tray UI + full local pipeline.")
    sub.add_parser("tray", help="Alias for app.")
    sub.add_parser("api", help="Run the local API + UI server.")
    sub.add_parser("worker", help="Run the OCR ingest worker loop only.")
    sub.add_parser("doctor", help="Run quick environment/self checks and exit.")
    sub.add_parser("print-config", help="Load config and print resolved values.")
    export = sub.add_parser("export", help="Export events + media for backup.")
    export.add_argument("--out", required=True, help="Output path for export bundle.")
    export.add_argument("--days", type=int, default=90, help="Export last N days.")
    export.add_argument(
        "--include-media",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include screenshots in export bundle.",
    )
    export.add_argument(
        "--decrypt-media",
        action="store_true",
        help="Decrypt media (requires encryption enabled).",
    )
    export.add_argument(
        "--zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a zip archive (default) or a folder.",
    )

    promptops = sub.add_parser("promptops", help="PromptOps automation utilities.")
    promptops_sub = promptops.add_subparsers(dest="promptops_cmd", required=True)
    promptops_sub.add_parser("run", help="Run PromptOps once.")
    promptops_sub.add_parser("status", help="Show latest PromptOps run.")
    promptops_sub.add_parser("list", help="List recent PromptOps runs.")

    db_cmd = sub.add_parser("db", help="Database utilities.")
    db_sub = db_cmd.add_subparsers(dest="db_cmd", required=True)
    db_sub.add_parser("encrypt", help="Encrypt the SQLite database with SQLCipher.")

    return p.parse_args(argv)


def _doctor(config: AppConfig) -> int:
    """Run full diagnostic suite and exit."""
    exit_code, _report = run_doctor(config)
    return exit_code


def _run_runtime(config: AppConfig) -> int:
    runtime = AppRuntime(config)
    runtime.start()
    log = get_logger("cli")
    log.info("Autocapture running. Press Ctrl+C to stop.")
    runtime.wait_forever()
    runtime.stop()
    return 0


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    cmd = args.cmd or "run"

    ensure_expected_interpreter()

    config_path = ensure_config_path(Path(args.config))
    config = load_config(config_path)
    configure_logging(args.log_dir or getattr(config, "logging", None))
    logger = get_logger("cli")
    if config.offline and not config.privacy.cloud_enabled and config.mode.mode == "remote":
        logger.warning("Offline guard disabled in remote mode (OIDC/JWKS requires outbound HTTPS).")
    apply_offline_guard(
        enabled=(
            config.offline and not config.privacy.cloud_enabled and config.mode.mode != "remote"
        ),
        allowed_hosts={"127.0.0.1", "::1", "localhost"},
    )

    if cmd == "print-config":
        # Avoid importing rich; keep it simple and predictable.
        logger.info("Resolved config loaded from {}", config_path)
        logger.info(
            "{}",
            config.model_dump() if hasattr(config, "model_dump") else config.dict(),
        )
        return

    if cmd == "doctor":
        raise SystemExit(_doctor(config))

    if cmd == "promptops":
        from .promptops import PromptOpsRunner
        from .storage.models import PromptOpsRunRecord

        db = DatabaseManager(config.database)
        runner = PromptOpsRunner(config, db)
        if args.promptops_cmd == "run":
            run = runner.run_once()
            if run is None:
                logger.info("PromptOps disabled; no run executed.")
                raise SystemExit(0)
            logger.info("PromptOps run {} status={}", run.run_id, run.status)
            raise SystemExit(0 if run.status != "failed" else 2)
        if args.promptops_cmd == "status":
            with db.session() as session:
                run = (
                    session.query(PromptOpsRunRecord).order_by(PromptOpsRunRecord.ts.desc()).first()
                )
            if not run:
                logger.info("No PromptOps runs recorded.")
                raise SystemExit(1)
            logger.info("Latest PromptOps run {} status={}", run.run_id, run.status)
            if run.pr_url:
                logger.info("PR: {}", run.pr_url)
            raise SystemExit(0)
        if args.promptops_cmd == "list":
            with db.session() as session:
                runs = (
                    session.query(PromptOpsRunRecord)
                    .order_by(PromptOpsRunRecord.ts.desc())
                    .limit(10)
                    .all()
                )
            for run in runs:
                logger.info("{} {} {}", run.run_id, run.status, run.pr_url or "")
            raise SystemExit(0)

    if cmd == "export":
        from .export import export_capture

        out_path = Path(args.out)
        export_capture(
            config,
            out_path=out_path,
            days=args.days,
            include_media=args.include_media,
            decrypt_media=args.decrypt_media,
            zip_output=args.zip,
        )
        logger.info("Export complete: {}", out_path)
        return

    if cmd == "db":
        if args.db_cmd == "encrypt":
            from .storage.sqlcipher_migrate import encrypt_sqlite_database

            encrypt_sqlite_database(config.database)
            logger.info("Database encryption complete.")
            return

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
                ssl_keyfile=(str(config.mode.tls_key_path) if config.mode.https_enabled else None),
            )
        )
        server.run()
        return

    if cmd == "worker":
        if not claim_single_instance():
            logger.warning("Autocapture worker already active in another interpreter. Exiting.")
            raise SystemExit(0)
        worker = WorkerSupervisor(config=config)
        logger.info("Worker supervisor running. Press Ctrl+C to stop.")
        worker.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down worker")
        finally:
            worker.stop()
        return

    if cmd in {"app", "tray"}:
        if not claim_single_instance():
            logger.warning("Autocapture already active in another interpreter. Exiting.")
            raise SystemExit(0)
        try:
            from .ui.tray import run_tray
        except ImportError:
            print(
                "PySide6 is not installed. Install with: poetry install --extras 'ui' "
                "(and 'windows' on Windows).",
                file=sys.stderr,
            )
            raise SystemExit(2)

        log_dir = Path(config.capture.data_dir) / "logs"
        run_tray(config_path, log_dir)
        return

    if not claim_single_instance():
        logger.warning("Autocapture orchestrator already active in another interpreter. Exiting.")
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
    # Bind host should be on the overlay interface, not wildcard or loopback.
    if config.api.bind_host in ("0.0.0.0", "::", ""):
        missing.append("api.bind_host must not be wildcard in remote mode")
    elif is_loopback_host(config.api.bind_host):
        missing.append("api.bind_host (overlay IP)")
    elif config.mode.overlay_interface:
        ips = overlay_interface_ips(config.mode.overlay_interface)
        if not ips:
            missing.append(f"overlay_interface has no usable IPs: {config.mode.overlay_interface}")
        elif config.api.bind_host not in ips:
            missing.append("api.bind_host must be an IP assigned to mode.overlay_interface")
    if missing:
        raise RuntimeError("Remote mode misconfigured. Missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
