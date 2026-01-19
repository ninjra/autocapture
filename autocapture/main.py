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
from .runtime_context import build_runtime_context
from .runtime_device import require_cuda_available
from .runtime_env import (
    RuntimeEnvConfig,
    apply_runtime_env_overrides,
    configure_cuda_visible_devices,
    load_runtime_env,
)
from .runtime_governor import RuntimeGovernor
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
    sub.add_parser("smoke", help="Run minimal smoke checks and exit.")
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

    keys = sub.add_parser("keys", help="Export/import encrypted keys for portability.")
    keys_sub = keys.add_subparsers(dest="keys_cmd", required=True)
    keys_export = keys_sub.add_parser("export", help="Export encrypted keys bundle.")
    keys_export.add_argument("--out", required=True, help="Output path for key bundle.")
    keys_export.add_argument("--password", required=True, help="Password for encrypting bundle.")
    keys_import = keys_sub.add_parser("import", help="Import encrypted keys bundle.")
    keys_import.add_argument("path", help="Input path for key bundle.")
    keys_import.add_argument("--password", required=True, help="Password for decrypting bundle.")

    promptops = sub.add_parser("promptops", help="PromptOps automation utilities.")
    promptops_sub = promptops.add_subparsers(dest="promptops_cmd", required=True)
    promptops_sub.add_parser("run", help="Run PromptOps once.")
    promptops_sub.add_parser("status", help="Show latest PromptOps run.")
    promptops_sub.add_parser("list", help="List recent PromptOps runs.")

    overlay = sub.add_parser("overlay-tracker", help="Overlay tracker utilities.")
    overlay_sub = overlay.add_subparsers(dest="overlay_cmd", required=True)
    overlay_sub.add_parser("status", help="Show overlay tracker status.")

    research = sub.add_parser("research", help="Research/model discovery utilities.")
    research_sub = research.add_subparsers(dest="research_cmd", required=True)
    scout = research_sub.add_parser("scout", help="Fetch model/paper updates and write report.")
    scout.add_argument("--out", required=True, help="Output path for scout report JSON.")
    scout.add_argument(
        "--append",
        default="docs/research/scout_log.md",
        help="Append a summary markdown log (set to empty string to disable).",
    )

    db_cmd = sub.add_parser("db", help="Database utilities.")
    db_sub = db_cmd.add_subparsers(dest="db_cmd", required=True)
    db_sub.add_parser("encrypt", help="Encrypt the SQLite database with SQLCipher.")
    backfill = db_sub.add_parser("backfill", help="Run resumable Phase 0 backfills.")
    backfill.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    backfill.add_argument("--batch-size", type=int, default=500)
    backfill.add_argument("--max-rows", type=int, default=None)
    backfill.add_argument("--frame-hash-days", type=int, default=7)
    backfill.add_argument("--fill-monotonic", action=argparse.BooleanOptionalAction, default=False)
    backfill.add_argument(
        "--task",
        action="append",
        default=[],
        help="Backfill task(s): captures, events, spans, embeddings.",
    )
    backfill.add_argument(
        "--reset-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    integrity = db_sub.add_parser("integrity-scan", help="Scan for index/storage orphans.")
    integrity.add_argument(
        "--include-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    prune = db_sub.add_parser("prune-indexes", help="Prune index entries missing events.")
    prune.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    prune.add_argument(
        "--include-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    memory = sub.add_parser("memory", help="Deterministic memory store utilities.")
    from .memory.cli import add_memory_subcommands

    add_memory_subcommands(memory)

    return p.parse_args(argv)


def _doctor(config: AppConfig) -> int:
    """Run full diagnostic suite and exit."""
    exit_code, _report = run_doctor(config)
    return exit_code


def _run_runtime(config: AppConfig, runtime_env: RuntimeEnvConfig) -> int:
    runtime = AppRuntime(config, runtime_env=runtime_env)
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
    runtime_env = load_runtime_env()
    configure_cuda_visible_devices(runtime_env)
    apply_runtime_env_overrides(config, runtime_env)
    require_cuda_available(runtime_env)
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

    if cmd == "smoke":
        from .smoke import run_smoke

        raise SystemExit(run_smoke(config))

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

    if cmd == "memory":
        from .memory.cli import run_memory_cli

        raise SystemExit(run_memory_cli(args, config))

    if cmd == "overlay-tracker":
        from .overlay_tracker.cli import overlay_status

        if args.overlay_cmd == "status":
            raise SystemExit(overlay_status(config))

    if cmd == "research":
        from .research.scout import append_report_log, run_scout, write_report

        out_path = Path(args.out)
        report = run_scout(config)
        write_report(report, out_path)
        if args.append:
            append_report_log(report, Path(args.append))
        logger.info("Research scout report written: {}", out_path)
        return

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

    if cmd == "keys":
        from .security.portable_keys import export_keys, import_keys

        if args.keys_cmd == "export":
            out_path = Path(args.out)
            export_keys(config, out_path, args.password)
            logger.info("Key export complete: {}", out_path)
            return
        if args.keys_cmd == "import":
            in_path = Path(args.path)
            import_keys(config, in_path, args.password)
            logger.info("Key import complete: {}", in_path)
            return

    if cmd == "db":
        db = DatabaseManager(config.database)
        if args.db_cmd == "encrypt":
            from .storage.sqlcipher_migrate import encrypt_sqlite_database

            encrypt_sqlite_database(config.database)
            logger.info("Database encryption complete.")
            return
        if args.db_cmd == "backfill":
            from .storage.backfill import BackfillRunner

            runner = BackfillRunner(config, db=db)
            counts = runner.run(
                tasks=args.task or None,
                dry_run=bool(args.dry_run),
                batch_size=int(args.batch_size),
                max_rows=args.max_rows,
                frame_hash_days=int(args.frame_hash_days),
                fill_monotonic=bool(args.fill_monotonic),
                reset_checkpoints=bool(args.reset_checkpoints),
            )
            logger.info(
                "Backfill complete. captures={} events={} spans={} embeddings={} hashes={} normalized={}",
                counts.captures_updated,
                counts.events_updated,
                counts.spans_updated,
                counts.embeddings_updated,
                counts.hashes_computed,
                counts.normalized_texts,
            )
            return
        if args.db_cmd == "integrity-scan":
            from .storage.integrity import scan_integrity
            from .embeddings.service import EmbeddingService
            from .indexing.vector_index import VectorIndex

            vector_index = None
            if args.include_vectors and config.qdrant.enabled:
                try:
                    embedder = EmbeddingService(config.embed)
                    vector_index = VectorIndex(config, embedder.dim)
                except Exception:
                    vector_index = None
            report = scan_integrity(db, vector_index=vector_index)
            logger.info(
                "Integrity scan: fts_orphans={} spans_orphans={} embedding_orphans={} vector_orphans={}",
                report.orphan_fts,
                report.orphan_spans,
                report.orphan_embeddings,
                report.orphan_vectors,
            )
            return
        if args.db_cmd == "prune-indexes":
            from .storage.integrity import find_fts_orphans
            from .indexing.pruner import IndexPruner
            from .embeddings.service import EmbeddingService
            from .indexing.vector_index import VectorIndex
            from .indexing.spans_v2 import SpansV2Index

            event_ids = find_fts_orphans(db)
            if not event_ids:
                logger.info("No orphan index entries detected.")
                return
            if args.dry_run:
                logger.info("Prune preview: {} orphaned event ids", len(event_ids))
                return
            vector_index = None
            spans_index = None
            if args.include_vectors and config.qdrant.enabled:
                try:
                    embedder = EmbeddingService(config.embed)
                    vector_index = VectorIndex(config, embedder.dim)
                    spans_index = SpansV2Index(config, embedder.dim)
                except Exception:
                    vector_index = None
                    spans_index = None
            pruner = IndexPruner(db, vector_index=vector_index, spans_index=spans_index)
            pruner.prune_event_ids(event_ids)
            logger.info("Pruned {} orphaned event ids from indexes.", len(event_ids))
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
        runtime_context = build_runtime_context(config, runtime_env)
        profile_override = runtime_env.profile if runtime_env.profile_override else None
        runtime_governor = RuntimeGovernor(
            config.runtime,
            raw_input=None,
            pause_controller=runtime_context.pause,
            profile_override=profile_override,
            profile_scheduler=runtime_context.scheduler,
        )
        runtime_governor.start()
        worker = WorkerSupervisor(
            config=config,
            runtime_governor=runtime_governor,
            pause_controller=runtime_context.pause,
        )
        logger.info("Worker supervisor running. Press Ctrl+C to stop.")
        worker.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down worker")
        finally:
            worker.stop()
            runtime_governor.stop()
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
        raise SystemExit(_run_runtime(config, runtime_env))
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
