"""Application bootstrap / CLI.

This module is the canonical entrypoint for Autocapture.
It replaces earlier experimental bootstrap code and avoids circular imports.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from loguru import logger

from . import claim_single_instance, ensure_expected_interpreter
from .capture.orchestrator import CaptureOrchestrator
from .config import AppConfig, load_config
from .logging_utils import configure_logging
from .storage.database import DatabaseManager


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="autocapture")
    p.add_argument(
        "--config",
        default=os.environ.get("AUTOCAPTURE_CONFIG", "autocapture.yml"),
        help="Path to config YAML (default: autocapture.yml or AUTOCAPTURE_CONFIG).",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("run", help="Run the capture + OCR orchestrator (default).")
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

    # Encryption key (if enabled)
    if getattr(config, "encryption", None) and config.encryption.enabled:
        try:
            from .encryption import EncryptionManager

            _ = EncryptionManager(config.encryption)
        except Exception as exc:
            logger.error("Encryption init failed: %s", exc)
            ok = False

    logger.info("Doctor result: %s", "OK" if ok else "FAILED")
    return 0 if ok else 2


def _build_orchestrator(config: AppConfig) -> CaptureOrchestrator:
    db = DatabaseManager(config.database)
    cap = config.capture
    return CaptureOrchestrator(
        database=db,
        data_dir=Path(cap.data_dir),
        idle_grace_ms=cap.hid.idle_grace_ms,
        fps_soft_cap=cap.hid.fps_soft_cap,
        on_ocr_observation=None,
        on_vision_observation=None,
        vision_sample_rate=getattr(cap, "vision_sample_rate", 0.0),
    )


async def _run_async(config: AppConfig) -> int:
    orch = _build_orchestrator(config)
    orch.start()
    logger.info("Autocapture running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            orch.stop()
        except Exception:
            logger.exception("Error while stopping orchestrator")
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

    if not claim_single_instance("autocapture-orchestrator"):
        logger.warning(
            "Autocapture orchestrator already active in another interpreter. Exiting."
        )
        raise SystemExit(0)

    try:
        raise SystemExit(asyncio.run(_run_async(config)))
    except KeyboardInterrupt:
        logger.info("Shutting down")
        # Let asyncio loop close cleanly.
        time.sleep(0.1)


if __name__ == "__main__":
    main()

