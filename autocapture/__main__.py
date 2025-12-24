"""Module entry point for the autocapture CLI."""

from .main import main
"""Command-line entrypoints for autocapture."""
"""Autocapture CLI entrypoint for tray/worker/doctor modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from .logging_utils import configure_logging
from .store import apply_migrations, open_db


def _doctor(args: argparse.Namespace) -> None:
    configure_logging(args.log_dir, args.log_level)
    conn = open_db(args.data_dir)
    apply_migrations(conn)
    counts = {}
    for table in ("segments", "observations", "jobs", "segment_fts"):
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = int(cursor.fetchone()[0])
    print("SQLite store OK")
    for table, count in counts.items():
        print(f"{table}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autocapture tools")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Validate local SQLite store")
    args = parser.parse_args()
    if args.command == "doctor":
        _doctor(args)
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .logging_utils import configure_logging, get_logger

DEFAULT_CONFIG_NAME = "config.yml"

logger = get_logger("cli")


def _default_appdata_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "Autocapture"
    return Path.home() / "AppData" / "Local" / "Autocapture"


def _default_config_path() -> Path:
    return _default_appdata_dir() / DEFAULT_CONFIG_NAME


def _serialize_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _prompt_data_dir() -> Path:
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
    except Exception as exc:  # pragma: no cover - UI deps may be unavailable
        logger.warning("Folder picker unavailable (%s); using default data dir", exc)
        return Path.home() / "AutocaptureData"

    app = QApplication.instance() or QApplication([])
    directory = QFileDialog.getExistingDirectory(
        None,
        "Choose Autocapture data directory",
        str(Path.home()),
    )
    if directory:
        return Path(directory)
    return Path.home() / "AutocaptureData"


def ensure_default_config(path: Path) -> Path:
    if path.exists():
        return path

    data_dir = _prompt_data_dir()
    payload = {
        "data_dir": str(data_dir),
        "capture": {"backend": "dxcam", "fallback": "mss"},
        "ui": {"open_dashboard_on_start": True},
    }
    _serialize_config(path, payload)
    logger.info("Created default config at %s", path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autocapture local-first runner")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("tray", "worker", "doctor"),
        default="tray",
        help="Execution mode (default: tray).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config.yml/config.json.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _run_tray(config_path: Path) -> None:
    logger.info("Launching tray mode with config %s", config_path)
    logger.info("Tray mode is not yet implemented; stub runner exiting.")


def _run_worker(config_path: Path) -> None:
    logger.info("Launching worker mode with config %s", config_path)
    logger.info("Worker mode is not yet implemented; stub runner exiting.")


def _run_doctor(config_path: Path) -> None:
    logger.info("Running doctor checks with config %s", config_path)
    logger.info("Doctor mode is not yet implemented; stub runner exiting.")


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level)

    config_path = args.config or _default_config_path()
    config_path = ensure_default_config(config_path)

    if args.mode == "tray":
        _run_tray(config_path)
    elif args.mode == "worker":
        _run_worker(config_path)
    else:
        _run_doctor(config_path)


if __name__ == "__main__":
    main()
