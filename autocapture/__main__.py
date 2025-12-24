"""Command-line entrypoints for autocapture."""

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


if __name__ == "__main__":
    main()
