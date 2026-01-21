"""Verify native extension checksums against CHECKSUMS.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autocapture.security.extensions import sha256_file


def load_allowlist(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Checksum allowlist not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise ValueError("CHECKSUMS.json must contain a 'files' list")
    return files


def verify_allowlist(path: Path) -> list[str]:
    errors: list[str] = []
    files = load_allowlist(path)
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            errors.append("Invalid entry (not a dict)")
            continue
        file_path = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(file_path, str) or not isinstance(expected, str):
            errors.append(f"Invalid entry fields: {entry}")
            continue
        if file_path in seen:
            errors.append(f"Duplicate entry for {file_path}")
            continue
        seen.add(file_path)
        abs_path = Path(file_path)
        if not abs_path.exists():
            errors.append(f"Missing file: {file_path}")
            continue
        if abs_path.is_dir():
            errors.append(f"Path is a directory: {file_path}")
            continue
        actual = sha256_file(abs_path)
        if actual != expected:
            errors.append(f"Checksum mismatch for {file_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify native extension checksums.")
    parser.add_argument(
        "--path",
        default="CHECKSUMS.json",
        help="Path to CHECKSUMS.json (default: CHECKSUMS.json)",
    )
    args = parser.parse_args()
    try:
        errors = verify_allowlist(Path(args.path))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Checksum verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
