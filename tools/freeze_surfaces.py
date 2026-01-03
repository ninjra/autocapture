"""CLI for freezing or unfreezing stability surfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Unable to locate repo root with pyproject.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage frozen surfaces.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze", help="Freeze file surfaces")
    freeze_parser.add_argument("--reason", required=True, help="Reason for freezing")
    freeze_parser.add_argument("paths", nargs="+", help="Relative file paths")

    unfreeze_parser = subparsers.add_parser("unfreeze", help="Unfreeze file surfaces")
    unfreeze_parser.add_argument("--reason", required=True, help="Reason for unfreezing")
    unfreeze_parser.add_argument("paths", nargs="+", help="Relative file paths")

    subparsers.add_parser("verify", help="Verify frozen file hashes")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    sys.path.insert(0, str(repo_root))

    from autocapture.stability.freeze import (  # noqa: PLC0415
        freeze_files,
        unfreeze_files,
        verify_frozen,
    )

    if args.command == "freeze":
        freeze_files(repo_root, args.paths, args.reason)
        print("Frozen surfaces updated.")
        return 0
    if args.command == "unfreeze":
        unfreeze_files(repo_root, args.paths, args.reason)
        print("Frozen surfaces updated.")
        return 0
    if args.command == "verify":
        violations = verify_frozen(repo_root)
        if violations:
            print("Frozen surface violations detected:")
            for violation in violations:
                print(f"- {violation}")
            return 2
        print("All frozen surfaces match.")
        return 0
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
