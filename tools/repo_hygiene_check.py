"""Repo hygiene guard for tracked artifacts."""

from __future__ import annotations

import argparse
import subprocess
from typing import Iterable


def find_violations(paths: Iterable[str]) -> list[str]:
    violations: set[str] = set()
    for path in paths:
        if not path:
            continue
        normalized = path.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized.startswith(".idea/") or "/.idea/" in normalized:
            violations.add(path)
            continue
        parts = [part for part in normalized.split("/") if part]
        if "src" in parts:
            src_index = parts.index("src")
            tail = parts[src_index + 1 :]
            if "obj" in tail or "bin" in tail:
                violations.add(path)
    return sorted(violations)


def _git_tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for tracked build/IDE artifacts.")
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Optional list of paths to check (defaults to git ls-files).",
    )
    args = parser.parse_args()

    paths = args.paths or _git_tracked_paths()
    violations = find_violations(paths)
    if violations:
        print("Repo hygiene violations detected:")
        for path in violations:
            print(f"- {path}")
        return 2
    print("Repo hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
