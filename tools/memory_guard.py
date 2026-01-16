"""Compatibility wrapper for the memory guard CLI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    guard_path = repo_root / ".tools" / "memory_guard.py"
    if len(sys.argv) == 1:
        sys.argv.append("--check")
    runpy.run_path(str(guard_path), run_name="__main__")


if __name__ == "__main__":
    main()
