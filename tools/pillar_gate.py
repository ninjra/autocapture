"""Pillar declaration enforcement gate."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _ref_exists(ref: str) -> bool:
    result = _run(["git", "rev-parse", "--verify", "--quiet", ref])
    return result.returncode == 0


def _resolve_base_ref() -> str:
    base_ref = os.environ.get("GITHUB_BASE_REF")
    candidates = []
    if base_ref:
        candidates.append(f"origin/{base_ref}")
        candidates.append(base_ref)
    candidates.extend(["origin/main", "origin/master", "main", "master", "HEAD~1"])
    for ref in candidates:
        if _ref_exists(ref):
            return ref
    return "HEAD~1"


def _changed_files() -> list[str]:
    base = _resolve_base_ref()
    result = _run(["git", "diff", "--name-only", f"{base}...HEAD"])
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    changed = _changed_files()
    if not changed:
        print("No changed files detected; pillar gate skipped.")
        return 0

    def _is_pillar(path: str) -> bool:
        normalized = path.replace("\\", "/")
        return normalized.startswith("docs/pillars/") or normalized == "PILLARS.md"

    def _is_enforcement(path: str) -> bool:
        normalized = path.replace("\\", "/")
        if normalized.startswith("tests/"):
            return True
        if normalized.startswith("config/defaults/"):
            return True
        if normalized.startswith("tools/") and normalized.endswith("_gate.py"):
            return True
        return False

    has_pillar = any(_is_pillar(path) for path in changed)
    has_enforcement = any(_is_enforcement(path) for path in changed)

    if not has_pillar:
        print("Missing pillar declaration: add docs/pillars/*.md or update PILLARS.md.")
        return 2
    if not has_enforcement:
        print("Missing enforcement artifact: add/modify tests or gate/policy configs.")
        return 2
    print("Pillar gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
