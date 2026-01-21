"""Guardrails against Elastic-licensed dependencies or code."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


DENYLIST_DEP_PATTERNS = [
    re.compile(r"^elasticsearch$", re.IGNORECASE),
    re.compile(r"^elastic$", re.IGNORECASE),
    re.compile(r"elastic[-_]?license", re.IGNORECASE),
]

DENYLIST_TEXT_PATTERNS = [
    re.compile(r"\bElastic\\s+License\b", re.IGNORECASE),
    re.compile(r"\bServer\\s+Side\\s+Public\\s+License\b", re.IGNORECASE),
    re.compile(r"SSPL(?:\\b|v\\d+)", re.IGNORECASE),
]


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except Exception:
        return None


def _git_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _load_dependency_names(pyproject_path: Path) -> list[str]:
    if not pyproject_path.exists():
        return []
    import tomllib

    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    tool = payload.get("tool", {})
    poetry = tool.get("poetry", {})
    deps = poetry.get("dependencies", {}) if isinstance(poetry, dict) else {}
    groups = poetry.get("group", {}) if isinstance(poetry, dict) else {}
    names = [name for name in deps.keys() if name != "python"]
    if isinstance(groups, dict):
        for group in groups.values():
            if isinstance(group, dict):
                group_deps = group.get("dependencies", {})
                if isinstance(group_deps, dict):
                    names.extend(group_deps.keys())
    return sorted(set(names))


def _scan_dependencies(pyproject_path: Path, lock_path: Path) -> list[str]:
    violations: list[str] = []
    dep_names = _load_dependency_names(pyproject_path)
    for name in dep_names:
        for pattern in DENYLIST_DEP_PATTERNS:
            if pattern.search(name):
                violations.append(f"dependency:{name}")
                break
    if lock_path.exists():
        text = lock_path.read_text(encoding="utf-8", errors="ignore")
        for pattern in DENYLIST_TEXT_PATTERNS:
            if pattern.search(text):
                violations.append(f"lockfile:{pattern.pattern}")
                break
    return violations


def _scan_files(files: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in files:
        text = _read_text(path)
        if text is None:
            continue
        for pattern in DENYLIST_TEXT_PATTERNS:
            if pattern.search(text):
                violations.append(f"file:{path}:{pattern.pattern}")
                break
    return violations


def _load_adr_exceptions() -> list[str]:
    candidates = []
    for base in (Path("docs/adr"),):
        if base.exists():
            candidates.extend(base.rglob("*.md"))
    candidates.append(Path(".memory/DECISIONS.md"))
    exceptions: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        text = _read_text(path)
        if not text:
            continue
        for line in text.splitlines():
            if "ELASTIC_EXCEPTION" in line:
                exceptions.append(line.strip())
    return exceptions


def _has_exception(violation: str, exceptions: list[str]) -> bool:
    token = violation.split(":", 1)[-1]
    for line in exceptions:
        if token in line:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for Elastic-licensed dependencies/code.")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--lockfile", default="poetry.lock")
    args = parser.parse_args()

    violations = []
    violations.extend(_scan_dependencies(Path(args.pyproject), Path(args.lockfile)))
    violations.extend(_scan_files(_git_files()))

    if not violations:
        print("License guardrails passed.")
        return 0

    exceptions = _load_adr_exceptions()
    unapproved = [v for v in violations if not _has_exception(v, exceptions)]
    if unapproved:
        for violation in unapproved:
            print(f"ERROR: Elastic-licensed content detected: {violation}", file=sys.stderr)
        print(
            "ERROR: Add an ADR with ELASTIC_EXCEPTION referencing the violation to proceed.",
            file=sys.stderr,
        )
        return 2
    print("License guardrails passed with ADR exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
