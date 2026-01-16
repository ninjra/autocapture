"""Show Codex loop status from .codex files."""

from __future__ import annotations

import argparse
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Unable to locate repo root with pyproject.toml")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tail(text: str, lines: int) -> str:
    if lines <= 0:
        return ""
    chunks = text.splitlines()
    return "\n".join(chunks[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Show Codex status from .codex files.")
    parser.add_argument(
        "--session-tail",
        type=int,
        default=20,
        help="Lines to include from SESSION_LOG.md (0 to skip).",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    codex_dir = repo_root / ".codex"
    state_path = codex_dir / "STATE.md"
    session_path = codex_dir / "SESSION_LOG.md"

    missing = [path for path in [codex_dir, state_path, session_path] if not path.exists()]
    if missing:
        print("Missing required Codex files:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("STATE.md")
    print("========")
    print(_read_text(state_path).rstrip())

    if args.session_tail:
        print("\nSESSION_LOG.md (tail)")
        print("====================")
        print(_tail(_read_text(session_path), args.session_tail))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
