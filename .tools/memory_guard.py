"""Memory guard for repo-native critical memory artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ALLOWED_TYPES = {
    "constraint",
    "lesson",
    "decision",
    "fact",
    "failure",
    "fix",
    "todo",
    "note",
}
ALLOWED_SOURCES = {"agent", "user", "ci", "test", "manual"}

SCOPE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")

MAX_LEDGER_BYTES = 5 * 1024 * 1024
MAX_STATE_BYTES = 512 * 1024
MAX_DECISIONS_BYTES = 512 * 1024
MAX_APPEND_BATCH = 30

BANNED_PATTERN_STRINGS = [
    r"BEGIN PRIVATE KEY",
    r"BEGIN RSA PRIVATE KEY",
    r"AKIA[0-9A-Z]{16}",
    r"ASIA[0-9A-Z]{16}",
    r"ghp_[A-Za-z0-9]{36,}",
    r"sk-[A-Za-z0-9-]{10,}",
    r"eyJ[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+",
    r"(?i)password\\s*[:=]\\s*\\S+",
    r"(?i)token\\s*[:=]\\s*\\S+",
    r"(?i)secret\\s*[:=]\\s*\\S+",
    r"[A-Za-z0-9+/]{80,}={0,2}",
]
BANNED_PATTERNS = [re.compile(pattern) for pattern in BANNED_PATTERN_STRINGS]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?){2}\d{4}")


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Unable to locate repo root with pyproject.toml")


def _parse_iso(ts: str) -> dt.datetime:
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, val in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(val)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, str):
        yield value


def _is_redacted(value: str) -> bool:
    lowered = value.lower()
    return "redacted" in lowered or "***" in lowered


def scan_forbidden(value: object, *, allow_pii: bool) -> list[str]:
    issues: list[str] = []
    for text in _iter_strings(value):
        if not text or _is_redacted(text):
            continue
        for pattern in BANNED_PATTERNS:
            if pattern.search(text):
                issues.append(f"Forbidden pattern detected: {pattern.pattern}")
                break
        if allow_pii:
            continue
        if EMAIL_RE.search(text):
            issues.append("Email address detected; redact before storing.")
        if PHONE_RE.search(text):
            issues.append("Phone number detected; redact before storing.")
    return issues


def _validate_entry(entry: dict, *, allow_pii: bool) -> list[str]:
    errors: list[str] = []
    for field in ("ts", "type", "scope", "key", "value", "why", "source"):
        if field not in entry:
            errors.append(f"Missing required field: {field}")
    if errors:
        return errors

    if entry["type"] not in ALLOWED_TYPES:
        errors.append(f"Invalid type: {entry['type']!r}")
    if entry["source"] not in ALLOWED_SOURCES:
        errors.append(f"Invalid source: {entry['source']!r}")
    if not isinstance(entry["scope"], str) or not SCOPE_RE.match(entry["scope"]):
        errors.append("Invalid scope format.")
    if not isinstance(entry["key"], str) or not KEY_RE.match(entry["key"]):
        errors.append("Invalid key format.")
    why = entry.get("why")
    if not isinstance(why, str) or len(why) > 200:
        errors.append("Field 'why' must be <= 200 chars.")

    try:
        _parse_iso(entry["ts"])
    except Exception:
        errors.append("Field 'ts' must be ISO8601 with timezone.")

    if entry["type"] == "decision":
        if not isinstance(entry.get("id"), str) or not entry.get("id"):
            errors.append("Decision entries must include a non-empty 'id'.")

    if "tags" in entry:
        tags = entry["tags"]
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            errors.append("Field 'tags' must be a list of strings.")
    if "supersedes" in entry:
        supersedes = entry["supersedes"]
        if not isinstance(supersedes, list) or not all(isinstance(tag, str) for tag in supersedes):
            errors.append("Field 'supersedes' must be a list of strings.")
    if "evidence" in entry and not isinstance(entry["evidence"], str):
        errors.append("Field 'evidence' must be a string.")

    issues = scan_forbidden(entry, allow_pii=allow_pii)
    if issues:
        errors.extend(issues)
    return errors


def _load_ledger(path: Path, *, allow_pii: bool) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError("Ledger missing.")
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ledger line {idx} is not valid JSON: {exc}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"Ledger line {idx} is not a JSON object.")
            errors = _validate_entry(entry, allow_pii=allow_pii)
            if errors:
                joined = "; ".join(errors)
                raise ValueError(f"Ledger line {idx} invalid: {joined}")
            entries.append(entry)
    return entries


def _build_state(entries: list[dict]) -> dict:
    keys: dict[str, dict] = {}
    decisions: list[dict] = []
    last_ts = None
    for entry in entries:
        scope_key = f"{entry['scope']}.{entry['key']}"
        keys[scope_key] = {
            "value": entry["value"],
            "ts": entry["ts"],
            "type": entry["type"],
            "why": entry["why"],
            "source": entry["source"],
        }
        if entry["type"] == "decision":
            title = None
            value = entry.get("value")
            if isinstance(value, dict):
                title = value.get("title")
            if not title:
                title = entry["key"]
            decisions.append({"id": entry.get("id", ""), "title": title})
        last_ts = entry["ts"]
    updated_at = last_ts or dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "version": 1,
        "updated_at": updated_at,
        "keys": {key: keys[key] for key in sorted(keys)},
        "decisions": decisions,
        "banned_patterns": list(BANNED_PATTERN_STRINGS),
    }


def _write_state(path: Path, state: dict) -> None:
    payload = json.dumps(state, sort_keys=True, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")


def _check_sizes(repo_root: Path) -> list[str]:
    errors: list[str] = []
    ledger = repo_root / ".memory" / "LEDGER.ndjson"
    state = repo_root / ".memory" / "STATE.json"
    decisions = repo_root / ".memory" / "DECISIONS.md"
    if ledger.exists() and ledger.stat().st_size > MAX_LEDGER_BYTES:
        errors.append("LEDGER.ndjson exceeds size limit.")
    if state.exists() and state.stat().st_size > MAX_STATE_BYTES:
        errors.append("STATE.json exceeds size limit.")
    if decisions.exists() and decisions.stat().st_size > MAX_DECISIONS_BYTES:
        errors.append("DECISIONS.md exceeds size limit.")
    return errors


def _ensure_memory_files(repo_root: Path) -> list[str]:
    errors: list[str] = []
    required = [
        repo_root / ".memory" / "CANONICAL.md",
        repo_root / ".memory" / "LEDGER.ndjson",
        repo_root / ".memory" / "STATE.json",
        repo_root / ".memory" / "DECISIONS.md",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"Missing memory artifact: {path}")
    return errors


def run_check(repo_root: Path, *, allow_pii: bool) -> int:
    errors = []
    errors.extend(_ensure_memory_files(repo_root))
    errors.extend(_check_sizes(repo_root))
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 2

    ledger_path = repo_root / ".memory" / "LEDGER.ndjson"
    state_path = repo_root / ".memory" / "STATE.json"
    try:
        entries = _load_ledger(ledger_path, allow_pii=allow_pii)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    state_expected = _build_state(entries)
    try:
        state_actual = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: STATE.json invalid: {exc}")
        return 2
    if state_expected != state_actual:
        print("ERROR: STATE.json does not match ledger; run --rebuild-state.")
        return 2
    return 0


def run_rebuild(repo_root: Path, *, allow_pii: bool) -> int:
    ledger_path = repo_root / ".memory" / "LEDGER.ndjson"
    try:
        entries = _load_ledger(ledger_path, allow_pii=allow_pii)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    state_path = repo_root / ".memory" / "STATE.json"
    _write_state(state_path, _build_state(entries))
    print("STATE.json rebuilt.")
    return 0


def run_append(repo_root: Path, *, allow_pii: bool, payload: str | None) -> int:
    if not payload:
        payload = sys.stdin.read()
    if not payload.strip():
        print("ERROR: No entry JSON provided (stdin or --entry).")
        return 2
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON payload: {exc}")
        return 2
    entries: list[dict]
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = [data]
    else:
        print("ERROR: Entry payload must be a JSON object or list.")
        return 2
    if len(entries) > MAX_APPEND_BATCH:
        print(f"ERROR: Append batch exceeds limit ({MAX_APPEND_BATCH}).")
        return 2

    ledger_path = repo_root / ".memory" / "LEDGER.ndjson"
    errors: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("Entry must be a JSON object.")
            continue
        errors.extend(_validate_entry(entry, allow_pii=allow_pii))
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        print("Store a redacted summary instead of sensitive data.")
        return 2

    with ledger_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    return run_rebuild(repo_root, allow_pii=allow_pii)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate repo memory artifacts.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Validate memory artifacts.")
    group.add_argument("--rebuild-state", action="store_true", help="Rebuild STATE.json.")
    group.add_argument("--append", action="store_true", help="Append entry to ledger.")
    parser.add_argument("--entry", help="Entry JSON payload for --append.")
    parser.add_argument(
        "--allow-pii",
        action="store_true",
        help="Allow emails/phones in entries (default: reject).",
    )
    parser.add_argument("--root", help="Repo root override for tests.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(args.root).resolve() if args.root else _find_repo_root(Path.cwd())
    if args.check:
        return run_check(repo_root, allow_pii=args.allow_pii)
    if args.rebuild_state:
        return run_rebuild(repo_root, allow_pii=args.allow_pii)
    if args.append:
        return run_append(repo_root, allow_pii=args.allow_pii, payload=args.entry)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
