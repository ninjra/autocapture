from __future__ import annotations

from pathlib import Path
import importlib.util


def _load_guard():
    root = Path(__file__).resolve().parents[1]
    guard_path = root / ".tools" / "memory_guard.py"
    spec = importlib.util.spec_from_file_location("memory_guard", guard_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_entry() -> dict:
    return {
        "ts": "2026-01-15T00:00:00Z",
        "type": "note",
        "scope": "test",
        "key": "memory-guard",
        "value": "redacted",
        "why": "unit test",
        "source": "test",
    }


def test_memory_guard_rejects_secret() -> None:
    guard = _load_guard()
    entry = _base_entry()
    entry["value"] = "sk-test-1234567890"
    errors = guard._validate_entry(entry, allow_pii=False)
    assert any("Forbidden pattern" in err for err in errors)


def test_memory_guard_accepts_redacted() -> None:
    guard = _load_guard()
    entry = _base_entry()
    entry["value"] = "sk-REDACTED"
    errors = guard._validate_entry(entry, allow_pii=False)
    assert errors == []
