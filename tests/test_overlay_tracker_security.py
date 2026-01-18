from __future__ import annotations

from pathlib import Path

from autocapture.overlay_tracker.schemas import OverlayCollectorEvent, OverlayPersistEvent

BANNED_PATTERNS = [
    "SetWindowsHookEx",
    "WH_KEYBOARD_LL",
    "keyCode",
    "scanCode",
    "typedText",
    "clipboard",
    "keystroke",
]


def _field_names(model) -> set[str]:
    if hasattr(model, "model_fields"):
        return set(model.model_fields.keys())  # type: ignore[attr-defined]
    return set(model.__fields__.keys())  # type: ignore[attr-defined]


def test_overlay_tracker_schema_forbids_key_fields() -> None:
    names = {name.lower() for name in _field_names(OverlayPersistEvent)}
    names.update({name.lower() for name in _field_names(OverlayCollectorEvent)})
    for banned in ["keycode", "scancode", "typedtext", "clipboard", "keystroke"]:
        assert not any(banned in name for name in names)


def test_overlay_tracker_source_forbids_low_level_hooks() -> None:
    root = Path(__file__).resolve().parents[1] / "autocapture" / "overlay_tracker"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in BANNED_PATTERNS:
            assert pattern not in text, f"Forbidden pattern {pattern} in {path}"
