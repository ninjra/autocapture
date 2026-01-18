from __future__ import annotations

from pathlib import Path

from autocapture.promptops.validate import validate_prompt_templates


def _write_prompt(path: Path, *, name: str, system_prompt: str, raw_template: str) -> None:
    content = (
        "name: "
        + name
        + "\nversion: v1\nsystem_prompt: |\n  "
        + system_prompt
        + "\nraw_template: |\n  "
        + raw_template
        + "\nderived_template: |\n  "
        + raw_template
        + "\ntags: []\n"
    )
    path.write_text(content, encoding="utf-8")


def test_prompt_template_validator_blocks_payload(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_prompt(
        raw_dir / "blocked.yaml",
        name="BLOCKED",
        system_prompt="safe",
        raw_template="{{ __class__ }}",
    )
    ok, errors = validate_prompt_templates(raw_dir=raw_dir, derived_dir=raw_dir)
    assert not ok
    assert errors


def test_prompt_template_validator_accepts_safe(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_prompt(
        raw_dir / "safe.yaml",
        name="SAFE",
        system_prompt="Just a prompt.",
        raw_template="Hello {{ user }}",
    )
    ok, errors = validate_prompt_templates(raw_dir=raw_dir, derived_dir=raw_dir)
    assert ok
    assert not errors
