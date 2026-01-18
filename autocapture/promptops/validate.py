"""Prompt template validation gate for CI and local checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import (
    SYSTEM_PROMPT_MAX_CHARS,
    TEMPLATE_MAX_CHARS,
    _load_prompt_specs,
    _validate_prompt_template,
)


def _validate_directory(path: Path, *, label: str) -> list[str]:
    errors: list[str] = []
    for spec in _load_prompt_specs(path):
        for field, template, limit in (
            ("system_prompt", spec.system_prompt, SYSTEM_PROMPT_MAX_CHARS),
            ("raw_template", spec.raw_template, TEMPLATE_MAX_CHARS),
            ("derived_template", spec.derived_template, TEMPLATE_MAX_CHARS),
        ):
            try:
                _validate_prompt_template(
                    template,
                    limit,
                    label=f"{label}:{spec.name}:{field}",
                )
            except Exception as exc:
                errors.append(f"{label}:{spec.name}:{field}: {exc}")
    return errors


def validate_prompt_templates(*, raw_dir: Path, derived_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if raw_dir.exists():
        errors.extend(_validate_directory(raw_dir, label="raw"))
    if derived_dir.exists():
        errors.extend(_validate_directory(derived_dir, label="derived"))
    return not errors, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate prompt templates for safety.")
    parser.add_argument(
        "--raw-dir",
        default="prompts/raw",
        help="Directory containing raw prompt YAML files.",
    )
    parser.add_argument(
        "--derived-dir",
        default="autocapture/prompts/derived",
        help="Directory containing derived prompt YAML files.",
    )
    args = parser.parse_args(argv)

    ok, errors = validate_prompt_templates(
        raw_dir=Path(args.raw_dir),
        derived_dir=Path(args.derived_dir),
    )
    if ok:
        return 0
    for err in errors:
        print(err)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
