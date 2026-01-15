from __future__ import annotations

from pathlib import Path

import pytest

from autocapture.promptops.runner import _load_prompt_specs, _validate_prompt_template


def test_promptops_loads_all_raw_prompts() -> None:
    specs = _load_prompt_specs(Path("prompts/raw"))
    names = {spec.name for spec in specs}
    assert "ANSWER_WITH_CONTEXT_PACK" in names
    assert "QUERY_REFINEMENT" in names


def test_promptops_template_validation_accepts_safe_template() -> None:
    _validate_prompt_template("Just a plain template.", 200, label="raw_template")


def test_promptops_template_validation_rejects_import() -> None:
    with pytest.raises(ValueError, match="forbidden Jinja2 construct"):
        _validate_prompt_template("{% import 'x' as y %}", 200, label="raw_template")


def test_promptops_template_validation_rejects_macro() -> None:
    with pytest.raises(ValueError, match="forbidden Jinja2 construct"):
        _validate_prompt_template("{% macro boom() %}x{% endmacro %}", 200, label="raw_template")


def test_promptops_template_validation_rejects_dunder() -> None:
    with pytest.raises(ValueError, match="dunder"):
        _validate_prompt_template("{{ __class__ }}", 200, label="raw_template")
