from __future__ import annotations

from autocapture.promptops.runner import PromptSpec, _parse_promptops_response


def test_parse_no_semantic_change_keeps_version() -> None:
    current = PromptSpec(
        name="TEST",
        version="v3",
        system_prompt="system",
        raw_template="raw",
        derived_template="derived",
        tags=["a"],
    )
    response = "name: TEST\nversion: v99\nrationale: keep\n"
    parsed = _parse_promptops_response(response, current)
    assert parsed.version == "v3"
    assert parsed.raw_template == "raw"
    assert parsed.derived_template == "derived"


def test_parse_bumps_version_on_change() -> None:
    current = PromptSpec(
        name="TEST",
        version="v1",
        system_prompt="system",
        raw_template="raw",
        derived_template="derived",
        tags=["a"],
    )
    response = "name: TEST\nsystem_prompt: updated\n"
    parsed = _parse_promptops_response(response, current)
    assert parsed.version == "v2"
    assert parsed.system_prompt == "updated"


def test_parse_defaults_templates_to_current() -> None:
    current = PromptSpec(
        name="TEST",
        version="v5",
        system_prompt="system",
        raw_template="raw",
        derived_template="derived",
        tags=["a"],
    )
    response = "name: TEST\nsystem_prompt: updated\n"
    parsed = _parse_promptops_response(response, current)
    assert parsed.raw_template == "raw"
    assert parsed.derived_template == "derived"
