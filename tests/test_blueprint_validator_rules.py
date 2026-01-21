from __future__ import annotations

from pathlib import Path

from tools.validate_blueprint import validate_blueprint


def _minimal_blueprint(extra: str = "") -> str:
    return (
        "# 1. System Context & Constraints\n"
        "## Validation_Checklist\n"
        "- [ ] item\n"
        "## Source_Index:\n"
        "- SRC-001: README.md\n"
        "## Coverage_Map:\n"
        "- 1.System_Context_And_Constraints -> SRC-001\n"
        "\n"
        "# 2. Functional Modules & Logic\n"
        "\n"
        "# 3. Architecture Decision Records (ADRs)\n"
        "\n"
        "# 4. Grounding Data (Few-Shot Samples)\n"
        f"{extra}\n"
    )


def test_validator_blocks_work_later_language(tmp_path: Path) -> None:
    content = _minimal_blueprint(extra="Work-later items here.")
    path = tmp_path / "BLUEPRINT.md"
    path.write_text(content, encoding="utf-8")
    errors = validate_blueprint(path)
    assert any("work-later" in error for error in errors)


def test_validator_requires_sources_in_object_blocks(tmp_path: Path) -> None:
    content = _minimal_blueprint(extra="Object_ID: OBJ-001\n- Field: Value\n")
    path = tmp_path / "BLUEPRINT.md"
    path.write_text(content, encoding="utf-8")
    errors = validate_blueprint(path)
    assert any("Missing Sources" in error for error in errors)


def test_validator_requires_three_table_rows(tmp_path: Path) -> None:
    table = "| Input | Output |\n" "| --- | --- |\n" "| A | B |\n"
    content = _minimal_blueprint(extra=table)
    path = tmp_path / "BLUEPRINT.md"
    path.write_text(content, encoding="utf-8")
    errors = validate_blueprint(path)
    assert any("Few-shot table" in error for error in errors)
