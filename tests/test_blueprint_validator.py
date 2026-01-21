from __future__ import annotations

from pathlib import Path

from tools.validate_blueprint import validate_blueprint


def test_blueprint_validator_passes() -> None:
    blueprint_path = Path(__file__).resolve().parents[1] / "BLUEPRINT.md"
    errors = validate_blueprint(blueprint_path)
    assert errors == []
