from __future__ import annotations

from pathlib import Path

import yaml

from autocapture.setup import plan_setup, run_setup


def test_plan_setup_full_includes_core_changes(tmp_path: Path) -> None:
    config_path = tmp_path / "autocapture.yml"
    config_path.write_text("{}", encoding="utf-8")

    plan = plan_setup(config_path, profile="full")
    change_paths = {change.path for change in plan.changes}

    assert "offline" in change_paths
    assert "database.encryption_enabled" in change_paths
    assert "tracking.encryption_enabled" in change_paths
    assert "ocr.device" in change_paths
    assert "presets.active_preset" in change_paths


def test_run_setup_applies_and_creates_backup(tmp_path: Path) -> None:
    config_path = tmp_path / "autocapture.yml"
    config_path.write_text("offline: true\n", encoding="utf-8")

    code = run_setup(config_path, profile="full", apply=True, json_output=True)
    assert code == 0

    backups = list(tmp_path.glob("autocapture.yml.bak.*"))
    assert backups, "Expected a backup file to be created"

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["offline"] is False
