from __future__ import annotations

import json
from pathlib import Path

from autocapture.stability.freeze import freeze_files, unfreeze_files, verify_frozen


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version": 1, "frozen": {}}\n', encoding="utf-8")


def test_freeze_and_verify_roundtrip(tmp_path: Path) -> None:
    repo_root = tmp_path
    manifest_path = repo_root / "autocapture" / "stability" / "frozen_manifest.json"
    _write_manifest(manifest_path)

    target = repo_root / "sample.txt"
    target.write_text("alpha", encoding="utf-8")

    freeze_files(repo_root, ["sample.txt"], "unit test freeze")
    assert verify_frozen(repo_root) == []

    target.write_text("beta", encoding="utf-8")
    violations = verify_frozen(repo_root)
    assert violations
    assert "expected" in violations[0]
    assert "actual" in violations[0]

    unfreeze_files(repo_root, ["sample.txt"], "unit test unfreeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "sample.txt" not in manifest.get("frozen", {})
