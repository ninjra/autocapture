from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_frozen_surfaces_manifest_matches() -> None:
    if os.environ.get("STRICT_FROZEN_SURFACES") != "1":
        print("Skipping frozen surfaces check; set STRICT_FROZEN_SURFACES=1 to enable.")
        return
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "autocapture" / "stability" / "frozen_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = data.get("frozen", {})
    if not frozen:
        return

    mismatches: list[str] = []
    for rel_path, metadata in frozen.items():
        expected = metadata.get("sha256")
        path = repo_root / rel_path
        if not path.exists() or not path.is_file():
            mismatches.append(f"{rel_path}: expected {expected}, actual missing")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            mismatches.append(f"{rel_path}: expected {expected}, actual {actual}")

    assert not mismatches, "Frozen surface violations:\n" + "\n".join(mismatches)
