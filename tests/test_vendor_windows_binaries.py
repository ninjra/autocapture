from __future__ import annotations

from pathlib import Path

import pytest

from tools.vendor_windows_binaries import _sha256, _verify_sha256


def test_verify_sha256_accepts_matching_hash(tmp_path: Path) -> None:
    archive = tmp_path / "file.zip"
    archive.write_bytes(b"test-payload")
    digest = _sha256(archive)
    sha_file = tmp_path / "file.zip.sha256"
    sha_file.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")

    _verify_sha256(archive, sha_file)


def test_verify_sha256_rejects_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "file.zip"
    archive.write_bytes(b"test-payload")
    sha_file = tmp_path / "file.zip.sha256"
    sha_file.write_text("deadbeef  file.zip\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _verify_sha256(archive, sha_file)
