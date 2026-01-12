from __future__ import annotations

import sys
from pathlib import Path

from autocapture.config import FFmpegConfig, QdrantConfig
from autocapture.paths import resource_root, resolve_ffmpeg_path, resolve_qdrant_path


def test_resource_root_uses_meipass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert resource_root() == tmp_path


def test_resolve_ffmpeg_prefers_bundled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    bundled = tmp_path / "ffmpeg" / "bin" / "ffmpeg.exe"
    bundled.parent.mkdir(parents=True, exist_ok=True)
    bundled.write_text("stub", encoding="utf-8")
    config = FFmpegConfig(
        enabled=True,
        require_bundled=False,
        allow_system=False,
        allow_disable=False,
    )
    assert resolve_ffmpeg_path(config) == bundled


def test_resolve_qdrant_missing_returns_none(tmp_path: Path) -> None:
    config = QdrantConfig(binary_path=tmp_path / "missing.exe")
    assert resolve_qdrant_path(config) is None
