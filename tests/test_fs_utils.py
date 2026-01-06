from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from autocapture.fs_utils import atomic_publish, safe_replace


def test_safe_replace_handles_exdev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("payload", encoding="utf-8")

    real_replace = os.replace

    def fake_replace(src: str, dst: str) -> None:
        if Path(src) == source:
            raise OSError(errno.EXDEV, "Cross-device link")
        real_replace(src, dst)

    monkeypatch.setattr("autocapture.fs_utils.os.replace", fake_replace)

    safe_replace(source, destination)

    assert destination.read_text(encoding="utf-8") == "payload"
    assert not source.exists()


def test_atomic_publish_does_not_clobber_on_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")

    def failing_writer(_src: Path, dest: Path) -> None:
        dest.write_bytes(b"partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_publish(source, destination, writer=failing_writer, remove_source=False)

    assert destination.read_bytes() == b"old"
