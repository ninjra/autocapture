from __future__ import annotations

import sys
from pathlib import Path

from autocapture.paths import resource_root


def test_resource_root_uses_meipass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert resource_root() == tmp_path
