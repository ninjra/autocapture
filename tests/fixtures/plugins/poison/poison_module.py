"""Poison plugin module for tests."""

from __future__ import annotations

import os
from pathlib import Path

sentinel = os.environ.get("POISON_SENTINEL")
if sentinel:
    Path(sentinel).write_text("imported", encoding="utf-8")


def create_embedder(_context, **_kwargs):
    return None
