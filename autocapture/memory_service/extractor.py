"""Stub proposal extractor for Memory Service write hook."""

from __future__ import annotations

from .schemas import MemoryProposal


def extract_proposals(_text: str, *_args, **_kwargs) -> list[MemoryProposal]:
    # Deterministic stub: no proposals.
    return []
