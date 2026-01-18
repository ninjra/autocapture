"""Index pruning utilities for retention and deletions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..indexing.lexical_index import LexicalIndex
from ..indexing.vector_index import VectorIndex
from ..indexing.spans_v2 import SpansV2Index


@dataclass(frozen=True)
class PruneCounts:
    lexical_deleted: int = 0
    vector_deleted: int = 0
    spans_v2_deleted: int = 0


class IndexPruner:
    def __init__(
        self,
        db: DatabaseManager,
        *,
        vector_index: VectorIndex | None = None,
        spans_index: SpansV2Index | None = None,
    ) -> None:
        self._db = db
        self._lexical = LexicalIndex(db)
        self._vector = vector_index
        self._spans = spans_index
        self._log = get_logger("index.prune")

    def prune_event_ids(self, event_ids: Iterable[str]) -> PruneCounts:
        ids = sorted({str(item) for item in event_ids if item})
        if not ids:
            return PruneCounts()
        lexical_deleted = self._lexical.delete_events(ids)
        vector_deleted = 0
        spans_deleted = 0
        if self._vector is not None:
            vector_deleted = self._vector.delete_event_ids(ids)
        if self._spans is not None:
            spans_deleted = self._spans.delete_event_ids(ids)
        self._log.info(
            "Pruned indexes for {} events (lexical={}, vector={}, spans_v2={})",
            len(ids),
            lexical_deleted,
            vector_deleted,
            spans_deleted,
        )
        return PruneCounts(
            lexical_deleted=lexical_deleted,
            vector_deleted=vector_deleted,
            spans_v2_deleted=spans_deleted,
        )
