"""Citation integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from ..storage.database import DatabaseManager
from ..storage.models import (
    AnswerCitationRecord,
    AnswerRecord,
    CitableSpanRecord,
    FrameRecord,
    ProvenanceLedgerEntryRecord,
)


@dataclass(frozen=True)
class IntegrityResult:
    valid_span_ids: set[str]
    invalid_span_ids: dict[str, str]


def check_citations(db: DatabaseManager, span_ids: list[str]) -> IntegrityResult:
    if not span_ids:
        return IntegrityResult(valid_span_ids=set(), invalid_span_ids={})
    span_ids = list(dict.fromkeys(span_ids))
    with db.session() as session:
        spans = (
            session.execute(
                select(CitableSpanRecord).where(CitableSpanRecord.span_id.in_(span_ids))
            )
            .scalars()
            .all()
        )
        spans_by_id = {span.span_id: span for span in spans}
        frame_ids = [span.frame_id for span in spans]
        frames = (
            session.execute(select(FrameRecord).where(FrameRecord.frame_id.in_(frame_ids)))
            .scalars()
            .all()
        )
        frames_by_id = {frame.frame_id: frame for frame in frames}
        ledger_span_ids = {
            row.payload_json.get("span_id")
            for row in session.execute(
                select(ProvenanceLedgerEntryRecord).where(
                    ProvenanceLedgerEntryRecord.entry_type == "span"
                )
            )
            .scalars()
            .all()
        }
    valid: set[str] = set()
    invalid: dict[str, str] = {}
    for span_id in span_ids:
        span = spans_by_id.get(span_id)
        if span is None or span.tombstoned:
            invalid[span_id] = "missing_span"
            continue
        if span_id not in ledger_span_ids:
            invalid[span_id] = "missing_ledger"
            continue
        frame = frames_by_id.get(span.frame_id)
        if frame is None or not frame.media_path:
            invalid[span_id] = "missing_frame"
            continue
        if not Path(frame.media_path).exists():
            invalid[span_id] = "missing_media"
            continue
        valid.add(span_id)
    return IntegrityResult(valid_span_ids=valid, invalid_span_ids=invalid)


def scan_recent_answers(
    db: DatabaseManager,
    *,
    limit: int = 25,
    update_stale: bool = False,
) -> dict:
    report: dict[str, object] = {"answers_scanned": 0, "answers_with_invalid": 0}
    with db.session() as session:
        answers = (
            session.execute(
                select(AnswerRecord)
                .order_by(AnswerRecord.created_at.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        report["answers_scanned"] = len(answers)
        for answer in answers:
            citations = (
                session.execute(
                    select(AnswerCitationRecord.span_id).where(
                        AnswerCitationRecord.answer_id == answer.answer_id
                    )
                )
                .scalars()
                .all()
            )
            result = check_citations(db, [span_id for span_id in citations if span_id])
            if result.invalid_span_ids:
                report["answers_with_invalid"] = int(report["answers_with_invalid"]) + 1
                if update_stale and not answer.stale:
                    answer.stale = True
    return report
