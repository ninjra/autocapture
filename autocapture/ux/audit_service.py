"""Audit reporting for recent requests/answers."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict


from sqlalchemy import func, select

from .models import (
    AuditAnswerDetail,
    AuditAnswerResponse,
    AuditClaim,
    AuditClaimCitation,
    AuditRequestSummary,
    AuditSummaryResponse,
)
from .redaction import redact_payload
from ..storage.database import DatabaseManager
from ..storage.models import (
    AnswerCitationRecord,
    AnswerClaimCitationRecord,
    AnswerClaimRecord,
    AnswerRecord,
    EvidenceItemRecord,
    ProviderCallRecord,
    RequestRunRecord,
)


class AuditService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def list_requests(self, *, limit: int = 20) -> AuditSummaryResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._db.session() as session:
            requests = (
                session.execute(
                    select(RequestRunRecord)
                    .order_by(RequestRunRecord.started_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        request_ids = [req.request_id for req in requests]
        query_ids = [req.query_id for req in requests if req.query_id]

        evidence_counts: dict[str, int] = {}
        provider_counts: dict[str, int] = {}
        citation_counts: dict[str, int] = {}
        answers_by_query: dict[str, AnswerRecord] = {}

        if request_ids:
            with self._db.session() as session:
                rows = session.execute(
                    select(EvidenceItemRecord.request_id, func.count(EvidenceItemRecord.item_id))
                    .where(EvidenceItemRecord.request_id.in_(request_ids))
                    .group_by(EvidenceItemRecord.request_id)
                ).all()
                evidence_counts = {row[0]: int(row[1] or 0) for row in rows}

        if query_ids:
            with self._db.session() as session:
                rows = session.execute(
                    select(ProviderCallRecord.query_id, func.count(ProviderCallRecord.call_id))
                    .where(ProviderCallRecord.query_id.in_(query_ids))
                    .group_by(ProviderCallRecord.query_id)
                ).all()
                provider_counts = {row[0]: int(row[1] or 0) for row in rows}

                answers = (
                    session.execute(
                        select(AnswerRecord)
                        .where(AnswerRecord.query_id.in_(query_ids))
                        .order_by(AnswerRecord.created_at.desc())
                    )
                    .scalars()
                    .all()
                )
                for answer in answers:
                    if answer.query_id and answer.query_id not in answers_by_query:
                        answers_by_query[answer.query_id] = answer

                answer_ids = [answer.answer_id for answer in answers_by_query.values()]
                if answer_ids:
                    rows = session.execute(
                        select(AnswerCitationRecord.answer_id, func.count(AnswerCitationRecord.id))
                        .where(AnswerCitationRecord.answer_id.in_(answer_ids))
                        .group_by(AnswerCitationRecord.answer_id)
                    ).all()
                    citation_counts = {row[0]: int(row[1] or 0) for row in rows}

        summaries: list[AuditRequestSummary] = []
        for req in requests:
            answer = answers_by_query.get(req.query_id or "") if req.query_id else None
            citations_count = citation_counts.get(answer.answer_id, 0) if answer else 0
            summaries.append(
                AuditRequestSummary(
                    request_id=req.request_id,
                    query_id=req.query_id,
                    query_text=req.query_text,
                    status=req.status,
                    started_at_utc=_to_iso(req.started_at),
                    completed_at_utc=_to_iso(req.completed_at),
                    warnings=redact_payload(req.warnings_json or {}),
                    evidence_count=evidence_counts.get(req.request_id, 0),
                    provider_calls=provider_counts.get(req.query_id, 0) if req.query_id else 0,
                    answer_id=answer.answer_id if answer else None,
                    answer_mode=answer.mode if answer else None,
                    citations_count=citations_count,
                )
            )
        return AuditSummaryResponse(requests=summaries, generated_at_utc=now)

    def answer_detail(self, answer_id: str, *, verbose: bool = False) -> AuditAnswerResponse:
        with self._db.session() as session:
            answer = session.get(AnswerRecord, answer_id)
            if not answer:
                raise ValueError("Answer not found")
            claims = (
                session.execute(
                    select(AnswerClaimRecord)
                    .where(AnswerClaimRecord.answer_id == answer_id)
                    .order_by(AnswerClaimRecord.claim_index.asc())
                )
                .scalars()
                .all()
            )
            claim_ids = [claim.claim_id for claim in claims]
            citations: dict[str, list[AuditClaimCitation]] = defaultdict(list)
            if claim_ids:
                rows = (
                    session.execute(
                        select(AnswerClaimCitationRecord).where(
                            AnswerClaimCitationRecord.claim_id.in_(claim_ids)
                        )
                    )
                    .scalars()
                    .all()
                )
                for row in rows:
                    citations[row.claim_id].append(
                        AuditClaimCitation(
                            evidence_id=row.evidence_id,
                            span_id=row.span_id,
                            line_start=row.line_start,
                            line_end=row.line_end,
                            confidence=row.confidence,
                        )
                    )
            citation_count = session.execute(
                select(func.count(AnswerCitationRecord.id)).where(
                    AnswerCitationRecord.answer_id == answer_id
                )
            ).scalar_one()

        audit_claims = [
            AuditClaim(
                claim_id=claim.claim_id,
                claim_index=claim.claim_index,
                text=str(redact_payload(claim.claim_text)),
                entailment_verdict=claim.entailment_verdict,
                entailment_rationale=(
                    str(redact_payload(claim.entailment_rationale))
                    if claim.entailment_rationale
                    else None
                ),
                citations=citations.get(claim.claim_id, []),
            )
            for claim in claims
        ]
        detail = AuditAnswerDetail(
            answer_id=answer.answer_id,
            query_id=answer.query_id,
            mode=answer.mode,
            created_at_utc=_to_iso(answer.created_at),
            coverage=redact_payload(answer.coverage_json or {}),
            confidence=redact_payload(answer.confidence_json or {}),
            budgets=redact_payload(answer.budgets_json or {}),
            answer_text=str(redact_payload(answer.answer_text)) if verbose else None,
            claims=audit_claims,
            citations_count=int(citation_count or 0),
        )
        return AuditAnswerResponse(
            answer=detail,
            generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )


def _to_iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.isoformat()
