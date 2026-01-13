"""Agent job queue helpers with leasing and retries."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import AgentJobRecord, AgentResultRecord


TERMINAL_STATUSES = {"completed", "failed", "skipped"}


@dataclass(frozen=True)
class LeaseResult:
    job: AgentJobRecord | None
    lease_expires_at: dt.datetime | None


class AgentJobQueue:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger("agents.jobs")

    def enqueue(
        self,
        *,
        job_key: str,
        job_type: str,
        event_id: str | None = None,
        day: str | None = None,
        payload: dict | None = None,
        scheduled_for: dt.datetime | None = None,
        max_attempts: int = 3,
        max_pending: int | None = None,
    ) -> str:
        payload = payload or {}
        scheduled_for = scheduled_for or dt.datetime.now(dt.timezone.utc)

        def _insert(session) -> str:
            if max_pending is not None:
                pending = session.execute(
                    select(AgentJobRecord.id).where(AgentJobRecord.status == "pending")
                ).all()
                if len(pending) >= max_pending:
                    self._log.warning(
                        "Agent job backlog at limit ({}); skipping enqueue for {}",
                        max_pending,
                        job_key,
                    )
                    existing = session.execute(
                        select(AgentJobRecord.id).where(AgentJobRecord.job_key == job_key)
                    ).scalar_one_or_none()
                    return str(existing) if existing else ""
            job_id = str(uuid.uuid4())
            values = dict(
                id=job_id,
                job_key=job_key,
                job_type=job_type,
                status="pending",
                event_id=event_id,
                day=day,
                payload_json=payload,
                scheduled_for=scheduled_for,
                max_attempts=max_attempts,
                attempts=0,
            )
            dialect = session.bind.dialect.name if session.bind else "sqlite"
            if dialect == "sqlite":
                stmt = sqlite_insert(AgentJobRecord).values(**values)
                stmt = stmt.on_conflict_do_nothing(index_elements=["job_key"])
                session.execute(stmt)
            else:
                existing = session.execute(
                    select(AgentJobRecord.id).where(AgentJobRecord.job_key == job_key)
                ).scalar_one_or_none()
                if not existing:
                    session.add(AgentJobRecord(**values))
            existing = session.execute(
                select(AgentJobRecord.id).where(AgentJobRecord.job_key == job_key)
            ).scalar_one()
            return str(existing)

        return self._db.transaction(_insert)

    def lease_next(
        self,
        *,
        worker_id: str,
        lease_ms: int,
        job_types: Iterable[str] | None = None,
    ) -> LeaseResult:
        now = dt.datetime.now(dt.timezone.utc)
        deadline = now + dt.timedelta(milliseconds=lease_ms)
        job_types = list(job_types or [])

        def _select_candidate(session) -> AgentJobRecord | None:
            stmt = select(AgentJobRecord).where(
                AgentJobRecord.status == "pending",
                AgentJobRecord.scheduled_for <= now,
                AgentJobRecord.attempts < AgentJobRecord.max_attempts,
            )
            if job_types:
                stmt = stmt.where(AgentJobRecord.job_type.in_(job_types))
            stmt = stmt.order_by(
                AgentJobRecord.scheduled_for.asc(), AgentJobRecord.created_at.asc()
            )
            return session.execute(stmt).scalars().first()

        def _lease(session) -> LeaseResult:
            job = _select_candidate(session)
            if not job:
                return LeaseResult(None, None)
            updated = session.execute(
                update(AgentJobRecord)
                .where(
                    AgentJobRecord.id == job.id,
                    AgentJobRecord.status == "pending",
                )
                .values(
                    status="leased",
                    leased_by=worker_id,
                    leased_at=now,
                    lease_expires_at=deadline,
                    attempts=AgentJobRecord.attempts + 1,
                    updated_at=now,
                )
            )
            if updated.rowcount != 1:
                return LeaseResult(None, None)
            job.status = "leased"
            job.lease_expires_at = deadline
            job.leased_by = worker_id
            return LeaseResult(job, deadline)

        return self._db.transaction(_lease)

    def complete_job(self, job_id: str, result: AgentResultRecord | None = None) -> None:
        now = dt.datetime.now(dt.timezone.utc)

        def _update(session) -> None:
            session.execute(
                update(AgentJobRecord)
                .where(AgentJobRecord.id == job_id)
                .values(
                    status="completed",
                    result_id=(result.id if result else None),
                    updated_at=now,
                )
            )

        self._db.transaction(_update)

    def mark_failed(self, job_id: str, error: str, *, retry_after_s: float | None = None) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        retry_at = now + dt.timedelta(seconds=retry_after_s or 0)

        def _update(session) -> None:
            job = session.get(AgentJobRecord, job_id)
            if not job:
                return
            status = "pending" if job.attempts < job.max_attempts else "failed"
            session.execute(
                update(AgentJobRecord)
                .where(AgentJobRecord.id == job_id)
                .values(
                    status=status,
                    last_error=error,
                    scheduled_for=retry_at if status == "pending" else job.scheduled_for,
                    updated_at=now,
                )
            )

        self._db.transaction(_update)

    def mark_skipped(self, job_id: str, reason: str) -> None:
        now = dt.datetime.now(dt.timezone.utc)

        def _update(session) -> None:
            session.execute(
                update(AgentJobRecord)
                .where(AgentJobRecord.id == job_id)
                .values(status="skipped", last_error=reason, updated_at=now)
            )

        self._db.transaction(_update)

    def insert_result(
        self,
        *,
        job_id: str | None,
        job_type: str,
        event_id: str | None,
        day: str | None,
        schema_version: str,
        output_json: dict,
        provenance: dict,
    ) -> AgentResultRecord:
        def _insert(session) -> AgentResultRecord:
            record = AgentResultRecord(
                id=str(uuid.uuid4()),
                job_id=job_id,
                job_type=job_type,
                event_id=event_id,
                day=day,
                schema_version=schema_version,
                output_json=output_json,
                provenance=provenance,
            )
            session.add(record)
            return record

        return self._db.transaction(_insert)
