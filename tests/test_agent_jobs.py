from __future__ import annotations

from autocapture.agents.jobs import AgentJobQueue
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import AgentJobRecord


def _db() -> DatabaseManager:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    return DatabaseManager(config.database)


def test_agent_job_enqueue_idempotent() -> None:
    db = _db()
    queue = AgentJobQueue(db)
    job_id1 = queue.enqueue(job_key="enrich:1", job_type="event_enrichment")
    job_id2 = queue.enqueue(job_key="enrich:1", job_type="event_enrichment")
    assert job_id1 == job_id2


def test_agent_job_leasing() -> None:
    db = _db()
    queue = AgentJobQueue(db)
    queue.enqueue(job_key="vision:1", job_type="vision_caption")
    lease = queue.lease_next(worker_id="worker-1", lease_ms=1000)
    assert lease.job is not None
    lease2 = queue.lease_next(worker_id="worker-2", lease_ms=1000)
    assert lease2.job is None


def test_agent_job_retry_to_failed() -> None:
    db = _db()
    queue = AgentJobQueue(db)
    job_id = queue.enqueue(job_key="job:1", job_type="event_enrichment", max_attempts=1)
    queue.lease_next(worker_id="worker-1", lease_ms=1000)
    queue.mark_failed(job_id, "boom")
    with db.session() as session:
        job = session.get(AgentJobRecord, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.last_error == "boom"
