"""Background enrichment scheduler for events and threads."""

from __future__ import annotations

import datetime as dt
import threading

from sqlalchemy import select, update

from ..agents import AGENT_JOB_ENRICH_EVENT, AGENT_JOB_THREAD_SUMMARY, AGENT_JOB_VISION_EXTRACT
from ..agents.jobs import AgentJobQueue
from ..agents.schemas import EventEnrichmentV2, ThreadSummaryV1
from ..config import AppConfig
from ..embeddings.service import EmbeddingService
from ..enrichment.sql_artifacts import extract_sql_artifacts
from ..indexing.vector_index import SpanEmbeddingUpsert, VectorIndex
from ..logging_utils import get_logger
from ..memory.prompts import PromptRegistry
from ..memory.threads import ThreadSegmenter, ThreadStore
from ..observability.metrics import (
    enrichment_backlog,
    enrichment_at_risk,
    enrichment_oldest_age_hours,
    enrichment_jobs_enqueued_total,
    enrichment_failures_total,
)
from ..storage.database import DatabaseManager
from ..storage.models import (
    AgentResultRecord,
    EventEnrichmentRecord,
    EventRecord,
    ThreadSummaryRecord,
)
from ..vision.types import VISION_SCHEMA_VERSION


class EnrichmentScheduler:
    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        queue: AgentJobQueue,
        *,
        embedder: EmbeddingService | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._queue = queue
        self._embedder = embedder or EmbeddingService(config.embed)
        self._vector_index = vector_index or VectorIndex(config, self._embedder.dim)
        self._log = get_logger("enrichment.scheduler")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prompt_registry = PromptRegistry.from_package("autocapture.prompts.derived")
        self._thread_segmenter = ThreadSegmenter(
            max_gap_minutes=config.threads.max_gap_minutes,
            app_similarity=config.threads.app_similarity_threshold,
            title_similarity=config.threads.title_similarity_threshold,
        )
        self._thread_store = ThreadStore(db)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        interval = self._config.enrichment.scan_interval_s
        while not self._stop.wait(interval):
            try:
                self.run_once()
            except Exception as exc:
                enrichment_failures_total.inc()
                self._log.warning("Enrichment scan failed: {}", exc)

    def run_once(self) -> None:
        if not self._config.enrichment.enabled:
            return
        now = dt.datetime.now(dt.timezone.utc)
        window_days = (
            self._config.enrichment.window_days or self._config.retention.screenshot_ttl_days
        )
        window_start = now - dt.timedelta(days=window_days)
        at_risk_cutoff = now - dt.timedelta(hours=self._config.enrichment.at_risk_hours)

        with self._db.session() as session:
            events = (
                session.execute(
                    select(EventRecord)
                    .where(EventRecord.ts_start >= window_start)
                    .order_by(EventRecord.ts_start.desc())
                    .limit(self._config.enrichment.max_events_per_scan)
                )
                .scalars()
                .all()
            )
        if not events:
            enrichment_backlog.set(0)
            enrichment_at_risk.set(0)
            return

        event_ids = [event.event_id for event in events]
        enrichment_map, result_map = _load_enrichment_state(self._db, event_ids)
        backlog = 0
        at_risk = 0
        oldest_missing: float | None = None

        prompt_enrich = _prompt_revision(self._prompt_registry, "EVENT_ENRICHMENT")
        prompt_thread = _prompt_revision(self._prompt_registry, "THREAD_SUMMARY")
        prompt_vision = "vision_extract:v1"
        llm_model_id = _llm_model_id(self._config)
        vision_model_id = _vision_model_id(self._config)

        for event in events:
            tags = event.tags or {}
            vision_ok = (
                isinstance(tags.get("vision_extract"), dict)
                and tags["vision_extract"].get("schema_version") == VISION_SCHEMA_VERSION
            )
            if not vision_ok and event.screenshot_path:
                self._enqueue_job(
                    stage=AGENT_JOB_VISION_EXTRACT,
                    entity_id=event.event_id,
                    schema_version=VISION_SCHEMA_VERSION,
                    prompt_revision=prompt_vision,
                    model_id=vision_model_id,
                    payload={"event_id": event.event_id},
                )
                backlog += 1
                if event.ts_start <= at_risk_cutoff:
                    at_risk += 1
                oldest_missing = _oldest_missing_ts(oldest_missing, event.ts_start, now)

            if (
                event.embedding_status != "done"
                or event.embedding_model != self._config.embed.text_model
            ):
                self._db.transaction(
                    lambda session: session.execute(
                        update(EventRecord)
                        .where(EventRecord.event_id == event.event_id)
                        .values(embedding_status="pending")
                    )
                )
                backlog += 1

            if not _enrichment_ok(
                enrichment_map.get(event.event_id),
                result_map,
                schema_version=EventEnrichmentV2.model_fields["schema_version"].default,
                prompt_revision=prompt_enrich,
                model_id=llm_model_id,
            ):
                self._enqueue_job(
                    stage=AGENT_JOB_ENRICH_EVENT,
                    entity_id=event.event_id,
                    schema_version=EventEnrichmentV2.model_fields["schema_version"].default,
                    prompt_revision=prompt_enrich,
                    model_id=llm_model_id,
                    payload={"event_id": event.event_id},
                )
                backlog += 1
                if event.ts_start <= at_risk_cutoff:
                    at_risk += 1
                oldest_missing = _oldest_missing_ts(oldest_missing, event.ts_start, now)

            sql_artifacts = tags.get("sql_artifacts")
            if not isinstance(sql_artifacts, dict):
                sql_result = extract_sql_artifacts(
                    event.ocr_text or "", tags.get("vision_extract", {}).get("regions", [])
                )
                if sql_result.code_blocks or sql_result.sql_statements:
                    self._merge_event_tags(event.event_id, {"sql_artifacts": sql_result.as_tags()})
                    sql_artifacts = sql_result.as_tags()
            if isinstance(sql_artifacts, dict):
                artifact_text = str(sql_artifacts.get("artifact_text") or "")
                embed_model = sql_artifacts.get("embedding_model")
                if artifact_text and embed_model != self._embedder.model_name:
                    self._index_sql_artifact(event.event_id, artifact_text, sql_artifacts)

        enrichment_backlog.set(backlog)
        enrichment_at_risk.set(at_risk)
        enrichment_oldest_age_hours.set(oldest_missing or 0.0)
        self._refresh_threads(events, prompt_thread, llm_model_id)

    def _enqueue_job(
        self,
        *,
        stage: str,
        entity_id: str,
        schema_version: str,
        prompt_revision: str,
        model_id: str,
        payload: dict,
    ) -> None:
        job_key = _job_key(stage, entity_id, schema_version, prompt_revision, model_id)
        job_id = self._queue.enqueue(
            job_key=job_key,
            job_type=stage,
            event_id=entity_id,
            payload=payload,
            max_pending=self._config.agents.max_pending_jobs,
        )
        if job_id:
            enrichment_jobs_enqueued_total.labels(stage).inc()

    def _index_sql_artifact(self, event_id: str, text: str, tags: dict) -> None:
        try:
            vector = self._embedder.embed_texts([text])[0]
            upsert = SpanEmbeddingUpsert(
                capture_id=event_id,
                span_key="sql_artifact",
                vector=vector,
                payload={"event_id": event_id, "span_key": "sql_artifact", "kind": "sql_artifact"},
                embedding_model=self._embedder.model_name,
            )
            self._vector_index.upsert_spans([upsert])
            tags["embedding_model"] = self._embedder.model_name
            self._merge_event_tags(event_id, {"sql_artifacts": tags})
        except Exception as exc:
            self._log.debug("SQL artifact embedding failed: {}", exc)

    def _merge_event_tags(self, event_id: str, tags: dict) -> None:
        def _update(session) -> None:
            event = session.get(EventRecord, event_id)
            if not event:
                return
            event.tags = _merge_tags(event.tags or {}, tags)

        self._db.transaction(_update)

    def _refresh_threads(
        self, events: list[EventRecord], prompt_revision: str, model_id: str
    ) -> None:
        if not self._config.threads.enabled:
            return
        segments = self._thread_segmenter.segment(events)
        self._thread_store.persist(segments)
        thread_ids = [segment.thread_id for segment in segments]
        if not thread_ids:
            return
        summaries = _load_thread_summaries(self._db, thread_ids)
        for thread_id in thread_ids:
            summary = summaries.get(thread_id)
            if not _thread_summary_ok(
                summary,
                schema_version=ThreadSummaryV1.model_fields["schema_version"].default,
                prompt_revision=prompt_revision,
                model_id=model_id,
            ):
                self._enqueue_job(
                    stage=AGENT_JOB_THREAD_SUMMARY,
                    entity_id=thread_id,
                    schema_version=ThreadSummaryV1.model_fields["schema_version"].default,
                    prompt_revision=prompt_revision,
                    model_id=model_id,
                    payload={"thread_id": thread_id},
                )


def _job_key(
    stage: str, entity_id: str, schema_version: str, prompt_revision: str, model_id: str
) -> str:
    return (
        f"{stage}:{entity_id}:{schema_version}:{_sanitize(prompt_revision)}:{_sanitize(model_id)}"
    )


def _sanitize(value: str, max_len: int = 32) -> str:
    cleaned = "".join(ch for ch in (value or "") if ch.isalnum() or ch in {".", "-", "_"})
    if not cleaned:
        return "none"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len]


def _prompt_revision(registry: PromptRegistry, name: str) -> str:
    try:
        prompt = registry.get(name)
        return f"{prompt.name}:{prompt.version}"
    except Exception:
        return f"{name}:unknown"


def _llm_model_id(config: AppConfig) -> str:
    provider = config.llm.provider
    if provider == "openai":
        model = config.llm.openai_model
    elif provider == "openai_compatible":
        model = config.llm.openai_compatible_model
    else:
        model = config.llm.ollama_model
    return f"{provider}:{model}"


def _vision_model_id(config: AppConfig) -> str:
    backend = config.vision_extract.vlm
    return f"{backend.provider}:{backend.model}"


def _load_enrichment_state(
    db: DatabaseManager, event_ids: list[str]
) -> tuple[dict[str, EventEnrichmentRecord], dict[str, AgentResultRecord]]:
    if not event_ids:
        return {}, {}
    with db.session() as session:
        enrichments = (
            session.execute(
                select(EventEnrichmentRecord).where(EventEnrichmentRecord.event_id.in_(event_ids))
            )
            .scalars()
            .all()
        )
        result_ids = [record.result_id for record in enrichments]
        results = (
            session.execute(select(AgentResultRecord).where(AgentResultRecord.id.in_(result_ids)))
            .scalars()
            .all()
        )
    return {record.event_id: record for record in enrichments}, {
        result.id: result for result in results
    }


def _enrichment_ok(
    record: EventEnrichmentRecord | None,
    results: dict[str, AgentResultRecord],
    *,
    schema_version: str,
    prompt_revision: str,
    model_id: str,
) -> bool:
    if not record or record.schema_version != schema_version:
        return False
    result = results.get(record.result_id)
    if not result:
        return False
    provenance = result.provenance or {}
    if provenance.get("prompt") != prompt_revision:
        return False
    if provenance.get("model") != model_id.split(":", 1)[-1]:
        return False
    return True


def _load_thread_summaries(
    db: DatabaseManager, thread_ids: list[str]
) -> dict[str, ThreadSummaryRecord]:
    if not thread_ids:
        return {}
    with db.session() as session:
        summaries = (
            session.execute(
                select(ThreadSummaryRecord).where(ThreadSummaryRecord.thread_id.in_(thread_ids))
            )
            .scalars()
            .all()
        )
    return {summary.thread_id: summary for summary in summaries}


def _thread_summary_ok(
    summary: ThreadSummaryRecord | None,
    *,
    schema_version: str,
    prompt_revision: str,
    model_id: str,
) -> bool:
    if not summary or summary.schema_version != schema_version:
        return False
    provenance = summary.provenance or {}
    if provenance.get("prompt") != prompt_revision:
        return False
    if provenance.get("model") != model_id.split(":", 1)[-1]:
        return False
    return True


def _merge_tags(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_tags(merged[key], value)
        else:
            merged[key] = value
    return merged


def _oldest_missing_ts(
    oldest_hours: float | None, ts_start: dt.datetime | None, now: dt.datetime
) -> float | None:
    if not ts_start:
        return oldest_hours
    age_hours = max(0.0, (now - ts_start).total_seconds() / 3600.0)
    if oldest_hours is None:
        return age_hours
    return max(oldest_hours, age_hours)
