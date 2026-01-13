"""Agent job worker for enrichment, vision captioning, and highlights."""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path

from sqlalchemy import select, update

from ..agents import (
    AGENT_JOB_DAILY_HIGHLIGHTS,
    AGENT_JOB_ENRICH_EVENT,
    AGENT_JOB_VISION_CAPTION,
)
from ..agents.jobs import AgentJobQueue
from ..agents.llm_client import AgentLLMClient
from ..agents.schemas import DailyHighlightsV1, EventEnrichmentV1, VisionCaptionV1
from ..agents.structured_output import parse_structured_output
from ..config import AppConfig
from ..embeddings.service import EmbeddingService
from ..indexing.lexical_index import LexicalIndex
from ..indexing.vector_index import SpanEmbeddingUpsert, VectorIndex
from ..logging_utils import get_logger
from ..observability.metrics import worker_errors_total
from ..memory.entities import EntityResolver, SecretStore
from ..security.token_vault import TokenVaultStore
from ..storage.database import DatabaseManager
from ..storage.models import (
    AgentJobRecord,
    DailyAggregateRecord,
    DailyHighlightsRecord,
    EventEnrichmentRecord,
    EventRecord,
)


class AgentJobWorker:
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager | None = None,
        *,
        embedder: EmbeddingService | None = None,
        vector_index: VectorIndex | None = None,
        llm_client: AgentLLMClient | None = None,
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.agents")
        self._queue = AgentJobQueue(self._db)
        self._embedder = embedder or EmbeddingService(config.embed)
        self._vector_index = vector_index or VectorIndex(config, self._embedder.dim)
        self._lexical = LexicalIndex(self._db)
        self._llm = llm_client or AgentLLMClient(config)
        secret = SecretStore(Path(config.capture.data_dir)).get_or_create()
        self._entities = EntityResolver(
            self._db,
            secret,
            token_vault=TokenVaultStore(config, self._db),
        )
        self._lease_timeout_s = config.worker.lease_ms / 1000
        self._max_task_runtime_s = config.worker.max_task_runtime_s

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        poll_interval = self._config.worker.poll_interval_s
        backoff_s = 1.0
        while True:
            if stop_event and stop_event.is_set():
                return
            try:
                processed = self.process_batch()
                backoff_s = 1.0
            except Exception as exc:
                self._log.exception("Agent worker loop failed: {}", exc)
                worker_errors_total.labels("agents").inc()
                if stop_event and stop_event.is_set():
                    return
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30.0)
                continue
            if processed == 0:
                time.sleep(poll_interval)

    def process_batch(self) -> int:
        if not self._config.agents.enabled:
            return 0
        self._recover_stale_leases()
        worker_id = "agent-worker"
        lease = self._queue.lease_next(
            worker_id=worker_id,
            lease_ms=int(self._lease_timeout_s * 1000),
        )
        job = lease.job
        if not job:
            return 0
        try:
            if job.job_type == AGENT_JOB_ENRICH_EVENT:
                self._handle_enrichment(job)
            elif job.job_type == AGENT_JOB_VISION_CAPTION:
                self._handle_vision(job)
            elif job.job_type == AGENT_JOB_DAILY_HIGHLIGHTS:
                self._handle_highlights(job)
            else:
                self._queue.mark_skipped(job.id, f"Unknown job type {job.job_type}")
        except Exception as exc:
            self._log.warning("Agent job failed: {}", exc)
            worker_errors_total.labels("agents").inc()
            self._queue.mark_failed(job.id, str(exc), retry_after_s=5.0)
        return 1

    def _recover_stale_leases(self) -> None:
        now = dt.datetime.now(dt.timezone.utc)

        def _recover(session) -> None:
            session.execute(
                update(AgentJobRecord)
                .where(
                    AgentJobRecord.status == "leased",
                    AgentJobRecord.lease_expires_at.is_not(None),
                    AgentJobRecord.lease_expires_at < now,
                )
                .values(status="pending", leased_by=None, leased_at=None, lease_expires_at=None)
            )

        self._db.transaction(_recover)

    def _handle_enrichment(self, job: AgentJobRecord) -> None:
        if not self._cloud_allowed():
            self._queue.mark_skipped(job.id, "Cloud agents disabled")
            return
        if not job.event_id:
            self._queue.mark_skipped(job.id, "Missing event_id")
            return
        with self._db.session() as session:
            event = session.get(EventRecord, job.event_id)
        if not event or not event.ocr_text:
            self._queue.mark_skipped(job.id, "Event missing OCR text")
            return
        system_prompt = "You are a memory assistant. Respond with JSON only."
        context_payload = {
            "event_id": event.event_id,
            "app_name": event.app_name,
            "window_title": event.window_title,
            "url": event.url,
            "domain": event.domain,
            "ocr_text": event.ocr_text,
        }
        if self._sanitize_for_cloud():
            context_payload = self._sanitize_payload(context_payload)
        context = json.dumps(
            {
                **context_payload,
            },
            ensure_ascii=False,
        )
        user_prompt = (
            "Generate an enrichment JSON with keys matching EventEnrichmentV1. "
            "Be concise, memory-first, and only use evidence in the input."
        )
        response = self._llm.generate_text(system_prompt, user_prompt, context)

        def _repair(text: str) -> str:
            repair_prompt = (
                "Your last response was invalid JSON. Output only valid JSON matching the schema. "
                f"Invalid output:\n{text}"
            )
            return self._llm.generate_text(system_prompt, repair_prompt, context).text

        result = parse_structured_output(response.text, EventEnrichmentV1, repair_fn=_repair)
        if result.event_id != event.event_id:
            result.event_id = event.event_id
        record = self._queue.insert_result(
            job_id=job.id,
            job_type=job.job_type,
            event_id=event.event_id,
            day=None,
            schema_version=result.schema_version,
            output_json=result.model_dump(),
            provenance=result.provenance.model_dump(),
        )
        self._queue.complete_job(job.id, record)
        self._upsert_event_enrichment(event.event_id, record.id, result.schema_version)
        self._merge_event_tags(
            event.event_id,
            {
                "agents": {
                    "enrichment": {
                        result.schema_version: {
                            "short_summary": result.short_summary,
                            "topics": result.topics,
                            "tasks": [
                                {"title": item.title, "status": item.status}
                                for item in result.tasks
                            ],
                            "importance": result.importance,
                        }
                    }
                }
            },
        )
        synthetic_text = " ".join(
            [
                result.short_summary,
                " ".join(result.keywords),
                " ".join(t.title for t in result.tasks),
            ]
        ).strip()
        if synthetic_text:
            self._index_synthetic_span(
                event.event_id,
                f"enrich:{event.event_id}:v1",
                synthetic_text,
            )

    def _handle_vision(self, job: AgentJobRecord) -> None:
        if not self._vision_allowed():
            self._queue.mark_failed(job.id, "vision_throttled", retry_after_s=3600)
            return
        if not job.event_id:
            self._queue.mark_skipped(job.id, "Missing event_id")
            return
        with self._db.session() as session:
            event = session.get(EventRecord, job.event_id)
        if not event or not event.screenshot_path:
            self._queue.mark_skipped(job.id, "Screenshot missing")
            return
        image_path = event.screenshot_path
        try:
            with open(image_path, "rb") as handle:
                image_bytes = handle.read()
        except FileNotFoundError:
            self._queue.mark_skipped(job.id, "Screenshot missing")
            return
        if self._config.agents.vision.provider != "ollama":
            if (
                not self._config.privacy.cloud_enabled
                or not self._config.privacy.allow_cloud_images
            ):
                self._queue.mark_skipped(job.id, "Cloud images not permitted")
                return
        system_prompt = "You are a memory assistant. Respond with JSON only."
        user_prompt = (
            "Summarize the screenshot with a short caption and UI elements per VisionCaptionV1."
        )
        response = self._llm.generate_vision(system_prompt, user_prompt, image_bytes)

        def _repair(text: str) -> str:
            repair_prompt = (
                "Your last response was invalid JSON. Output only valid JSON matching the schema. "
                f"Invalid output:\n{text}"
            )
            return self._llm.generate_text(system_prompt, repair_prompt, "").text

        result = parse_structured_output(response.text, VisionCaptionV1, repair_fn=_repair)
        if result.event_id != event.event_id:
            result.event_id = event.event_id
        record = self._queue.insert_result(
            job_id=job.id,
            job_type=job.job_type,
            event_id=event.event_id,
            day=None,
            schema_version=result.schema_version,
            output_json=result.model_dump(),
            provenance=result.provenance.model_dump(),
        )
        self._queue.complete_job(job.id, record)
        self._merge_event_tags(
            event.event_id,
            {
                "agents": {
                    "vision_caption": {
                        result.schema_version: {
                            "caption": result.caption,
                            "ui_elements": result.ui_elements,
                        }
                    }
                }
            },
        )
        synthetic_text = " ".join(
            [result.caption, " ".join(result.ui_elements), result.visible_text_summary]
        )
        if synthetic_text:
            self._index_synthetic_span(
                event.event_id,
                f"vision:{event.event_id}:v1",
                synthetic_text,
            )

    def _vision_allowed(self) -> bool:
        config = self._config.agents.vision
        now = dt.datetime.now(dt.timezone.utc)
        if config.run_only_when_idle:
            hour = now.hour
            start = config.idle_hours_start
            end = config.idle_hours_end
            in_window = start <= hour < end if start < end else hour >= start or hour < end
            if not in_window:
                return False
        if config.max_jobs_per_hour <= 0:
            return True
        cutoff = now - dt.timedelta(hours=1)
        with self._db.session() as session:
            recent = (
                session.execute(
                    select(AgentJobRecord.id).where(
                        AgentJobRecord.job_type == AGENT_JOB_VISION_CAPTION,
                        AgentJobRecord.status == "completed",
                        AgentJobRecord.updated_at >= cutoff,
                    )
                )
                .scalars()
                .all()
            )
        return len(recent) < config.max_jobs_per_hour

    def _handle_highlights(self, job: AgentJobRecord) -> None:
        if not self._cloud_allowed():
            self._queue.mark_skipped(job.id, "Cloud agents disabled")
            return
        day = job.day or job.payload_json.get("day")
        if not day:
            self._queue.mark_skipped(job.id, "Missing day")
            return
        start = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(days=1)
        with self._db.session() as session:
            events = (
                session.execute(
                    select(EventRecord)
                    .where(EventRecord.ts_start >= start, EventRecord.ts_start < end)
                    .order_by(EventRecord.ts_start.asc())
                )
                .scalars()
                .all()
            )
            aggregates = (
                session.execute(select(DailyAggregateRecord).where(DailyAggregateRecord.day == day))
                .scalars()
                .all()
            )
        event_payload = [
            {
                "event_id": event.event_id,
                "ts_start": event.ts_start.isoformat(),
                "app_name": event.app_name,
                "window_title": event.window_title,
                "url": event.url,
                "tags": event.tags or {},
            }
            for event in events
        ]
        aggregate_payload = [
            {
                "app_name": agg.app_name,
                "domain": agg.domain,
                "metric_name": agg.metric_name,
                "metric_value": agg.metric_value,
            }
            for agg in aggregates
        ]
        context_payload = {"day": day, "events": event_payload, "aggregates": aggregate_payload}
        if self._sanitize_for_cloud():
            context_payload = self._sanitize_payload(context_payload)
        context = json.dumps(context_payload, ensure_ascii=False)
        system_prompt = "You are a memory assistant. Respond with JSON only."
        user_prompt = (
            "Create daily highlights with summary, highlights, projects, open loops, people, "
            "context switches, and time spent by app per DailyHighlightsV1."
        )
        response = self._llm.generate_text(system_prompt, user_prompt, context)

        def _repair(text: str) -> str:
            repair_prompt = (
                "Your last response was invalid JSON. Output only valid JSON matching the schema. "
                f"Invalid output:\n{text}"
            )
            return self._llm.generate_text(system_prompt, repair_prompt, context).text

        result = parse_structured_output(response.text, DailyHighlightsV1, repair_fn=_repair)
        if result.day != day:
            result.day = day
        record = self._queue.insert_result(
            job_id=job.id,
            job_type=job.job_type,
            event_id=None,
            day=day,
            schema_version=result.schema_version,
            output_json=result.model_dump(),
            provenance=result.provenance.model_dump(),
        )
        self._queue.complete_job(job.id, record)
        self._upsert_daily_highlights(day, result, record.provenance)

    def _merge_event_tags(self, event_id: str, extra: dict) -> None:
        def _merge(a: dict, b: dict) -> dict:
            merged = dict(a)
            for key, value in b.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = _merge(merged[key], value)
                else:
                    merged[key] = value
            return merged

        def _update(session) -> None:
            event = session.get(EventRecord, event_id)
            if not event:
                return
            tags = event.tags or {}
            event.tags = _merge(tags, extra)
            session.add(event)

        self._db.transaction(_update)

    def _upsert_event_enrichment(self, event_id: str, result_id: str, schema_version: str) -> None:
        def _update(session) -> None:
            existing = session.execute(
                select(EventEnrichmentRecord).where(EventEnrichmentRecord.event_id == event_id)
            ).scalar_one_or_none()
            now = dt.datetime.now(dt.timezone.utc)
            if existing:
                existing.result_id = result_id
                existing.schema_version = schema_version
                existing.created_at = now
                session.add(existing)
            else:
                session.add(
                    EventEnrichmentRecord(
                        event_id=event_id,
                        result_id=result_id,
                        schema_version=schema_version,
                        created_at=now,
                    )
                )

        self._db.transaction(_update)

    def _upsert_daily_highlights(
        self, day: str, result: DailyHighlightsV1, provenance: dict
    ) -> None:
        def _update(session) -> None:
            existing = session.execute(
                select(DailyHighlightsRecord).where(DailyHighlightsRecord.day == day)
            ).scalar_one_or_none()
            now = dt.datetime.now(dt.timezone.utc)
            if existing:
                existing.data_json = result.model_dump()
                existing.schema_version = result.schema_version
                existing.updated_at = now
                existing.provenance = provenance
                session.add(existing)
            else:
                session.add(
                    DailyHighlightsRecord(
                        day=day,
                        schema_version=result.schema_version,
                        data_json=result.model_dump(),
                        provenance=provenance,
                        created_at=now,
                        updated_at=now,
                    )
                )

        self._db.transaction(_update)

    def _index_synthetic_span(self, event_id: str, span_key: str, text: str) -> None:
        try:
            vector = self._embedder.embed_texts([text])[0]
            upsert = SpanEmbeddingUpsert(
                capture_id=event_id,
                span_key=span_key,
                vector=vector,
                payload={"event_id": event_id, "span_key": span_key},
                embedding_model=self._embedder.model_name,
            )
            self._vector_index.upsert_spans([upsert])
        except Exception as exc:
            self._log.debug("Synthetic span embedding failed: {}", exc)
        try:
            self._lexical.upsert_agent_text(event_id, text)
        except Exception as exc:
            self._log.debug("Synthetic span lexical index failed: {}", exc)

    def _sanitize_for_cloud(self) -> bool:
        return self._config.llm.provider == "openai" and self._config.privacy.sanitize_default

    def _cloud_allowed(self) -> bool:
        if self._config.llm.provider == "openai" and not self._config.privacy.cloud_enabled:
            return False
        return True

    def _sanitize_payload(self, payload: dict) -> dict:
        def _sanitize(value):
            if isinstance(value, str):
                return self._entities.pseudonymize_text(value)
            if isinstance(value, list):
                return [_sanitize(item) for item in value]
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            return value

        return {key: _sanitize(value) for key, value in payload.items()}
