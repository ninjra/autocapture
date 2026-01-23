"""Agent job worker for enrichment, vision captioning, and highlights."""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
import errno
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from ..agents import (
    AGENT_JOB_DAILY_HIGHLIGHTS,
    AGENT_JOB_ENRICH_EVENT,
    AGENT_JOB_THREAD_SUMMARY,
    AGENT_JOB_VISION_CAPTION,
    AGENT_JOB_VISION_EXTRACT,
)
from ..agents.jobs import AgentJobQueue
from ..agents.llm_client import AgentLLMClient
from ..agents.schemas import (
    CodeBlock,
    DailyHighlightsV1,
    EventEnrichmentV2,
    ProvenanceInfo,
    SensitivityInfo,
    SqlStatement,
    TaskItem,
    ThreadCitation,
    ThreadSummaryV1,
    VisionCaptionV1,
)
from ..agents.structured_output import (
    StructuredOutputError,
    extract_json_payload,
    parse_structured_output,
)
from ..config import AppConfig, is_loopback_host
from ..embeddings.service import EmbeddingService
from ..indexing.lexical_index import LexicalIndex
from ..indexing.thread_index import ThreadLexicalIndex
from ..indexing.vector_index import SpanEmbeddingUpsert, VectorIndex
from ..image_utils import ensure_rgb
from ..logging_utils import get_logger
from ..observability.metrics import worker_errors_total
from ..memory.entities import EntityResolver, SecretStore
from ..memory.prompts import PromptRegistry
from ..security.token_vault import TokenVaultStore
from ..storage.database import DatabaseManager
from ..storage.models import (
    AgentJobRecord,
    DailyAggregateRecord,
    DailyHighlightsRecord,
    EmbeddingRecord,
    EventEnrichmentRecord,
    EventRecord,
    OCRSpanRecord,
    ThreadEventRecord,
    ThreadRecord,
    ThreadSummaryRecord,
)
from ..media.store import MediaStore
from ..runtime_governor import RuntimeGovernor
from ..runtime_pause import PauseController, paused_guard
from ..vision.extractors import ScreenExtractorRouter
from ..enrichment.sql_artifacts import extract_sql_artifacts
from ..plugins import PluginManager


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _truncate_text(value: str, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _derive_summary(context: dict) -> tuple[str, str]:
    app_name = _normalize_text(context.get("app_name"))
    window_title = _normalize_text(context.get("window_title"))
    ocr_text = _normalize_text(context.get("ocr_text"))
    if app_name and window_title:
        summary = f"Working in {app_name}: {window_title}"
    elif window_title:
        summary = window_title
    elif app_name:
        summary = f"Working in {app_name}"
    elif ocr_text:
        summary = _truncate_text(ocr_text, 160)
    else:
        summary = "Working on captured screen"
    what = _truncate_text(ocr_text, 320) if ocr_text else summary
    return _truncate_text(summary, 200), what


def _coerce_importance(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.1
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _coerce_string_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [item for item in (_normalize_text(v) for v in value) if item]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


def _coerce_string_value(value: object, default: str = "") -> str:
    if isinstance(value, str):
        return value.strip()
    return default


def _coerce_tasks(value: object) -> list[TaskItem]:
    if not isinstance(value, list):
        return []
    tasks: list[TaskItem] = []
    for item in value:
        if isinstance(item, TaskItem):
            tasks.append(item)
            continue
        if isinstance(item, dict):
            title = _normalize_text(item.get("title"))
            if not title:
                continue
            tasks.append(
                TaskItem(
                    title=title,
                    status=_normalize_text(item.get("status")) or "unknown",
                    evidence=_coerce_string_list(item.get("evidence")),
                )
            )
            continue
        title = _normalize_text(item)
        if title:
            tasks.append(TaskItem(title=title))
    return tasks


def _coerce_code_blocks(value: object) -> list[CodeBlock]:
    if not isinstance(value, list):
        return []
    blocks: list[CodeBlock] = []
    for item in value:
        if isinstance(item, CodeBlock):
            blocks.append(item)
            continue
        if isinstance(item, dict):
            language = _normalize_text(item.get("language")) or "text"
            text = _normalize_text(item.get("text"))
            if text:
                blocks.append(CodeBlock(language=language, text=text))
    return blocks


def _coerce_sql_statements(value: object) -> list[SqlStatement]:
    if not isinstance(value, list):
        return []
    statements: list[SqlStatement] = []
    for item in value:
        if isinstance(item, SqlStatement):
            statements.append(item)
            continue
        if isinstance(item, dict):
            text = _normalize_text(item.get("text"))
            operation = _normalize_text(item.get("operation")) or "other"
            if text:
                statements.append(
                    SqlStatement(
                        text=text,
                        operation=operation,
                        tables=_coerce_string_list(item.get("tables")),
                        parse_error=_normalize_text(item.get("parse_error")) or None,
                    )
                )
    return statements


def _fallback_enrichment_payload(
    raw_text: str,
    context: dict,
    *,
    response_model: str,
    response_provider: str,
    prompt_id: str,
) -> EventEnrichmentV2:
    payload: dict = {}
    try:
        raw_json = extract_json_payload(raw_text)
        payload = json.loads(raw_json)
    except Exception:
        payload = {}

    short_summary_default, what_default = _derive_summary(context)
    app_name = _normalize_text(context.get("app_name"))

    short_summary = _coerce_string_value(payload.get("short_summary"), short_summary_default)
    what_i_was_doing = _coerce_string_value(payload.get("what_i_was_doing"), what_default)
    if not short_summary:
        short_summary = short_summary_default
    if not what_i_was_doing:
        what_i_was_doing = what_default

    apps_and_tools = _coerce_string_list(payload.get("apps_and_tools"))
    if not apps_and_tools and app_name:
        apps_and_tools = [app_name]

    sensitivity = payload.get("sensitivity") if isinstance(payload.get("sensitivity"), dict) else {}
    sensitivity_payload = {
        "contains_pii": bool(sensitivity.get("contains_pii", False)),
        "contains_secrets": bool(sensitivity.get("contains_secrets", False)),
        "notes": _coerce_string_list(sensitivity.get("notes")),
    }

    return EventEnrichmentV2(
        schema_version="v2",
        event_id=_coerce_string_value(payload.get("event_id"), context.get("event_id", "") or ""),
        short_summary=short_summary,
        what_i_was_doing=what_i_was_doing,
        apps_and_tools=apps_and_tools,
        topics=_coerce_string_list(payload.get("topics")),
        tasks=_coerce_tasks(payload.get("tasks")),
        people=_coerce_string_list(payload.get("people")),
        projects=_coerce_string_list(payload.get("projects")),
        next_actions=_coerce_string_list(payload.get("next_actions")),
        importance=_coerce_importance(payload.get("importance")),
        sensitivity=SensitivityInfo.model_validate(sensitivity_payload),
        keywords=_coerce_string_list(payload.get("keywords")),
        code_blocks=_coerce_code_blocks(payload.get("code_blocks")),
        sql_statements=_coerce_sql_statements(payload.get("sql_statements")),
        provenance=_build_provenance(response_model, response_provider, prompt_id),
    )


def _build_provenance(model: str, provider: str, prompt_id: str) -> ProvenanceInfo:
    return ProvenanceInfo(
        model=model or "unknown",
        provider=provider or "unknown",
        prompt=prompt_id,
    )


def _fallback_vision_payload(
    event: EventRecord,
    *,
    response_model: str,
    response_provider: str,
    prompt_id: str,
) -> VisionCaptionV1:
    ocr_text = _normalize_text(event.ocr_text)
    caption = _truncate_text(ocr_text, 160) if ocr_text else "Screenshot captured"
    visible = _truncate_text(ocr_text, 320) if ocr_text else ""
    payload = {
        "schema_version": "v1",
        "event_id": event.event_id,
        "caption": caption,
        "ui_elements": [],
        "visible_text_summary": visible,
        "sensitivity": {"contains_pii": False, "contains_secrets": False, "notes": []},
        "provenance": _build_provenance(response_model, response_provider, prompt_id),
    }
    return VisionCaptionV1.model_validate(payload)


def _fallback_thread_summary_payload(
    thread_id: str,
    title_hint: str,
    events: list[EventRecord],
    *,
    response_model: str,
    response_provider: str,
    prompt_id: str,
) -> ThreadSummaryV1:
    title = _normalize_text(title_hint) or "Thread summary"
    event_titles = [
        _normalize_text(event.window_title) or _normalize_text(event.app_name)
        for event in events
    ]
    summary = _truncate_text(" / ".join([t for t in event_titles if t]), 300) or title
    citations = [
        ThreadCitation(
            event_id=event.event_id,
            ts_start=event.ts_start.isoformat(),
            ts_end=event.ts_end.isoformat() if event.ts_end else None,
        )
        for event in events[:3]
    ]
    payload = {
        "schema_version": "v1",
        "thread_id": thread_id,
        "title": title,
        "summary": summary,
        "key_entities": [],
        "tasks": [],
        "citations": citations,
        "provenance": _build_provenance(response_model, response_provider, prompt_id),
    }
    return ThreadSummaryV1.model_validate(payload)


def _fallback_daily_highlights_payload(
    day: str,
    events: list[EventRecord],
    aggregates: list[DailyAggregateRecord],
    *,
    response_model: str,
    response_provider: str,
    prompt_id: str,
) -> DailyHighlightsV1:
    highlights: list[str] = []
    for event in events[:5]:
        title = _normalize_text(event.window_title) or _normalize_text(event.app_name)
        if title:
            highlights.append(_truncate_text(title, 120))
    summary = _truncate_text(" / ".join(highlights), 320) if highlights else f"Daily summary for {day}"
    time_spent: dict[str, float] = {}
    for agg in aggregates:
        if agg.metric_name == "seconds_active" and agg.metric_value:
            time_spent[str(agg.app_name)] = float(agg.metric_value)
    payload = {
        "schema_version": "v1",
        "day": day,
        "summary": summary,
        "highlights": highlights,
        "projects": [],
        "open_loops": [],
        "people": [],
        "context_switches": [],
        "time_spent_by_app": time_spent,
        "provenance": _build_provenance(response_model, response_provider, prompt_id),
    }
    return DailyHighlightsV1.model_validate(payload)


class AgentJobWorker:
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager | None = None,
        *,
        embedder: EmbeddingService | None = None,
        vector_index: VectorIndex | None = None,
        llm_client: AgentLLMClient | None = None,
        runtime_governor: RuntimeGovernor | None = None,
        pause_controller: PauseController | None = None,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.agents")
        self._queue = AgentJobQueue(self._db)
        self._embedder = embedder or EmbeddingService(
            config.embed, pause_controller=pause_controller
        )
        self._vector_index = vector_index or VectorIndex(config, self._embedder.dim)
        self._runtime = runtime_governor
        self._pause = pause_controller
        self._lexical = LexicalIndex(self._db)
        self._thread_lexical = ThreadLexicalIndex(self._db)
        self._llm = llm_client or AgentLLMClient(config)
        plugins = plugin_manager or PluginManager(config)
        self._prompt_registry = PromptRegistry.from_package(
            "autocapture.prompts.derived",
            hardening_enabled=config.templates.enabled,
            log_provenance=config.templates.log_provenance,
            extra_dirs=plugins.prompt_bundles(),
            allow_external=True,
        )
        self._media_store = MediaStore(config.capture, config.encryption)
        self._screen_extractor = ScreenExtractorRouter(
            config, runtime_governor=runtime_governor, plugin_manager=plugins
        )
        secret = SecretStore(Path(config.capture.data_dir)).get_or_create()
        self._entities = EntityResolver(
            self._db,
            secret,
            token_vault=TokenVaultStore(config, self._db),
        )
        self._lease_timeout_s = config.worker.lease_ms / 1000
        self._max_task_runtime_s = config.worker.max_task_runtime_s
        self._llm_unreachable_until = 0.0
        self._llm_unreachable_log_at = 0.0
        self._llm_unreachable_backoff_s = 300.0
        self._fallback_log_at: dict[str, float] = {}
        self._fallback_log_cooldown_s = 60.0

    def _allow_work(self) -> bool:
        if not self._runtime:
            return True
        if self._runtime.allow_workers():
            return True
        self._log.debug("Agent worker paused by runtime governor")
        return False

    def _llm_backoff_active(self) -> bool:
        if not self._llm_unreachable_until:
            return False
        now = time.monotonic()
        if now >= self._llm_unreachable_until:
            self._llm_unreachable_until = 0.0
            return False
        if now - self._llm_unreachable_log_at > 30.0:
            self._llm_unreachable_log_at = now
            remaining = max(0.0, self._llm_unreachable_until - now)
            self._log.warning(
                "LLM endpoint unavailable; pausing agent jobs for {:.0f}s.",
                remaining,
            )
        return True

    def _register_llm_unreachable(self, exc: Exception) -> bool:
        if not self._is_connection_refused(exc):
            return False
        self._llm_unreachable_until = time.monotonic() + self._llm_unreachable_backoff_s
        self._llm_unreachable_log_at = 0.0
        return True

    @staticmethod
    def _is_connection_refused(exc: Exception) -> bool:
        seen = set()
        current: Exception | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, OSError):
                if getattr(current, "winerror", None) == 10061:
                    return True
                err = getattr(current, "errno", None)
                if err in (errno.ECONNREFUSED, 61, 10061):
                    return True
            message = str(current).lower()
            if "connection refused" in message or "10061" in message:
                return True
            current = current.__cause__ or current.__context__
        return False

    def _log_fallback(self, label: str, exc: Exception) -> None:
        now = time.monotonic()
        last = self._fallback_log_at.get(label, 0.0)
        if now - last >= self._fallback_log_cooldown_s:
            self._fallback_log_at[label] = now
            self._log.warning("{} output invalid; using fallback: {}", label, exc)
        else:
            self._log.debug("{} output invalid; using fallback: {}", label, exc)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        backoff_s = 1.0
        while True:
            if stop_event and stop_event.is_set():
                return
            if paused_guard(self._pause, stop_event):
                return
            if not self._allow_work():
                sleep_ms = self._runtime.qos_budget().sleep_ms if self._runtime else int(1000)
                time.sleep(max(0.01, sleep_ms / 1000.0))
                continue
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
                poll_interval = self._config.worker.poll_interval_s
                if self._runtime:
                    poll_interval = self._runtime.poll_interval_s(poll_interval)
                time.sleep(poll_interval)

    def process_batch(self) -> int:
        if not self._config.agents.enabled:
            return 0
        if paused_guard(self._pause):
            return 0
        if self._llm_backoff_active():
            return 0
        if not self._allow_work():
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
            elif job.job_type == AGENT_JOB_VISION_EXTRACT:
                self._handle_vision_extract(job)
            elif job.job_type == AGENT_JOB_VISION_CAPTION:
                self._handle_vision(job)
            elif job.job_type == AGENT_JOB_THREAD_SUMMARY:
                self._handle_thread_summary(job)
            elif job.job_type == AGENT_JOB_DAILY_HIGHLIGHTS:
                self._handle_highlights(job)
            else:
                self._queue.mark_skipped(job.id, f"Unknown job type {job.job_type}")
        except Exception as exc:
            if self._register_llm_unreachable(exc):
                self._queue.mark_failed(
                    job.id,
                    "LLM endpoint unavailable",
                    retry_after_s=self._llm_unreachable_backoff_s,
                )
            else:
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
        prompt = None
        try:
            prompt = self._prompt_registry.get("EVENT_ENRICHMENT")
        except Exception:
            prompt = None
        system_prompt = (
            prompt.system_prompt
            if prompt
            else "You are a memory assistant. Respond with JSON only."
        )
        prompt_id = f"{prompt.name}:{prompt.version}" if prompt else "event_enrichment:default"
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
            "Generate an enrichment JSON matching EventEnrichmentV2. "
            "Required keys: schema_version, event_id, short_summary, what_i_was_doing, "
            "importance (0-1), provenance. Use empty arrays for optional lists. "
            "Be concise, memory-first, and only use evidence in the input."
        )
        response = self._llm.generate_text(system_prompt, user_prompt, context)

        def _repair(text: str) -> str:
            repair_prompt = (
                "Your last response was invalid JSON. Output only valid JSON matching the schema. "
                f"Invalid output:\n{text}"
            )
            return self._llm.generate_text(system_prompt, repair_prompt, context).text

        response_model = getattr(response, "model", "unknown")
        response_provider = getattr(response, "provider", "unknown")
        try:
            result = parse_structured_output(response.text, EventEnrichmentV2, repair_fn=_repair)
        except Exception as exc:
            self._log_fallback("Enrichment", exc)
            try:
                result = _fallback_enrichment_payload(
                    response.text,
                    context_payload,
                    response_model=response_model,
                    response_provider=response_provider,
                    prompt_id=prompt_id,
                )
            except Exception as fallback_exc:
                self._log.warning("Enrichment fallback failed; using minimal output: {}", fallback_exc)
                short_summary, what_i_was_doing = _derive_summary(context_payload)
                result = EventEnrichmentV2(
                    schema_version="v2",
                    event_id=event.event_id,
                    short_summary=short_summary,
                    what_i_was_doing=what_i_was_doing,
                    importance=0.1,
                    provenance=_build_provenance(response_model, response_provider, prompt_id),
                )
        else:
            fallback = _fallback_enrichment_payload(
                "{}",
                context_payload,
                response_model=response_model,
                response_provider=response_provider,
                prompt_id=prompt_id,
            )
            if not result.short_summary.strip():
                result.short_summary = fallback.short_summary
            if not result.what_i_was_doing.strip():
                result.what_i_was_doing = fallback.what_i_was_doing
            if result.importance is None:
                result.importance = fallback.importance
        if result.event_id != event.event_id:
            result.event_id = event.event_id
        result.provenance.model = response_model or result.provenance.model
        result.provenance.provider = response_provider or result.provenance.provider
        result.provenance.prompt = prompt_id
        sql_artifacts = extract_sql_artifacts(
            event.ocr_text or "",
            (event.tags or {}).get("vision_extract", {}).get("regions", []),
        )
        if sql_artifacts.code_blocks:
            result.code_blocks = sql_artifacts.code_blocks
        if sql_artifacts.sql_statements:
            result.sql_statements = sql_artifacts.sql_statements
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
        if sql_artifacts.code_blocks or sql_artifacts.sql_statements:
            self._merge_event_tags(
                event.event_id,
                {"sql_artifacts": sql_artifacts.as_tags()},
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
                f"enrich:{event.event_id}:v2",
                synthetic_text,
            )
        if sql_artifacts.artifact_text:
            self._index_synthetic_span(
                event.event_id,
                f"sql_artifact:{event.event_id}",
                sql_artifacts.artifact_text,
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
        prompt_id = "vision_caption:default"
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

        response_model = getattr(response, "model", "unknown")
        response_provider = getattr(response, "provider", "unknown")
        try:
            result = parse_structured_output(response.text, VisionCaptionV1, repair_fn=_repair)
        except Exception as exc:
            self._log_fallback("Vision caption", exc)
            result = _fallback_vision_payload(
                event,
                response_model=response_model,
                response_provider=response_provider,
                prompt_id=prompt_id,
            )
        if result.event_id != event.event_id:
            result.event_id = event.event_id
        result.provenance.model = response_model or result.provenance.model
        result.provenance.provider = response_provider or result.provenance.provider
        result.provenance.prompt = prompt_id
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

    def _handle_vision_extract(self, job: AgentJobRecord) -> None:
        if not job.event_id:
            self._queue.mark_skipped(job.id, "Missing event_id")
            return
        with self._db.session() as session:
            event = session.get(EventRecord, job.event_id)
        if not event or not event.screenshot_path:
            self._queue.mark_skipped(job.id, "Missing screenshot")
            return
        path = Path(event.screenshot_path)
        if not path.exists():
            self._queue.mark_skipped(job.id, "Screenshot missing on disk")
            return
        image = ensure_rgb(self._media_store.read_image(path))
        paused_guard(self._pause)
        result = self._screen_extractor.extract(image)
        sql_artifacts = extract_sql_artifacts(
            result.text,
            (result.tags or {}).get("vision_extract", {}).get("regions", []),
        )

        def _write(session) -> None:
            record = session.get(EventRecord, job.event_id)
            if not record:
                return
            if not record.ocr_text:
                record.ocr_text = result.text
            if result.tags:
                record.tags = _merge_tags(record.tags or {}, result.tags)
            if sql_artifacts.code_blocks or sql_artifacts.sql_statements:
                record.tags = _merge_tags(
                    record.tags or {}, {"sql_artifacts": sql_artifacts.as_tags()}
                )
            if result.spans and not _has_spans(session, job.event_id):
                self._upsert_spans(session, job.event_id, result.spans)
                self._upsert_embeddings(session, job.event_id, result.spans)

        self._db.transaction(_write)
        with self._db.session() as session:
            refreshed = session.get(EventRecord, job.event_id)
        if refreshed:
            self._lexical.upsert_event(refreshed)
            if sql_artifacts.artifact_text:
                self._lexical.upsert_agent_text(job.event_id, sql_artifacts.artifact_text)
        self._queue.complete_job(job.id)

    def _handle_thread_summary(self, job: AgentJobRecord) -> None:
        payload = job.payload_json or {}
        thread_id = payload.get("thread_id") or job.event_id
        if not thread_id:
            self._queue.mark_skipped(job.id, "Missing thread_id")
            return
        with self._db.session() as session:
            thread = session.get(ThreadRecord, thread_id)
            if not thread:
                self._queue.mark_skipped(job.id, "Thread missing")
                return
            rows = session.execute(
                select(EventRecord, ThreadEventRecord.position)
                .join(ThreadEventRecord, ThreadEventRecord.event_id == EventRecord.event_id)
                .where(ThreadEventRecord.thread_id == thread_id)
                .order_by(ThreadEventRecord.position.asc())
            ).all()
        events = [row[0] for row in rows]
        if not events:
            self._queue.mark_skipped(job.id, "Thread has no events")
            return
        max_events = self._config.threads.max_events_per_thread
        if len(events) > max_events:
            events = events[-max_events:]
        prompt = None
        try:
            prompt = self._prompt_registry.get("THREAD_SUMMARY")
        except Exception:
            prompt = None
        system_prompt = (
            prompt.system_prompt
            if prompt
            else "You are a memory assistant. Respond with JSON only."
        )
        context_payload = {
            "thread_id": thread_id,
            "title_hint": thread.window_title,
            "events": [
                {
                    "event_id": event.event_id,
                    "ts_start": event.ts_start.isoformat(),
                    "ts_end": event.ts_end.isoformat() if event.ts_end else None,
                    "app_name": event.app_name,
                    "window_title": event.window_title,
                    "ocr_text": (event.ocr_text or "")[:800],
                }
                for event in events
            ],
        }
        if self._sanitize_for_cloud():
            context_payload = self._sanitize_payload(context_payload)
        context = json.dumps(context_payload, ensure_ascii=False)
        user_prompt = (
            "Summarize the thread into ThreadSummaryV1 JSON with citations. "
            "Citations must reference event_id and timestamps from the input."
        )
        response = self._llm.generate_text(system_prompt, user_prompt, context)

        def _repair(text: str) -> str:
            repair_prompt = (
                "Your last response was invalid JSON. Output only valid JSON matching ThreadSummaryV1. "
                f"Invalid output:\n{text}"
            )
            return self._llm.generate_text(system_prompt, repair_prompt, context).text

        response_model = getattr(response, "model", "unknown")
        response_provider = getattr(response, "provider", "unknown")
        prompt_id = f"{prompt.name}:{prompt.version}" if prompt else "thread_summary:default"
        try:
            result = parse_structured_output(response.text, ThreadSummaryV1, repair_fn=_repair)
        except Exception as exc:
            self._log_fallback("Thread summary", exc)
            result = _fallback_thread_summary_payload(
                thread_id,
                thread.window_title or "",
                events,
                response_model=response_model,
                response_provider=response_provider,
                prompt_id=prompt_id,
            )
        if result.thread_id != thread_id:
            result.thread_id = thread_id
        if not result.citations:
            result.citations = [
                ThreadCitation(
                    event_id=event.event_id,
                    ts_start=event.ts_start.isoformat(),
                    ts_end=event.ts_end.isoformat() if event.ts_end else None,
                )
                for event in events[:3]
            ]
        result.provenance.model = response_model or result.provenance.model
        result.provenance.provider = response_provider or result.provenance.provider
        result.provenance.prompt = prompt_id
        result_record = self._queue.insert_result(
            job_id=job.id,
            job_type=job.job_type,
            event_id=None,
            day=None,
            schema_version=result.schema_version,
            output_json=result.model_dump(),
            provenance=result.provenance.model_dump(),
        )
        self._upsert_thread_summary(thread_id, result, result_record.provenance)
        self._queue.complete_job(job.id, result_record)
        self._index_thread_summary(thread_id, result)

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

        response_model = getattr(response, "model", "unknown")
        response_provider = getattr(response, "provider", "unknown")
        prompt_id = "daily_highlights:default"
        try:
            result = parse_structured_output(response.text, DailyHighlightsV1, repair_fn=_repair)
        except Exception as exc:
            self._log_fallback("Daily highlights", exc)
            result = _fallback_daily_highlights_payload(
                day,
                events,
                aggregates,
                response_model=response_model,
                response_provider=response_provider,
                prompt_id=prompt_id,
            )
        if result.day != day:
            result.day = day
        result.provenance.model = response_model or result.provenance.model
        result.provenance.provider = response_provider or result.provenance.provider
        result.provenance.prompt = prompt_id
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

    def _upsert_thread_summary(
        self, thread_id: str, result: ThreadSummaryV1, provenance: dict
    ) -> None:
        def _update(session) -> None:
            existing = session.get(ThreadSummaryRecord, thread_id)
            now = dt.datetime.now(dt.timezone.utc)
            if existing:
                existing.data_json = result.model_dump()
                existing.schema_version = result.schema_version
                existing.updated_at = now
                existing.provenance = provenance
                session.add(existing)
            else:
                session.add(
                    ThreadSummaryRecord(
                        thread_id=thread_id,
                        schema_version=result.schema_version,
                        data_json=result.model_dump(),
                        provenance=provenance,
                        created_at=now,
                        updated_at=now,
                    )
                )

        self._db.transaction(_update)

    def _index_thread_summary(self, thread_id: str, result: ThreadSummaryV1) -> None:
        entities = result.key_entities or []
        tasks = [task.title for task in result.tasks]
        self._thread_lexical.upsert_thread(
            thread_id=thread_id,
            title=result.title,
            summary=result.summary,
            entities=entities,
            tasks=tasks,
        )
        text = " ".join([result.title, result.summary, " ".join(entities), " ".join(tasks)]).strip()
        if not text:
            return
        try:
            paused_guard(self._pause)
            vector = self._embedder.embed_texts([text])[0]
            upsert = SpanEmbeddingUpsert(
                capture_id=thread_id,
                span_key="summary",
                vector=vector,
                payload={"thread_id": thread_id, "span_key": "summary", "kind": "thread"},
                embedding_model=self._embedder.model_name,
            )
            self._vector_index.upsert_spans([upsert])
        except Exception as exc:
            self._log.debug("Thread summary embedding failed: {}", exc)

    def _upsert_spans(self, session, capture_id: str, spans: list[dict]) -> None:
        event = session.get(EventRecord, capture_id)
        frame_hash = getattr(event, "frame_hash", None) if event else None
        if event and not frame_hash:
            frame_hash = event.screenshot_hash
        rows = [
            {
                "capture_id": capture_id,
                "span_key": str(span.get("span_key")),
                "start": int(span.get("start", 0)),
                "end": int(span.get("end", 0)),
                "text": str(span.get("text", "")),
                "confidence": float(span.get("conf", 0.0)),
                "bbox": span.get("bbox", []),
                "engine": str(span.get("engine") or "agent"),
                "frame_hash": frame_hash,
                "schema_version": "v1",
            }
            for span in spans
        ]
        if not rows:
            return
        dialect = session.bind.dialect.name if session.bind else ""
        if dialect == "sqlite":
            stmt = (
                sqlite_insert(OCRSpanRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["capture_id", "span_key"])
            )
            session.execute(stmt)
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = (
                pg_insert(OCRSpanRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["capture_id", "span_key"])
            )
            session.execute(stmt)
        else:
            for row in rows:
                with session.begin_nested():
                    session.add(OCRSpanRecord(**row))
                    try:
                        session.flush()
                    except IntegrityError:
                        session.rollback()

    def _upsert_embeddings(self, session, capture_id: str, spans: list[dict]) -> None:
        event = session.get(EventRecord, capture_id)
        frame_hash = getattr(event, "frame_hash", None) if event else None
        if event and not frame_hash:
            frame_hash = event.screenshot_hash
        rows = [
            {
                "capture_id": capture_id,
                "vector": None,
                "model": self._config.embed.text_model,
                "status": "pending",
                "span_key": str(span.get("span_key")),
                "frame_hash": frame_hash,
            }
            for span in spans
        ]
        if not rows:
            return
        dialect = session.bind.dialect.name if session.bind else ""
        if dialect == "sqlite":
            stmt = (
                sqlite_insert(EmbeddingRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["capture_id", "span_key", "model"])
            )
            session.execute(stmt)
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = (
                pg_insert(EmbeddingRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["capture_id", "span_key", "model"])
            )
            session.execute(stmt)
        else:
            for row in rows:
                with session.begin_nested():
                    session.add(EmbeddingRecord(**row))
                    try:
                        session.flush()
                    except IntegrityError:
                        session.rollback()

    def _index_synthetic_span(self, event_id: str, span_key: str, text: str) -> None:
        try:
            paused_guard(self._pause)
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
        if not self._config.privacy.sanitize_default:
            return False
        provider = self._config.llm.provider
        if provider == "openai":
            return True
        if provider == "openai_compatible":
            base_url = self._config.llm.openai_compatible_base_url or ""
            host = urlparse(base_url).hostname if base_url else ""
            return bool(host and not is_loopback_host(host))
        return False

    def _cloud_allowed(self) -> bool:
        provider = self._config.llm.provider
        if provider == "openai":
            return bool(self._config.privacy.cloud_enabled)
        if provider == "openai_compatible":
            base_url = self._config.llm.openai_compatible_base_url or ""
            host = urlparse(base_url).hostname if base_url else ""
            if host and not is_loopback_host(host):
                return bool(self._config.privacy.cloud_enabled)
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


def _merge_tags(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_tags(merged[key], value)
        else:
            merged[key] = value
    return merged


def _has_spans(session, capture_id: str) -> bool:
    return (
        session.execute(
            select(OCRSpanRecord.id).where(OCRSpanRecord.capture_id == capture_id).limit(1)
        ).scalar_one_or_none()
        is not None
    )
