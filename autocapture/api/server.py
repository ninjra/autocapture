"""Local FastAPI server for Personal Activity Memory Engine."""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, model_validator
from sqlalchemy import select

from ..config import AppConfig, ProviderRoutingConfig
from ..logging_utils import get_logger
from ..memory.compression import extractive_answer
from ..memory.context_pack import EvidenceItem, EvidenceSpan, build_context_pack
from ..memory.entities import EntityResolver, SecretStore
from ..memory.prompts import PromptLibraryService, PromptRegistry
from ..memory.retrieval import RetrieveFilters, RetrievalService
from ..memory.router import ProviderRouter
from ..memory.verification import Claim, RulesVerifier
from ..security.oidc import GoogleOIDCVerifier
from ..storage.database import DatabaseManager
from ..storage.models import EventRecord, QueryHistoryRecord
from ..storage.retention import RetentionManager


class RetrieveRequest(BaseModel):
    query: str
    time_range: Optional[tuple[dt.datetime, dt.datetime]] = None
    filters: Optional[dict[str, list[str]]] = None
    k: int = Field(8, ge=1, le=100)
    sanitize: Optional[bool] = None
    extractive_only: Optional[bool] = None
    include_screenshots: bool = False


class RetrieveResponse(BaseModel):
    evidence: list[dict[str, Any]]


class ContextPackRequest(BaseModel):
    query: str
    time_range: Optional[tuple[dt.datetime, dt.datetime]] = None
    filters: Optional[dict[str, list[str]]] = None
    k: int = Field(8, ge=1, le=100)
    sanitize: Optional[bool] = None
    extractive_only: Optional[bool] = None
    pack_format: str = Field("json", description="json or text")


class ContextPackResponse(BaseModel):
    pack: dict[str, Any]
    text: Optional[str] = None


class AnswerRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    query: Optional[str] = None
    q: Optional[str] = None
    routing: Optional[dict[str, str]] = None
    sanitize: Optional[bool] = None
    extractive_only: Optional[bool] = None
    model: Optional[str] = None
    time_range: Optional[tuple[dt.datetime, dt.datetime]] = None
    filters: Optional[dict[str, list[str]]] = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_query(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if not values.get("query") and values.get("q"):
                values["query"] = values["q"]
        return values


class AnswerResponse(BaseModel):
    answer: str
    citations: list[str]
    used_context_pack: dict[str, Any]
    latency_ms: float


class EventResponse(BaseModel):
    event_id: str
    ts_start: dt.datetime
    ts_end: Optional[dt.datetime]
    app_name: str
    window_title: str
    url: Optional[str]
    domain: Optional[str]
    screenshot_path: Optional[str]
    screenshot_hash: str
    ocr_text: str
    ocr_spans: list[dict[str, Any]]
    tags: dict[str, Any]


class SettingsRequest(BaseModel):
    settings: dict[str, Any]


class SettingsResponse(BaseModel):
    status: str


class SuggestRequest(BaseModel):
    q: str


def create_app(
    config: AppConfig,
    db_manager: DatabaseManager | None = None,
) -> FastAPI:
    app = FastAPI(title="Autocapture Memory Engine")
    db = db_manager or DatabaseManager(config.database)
    retrieval = RetrievalService(db, config)
    secret = SecretStore(Path(config.capture.data_dir)).get_or_create()
    entities = EntityResolver(db, secret)
    prompt_registry = PromptRegistry.from_package("autocapture.prompts.derived")
    PromptLibraryService(db).sync_registry(prompt_registry)
    retention = RetentionManager(config.storage, config.retention, db, Path(config.capture.data_dir))
    log = get_logger("api")
    oidc_verifier: GoogleOIDCVerifier | None = None
    if config.mode.mode == "remote":
        oidc_verifier = GoogleOIDCVerifier(
            config.mode.google_oauth_client_id or "",
            config.mode.google_allowed_emails,
        )

    base_dir = Path(__file__).resolve().parents[1]
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_dir = Path(getattr(sys, "_MEIPASS")) / "autocapture"
    ui_dir = base_dir / "ui" / "web"
    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=ui_dir), name="static")

    if config.mode.mode == "remote":
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):  # type: ignore[no-redef]
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or not oidc_verifier:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            token = auth.split(" ", 1)[1]
            try:
                oidc_verifier.verify(token)
            except Exception as exc:
                log.warning("OIDC verification failed: {}", exc)
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            return await call_next(request)

    @app.on_event("startup")
    async def startup() -> None:
        await asyncio.to_thread(retention.enforce_screenshot_ttl)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "mode": config.mode.mode}

    @app.get("/")
    async def index() -> HTMLResponse:
        if not ui_dir.exists():
            return HTMLResponse("<h2>UI not available</h2>")
        return FileResponse(ui_dir / "index.html")

    @app.get("/dashboard")
    async def dashboard_redirect() -> RedirectResponse:
        return RedirectResponse(url="/")

    @app.post("/api/retrieve")
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        _record_query_history(db, request.query)
        evidence, events = _build_evidence(
            retrieval,
            entities,
            request.query,
            request.time_range,
            request.filters,
            request.k,
            sanitized=_resolve_bool(request.sanitize, config.privacy.sanitize_default),
        )
        event_map = {event.event_id: event for event in events}
        return RetrieveResponse(
            evidence=[
                _evidence_to_json(item, event_map.get(item.event_id), request.include_screenshots)
                for item in evidence
            ]
        )

    @app.post("/api/context-pack")
    def context_pack(request: ContextPackRequest) -> ContextPackResponse:
        sanitized = _resolve_bool(request.sanitize, config.privacy.sanitize_default)
        evidence, events = _build_evidence(
            retrieval,
            entities,
            request.query,
            request.time_range,
            request.filters,
            request.k,
            sanitized=sanitized,
        )
        routing_data = _merge_routing(config.routing, request.routing)
        pack = build_context_pack(
            query=request.query,
            evidence=evidence,
            entity_tokens=entities.tokens_for_events(events),
            routing=_model_dump(routing_data),
            filters={
                "time_range": request.time_range,
                "apps": request.filters.get("app") if request.filters else None,
                "domains": request.filters.get("domain") if request.filters else None,
            },
            sanitized=sanitized,
        )
        text_pack = None
        if request.pack_format == "text":
            text_pack = pack.to_text(
                extractive_only=_resolve_bool(
                    request.extractive_only, config.privacy.extractive_only_default
                )
            )
        return ContextPackResponse(pack=pack.to_json(), text=text_pack)

    @app.post("/api/answer")
    async def answer(request: AnswerRequest) -> AnswerResponse:
        sanitized = _resolve_bool(request.sanitize, config.privacy.sanitize_default)
        extractive_only = _resolve_bool(
            request.extractive_only, config.privacy.extractive_only_default
        )
        query_text = request.query or ""
        _record_query_history(db, query_text)
        evidence, events = await asyncio.to_thread(
            _build_evidence,
            retrieval,
            entities,
            query_text,
            request.time_range,
            request.filters,
            12,
            sanitized,
        )
        routing_data = _merge_routing(config.routing, request.routing)
        pack = build_context_pack(
            query=query_text,
            evidence=evidence,
            entity_tokens=entities.tokens_for_events(events),
            routing=_model_dump(routing_data),
            filters={
                "time_range": request.time_range,
                "apps": request.filters.get("app") if request.filters else None,
                "domains": request.filters.get("domain") if request.filters else None,
            },
            sanitized=sanitized,
        )
        start = dt.datetime.now(dt.timezone.utc)
        if extractive_only:
            compressed = extractive_answer(evidence)
            answer_text = compressed.answer
            citations = compressed.citations
        else:
            provider, decision = ProviderRouter(routing_data, config.llm).select_llm()
            system_prompt = prompt_registry.get("ANSWER_WITH_CONTEXT_PACK").system_prompt
            answer_text = await provider.generate_answer(
                system_prompt,
                query_text,
                pack.to_text(extractive_only=False),
            )
            citations = _extract_citations(answer_text)
            if not _valid_citations(citations, evidence):
                retry_prompt = (
                    system_prompt
                    + "\n\nYou must cite evidence IDs in the form [E1], [E2], etc. "
                    "Only cite IDs that appear in the provided context pack."
                )
                answer_text = await provider.generate_answer(
                    retry_prompt,
                    query_text,
                    pack.to_text(extractive_only=False),
                )
                citations = _extract_citations(answer_text)
            if not _valid_citations(citations, evidence):
                compressed = extractive_answer(evidence)
                answer_text = compressed.answer
                citations = compressed.citations
            else:
                verifier = RulesVerifier()
                verifier.verify(
                    [Claim(text=answer_text, evidence_ids=citations, entity_tokens=[])],
                    {item.evidence_id for item in evidence},
                    set(),
                )
            log.info("LLM routed to {}", decision.llm_provider)
        latency = (dt.datetime.now(dt.timezone.utc) - start).total_seconds() * 1000
        return AnswerResponse(
            answer=answer_text,
            citations=citations,
            used_context_pack=pack.to_json(),
            latency_ms=latency,
        )

    @app.post("/api/suggest")
    def suggest(request: SuggestRequest) -> list[dict[str, Any]]:
        query = request.q.strip()
        if not query:
            return []
        normalized = _normalize_query(query)
        with db.session() as session:
            stmt = (
                select(QueryHistoryRecord)
                .where(QueryHistoryRecord.normalized_text.like(f"{normalized}%"))
                .order_by(QueryHistoryRecord.last_used_at.desc())
                .limit(30)
            )
            rows = session.execute(stmt).scalars().all()
        scored = []
        now = dt.datetime.now(dt.timezone.utc)
        for row in rows:
            prefix_boost = 1.0 if row.normalized_text.startswith(normalized) else 0.0
            last_used = _ensure_aware(row.last_used_at)
            age_hours = max((now - last_used).total_seconds() / 3600, 0.0)
            recency = 1 / (1 + age_hours)
            score = prefix_boost * 1.0 + (row.count ** 0.5) * 0.3 + recency * 0.3
            scored.append((score, row.query_text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"snippet": text} for _, text in scored[:8]]

    @app.get("/api/event/{event_id}")
    def event_detail(event_id: str) -> EventResponse:
        with db.session() as session:
            event = session.get(EventRecord, event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        return EventResponse(
            event_id=event.event_id,
            ts_start=event.ts_start,
            ts_end=event.ts_end,
            app_name=event.app_name,
            window_title=event.window_title,
            url=event.url,
            domain=event.domain,
            screenshot_path=event.screenshot_path,
            screenshot_hash=event.screenshot_hash,
            ocr_text=event.ocr_text,
            ocr_spans=event.ocr_spans,
            tags=event.tags,
        )

    @app.post("/api/settings")
    def settings(request: SettingsRequest) -> SettingsResponse:
        settings_path = Path(config.capture.data_dir) / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = settings_path.with_suffix(".tmp")
        tmp_path.write_text(_safe_json(request.settings), encoding="utf-8")
        tmp_path.replace(settings_path)
        return SettingsResponse(status="ok")

    return app


def _resolve_bool(value: Optional[bool], default: bool) -> bool:
    return default if value is None else value


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _merge_routing(base: ProviderRoutingConfig, override: Optional[dict[str, str]]) -> ProviderRoutingConfig:
    data = _model_dump(base)
    if override:
        data.update({k: v for k, v in override.items() if v})
    return ProviderRoutingConfig(**data)


def _safe_json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def _build_evidence(
    retrieval: RetrievalService,
    entities: EntityResolver,
    query: str,
    time_range: Optional[tuple[dt.datetime, dt.datetime]],
    filters: Optional[dict[str, list[str]]],
    k: int,
    sanitized: bool,
) -> tuple[list[EvidenceItem], list[EventRecord]]:
    retrieve_filters = None
    if filters:
        retrieve_filters = RetrieveFilters(
            apps=filters.get("app"), domains=filters.get("domain")
        )
    results = retrieval.retrieve(query, time_range, retrieve_filters, limit=k)
    evidence: list[EvidenceItem] = []
    events: list[EventRecord] = []
    for idx, result in enumerate(results, start=1):
        event = result.event
        events.append(event)
        snippet, snippet_offset = _snippet_for_query(event.ocr_text, query)
        spans = _spans_for_event(
            event.ocr_spans,
            snippet,
            snippet_offset,
            query,
            result.matched_span_keys,
        )
        app_name = event.app_name
        title = event.window_title
        domain = event.domain
        if sanitized:
            snippet, mapping = entities.pseudonymize_text_with_mapping(snippet)
            spans = _remap_spans(spans, mapping, len(snippet))
            app_name = entities.pseudonymize_text(app_name)
            title = entities.pseudonymize_text(title)
            if domain:
                domain = entities.pseudonymize_text(domain)
        evidence.append(
            EvidenceItem(
                evidence_id=f"E{idx}",
                event_id=event.event_id,
                timestamp=event.ts_start.isoformat(),
                app=app_name,
                title=title,
                domain=domain,
                score=result.score,
                spans=spans,
                text=snippet,
            )
        )
    return evidence, events


def _snippet_for_query(text: str, query: str, window: int = 200) -> tuple[str, int]:
    if not text:
        return "", 0
    lower = text.lower()
    q = query.lower()
    idx = lower.find(q)
    if idx == -1:
        return text[: min(400, len(text))], 0
    start = max(idx - window, 0)
    end = min(idx + len(q) + window, len(text))
    return text[start:end], start


def _spans_for_event(
    spans: list[dict],
    snippet: str,
    snippet_offset: int,
    query: str,
    matched_span_keys: list[str],
) -> list[EvidenceSpan]:
    evidence_spans: list[EvidenceSpan] = []
    query_lower = query.lower()
    candidate_spans = spans
    if matched_span_keys:
        candidate_spans = [
            span for span in spans if str(span.get("span_key")) in set(matched_span_keys)
        ]
    elif query_lower:
        candidate_spans = [
            span for span in spans if query_lower in str(span.get("text", "")).lower()
        ]
    for span in candidate_spans:
        start = int(span.get("start", 0)) - snippet_offset
        end = int(span.get("end", 0)) - snippet_offset
        if start < 0 or end > len(snippet) or end <= start:
            continue
        evidence_spans.append(
            EvidenceSpan(
                span_id=str(span.get("span_id", "S?")),
                start=start,
                end=end,
                conf=float(span.get("conf", span.get("confidence", 0.9))),
            )
        )
    if not evidence_spans:
        evidence_spans.append(EvidenceSpan(span_id="S0", start=0, end=len(snippet), conf=0.5))
    return evidence_spans


def _remap_spans(
    spans: list[EvidenceSpan],
    replacements: list[tuple[int, int, int, int]],
    text_len: int,
) -> list[EvidenceSpan]:
    if not replacements:
        return spans
    replacements = sorted(replacements, key=lambda item: item[0])

    def delta_before(idx: int) -> int:
        delta = 0
        for start, end, new_start, new_end in replacements:
            if idx <= start:
                break
            delta += (new_end - new_start) - (end - start)
        return delta

    remapped: list[EvidenceSpan] = []
    for span in spans:
        overlaps = [
            rep
            for rep in replacements
            if span.start < rep[1] and span.end > rep[0]
        ]
        if overlaps:
            new_start = min(rep[2] for rep in overlaps)
            new_end = max(rep[3] for rep in overlaps)
        else:
            new_start = span.start + delta_before(span.start)
            new_end = span.end + delta_before(span.end)
        new_start = max(0, min(new_start, text_len))
        new_end = max(new_start + 1, min(new_end, text_len))
        remapped.append(
            EvidenceSpan(
                span_id=span.span_id,
                start=new_start,
                end=new_end,
                conf=span.conf,
            )
        )
    return remapped


def _extract_citations(answer_text: str) -> list[str]:
    import re

    citations = re.findall(r"E\d+", answer_text or "")
    seen = []
    for cite in citations:
        if cite not in seen:
            seen.append(cite)
    return seen


def _valid_citations(citations: list[str], evidence: list[EvidenceItem]) -> bool:
    if not citations:
        return False
    valid = {item.evidence_id for item in evidence}
    return set(citations).issubset(valid)


def _normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _ensure_aware(timestamp: dt.datetime) -> dt.datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp


def _record_query_history(db: DatabaseManager, query: str) -> None:
    if not query.strip():
        return
    normalized = _normalize_query(query)
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        record = (
            session.execute(
                select(QueryHistoryRecord).where(
                    QueryHistoryRecord.normalized_text == normalized
                )
            )
            .scalars()
            .first()
        )
        if record:
            record.count += 1
            record.last_used_at = now
            record.query_text = query
        else:
            session.add(
                QueryHistoryRecord(
                    query_text=query,
                    normalized_text=normalized,
                    count=1,
                    last_used_at=now,
                )
            )


def _evidence_to_json(
    item: EvidenceItem, event: Optional[EventRecord], include_screenshots: bool
) -> dict[str, Any]:
    payload = {
        "evidence_id": item.evidence_id,
        "event_id": item.event_id,
        "timestamp": item.timestamp,
        "app": item.app,
        "title": item.title,
        "domain": item.domain,
        "score": item.score,
        "spans": [
            {"span_id": span.span_id, "start": span.start, "end": span.end, "conf": span.conf}
            for span in item.spans
        ],
        "text": item.text,
    }
    if include_screenshots and event:
        payload["screenshot_path"] = event.screenshot_path
    return payload
