"""Local FastAPI server for Personal Activity Memory Engine."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import AppConfig, ProviderRoutingConfig
from ..logging_utils import get_logger
from ..memory.compression import extractive_answer
from ..memory.context_pack import EvidenceItem, EvidenceSpan, build_context_pack
from ..memory.entities import EntityResolver, SecretStore
from ..memory.prompts import PromptLibraryService, PromptRegistry
from ..memory.retrieval import RetrieveFilters, RetrievalService
from ..memory.router import ProviderRouter
from ..security.oidc import GoogleOIDCVerifier
from ..storage.database import DatabaseManager
from ..storage.models import EventRecord
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
    query: str
    routing: Optional[dict[str, str]] = None
    sanitize: Optional[bool] = None
    extractive_only: Optional[bool] = None
    model: Optional[str] = None
    time_range: Optional[tuple[dt.datetime, dt.datetime]] = None
    filters: Optional[dict[str, list[str]]] = None


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


def create_app(
    config: AppConfig,
    db_manager: DatabaseManager | None = None,
) -> FastAPI:
    app = FastAPI(title="Autocapture Memory Engine")
    db = db_manager or DatabaseManager(config.database)
    retrieval = RetrievalService(db)
    secret = SecretStore(Path(config.capture.data_dir)).get_or_create()
    entities = EntityResolver(db, secret)
    prompt_registry = PromptRegistry(Path(__file__).resolve().parents[2] / "prompts" / "derived")
    PromptLibraryService(db).sync_registry(prompt_registry)
    retention = RetentionManager(config.storage, config.retention, db, Path(config.capture.data_dir))
    log = get_logger("api")
    oidc_verifier: GoogleOIDCVerifier | None = None
    if config.mode.mode == "remote":
        oidc_verifier = GoogleOIDCVerifier(
            config.mode.google_oauth_client_id or "",
            config.mode.google_allowed_emails,
        )

    ui_dir = Path(__file__).resolve().parents[1] / "ui" / "web"
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
                log.warning("OIDC verification failed: %s", exc)
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

    @app.post("/api/retrieve")
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
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
        evidence, events = await asyncio.to_thread(
            _build_evidence,
            retrieval,
            entities,
            request.query,
            request.time_range,
            request.filters,
            12,
            sanitized,
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
                request.query,
                pack.to_text(extractive_only=False),
            )
            citations = [item.evidence_id for item in evidence]
            log.info("LLM routed to %s", decision.llm_provider)
        latency = (dt.datetime.now(dt.timezone.utc) - start).total_seconds() * 1000
        return AnswerResponse(
            answer=answer_text,
            citations=citations,
            used_context_pack=pack.to_json(),
            latency_ms=latency,
        )

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
        snippet = _snippet_for_query(event.ocr_text, query)
        spans = _spans_for_event(event.ocr_spans, snippet)
        app_name = event.app_name
        title = event.window_title
        domain = event.domain
        if sanitized:
            snippet = entities.pseudonymize_text(snippet)
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


def _snippet_for_query(text: str, query: str, window: int = 200) -> str:
    if not text:
        return ""
    lower = text.lower()
    q = query.lower()
    idx = lower.find(q)
    if idx == -1:
        return text[: min(400, len(text))]
    start = max(idx - window, 0)
    end = min(idx + len(q) + window, len(text))
    return text[start:end]


def _spans_for_event(spans: list[dict], snippet: str) -> list[EvidenceSpan]:
    evidence_spans: list[EvidenceSpan] = []
    for span in spans[:3]:
        evidence_spans.append(
            EvidenceSpan(
                span_id=str(span.get("span_id", "S?")),
                start=int(span.get("start", 0)),
                end=int(span.get("end", len(snippet))),
                conf=float(span.get("conf", span.get("confidence", 0.9))),
            )
        )
    if not evidence_spans:
        evidence_spans.append(EvidenceSpan(span_id="S0", start=0, end=len(snippet), conf=0.5))
    return evidence_spans


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
