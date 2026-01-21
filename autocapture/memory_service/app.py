"""FastAPI app for the Memory Service."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import AppConfig
from ..logging_utils import get_logger
from ..observability.metrics import (
    memory_feedback_total,
    memory_ingest_reject_total,
    memory_ingest_total,
    memory_query_latency_ms,
    memory_query_total,
)
from ..observability.otel import init_otel
from ..storage.database import DatabaseManager
from .providers import build_embedder, build_reranker
from .schemas import (
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    MemoryHealthResponse,
    MemoryIngestRequest,
    MemoryIngestResponse,
    MemoryQueryRequest,
    MemoryQueryResponse,
)
from .store import MemoryServiceStore, SqliteMemoryServiceStore

_LOG = get_logger("memory.api")


def resolve_memory_service_db_url(config: AppConfig) -> str:
    data_dir = Path(config.capture.data_dir)
    path = data_dir / "memory_service.db"
    return f"sqlite:///{path.as_posix()}"


def create_memory_service_app(
    config: AppConfig,
    *,
    db_manager: DatabaseManager | None = None,
) -> FastAPI:
    init_otel(config.features.enable_otel)
    mem_cfg = config.memory_service
    if not mem_cfg.enabled:
        _LOG.warning("Memory Service disabled; starting anyway for explicit run.")

    db_config = config.database
    if mem_cfg.database_url:
        db_config = db_config.model_copy(update={"url": mem_cfg.database_url})
    else:
        db_config = db_config.model_copy(update={"url": resolve_memory_service_db_url(config)})
    db = db_manager or DatabaseManager(db_config)

    embedder = build_embedder(mem_cfg.embedder, allow_local=not config.offline)
    reranker = build_reranker(mem_cfg.reranker)
    if db.engine.dialect.name == "sqlite":
        store = SqliteMemoryServiceStore(db, mem_cfg, embedder, reranker)
    else:
        store = MemoryServiceStore(db, mem_cfg, embedder, reranker)

    app = FastAPI(title="Autocapture Memory Service")

    @app.middleware("http")
    async def _limit_body(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
            except ValueError:
                size = None
            if size is not None and size > mem_cfg.max_body_bytes:
                return Response(status_code=413, content="Payload too large")
        return await call_next(request)

    def _require_api_key(request: Request) -> None:
        if not mem_cfg.require_api_key:
            return
        api_key = mem_cfg.api_key or ""
        header = request.headers.get("Authorization") or ""
        if not api_key or header != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/v1/memory/health", response_model=MemoryHealthResponse)
    def health() -> MemoryHealthResponse:
        result = store.health()
        return MemoryHealthResponse(**result)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/memory/ingest", response_model=MemoryIngestResponse)
    def ingest(payload: MemoryIngestRequest, request: Request) -> MemoryIngestResponse:
        _require_api_key(request)
        if not mem_cfg.enable_ingest:
            raise HTTPException(status_code=403, detail="ingest_disabled")
        try:
            result = store.ingest(payload)
            memory_ingest_total.labels("accepted").inc(result.accepted)
            memory_ingest_total.labels("deduped").inc(result.deduped)
            memory_ingest_total.labels("rejected").inc(result.rejected)
            for reject in result.rejects:
                for reason in reject.reasons:
                    memory_ingest_reject_total.labels(reason).inc()
            return result
        except ValueError as exc:
            memory_ingest_total.labels("error").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            memory_ingest_total.labels("error").inc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/memory/query", response_model=MemoryQueryResponse)
    def query(payload: MemoryQueryRequest, request: Request) -> MemoryQueryResponse:
        _require_api_key(request)
        if not mem_cfg.enable_query:
            raise HTTPException(status_code=403, detail="query_disabled")
        start = time.monotonic()
        try:
            result = store.query(payload)
            memory_query_total.labels("200").inc()
            return result
        except ValueError as exc:
            memory_query_total.labels("400").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            memory_query_total.labels("500").inc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            memory_query_latency_ms.observe((time.monotonic() - start) * 1000)

    @app.post("/v1/memory/feedback", response_model=MemoryFeedbackResponse)
    def feedback(payload: MemoryFeedbackRequest, request: Request) -> MemoryFeedbackResponse:
        _require_api_key(request)
        if not mem_cfg.enable_feedback:
            raise HTTPException(status_code=403, detail="feedback_disabled")
        try:
            result = store.feedback(payload)
            memory_feedback_total.labels("stored").inc()
            return result
        except RuntimeError as exc:
            memory_feedback_total.labels("error").inc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
