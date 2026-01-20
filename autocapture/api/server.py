"""Local FastAPI server for Personal Activity Memory Engine."""

from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import io
import json
import tempfile
import threading
import time
from uuid import uuid4

import numpy as np
from PIL import Image
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, model_validator
from sqlalchemy import func, select

from ..agents.answer_graph import AnswerGraph
from ..agents import AGENT_JOB_DAILY_HIGHLIGHTS
from ..agents.jobs import AgentJobQueue
from ..answer.integrity import check_citations
from ..contracts_utils import sha256_text
from ..config import AppConfig, ProviderRoutingConfig, apply_settings_overrides, is_loopback_host
from ..encryption import EncryptionManager
from ..logging_utils import get_logger
from ..observability.metrics import get_gpu_snapshot, get_metrics_port
from ..observability.otel import init_otel
from ..paths import resource_root
from ..settings_store import read_settings, update_settings
from ..media.store import MediaStore
from ..capture.privacy_filter import apply_exclude_region_masks, should_skip_capture
from ..capture.frame_record import build_frame_record_v1, build_privacy_flags, capture_record_kwargs
from ..time_utils import elapsed_ms, monotonic_now, utc_now
from ..memory.compression import extractive_answer
from ..memory.context_pack import (
    EvidenceItem,
    EvidenceSpan,
    build_context_pack,
    build_evidence_payload,
)
from ..memory.prompt_injection import scan_prompt_injection
from ..memory.compiler import ContextCompiler
from ..memory.store import MemoryStore
from ..memory.utils import format_utc
from ..vision.citation_overlay import render_citation_overlay
from ..memory.entities import EntityResolver, SecretStore
from ..security.token_vault import TokenVaultStore
from ..memory.prompts import PromptLibraryService, PromptRegistry
from ..memory.retrieval import RetrieveFilters, RetrievalService
from ..memory.threads import ThreadRetrievalService
from ..memory.time_intent import (
    is_time_only_expression,
    resolve_time_range_for_query,
    resolve_timezone,
)
from ..memory_service.client import MemoryServiceClient
from ..memory_service.hooks import fetch_memory_cards
from ..embeddings.service import EmbeddingService
from ..indexing.vector_index import VectorIndex
from ..indexing.pruner import IndexPruner
from ..model_ops import StageRouter
from ..memory.verification import Claim, RulesVerifier
from ..plugins import PluginManager
from ..plugins.errors import PluginLockError, PluginResolutionError
from ..security.oidc import GoogleOIDCVerifier
from ..security.session import SecuritySessionManager, is_test_mode
from ..storage.database import DatabaseManager
from ..storage.deletion import delete_range as delete_range_records
from ..storage.models import (
    CaptureRecord,
    DailyHighlightsRecord,
    EventRecord,
    FrameRecord,
    OCRSpanRecord,
    PromptOpsRunRecord,
    QueryHistoryRecord,
)
from ..storage.retention import RetentionManager


class RetrieveRequest(BaseModel):
    query: str
    time_range: Optional[tuple[dt.datetime, dt.datetime]] = None
    filters: Optional[dict[str, list[str]]] = None
    k: int = Field(8, ge=1, le=100)
    page: int = Field(0, ge=0)
    page_size: Optional[int] = Field(None, ge=1)
    sanitize: Optional[bool] = None
    extractive_only: Optional[bool] = None
    include_screenshots: bool = False


class RetrieveResponse(BaseModel):
    evidence: list[dict[str, Any]]
    no_evidence: bool = False
    message: str | None = None


class IngestMetadata(BaseModel):
    observation_id: Optional[str] = None
    captured_at: Optional[dt.datetime] = None
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None
    monitor_id: Optional[str] = None
    is_fullscreen: bool = False


class IngestResponse(BaseModel):
    observation_id: str
    status: str


class StorageResponse(BaseModel):
    media_path: str
    media_usage_bytes: int
    screenshot_ttl_days: int


class ContextPackRequest(BaseModel):
    query: str
    time_range: Optional[tuple[dt.datetime, dt.datetime]] = None
    filters: Optional[dict[str, list[str]]] = None
    k: int = Field(8, ge=1, le=100)
    sanitize: Optional[bool] = None
    extractive_only: Optional[bool] = None
    pack_format: str = Field("json", description="json|text|tron")
    routing: Optional[dict[str, str]] = None
    include_memory_snapshot: Optional[bool] = None
    memory_hotness_mode: Optional[str] = None
    memory_hotness_as_of_utc: Optional[str] = None


class HighlightsSummary(BaseModel):
    day: str
    summary: str
    highlights: list[str]


class HighlightsDetail(BaseModel):
    day: str
    data: dict[str, Any]


class HighlightsRegenerateRequest(BaseModel):
    day: str


class ResolveTokensRequest(BaseModel):
    tokens: list[str]
    sanitize: Optional[bool] = None


class ContextPackResponse(BaseModel):
    pack: dict[str, Any]
    text: Optional[str] = None
    tron: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    message: str | None = None
    memory_snapshot: dict[str, Any] | None = None


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
    top_k: Optional[int] = Field(None, ge=1)
    output_format: Optional[str] = None
    context_pack_format: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_query(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if not values.get("query") and values.get("q"):
                values["query"] = values["q"]
        return values


class CitationOverlayRequest(BaseModel):
    event_id: str
    bboxes: list[list[float]] = Field(default_factory=list)
    bbox_format: str = Field("px", description="px|norm|auto")


class CitationValidateRequest(BaseModel):
    span_ids: list[str] = Field(default_factory=list)


class CitationValidateResponse(BaseModel):
    valid_span_ids: list[str]
    invalid_span_ids: dict[str, str]


class PromptStrategyInfo(BaseModel):
    strategy: str
    repeat_factor: int
    step_by_step_used: bool
    safe_mode_degraded: bool
    degraded_reason: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    citations: list[str]
    used_context_pack: dict[str, Any]
    latency_ms: float
    response_json: Optional[dict[str, Any]] = None
    response_tron: Optional[str] = None
    context_pack_tron: Optional[str] = None
    prompt_strategy: Optional[PromptStrategyInfo] = None
    no_evidence: bool = False
    message: str | None = None
    mode: str | None = None
    coverage: dict[str, Any] | None = None
    confidence: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    degraded_stages: list[str] | None = None
    hints: list[dict[str, Any]] | None = None
    actions: list[dict[str, Any]] | None = None
    conflict_summary: dict[str, Any] | None = None
    answer_id: str | None = None
    query_id: str | None = None


class EventResponse(BaseModel):
    event_id: str
    ts_start: dt.datetime
    ts_end: Optional[dt.datetime]
    app_name: str
    window_title: str
    url: Optional[str]
    domain: Optional[str]
    screenshot_path: Optional[str]
    focus_path: Optional[str]
    screenshot_hash: str
    ocr_text: str
    ocr_spans: list[dict[str, Any]]
    tags: dict[str, Any]


class SettingsRequest(BaseModel):
    settings: dict[str, Any]


class SettingsResponse(BaseModel):
    status: str


class SettingsSnapshot(BaseModel):
    settings: dict[str, Any]


class PluginEnableRequest(BaseModel):
    plugin_id: str
    accept_hashes: bool = False


class PluginDisableRequest(BaseModel):
    plugin_id: str


class PluginLockRequest(BaseModel):
    plugin_id: str


class PluginExtensionInfo(BaseModel):
    kind: str
    id: str
    name: str
    plugin_id: str
    plugin_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    pillars: dict[str, Any] | None = None
    ui: dict[str, Any] | None = None
    source: str | None = None
    enabled: bool = True


class PluginCatalogEntry(BaseModel):
    plugin_id: str
    name: str
    version: str
    description: str | None = None
    source: str
    enabled: bool
    blocked: bool
    reason: str | None = None
    lock_status: str
    lock_manifest: str | None = None
    lock_code: str | None = None
    manifest_sha256: str
    code_sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)
    extensions: list[PluginExtensionInfo] = Field(default_factory=list)


class PluginCatalogResponse(BaseModel):
    safe_mode: bool
    plugins: list[PluginCatalogEntry]
    warnings: list[str] = Field(default_factory=list)


class PluginExtensionsResponse(BaseModel):
    kind: str
    extensions: list[PluginExtensionInfo]


class SuggestRequest(BaseModel):
    q: str


class DeleteRangeRequest(BaseModel):
    start_utc: dt.datetime
    end_utc: dt.datetime
    process: Optional[str] = None
    window_title: Optional[str] = None


class DeleteRangeResponse(BaseModel):
    deleted_captures: int
    deleted_events: int
    deleted_segments: int
    deleted_files: int


class UnlockResponse(BaseModel):
    token: str
    expires_at: Optional[str] = None


class PromptOpsRunSummary(BaseModel):
    run_id: str
    ts: dt.datetime
    status: str
    pr_url: Optional[str] = None
    eval_results: dict[str, Any]


class PromptOpsRunsResponse(BaseModel):
    runs: list[PromptOpsRunSummary]


class _TokenBucket:
    def __init__(self, tokens: float, updated_at: float) -> None:
        self.tokens = tokens
        self.updated_at = updated_at


class _RateLimiter:
    def __init__(self, rps: float, burst: int, max_entries: int = 10_000) -> None:
        self._rps = rps
        self._burst = burst
        self._max_entries = max_entries
        self._buckets: OrderedDict[str, _TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(tokens=self._burst, updated_at=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rps)
                bucket.updated_at = now
                self._buckets.move_to_end(key)
            if len(self._buckets) > self._max_entries:
                self._buckets.popitem(last=False)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            retry_after = max(0.0, (1.0 - bucket.tokens) / self._rps)
            return False, retry_after


def create_app(
    config: AppConfig,
    db_manager: DatabaseManager | None = None,
    *,
    embedder: EmbeddingService | None = None,
    vector_index: VectorIndex | None = None,
    worker_supervisor: object | None = None,
    plugin_manager: PluginManager | None = None,
) -> FastAPI:
    init_otel(config.features.enable_otel)
    db_owned = db_manager is None
    db = db_manager or DatabaseManager(config.database)
    plugins = plugin_manager or PluginManager(config)
    embedder_obj = embedder
    if embedder_obj is None:
        embedder_id = (config.routing.embedding or "local").strip().lower()
        try:
            embedder_obj = plugins.resolve_extension("embedder.text", embedder_id)
        except Exception as exc:
            log = get_logger("api")
            log.warning("Embedder plugin failed ({}): {}", embedder_id, exc)
            embedder_obj = None
    if embedder_obj is None:
        embedder_obj = EmbeddingService(config.embed)
    dim = getattr(embedder_obj, "dim", None) or int(config.qdrant.text_vector_size)
    if vector_index is None:
        try:
            backend = plugins.resolve_extension(
                "vector.backend",
                "qdrant",
                factory_kwargs={"dim": dim},
            )
        except Exception as exc:
            log = get_logger("api")
            log.warning("Vector backend plugin failed: {}", exc)
            backend = None
        vector_index = VectorIndex(config, dim, backend=backend)
    reranker = None
    reranker_id = (config.routing.reranker or "disabled").strip().lower()
    try:
        reranker = plugins.resolve_extension("reranker", reranker_id)
    except Exception as exc:
        log = get_logger("api")
        log.warning("Reranker plugin failed ({}): {}", reranker_id, exc)
        reranker = None
    retrieval_id = (config.routing.retrieval or "local").strip().lower()
    try:
        retrieval = plugins.resolve_extension(
            "retrieval.strategy",
            retrieval_id,
            factory_kwargs={
                "db": db,
                "embedder": embedder_obj,
                "vector_index": vector_index,
                "reranker": reranker,
                "plugin_manager": plugins,
            },
        )
    except Exception as exc:
        log = get_logger("api")
        log.warning("Retrieval plugin failed ({}): {}", retrieval_id, exc)
        retrieval = RetrievalService(
            db,
            config,
            embedder=embedder_obj,
            vector_index=vector_index,
            reranker=reranker,
            plugin_manager=plugins,
        )
    thread_retrieval = ThreadRetrievalService(
        config,
        db,
        embedder=getattr(retrieval, "_embedder", None),
        vector_index=getattr(retrieval, "_vector", None),
    )
    encryption_mgr = EncryptionManager(config.encryption)
    secret = SecretStore(Path(config.capture.data_dir)).get_or_create()
    token_vault = TokenVaultStore(config, db)
    entities = EntityResolver(db, secret, token_vault=token_vault)
    agent_jobs = AgentJobQueue(db)
    prompt_registry = PromptRegistry.from_package(
        "autocapture.prompts.derived",
        hardening_enabled=config.templates.enabled,
        log_provenance=config.templates.log_provenance,
        extra_dirs=plugins.prompt_bundles(),
        allow_external=True,
    )
    PromptLibraryService(db).sync_registry(prompt_registry)
    memory_service_client: MemoryServiceClient | None = None
    if config.features.enable_memory_service_read_hook:
        memory_service_client = MemoryServiceClient(config.memory_service)
    answer_graph = AnswerGraph(
        config,
        retrieval,
        prompt_registry=prompt_registry,
        entities=entities,
        plugin_manager=plugins,
        memory_client=memory_service_client,
    )
    retention = RetentionManager(
        config.storage, config.retention, db, Path(config.capture.data_dir)
    )
    index_pruner = IndexPruner(
        db,
        vector_index=getattr(retrieval, "_vector", None),
        spans_index=getattr(retrieval, "_spans_v2", None),
    )
    media_store = MediaStore(config.capture, config.encryption)
    log = get_logger("api")
    memory_store: MemoryStore | None = None
    memory_compiler: ContextCompiler | None = None
    if config.memory.enabled:
        try:
            memory_store = MemoryStore(config.memory)
            memory_compiler = ContextCompiler(memory_store, config.memory)
        except Exception as exc:
            log.warning("Memory store init failed: {}", exc)
    storage_cache = {"ts": 0.0, "bytes": 0}
    storage_cache_ttl_s = 30.0
    media_root = Path(config.capture.data_dir) / "media"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(retention.enforce_screenshot_ttl)
        except Exception as exc:
            log.warning("Retention enforcement failed: {}", exc)
        yield
        if db_owned:
            try:
                db.engine.dispose()
            except Exception as exc:
                log.warning("Database dispose failed: {}", exc)

    app_kwargs: dict[str, Any] = {
        "title": "Autocapture Memory Engine",
        "lifespan": lifespan,
    }
    if config.api.require_api_key and config.mode.mode != "remote":
        app_kwargs["docs_url"] = None
        app_kwargs["redoc_url"] = None
        app_kwargs["openapi_url"] = None
    app = FastAPI(**app_kwargs)
    app.state.db = db
    app.state.vector_index = vector_index
    app.state.embedder = embedder_obj
    app.state.worker_supervisor = worker_supervisor
    app.state.memory_store = memory_store
    app.state.memory_compiler = memory_compiler
    app.state.memory_service_client = memory_service_client
    app.state.plugins = plugins

    oidc_verifier: GoogleOIDCVerifier | None = None
    if config.mode.mode == "remote":
        oidc_verifier = GoogleOIDCVerifier(
            config.mode.google_oauth_client_id or "",
            config.mode.google_allowed_emails,
        )
    rate_limiter = _RateLimiter(config.api.rate_limit_rps, config.api.rate_limit_burst)
    rate_limited_paths = {
        "/api/answer",
        "/api/retrieve",
        "/api/context-pack",
    }
    protected_prefixes = (
        "/api/answer",
        "/api/retrieve",
        "/api/context-pack",
        "/api/context_pack",
        "/api/screenshot/",
        "/api/context_pack/",
        "/api/context-pack/",
        "/api/event/",
        "/api/highlights",
        "/api/privacy/resolve_tokens",
        "/api/delete_range",
        "/api/delete_all",
        "/api/events/ingest",
        "/api/storage",
        "/api/plugins",
    )

    def _maybe_build_memory_snapshot(
        query: str,
        *,
        k: int,
        include: Optional[bool],
        hotness_mode: Optional[str] = None,
        hotness_as_of_utc: Optional[str] = None,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        include_flag = include if include is not None else config.memory.api_context_pack_enabled
        if not include_flag:
            return None, []
        if not config.memory.enabled or memory_compiler is None:
            return None, ["memory_disabled"]
        mode = (hotness_mode or "off").strip().lower()
        if mode not in {"off", "as_of", "dynamic"}:
            raise HTTPException(
                status_code=422,
                detail="memory_hotness_mode must be off, as_of, or dynamic",
            )
        if mode != "off" and not config.memory.hotness.enabled:
            raise HTTPException(status_code=400, detail="memory hotness is disabled")
        as_of_utc = hotness_as_of_utc
        if mode == "as_of" and not as_of_utc:
            raise HTTPException(status_code=400, detail="memory_hotness_as_of_utc required")
        if mode == "dynamic" and not as_of_utc:
            now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
            as_of_utc = format_utc(now)
        memory_k = min(int(k), int(config.memory.retrieval.max_k))
        try:
            result = memory_compiler.compile(
                query,
                k=memory_k,
                memory_hotness_mode=mode,
                memory_hotness_as_of_utc=as_of_utc,
            )
            snapshot_dir = Path(result.output_dir)
            context_md = (snapshot_dir / "context.md").read_text(encoding="utf-8")
            citations = json.loads((snapshot_dir / "citations.json").read_text(encoding="utf-8"))
            manifest = json.loads((snapshot_dir / "context.json").read_text(encoding="utf-8"))
            payload = result.model_dump(mode="json")
            payload.pop("output_dir", None)
            snapshot = {
                "result": payload,
                "context_md": context_md,
                "citations": citations,
                "manifest": manifest,
            }
            return snapshot, []
        except Exception as exc:
            log.warning("Memory snapshot failed: {}", exc)
            return None, ["memory_snapshot_failed"]

    security_manager: SecuritySessionManager | None = None
    if (
        config.security.local_unlock_enabled
        and config.mode.mode != "remote"
        and is_loopback_host(config.api.bind_host)
        and config.security.provider != "disabled"
    ):
        security_manager = SecuritySessionManager(
            ttl_seconds=config.security.session_ttl_seconds,
            provider=config.security.provider,
            test_mode_bypass=is_test_mode(),
        )

    ui_dir = resource_root() / "autocapture" / "ui" / "web"
    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=ui_dir), name="static")

    def _dump_model(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    def _plugin_status_map() -> dict[str, Any]:
        return {status.plugin.plugin_id: status for status in plugins.catalog()}

    def _extension_info_from_manifest(
        ext: Any,
        *,
        plugin_id: str,
        plugin_name: str | None,
        source: str | None,
        enabled: bool,
    ) -> PluginExtensionInfo:
        return PluginExtensionInfo(
            kind=ext.kind,
            id=ext.id,
            name=ext.name,
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            aliases=list(ext.aliases or []),
            pillars=_dump_model(ext.pillars),
            ui=_dump_model(ext.ui),
            source=source,
            enabled=enabled,
        )

    @app.get("/plugins/{plugin_id}/assets/{asset_path:path}")
    def plugin_assets(plugin_id: str, asset_path: str) -> Response:
        plugins.refresh()
        status = _plugin_status_map().get(plugin_id)
        if not status or not status.enabled or status.blocked:
            raise HTTPException(status_code=404, detail="Plugin assets unavailable")
        assets_root = status.plugin.source.assets_path
        if assets_root is None:
            raise HTTPException(status_code=404, detail="No assets for plugin")
        assets_path = Path(assets_root)
        try:
            resolved = (assets_path / asset_path).resolve()
        except Exception:
            raise HTTPException(status_code=404, detail="Invalid asset path")
        try:
            root_resolved = assets_path.resolve()
        except Exception:
            root_resolved = assets_path
        if root_resolved not in resolved.parents and resolved != root_resolved:
            raise HTTPException(status_code=404, detail="Invalid asset path")
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(resolved)

    @app.get("/healthz/deep")
    def deep_health() -> JSONResponse:
        checks: dict[str, dict[str, Any]] = {}
        overall_ok = True

        try:
            with db.session() as session:
                session.execute(select(1))
                _ = (
                    session.execute(
                        select(EventRecord.event_id).order_by(EventRecord.ts_start.desc()).limit(1)
                    )
                    .scalars()
                    .first()
                )
            checks["db"] = {"ok": True, "detail": "ok"}
        except Exception as exc:
            checks["db"] = {"ok": False, "detail": f"{exc}"}
            overall_ok = False

        if config.qdrant.enabled:
            try:
                from qdrant_client import QdrantClient

                client = QdrantClient(url=config.qdrant.url, timeout=2.0)
                collections = [config.qdrant.text_collection]
                if config.qdrant.image_collection not in collections:
                    collections.append(config.qdrant.image_collection)
                missing = [name for name in collections if not client.collection_exists(name)]
                if missing:
                    checks["qdrant"] = {
                        "ok": False,
                        "detail": f"missing collections: {', '.join(missing)}",
                        "skipped": False,
                    }
                    overall_ok = False
                else:
                    checks["qdrant"] = {
                        "ok": True,
                        "detail": "ok",
                        "skipped": False,
                    }
            except Exception as exc:
                checks["qdrant"] = {
                    "ok": False,
                    "detail": f"{exc}",
                    "skipped": False,
                }
                overall_ok = False
        else:
            checks["qdrant"] = {
                "ok": True,
                "detail": "disabled",
                "skipped": True,
            }

        embedder_for_health = embedder_obj
        if embedder_for_health is None and config.embed.text_model == "local-test":
            try:
                embedder_for_health = EmbeddingService(config.embed)
            except Exception as exc:
                checks["embedding"] = {
                    "ok": False,
                    "detail": f"{exc}",
                    "skipped": False,
                }
                overall_ok = False
        if embedder_for_health is None:
            checks["embedding"] = {
                "ok": True,
                "detail": "skipped (no local embedder)",
                "skipped": True,
            }
        elif "embedding" not in checks:
            try:
                vector = embedder_for_health.embed_texts(["health-check"])[0]
                if not vector:
                    raise RuntimeError("empty embedding vector")
                checks["embedding"] = {
                    "ok": True,
                    "detail": "ok",
                    "skipped": False,
                }
            except Exception as exc:
                checks["embedding"] = {
                    "ok": False,
                    "detail": f"{exc}",
                    "skipped": False,
                }
                overall_ok = False

        supervisor = app.state.worker_supervisor
        if supervisor is None:
            checks["workers"] = {
                "ok": True,
                "detail": "skipped (no supervisor)",
                "skipped": True,
            }
        else:
            snapshot = getattr(supervisor, "health_snapshot", lambda: {})()
            watchdog_ok = snapshot.get("watchdog_alive", False)
            workers_ok = snapshot.get("workers_alive", False)
            ok = bool(watchdog_ok and workers_ok)
            checks["workers"] = {
                "ok": ok,
                "detail": (
                    "watchdog ok"
                    if ok
                    else f"watchdog_alive={watchdog_ok}, workers_alive={workers_ok}"
                ),
                "skipped": False,
            }
            if not ok:
                overall_ok = False

        payload = {
            "ok": overall_ok,
            "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "checks": checks,
        }
        return JSONResponse(status_code=200 if overall_ok else 503, content=payload)

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):  # type: ignore[no-redef]
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        headers.setdefault("X-Robots-Tag", "noindex, nofollow")
        content_type = headers.get("content-type", "").lower()
        if content_type.startswith("text/html"):
            headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; "
                "img-src 'self' data:; "
                "script-src 'self'; "
                "style-src 'self'; "
                "object-src 'none'; "
                "base-uri 'none'; "
                "frame-ancestors 'none'",
            )
            headers.setdefault("Cache-Control", "no-store")
        if request.url.path.startswith("/api/"):
            headers.setdefault("Cache-Control", "no-store")
        return response

    if config.mode.mode == "remote":

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):  # type: ignore[no-redef]
            if request.url.path == "/api/events/ingest" and _bridge_token_valid(request, config):
                return await call_next(request)
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

    else:

        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next):  # type: ignore[no-redef]
            if request.url.path.startswith("/api/"):
                if request.url.path == "/api/events/ingest" and _bridge_token_valid(
                    request, config
                ):
                    return await call_next(request)
                _require_api_key(request, config)
            return await call_next(request)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):  # type: ignore[no-redef]
        if request.url.path in rate_limited_paths and request.method == "POST":
            client_host = request.client.host if request.client else "unknown"
            key = f"{client_host}:{request.url.path}"
            allowed, retry_after = rate_limiter.allow(key)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": f"{int(retry_after) + 1}"},
                    content={"detail": "Rate limit exceeded"},
                )
        return await call_next(request)

    @app.middleware("http")
    async def security_session_middleware(request: Request, call_next):  # type: ignore[no-redef]
        if security_manager and _needs_unlock(request, protected_prefixes):
            if request.url.path == "/api/events/ingest" and _bridge_token_valid(request, config):
                return await call_next(request)
            token = _extract_unlock_token(request)
            if not security_manager.is_unlocked(token):
                return JSONResponse(status_code=401, content={"detail": "Unlock required"})
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": config.mode.mode,
            "metrics_port": get_metrics_port() or config.observability.prometheus_port,
        }

    @app.get("/")
    async def index() -> HTMLResponse:
        if not ui_dir.exists():
            return HTMLResponse("<h2>UI not available</h2>")
        return FileResponse(ui_dir / "index.html")

    @app.get("/dashboard")
    async def dashboard_redirect() -> RedirectResponse:
        return RedirectResponse(url="/")

    @app.get("/api/status")
    def status_snapshot() -> dict:
        with db.session() as session:
            ocr_pending = session.execute(
                select(func.count())
                .select_from(CaptureRecord)
                .where(CaptureRecord.ocr_status == "pending")
            ).scalar_one()
            ocr_processing = session.execute(
                select(func.count())
                .select_from(CaptureRecord)
                .where(CaptureRecord.ocr_status == "processing")
            ).scalar_one()
            span_embed_pending = session.execute(
                select(func.count())
                .select_from(OCRSpanRecord)
                .where(OCRSpanRecord.embedding_status == "pending")
            ).scalar_one()
            event_embed_pending = session.execute(
                select(func.count())
                .select_from(EventRecord)
                .where(EventRecord.embedding_status == "pending")
            ).scalar_one()

        gpu = get_gpu_snapshot()
        return {
            "ok": True,
            "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ocr": {"pending": int(ocr_pending), "processing": int(ocr_processing)},
            "embeddings": {
                "pending_spans": int(span_embed_pending),
                "pending_events": int(event_embed_pending),
            },
            "gpu": gpu,
        }

    @app.get("/api/plugins/catalog")
    def plugins_catalog() -> PluginCatalogResponse:
        plugins.refresh()
        statuses = plugins.catalog()
        warnings: list[str] = []
        entries: list[PluginCatalogEntry] = []
        for status in statuses:
            plugin = status.plugin
            if plugin.plugin_id == "__discovery__":
                warnings.extend(plugin.warnings or [])
                continue
            manifest = plugin.manifest
            enabled = status.enabled and not status.blocked
            extensions = [
                _extension_info_from_manifest(
                    ext,
                    plugin_id=plugin.plugin_id,
                    plugin_name=manifest.name,
                    source=plugin.source.source_type.value,
                    enabled=enabled,
                )
                for ext in manifest.extensions
            ]
            entries.append(
                PluginCatalogEntry(
                    plugin_id=plugin.plugin_id,
                    name=manifest.name,
                    version=manifest.version,
                    description=manifest.description,
                    source=plugin.source.source_type.value,
                    enabled=status.enabled,
                    blocked=status.blocked,
                    reason=status.reason,
                    lock_status=status.lock_status,
                    lock_manifest=status.lock_manifest,
                    lock_code=status.lock_code,
                    manifest_sha256=status.manifest_sha256,
                    code_sha256=status.code_sha256 or None,
                    warnings=list(plugin.warnings or []),
                    extensions=sorted(extensions, key=lambda item: (item.kind, item.id)),
                )
            )
        entries.sort(key=lambda item: item.plugin_id)
        return PluginCatalogResponse(
            safe_mode=plugins.safe_mode,
            plugins=entries,
            warnings=warnings,
        )

    @app.get("/api/plugins/extensions")
    def plugins_extensions(kind: str, include_disabled: bool = False) -> PluginExtensionsResponse:
        if not kind:
            raise HTTPException(status_code=400, detail="kind is required")
        plugins.refresh()
        status_map = _plugin_status_map()
        extensions: list[PluginExtensionInfo] = []
        if include_disabled:
            for status in plugins.catalog():
                plugin = status.plugin
                if plugin.plugin_id == "__discovery__":
                    continue
                for ext in plugin.manifest.extensions:
                    if ext.kind != kind:
                        continue
                    extensions.append(
                        _extension_info_from_manifest(
                            ext,
                            plugin_id=plugin.plugin_id,
                            plugin_name=plugin.manifest.name,
                            source=plugin.source.source_type.value,
                            enabled=status.enabled and not status.blocked,
                        )
                    )
        else:
            for record in plugins.list_extensions(kind):
                status = status_map.get(record.plugin_id)
                plugin_name = status.plugin.manifest.name if status else None
                extensions.append(
                    _extension_info_from_manifest(
                        record.manifest,
                        plugin_id=record.plugin_id,
                        plugin_name=plugin_name,
                        source=record.source,
                        enabled=True,
                    )
                )
        extensions.sort(key=lambda item: (item.name.lower(), item.id))
        return PluginExtensionsResponse(kind=kind, extensions=extensions)

    @app.post("/api/plugins/enable")
    def plugins_enable(request: PluginEnableRequest) -> PluginCatalogEntry:
        plugin_id = request.plugin_id
        if not plugin_id:
            raise HTTPException(status_code=400, detail="plugin_id is required")
        try:
            status = plugins.enable_plugin(plugin_id, accept_hashes=request.accept_hashes)
        except PluginLockError:
            status = _plugin_status_map().get(plugin_id)
            detail = {"detail": "acceptance_required"}
            if status:
                detail.update(
                    {
                        "manifest_sha256": status.manifest_sha256,
                        "code_sha256": status.code_sha256,
                    }
                )
            raise HTTPException(status_code=409, detail=detail)
        except PluginResolutionError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        plugin = status.plugin
        manifest = plugin.manifest
        enabled = status.enabled and not status.blocked
        extensions = [
            _extension_info_from_manifest(
                ext,
                plugin_id=plugin.plugin_id,
                plugin_name=manifest.name,
                source=plugin.source.source_type.value,
                enabled=enabled,
            )
            for ext in manifest.extensions
        ]
        return PluginCatalogEntry(
            plugin_id=plugin.plugin_id,
            name=manifest.name,
            version=manifest.version,
            description=manifest.description,
            source=plugin.source.source_type.value,
            enabled=status.enabled,
            blocked=status.blocked,
            reason=status.reason,
            lock_status=status.lock_status,
            lock_manifest=status.lock_manifest,
            lock_code=status.lock_code,
            manifest_sha256=status.manifest_sha256,
            code_sha256=status.code_sha256 or None,
            warnings=list(plugin.warnings or []),
            extensions=sorted(extensions, key=lambda item: (item.kind, item.id)),
        )

    @app.post("/api/plugins/disable")
    def plugins_disable(request: PluginDisableRequest) -> PluginCatalogEntry:
        plugin_id = request.plugin_id
        if not plugin_id:
            raise HTTPException(status_code=400, detail="plugin_id is required")
        try:
            status = plugins.disable_plugin(plugin_id)
        except PluginResolutionError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        plugin = status.plugin
        manifest = plugin.manifest
        extensions = [
            _extension_info_from_manifest(
                ext,
                plugin_id=plugin.plugin_id,
                plugin_name=manifest.name,
                source=plugin.source.source_type.value,
                enabled=False,
            )
            for ext in manifest.extensions
        ]
        return PluginCatalogEntry(
            plugin_id=plugin.plugin_id,
            name=manifest.name,
            version=manifest.version,
            description=manifest.description,
            source=plugin.source.source_type.value,
            enabled=status.enabled,
            blocked=status.blocked,
            reason=status.reason,
            lock_status=status.lock_status,
            lock_manifest=status.lock_manifest,
            lock_code=status.lock_code,
            manifest_sha256=status.manifest_sha256,
            code_sha256=status.code_sha256 or None,
            warnings=list(plugin.warnings or []),
            extensions=sorted(extensions, key=lambda item: (item.kind, item.id)),
        )

    @app.post("/api/plugins/lock")
    def plugins_lock(request: PluginLockRequest) -> PluginCatalogEntry:
        plugin_id = request.plugin_id
        if not plugin_id:
            raise HTTPException(status_code=400, detail="plugin_id is required")
        try:
            status = plugins.lock_plugin(plugin_id)
        except PluginResolutionError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        plugin = status.plugin
        manifest = plugin.manifest
        enabled = status.enabled and not status.blocked
        extensions = [
            _extension_info_from_manifest(
                ext,
                plugin_id=plugin.plugin_id,
                plugin_name=manifest.name,
                source=plugin.source.source_type.value,
                enabled=enabled,
            )
            for ext in manifest.extensions
        ]
        return PluginCatalogEntry(
            plugin_id=plugin.plugin_id,
            name=manifest.name,
            version=manifest.version,
            description=manifest.description,
            source=plugin.source.source_type.value,
            enabled=status.enabled,
            blocked=status.blocked,
            reason=status.reason,
            lock_status=status.lock_status,
            lock_manifest=status.lock_manifest,
            lock_code=status.lock_code,
            manifest_sha256=status.manifest_sha256,
            code_sha256=status.code_sha256 or None,
            warnings=list(plugin.warnings or []),
            extensions=sorted(extensions, key=lambda item: (item.kind, item.id)),
        )

    @app.get("/api/plugins/health")
    def plugins_health() -> dict[str, Any]:
        plugins.refresh()
        return plugins.run_healthchecks()

    def _storage_usage_bytes() -> int:
        now = time.monotonic()
        if now - storage_cache["ts"] > storage_cache_ttl_s:
            storage_cache["bytes"] = RetentionManager._folder_size_bytes(media_root)
            storage_cache["ts"] = now
        return int(storage_cache["bytes"])

    @app.get("/api/storage")
    def storage_status() -> StorageResponse:
        return StorageResponse(
            media_path=str(media_root),
            media_usage_bytes=_storage_usage_bytes(),
            screenshot_ttl_days=config.retention.screenshot_ttl_days,
        )

    @app.post("/api/events/ingest")
    async def ingest_event(
        request: Request,
        metadata: str = Form(...),
        image: UploadFile = File(...),
    ) -> IngestResponse:
        try:
            parsed = IngestMetadata.model_validate_json(metadata)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid metadata JSON")

        observation_id = parsed.observation_id or str(uuid4())
        captured_at = parsed.captured_at or utc_now()
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=dt.timezone.utc)
        monotonic_ts = monotonic_now()
        app_name = parsed.app_name or "unknown"
        window_title = parsed.window_title or "unknown"
        monitor_id = parsed.monitor_id or "unknown"

        if _bridge_token_valid(request, config) is False:
            _require_api_key(request, config)

        if should_skip_capture(
            paused=config.privacy.paused,
            monitor_id=monitor_id,
            process_name=app_name,
            window_title=window_title,
            exclude_monitors=config.privacy.exclude_monitors,
            exclude_processes=config.privacy.exclude_processes,
            exclude_window_title_regex=config.privacy.exclude_window_title_regex,
        ):
            _record_skipped_capture(
                db=db,
                config=config,
                observation_id=observation_id,
                captured_at=captured_at,
                monotonic_ts=monotonic_ts,
                app_name=app_name,
                window_title=window_title,
                monitor_id=monitor_id,
                is_fullscreen=parsed.is_fullscreen,
                reason="privacy_filter",
                excluded=True,
                masked_regions_applied=bool(config.privacy.exclude_regions),
            )
            return IngestResponse(observation_id=observation_id, status="skipped")

        with db.session() as session:
            existing = session.get(CaptureRecord, observation_id)
            if existing:
                return IngestResponse(observation_id=observation_id, status="exists")

        payload = await image.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Image payload is empty")
        try:
            pil = Image.open(io.BytesIO(payload)).convert("RGB")
            image_array = np.array(pil)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid image payload") from exc

        masked_regions_applied = False
        if config.privacy.exclude_regions and monitor_id:
            masked_regions_applied = apply_exclude_region_masks(
                image_array,
                monitor_id=monitor_id,
                roi_origin_x=0,
                roi_origin_y=0,
                exclude_regions=config.privacy.exclude_regions,
            )

        path = media_store.write_fullscreen(image_array, captured_at, observation_id)
        if path is None:
            _record_skipped_capture(
                db=db,
                config=config,
                observation_id=observation_id,
                captured_at=captured_at,
                monotonic_ts=monotonic_ts,
                app_name=app_name,
                window_title=window_title,
                monitor_id=monitor_id,
                is_fullscreen=parsed.is_fullscreen,
                reason="disk_quota",
                excluded=False,
                masked_regions_applied=masked_regions_applied,
            )
            return IngestResponse(observation_id=observation_id, status="skipped")

        record_kwargs = None
        frame = None
        frame_hash = None
        if config.features.enable_frame_record_v1:
            if config.features.enable_frame_hash:
                try:
                    from ..image_utils import hash_rgb_image

                    frame_hash = hash_rgb_image(image_array)
                except Exception:
                    frame_hash = None
            privacy_flags = build_privacy_flags(
                config.privacy,
                excluded=False,
                masked_regions_applied=masked_regions_applied,
                capture_paused=False,
                offline=config.offline,
            )
            frame = build_frame_record_v1(
                frame_id=observation_id,
                event_id=observation_id,
                captured_at=captured_at,
                monotonic_ts=monotonic_ts,
                monitor_id=monitor_id,
                monitor_bounds=None,
                app_name=app_name,
                window_title=window_title,
                image_path=str(path),
                privacy_flags=privacy_flags,
                frame_hash=frame_hash,
            )
            record_kwargs = capture_record_kwargs(
                frame=frame,
                captured_at=captured_at,
                image_path=str(path),
                focus_path=None,
                foreground_process=app_name,
                foreground_window=window_title,
                monitor_id=monitor_id,
                is_fullscreen=parsed.is_fullscreen,
                ocr_status="pending",
            )
        if record_kwargs is None:
            record_kwargs = {
                "id": observation_id,
                "captured_at": captured_at,
                "image_path": str(path),
                "focus_path": None,
                "foreground_process": app_name,
                "foreground_window": window_title,
                "monitor_id": monitor_id,
                "is_fullscreen": parsed.is_fullscreen,
                "ocr_status": "pending",
            }

        frame_flags = frame.privacy_flags.model_dump() if frame else {}
        frame_hash_value = frame_hash

        def _write(session) -> None:
            session.add(CaptureRecord(**record_kwargs))
            session.add(
                FrameRecord(
                    frame_id=observation_id,
                    event_id=None,
                    captured_at_utc=captured_at,
                    monotonic_ts=monotonic_ts,
                    monitor_id=monitor_id,
                    monitor_bounds=None,
                    app_name=app_name,
                    window_title=window_title,
                    media_path=str(path),
                    privacy_flags=frame_flags,
                    frame_hash=frame_hash_value,
                    excluded=False,
                    masked=masked_regions_applied,
                    schema_version=1,
                    created_at=dt.datetime.now(dt.timezone.utc),
                )
            )

        db.transaction(_write)
        return IngestResponse(observation_id=observation_id, status="ok")

    @app.post("/api/unlock")
    def unlock_session() -> UnlockResponse:
        if not security_manager:
            return UnlockResponse(token="", expires_at=None)
        session = security_manager.unlock()
        if not session:
            raise HTTPException(status_code=401, detail="Unlock failed")
        return UnlockResponse(token=session.token, expires_at=session.expires_at.isoformat())

    @app.post("/api/lock")
    def lock_session() -> dict[str, Any]:
        if not security_manager:
            return {"status": "ok", "cleared": 0}
        cleared = security_manager.lock()
        return {"status": "ok", "cleared": cleared}

    @app.get("/api/promptops/latest")
    def promptops_latest() -> PromptOpsRunsResponse:
        with db.session() as session:
            run = session.query(PromptOpsRunRecord).order_by(PromptOpsRunRecord.ts.desc()).first()
        if not run:
            return PromptOpsRunsResponse(runs=[])
        return PromptOpsRunsResponse(runs=[_promptops_summary(run)])

    @app.get("/api/promptops/runs")
    def promptops_runs(limit: int = 20) -> PromptOpsRunsResponse:
        with db.session() as session:
            runs = (
                session.query(PromptOpsRunRecord)
                .order_by(PromptOpsRunRecord.ts.desc())
                .limit(min(limit, 100))
                .all()
            )
        return PromptOpsRunsResponse(runs=[_promptops_summary(run) for run in runs])

    @app.get("/api/screenshot/{event_id}")
    def screenshot(event_id: str, variant: str = "full") -> Response:
        with db.session() as session:
            event = session.get(EventRecord, event_id)
            if variant not in {"full", "focus"}:
                raise HTTPException(status_code=422, detail="variant must be full or focus")
            path_value = event.screenshot_path if variant == "full" else event.focus_path
            if not event or not path_value:
                raise HTTPException(status_code=404, detail="Screenshot not found")
            path = Path(path_value)

        if not path.is_absolute():
            path = config.capture.data_dir / path

        try:
            root = config.capture.data_dir.resolve()
            resolved = path.resolve()
            if root not in resolved.parents and resolved != root:
                raise HTTPException(status_code=403, detail="Invalid screenshot path")
            path = resolved
        except FileNotFoundError:
            pass

        if not path.exists():
            raise HTTPException(status_code=404, detail="Screenshot file missing")

        headers = {"Cache-Control": "private, max-age=60"}

        if path.suffix == ".acenc":
            with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                encryption_mgr.decrypt_file(path, tmp_path)
                data = tmp_path.read_bytes()
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return Response(content=data, media_type="image/webp", headers=headers)
        return FileResponse(path, headers=headers)

    @app.post("/api/citations/overlay")
    def citation_overlay(request: CitationOverlayRequest) -> Response:
        if not config.ui.overlay_citations_enabled:
            raise HTTPException(status_code=403, detail="Citation overlay disabled")
        with db.session() as session:
            event = session.get(EventRecord, request.event_id)
            if not event or not event.screenshot_path:
                raise HTTPException(status_code=404, detail="Screenshot not available")
            path_value = event.screenshot_path
            path = Path(path_value)
            if not path.is_absolute():
                path = config.capture.data_dir / path
            resolved = path.resolve()
            root = Path(config.capture.data_dir).resolve()
            if root not in resolved.parents and resolved != root:
                raise HTTPException(status_code=403, detail="Invalid screenshot path")
            if not resolved.exists():
                raise HTTPException(status_code=404, detail="Screenshot file missing")
            try:
                image = Image.open(resolved)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid screenshot: {exc}") from exc
        fmt = (request.bbox_format or "px").strip().lower()
        if fmt not in {"px", "norm", "auto"}:
            raise HTTPException(status_code=422, detail="bbox_format must be px, norm, or auto")
        normalized = True if fmt == "norm" else None if fmt == "auto" else False
        overlay = render_citation_overlay(image, request.bboxes, normalized=normalized)
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.post("/api/citations/validate")
    def citation_validate(request: CitationValidateRequest) -> CitationValidateResponse:
        if not request.span_ids:
            return CitationValidateResponse(valid_span_ids=[], invalid_span_ids={})
        result = check_citations(db, request.span_ids)
        return CitationValidateResponse(
            valid_span_ids=sorted(result.valid_span_ids),
            invalid_span_ids=result.invalid_span_ids,
        )

    @app.post("/api/retrieve")
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        _validate_query(request.query, config)
        offset, limit = _resolve_retrieve_paging(request, config)
        _record_query_history(db, request.query)
        evidence, events, no_evidence, message = _build_evidence(
            retrieval,
            entities,
            db,
            request.query,
            request.time_range,
            request.filters,
            limit,
            offset=offset,
            sanitized=_resolve_bool(request.sanitize, config.privacy.sanitize_default),
        )
        event_map = {event.event_id: event for event in events}
        return RetrieveResponse(
            evidence=[
                _evidence_to_json(item, event_map.get(item.event_id), request.include_screenshots)
                for item in evidence
            ],
            no_evidence=no_evidence,
            message=message,
        )

    @app.post("/api/context-pack")
    def context_pack(request: ContextPackRequest) -> ContextPackResponse:
        _validate_query(request.query, config)
        k = _cap_k(request.k, config)
        sanitized = _resolve_bool(request.sanitize, config.privacy.sanitize_default)
        evidence, events, no_evidence, message = _build_evidence(
            retrieval,
            entities,
            db,
            request.query,
            request.time_range,
            request.filters,
            k,
            sanitized=sanitized,
        )
        memory_snapshot, memory_warnings = _maybe_build_memory_snapshot(
            request.query,
            k=k,
            include=request.include_memory_snapshot,
            hotness_mode=request.memory_hotness_mode,
            hotness_as_of_utc=request.memory_hotness_as_of_utc,
        )
        memory_cards, memory_service_warnings = fetch_memory_cards(
            config,
            query=request.query,
            client=memory_service_client,
        )
        routing_data = _merge_routing(config.routing, request.routing)
        aggregates = _build_aggregates(db, request.time_range)
        thread_aggregates = _build_thread_aggregates(
            thread_retrieval, request.query, request.time_range
        )
        aggregates = _merge_aggregates(aggregates, thread_aggregates)
        if no_evidence and not evidence:
            empty_pack = build_context_pack(
                query=request.query,
                evidence=[],
                entity_tokens=[],
                routing=_model_dump(routing_data),
                filters={
                    "time_range": request.time_range,
                    "apps": request.filters.get("app") if request.filters else None,
                    "domains": request.filters.get("domain") if request.filters else None,
                },
                sanitized=sanitized,
                aggregates=aggregates,
                memory_cards=memory_cards,
            )
            payload = empty_pack.to_json()
            text_pack = None
            tron_pack = None
            if request.pack_format == "tron":
                tron_pack = empty_pack.to_text(extractive_only=False, format="tron")
            else:
                text_pack = empty_pack.to_text(extractive_only=False, format="json")
            return ContextPackResponse(
                pack=payload,
                text=text_pack,
                tron=tron_pack,
                warnings=["no_evidence", *memory_warnings, *memory_service_warnings],
                message=message
                or _no_evidence_message(request.query, bool(request.time_range), None),
                memory_snapshot=memory_snapshot,
            )
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
            aggregates=aggregates,
            memory_cards=memory_cards,
        )
        text_pack = None
        tron_pack = None
        if request.pack_format == "text":
            text_pack = pack.to_text(
                extractive_only=_resolve_bool(
                    request.extractive_only, config.privacy.extractive_only_default
                ),
                format="json",
            )
        elif request.pack_format == "tron":
            tron_pack = pack.to_text(
                extractive_only=_resolve_bool(
                    request.extractive_only, config.privacy.extractive_only_default
                ),
                format="tron",
            )
        return ContextPackResponse(
            pack=pack.to_json(),
            text=text_pack,
            tron=tron_pack,
            warnings=[*memory_warnings, *memory_service_warnings],
            memory_snapshot=memory_snapshot,
        )

    @app.post("/api/answer")
    async def answer(request: AnswerRequest) -> AnswerResponse:
        sanitized = _resolve_bool(request.sanitize, config.privacy.sanitize_default)
        extractive_only = _resolve_bool(
            request.extractive_only, config.privacy.extractive_only_default
        )
        output_format = _resolve_output_format(request.output_format, config.output.format)
        context_pack_format = _resolve_context_pack_format(
            request.context_pack_format, config.output.context_pack_format
        )
        query_text = request.query or ""
        _validate_query(query_text, config)
        k = _cap_k(request.top_k or 12, config)
        _record_query_history(db, query_text)
        tzinfo = resolve_timezone(config.time.timezone)
        resolved_time_range = resolve_time_range_for_query(
            query=query_text,
            time_range=request.time_range,
            now=dt.datetime.now(tzinfo),
            tzinfo=tzinfo,
        )
        time_only = is_time_only_expression(query_text)
        evidence_query = "" if time_only else query_text
        evidence, events, no_evidence, message = await asyncio.to_thread(
            _build_evidence,
            retrieval,
            entities,
            db,
            evidence_query,
            resolved_time_range,
            request.filters,
            k,
            sanitized,
        )
        routing_data = _merge_routing(config.routing, request.routing)
        routing_override = request.routing.get("llm") if request.routing else None
        aggregates = _build_aggregates(db, resolved_time_range)
        thread_aggregates = _build_thread_aggregates(
            thread_retrieval, query_text, resolved_time_range
        )
        aggregates = _merge_aggregates(aggregates, thread_aggregates)
        if no_evidence or not evidence:
            empty_pack = build_context_pack(
                query=query_text,
                evidence=[],
                entity_tokens=[],
                routing=_model_dump(routing_data),
                filters={
                    "time_range": resolved_time_range,
                    "apps": request.filters.get("app") if request.filters else None,
                    "domains": request.filters.get("domain") if request.filters else None,
                },
                sanitized=sanitized,
                aggregates=aggregates,
            )
            notice = message or _no_evidence_message(query_text, bool(resolved_time_range), None)
            response_json, response_tron = _build_answer_payload(
                notice,
                [],
                warnings=["no_evidence"],
                used_llm=False,
                context_pack=empty_pack.to_json(),
                output_format=output_format,
            )
            return AnswerResponse(
                answer=notice,
                citations=[],
                used_context_pack=empty_pack.to_json(),
                latency_ms=0.0,
                response_json=response_json,
                response_tron=response_tron,
                context_pack_tron=None,
                prompt_strategy=None,
                no_evidence=True,
                message=notice,
                mode="NO_EVIDENCE",
            )
        pack = build_context_pack(
            query=query_text,
            evidence=evidence,
            entity_tokens=entities.tokens_for_events(events),
            routing=_model_dump(routing_data),
            filters={
                "time_range": resolved_time_range,
                "apps": request.filters.get("app") if request.filters else None,
                "domains": request.filters.get("domain") if request.filters else None,
            },
            sanitized=sanitized,
            aggregates=aggregates,
        )
        pack_json_text = pack.to_text(extractive_only=False, format="json")
        pack_tron_text = (
            pack.to_text(extractive_only=False, format="tron")
            if context_pack_format == "tron"
            else None
        )
        pack_text = pack_tron_text or pack_json_text
        context_pack_tron = pack_tron_text if context_pack_format == "tron" else None
        start = monotonic_now()
        graph_attempted = False
        graph_used_llm = False
        prompt_strategy_info: PromptStrategyInfo | None = None
        graph_result = None
        if config.agents.answer_agent.enabled:
            try:
                graph_attempted = True
                result = await answer_graph.run(
                    query_text,
                    time_range=resolved_time_range,
                    filters=request.filters,
                    k=k,
                    sanitized=sanitized,
                    extractive_only=extractive_only,
                    routing=_model_dump(routing_data),
                    routing_override=routing_override,
                    aggregates=aggregates,
                    output_format=output_format,
                    context_pack_format=context_pack_format,
                )
                answer_text = result.answer
                citations = result.citations
                graph_used_llm = result.used_llm
                graph_result = result
                if result.prompt_strategy:
                    prompt_strategy_info = _prompt_strategy_info(result.prompt_strategy)
                pack = build_context_pack(
                    query=query_text,
                    evidence=evidence,
                    entity_tokens=entities.tokens_for_events(events),
                    routing=_model_dump(routing_data),
                    filters={
                        "time_range": resolved_time_range,
                        "apps": request.filters.get("app") if request.filters else None,
                        "domains": request.filters.get("domain") if request.filters else None,
                    },
                    sanitized=sanitized,
                    aggregates=aggregates,
                )
                pack_json_text = pack.to_text(extractive_only=False, format="json")
                pack_tron_text = (
                    pack.to_text(extractive_only=False, format="tron")
                    if context_pack_format == "tron"
                    else None
                )
                pack_text = pack_tron_text or pack_json_text
                context_pack_tron = result.context_pack_tron
                response_json = result.response_json
                response_tron = result.response_tron
            except Exception as exc:
                log.warning("Agentic answer failed; falling back to baseline: {}", exc)
                answer_text = ""
                citations = []
                graph_attempted = False
                response_json = None
                response_tron = None
        else:
            answer_text = ""
            citations = []
            response_json = None
            response_tron = None

        if time_only and resolved_time_range is not None:
            answer_text, citations = _build_timeline_answer(evidence)
            prompt_strategy_info = None
            response_json, response_tron = _build_answer_payload(
                answer_text,
                citations,
                warnings=[],
                used_llm=False,
                context_pack=pack.to_json(),
                output_format=output_format,
            )
        elif extractive_only:
            compressed = extractive_answer(evidence)
            answer_text = compressed.answer
            citations = compressed.citations
            prompt_strategy_info = None
            response_json, response_tron = _build_answer_payload(
                answer_text,
                citations,
                warnings=[],
                used_llm=False,
                context_pack=pack.to_json(),
                output_format=output_format,
            )
        elif graph_attempted and graph_used_llm:
            pass
        elif graph_attempted and getattr(graph_result, "mode", None) in {
            "BLOCKED",
            "NO_EVIDENCE",
            "CONFLICT",
        }:
            pass
        else:
            stage_router = StageRouter(config, plugin_manager=plugins)
            provider, decision = stage_router.select_llm(
                "final_answer", routing_override=routing_override
            )
            system_prompt = prompt_registry.get("ANSWER_WITH_CONTEXT_PACK").system_prompt
            try:
                pack_text = _select_context_pack_text(
                    config,
                    decision,
                    context_pack_format,
                    pack_json_text,
                    pack_tron_text,
                )
                answer_text = await provider.generate_answer(
                    system_prompt,
                    query_text,
                    pack_text,
                    temperature=decision.temperature,
                )
                prompt_strategy_info = _prompt_strategy_info(
                    getattr(provider, "last_prompt_metadata", None)
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
                        pack_text,
                        temperature=decision.temperature,
                    )
                    citations = _extract_citations(answer_text)
                if not _valid_citations(citations, evidence):
                    compressed = extractive_answer(evidence)
                    answer_text = compressed.answer
                    citations = compressed.citations
                else:
                    verifier = RulesVerifier()
                    verifier.verify(
                        [
                            Claim(
                                text=answer_text,
                                evidence_ids=citations,
                                entity_tokens=[],
                            )
                        ],
                        {item.evidence_id for item in evidence},
                        set(),
                    )
                log.info(
                    "LLM stage {} routed to {}",
                    decision.stage,
                    getattr(decision, "provider", "unknown"),
                )
                response_json, response_tron = _build_answer_payload(
                    answer_text,
                    citations,
                    warnings=[],
                    used_llm=True,
                    context_pack=pack.to_json(),
                    output_format=output_format,
                )
            except Exception as exc:
                log.warning("LLM unavailable; falling back to extractive answer: {}", exc)
                compressed = extractive_answer(evidence)
                answer_text = compressed.answer
                citations = compressed.citations
                prompt_strategy_info = None
                response_json, response_tron = _build_answer_payload(
                    answer_text,
                    citations,
                    warnings=[],
                    used_llm=False,
                    context_pack=pack.to_json(),
                    output_format=output_format,
                )
        latency = elapsed_ms(start)
        return AnswerResponse(
            answer=answer_text,
            citations=citations,
            used_context_pack=pack.to_json(),
            latency_ms=latency,
            response_json=response_json,
            response_tron=response_tron,
            context_pack_tron=context_pack_tron,
            prompt_strategy=prompt_strategy_info,
            no_evidence=False,
            message=None,
            mode=getattr(graph_result, "mode", None),
            coverage=getattr(graph_result, "coverage", None),
            confidence=getattr(graph_result, "confidence", None),
            budgets=getattr(graph_result, "budgets", None),
            degraded_stages=getattr(graph_result, "degraded_stages", None),
            hints=getattr(graph_result, "hints", None),
            actions=getattr(graph_result, "actions", None),
            conflict_summary=getattr(graph_result, "conflict_summary", None),
            answer_id=getattr(graph_result, "answer_id", None),
            query_id=getattr(graph_result, "query_id", None),
        )

    @app.get("/api/highlights")
    def list_highlights() -> list[HighlightsSummary]:
        with db.session() as session:
            rows = (
                session.execute(
                    select(DailyHighlightsRecord).order_by(DailyHighlightsRecord.day.desc())
                )
                .scalars()
                .all()
            )
        summaries = []
        for row in rows:
            data = row.data_json or {}
            summaries.append(
                HighlightsSummary(
                    day=row.day,
                    summary=data.get("summary", ""),
                    highlights=data.get("highlights", []),
                )
            )
        return summaries

    @app.get("/api/highlights/{day}")
    def get_highlights(day: str) -> HighlightsDetail:
        with db.session() as session:
            row = (
                session.execute(
                    select(DailyHighlightsRecord).where(DailyHighlightsRecord.day == day)
                )
                .scalars()
                .first()
            )
        if not row:
            raise HTTPException(status_code=404, detail="Highlights not found")
        return HighlightsDetail(day=row.day, data=row.data_json or {})

    @app.post("/api/highlights/regenerate")
    def regenerate_highlights(request: HighlightsRegenerateRequest, http_request: Request) -> dict:
        client_host = http_request.client.host if http_request.client else "unknown"
        if not is_loopback_host(client_host):
            _require_api_key(http_request, config)
        job_key = f"highlights:{request.day}:v1"
        job_id = agent_jobs.enqueue(
            job_key=job_key,
            job_type=AGENT_JOB_DAILY_HIGHLIGHTS,
            day=request.day,
            payload={"day": request.day},
            max_pending=config.agents.max_pending_jobs,
        )
        if not job_id:
            raise HTTPException(status_code=429, detail="Agent backlog exceeded")
        return {"job_id": job_id}

    @app.post("/api/privacy/resolve_tokens")
    def resolve_tokens(request: ResolveTokensRequest, http_request: Request) -> dict:
        if not config.privacy.allow_token_vault_decrypt:
            raise HTTPException(status_code=403, detail="Token vault decryption disabled")
        client_host = http_request.client.host if http_request.client else "unknown"
        if not is_loopback_host(client_host):
            _require_api_key(http_request, config)
        sanitized = _resolve_bool(request.sanitize, config.privacy.sanitize_default)
        if sanitized:
            return {"resolved": {}}
        resolved = token_vault.resolve_tokens(request.tokens)
        return {"resolved": resolved}

    @app.post("/api/suggest")
    def suggest(request: SuggestRequest) -> list[dict[str, Any]]:
        query = request.q.strip()
        if query:
            _validate_query(query, config)
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
            score = prefix_boost * 1.0 + (row.count**0.5) * 0.3 + recency * 0.3
            scored.append((score, row.query_text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"snippet": text} for _, text in scored[:8]]

    @app.get("/api/event/{event_id}")
    def event_detail(event_id: str) -> EventResponse:
        with db.session() as session:
            event = session.get(EventRecord, event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        spans = _fetch_spans(db, event.event_id)
        return EventResponse(
            event_id=event.event_id,
            ts_start=event.ts_start,
            ts_end=event.ts_end,
            app_name=event.app_name,
            window_title=event.window_title,
            url=event.url,
            domain=event.domain,
            screenshot_path=event.screenshot_path,
            focus_path=event.focus_path,
            screenshot_hash=event.screenshot_hash,
            ocr_text=event.ocr_text,
            ocr_spans=spans,
            tags=event.tags,
        )

    @app.post("/api/delete_range")
    def delete_range_endpoint(request: DeleteRangeRequest) -> DeleteRangeResponse:
        counts = delete_range_records(
            db,
            Path(config.capture.data_dir),
            start_utc=request.start_utc,
            end_utc=request.end_utc,
            process=request.process,
            window_title=request.window_title,
            index_pruner=index_pruner if config.features.enable_retention_prune else None,
        )
        return DeleteRangeResponse(
            deleted_captures=counts.deleted_captures,
            deleted_events=counts.deleted_events,
            deleted_segments=counts.deleted_segments,
            deleted_files=counts.deleted_files,
        )

    @app.post("/api/delete_all")
    def delete_all() -> DeleteRangeResponse:
        now = dt.datetime.now(dt.timezone.utc)
        start = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        counts = delete_range_records(
            db,
            Path(config.capture.data_dir),
            start_utc=start,
            end_utc=now,
            index_pruner=index_pruner if config.features.enable_retention_prune else None,
        )
        return DeleteRangeResponse(
            deleted_captures=counts.deleted_captures,
            deleted_events=counts.deleted_events,
            deleted_segments=counts.deleted_segments,
            deleted_files=counts.deleted_files,
        )

    @app.post("/api/settings")
    def settings(request: SettingsRequest) -> SettingsResponse:
        settings_path = Path(config.capture.data_dir) / "settings.json"
        incoming = request.settings if isinstance(request.settings, dict) else {}
        update_settings(
            settings_path,
            lambda current: {**current, **incoming},
        )
        apply_settings_overrides(config)
        plugins.refresh()
        return SettingsResponse(status="ok")

    @app.get("/api/settings")
    def settings_snapshot() -> SettingsSnapshot:
        settings_path = Path(config.capture.data_dir) / "settings.json"
        settings: dict[str, Any] = read_settings(settings_path)
        if "routing" not in settings:
            settings["routing"] = _model_dump(config.routing)
        if "privacy" not in settings:
            settings["privacy"] = {
                "paused": config.privacy.paused,
                "snooze_until_utc": _to_iso(config.privacy.snooze_until_utc),
            }
        if "active_preset" not in settings:
            settings["active_preset"] = config.presets.active_preset
        if "backup" not in settings:
            settings["backup"] = {"last_export_at_utc": None}
        if "llm" not in settings:
            settings["llm"] = _model_dump(config.llm)
        if "plugins" not in settings:
            settings["plugins"] = {
                "enabled": [],
                "disabled": [],
                "extension_overrides": {},
                "locks": {},
                "configs": {},
            }
        return SettingsSnapshot(settings=settings)

    return app


def _resolve_bool(value: Optional[bool], default: bool) -> bool:
    return default if value is None else value


def _needs_unlock(request: Request, prefixes: tuple[str, ...]) -> bool:
    path = request.url.path
    if not path.startswith("/api/"):
        return False
    return any(path.startswith(prefix) for prefix in prefixes)


def _extract_unlock_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1]
    return request.query_params.get("unlock")


def _bridge_token_valid(request: Request, config: AppConfig) -> bool:
    token = request.headers.get("X-Bridge-Token")
    expected = config.api.bridge_token
    if not token or not expected:
        return False
    return hmac.compare_digest(token, expected)


def _require_api_key(request: Request, config: AppConfig) -> None:
    if not config.api.require_api_key:
        return
    key = request.headers.get("X-API-Key")
    auth = request.headers.get("Authorization", "")
    if not key and auth.startswith("Bearer "):
        key = auth.split(" ", 1)[1]
    if not key or not config.api.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not hmac.compare_digest(key, config.api.api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _record_skipped_capture(
    *,
    db: DatabaseManager,
    config: AppConfig,
    observation_id: str,
    captured_at: dt.datetime,
    monotonic_ts: float,
    app_name: str,
    window_title: str,
    monitor_id: str,
    is_fullscreen: bool,
    reason: str,
    excluded: bool,
    masked_regions_applied: bool,
) -> None:
    record_kwargs = None
    if config.features.enable_frame_record_v1:
        privacy_flags = build_privacy_flags(
            config.privacy,
            excluded=excluded,
            masked_regions_applied=masked_regions_applied,
            capture_paused=False,
            offline=config.offline,
        )
        frame = build_frame_record_v1(
            frame_id=observation_id,
            event_id=None,
            captured_at=captured_at,
            monotonic_ts=monotonic_ts,
            monitor_id=monitor_id,
            monitor_bounds=None,
            app_name=app_name,
            window_title=window_title,
            image_path=None,
            privacy_flags=privacy_flags,
            frame_hash=None,
        )
        record_kwargs = capture_record_kwargs(
            frame=frame,
            captured_at=captured_at,
            image_path=None,
            focus_path=None,
            foreground_process=app_name,
            foreground_window=window_title,
            monitor_id=monitor_id,
            is_fullscreen=is_fullscreen,
            ocr_status="skipped",
            ocr_last_error=reason,
        )
    if record_kwargs is None:
        record_kwargs = {
            "id": observation_id,
            "captured_at": captured_at,
            "image_path": None,
            "focus_path": None,
            "foreground_process": app_name,
            "foreground_window": window_title,
            "monitor_id": monitor_id,
            "is_fullscreen": is_fullscreen,
            "ocr_status": "skipped",
            "ocr_last_error": reason,
        }

    def _write(session) -> None:
        session.add(CaptureRecord(**record_kwargs))
        session.add(
            FrameRecord(
                frame_id=observation_id,
                event_id=None,
                captured_at_utc=captured_at,
                monotonic_ts=monotonic_ts,
                monitor_id=monitor_id,
                monitor_bounds=None,
                app_name=app_name,
                window_title=window_title,
                media_path=None,
                privacy_flags=frame.privacy_flags.model_dump() if frame else {},
                frame_hash=None,
                excluded=excluded,
                masked=masked_regions_applied,
                schema_version=1,
                created_at=dt.datetime.now(dt.timezone.utc),
            )
        )

    db.transaction(_write)


def _validate_query(query: str, config: AppConfig) -> None:
    if not query or not query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty")
    if len(query) > config.api.max_query_chars:
        raise HTTPException(status_code=413, detail="Query too long")


def _cap_k(k: int, config: AppConfig) -> int:
    if k > config.api.max_context_k:
        raise HTTPException(status_code=422, detail="k exceeds maximum allowed")
    return k


def _resolve_retrieve_paging(request: RetrieveRequest, config: AppConfig) -> tuple[int, int]:
    fields_set = getattr(request, "model_fields_set", set())
    page = max(0, request.page)
    if "page_size" in fields_set:
        page_size = request.page_size or config.api.default_page_size
    elif "k" in fields_set:
        page_size = request.k
    else:
        page_size = config.api.default_page_size
    page_size = max(1, min(page_size, config.api.max_page_size))
    return page * page_size, page_size


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _resolve_output_format(requested: str | None, default: str) -> str:
    value = (requested or default or "text").strip().lower()
    allowed = {"text", "json", "tron"}
    if value not in allowed:
        raise HTTPException(status_code=422, detail="output_format must be text, json, or tron")
    return value


def _resolve_context_pack_format(requested: str | None, default: str) -> str:
    value = (requested or default or "json").strip().lower()
    allowed = {"json", "tron"}
    if value not in allowed:
        raise HTTPException(status_code=422, detail="context_pack_format must be json or tron")
    return value


def _merge_routing(
    base: ProviderRoutingConfig, override: Optional[dict[str, str]]
) -> ProviderRoutingConfig:
    data = _model_dump(base)
    if override:
        data.update({k: v for k, v in override.items() if v})
    return ProviderRoutingConfig(**data)


def _to_iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.isoformat()


def _promptops_summary(run: PromptOpsRunRecord) -> PromptOpsRunSummary:
    return PromptOpsRunSummary(
        run_id=run.run_id,
        ts=run.ts,
        status=run.status,
        pr_url=run.pr_url,
        eval_results=run.eval_results or {},
    )


def _build_evidence(
    retrieval: RetrievalService,
    entities: EntityResolver,
    db: DatabaseManager,
    query: str,
    time_range: Optional[tuple[dt.datetime, dt.datetime]],
    filters: Optional[dict[str, list[str]]],
    k: int,
    sanitized: bool,
    offset: int = 0,
) -> tuple[list[EvidenceItem], list[EventRecord], bool, str | None]:
    retrieve_filters = None
    if filters:
        retrieve_filters = RetrieveFilters(apps=filters.get("app"), domains=filters.get("domain"))
    batch = retrieval.retrieve(query, time_range, retrieve_filters, limit=k, offset=offset)
    results = list(batch.results)
    no_evidence = bool(batch.no_evidence)
    reason = batch.reason
    if not results and time_range and not query:
        batch = retrieval.retrieve("", time_range, retrieve_filters, limit=k, offset=offset)
        results = list(batch.results)
        no_evidence = bool(batch.no_evidence)
        reason = batch.reason
    if not results:
        return [], [], True, _no_evidence_message(query, bool(time_range), reason)
    raw_items: list[dict[str, Any]] = []
    events: list[EventRecord] = []
    seen_keys: set[tuple[str, str]] = set()
    for result in results:
        event = result.event
        snippet = result.snippet or ""
        snippet_offset = result.snippet_offset or 0
        if not snippet:
            snippet, snippet_offset = _snippet_for_query(event.ocr_text, query)
        spans_data = _fetch_spans(db, event.event_id, result.matched_span_keys)
        frame_size = _frame_size_from_tags(event.tags)
        spans = _spans_for_event(
            spans_data,
            snippet,
            snippet_offset,
            query,
            result.matched_span_keys,
            frame_size=frame_size,
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
        scan = scan_prompt_injection(snippet)
        redacted_text = scan.redacted_text
        content_hash = sha256_text(redacted_text)
        key = (event.event_id, content_hash)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        events.append(event)
        raw_items.append(
            {
                "key": key,
                "event": event,
                "app_name": app_name,
                "title": title,
                "domain": domain,
                "score": result.score,
                "spans": spans,
                "text": redacted_text,
                "raw_text": snippet,
                "redacted_text": redacted_text if scan.match_count else None,
                "injection_risk": scan.risk_score,
                "content_hash": content_hash,
                "non_citable": getattr(result, "non_citable", False),
                "retrieval": {
                    "engine": getattr(result, "engine", "hybrid"),
                    "rank": getattr(result, "rank", 0),
                    "rank_gap": getattr(result, "rank_gap", 0.0),
                    "lexical_score": getattr(result, "lexical_score", 0.0),
                    "vector_score": getattr(result, "vector_score", 0.0),
                    "sparse_score": getattr(result, "sparse_score", 0.0),
                    "late_score": getattr(result, "late_score", 0.0),
                    "rerank_score": getattr(result, "rerank_score", None),
                    "matched_spans": getattr(result, "matched_span_keys", []),
                    "snippet_offset": getattr(result, "snippet_offset", None),
                    "bbox": getattr(result, "bbox", None),
                    "non_citable": getattr(result, "non_citable", False),
                    "ts_start": event.ts_start.isoformat(),
                },
            }
        )
    if not raw_items:
        return [], [], True, _no_evidence_message(query, bool(time_range), reason)
    sorted_keys = sorted({item["key"] for item in raw_items})
    key_to_id = {key: f"E{idx}" for idx, key in enumerate(sorted_keys, start=1)}
    evidence: list[EvidenceItem] = []
    for item in raw_items:
        non_citable = bool(item["non_citable"])
        evidence.append(
            EvidenceItem(
                evidence_id=key_to_id[item["key"]],
                event_id=item["event"].event_id,
                timestamp=item["event"].ts_start.isoformat(),
                ts_end=item["event"].ts_end.isoformat() if item["event"].ts_end else None,
                app=item["app_name"],
                title=item["title"],
                domain=item["domain"],
                score=item["score"],
                spans=item["spans"],
                text=item["text"],
                raw_text=item["raw_text"],
                redacted_text=item["redacted_text"],
                kind="derived_summary" if non_citable else "source",
                citable=not non_citable,
                injection_risk=item["injection_risk"],
                content_hash=item["content_hash"],
                screenshot_path=item["event"].screenshot_path,
                screenshot_hash=item["event"].screenshot_hash,
                retrieval=item["retrieval"],
            )
        )
    if not evidence and no_evidence:
        message = _no_evidence_message(query, bool(time_range), reason)
        return [], [], True, message
    return evidence, events, no_evidence, None


def _build_aggregates(
    db: DatabaseManager,
    time_range: Optional[tuple[dt.datetime, dt.datetime]],
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {"time_spent_by_app": [], "notable_changes": []}
    if not time_range:
        return aggregates
    start, end = time_range
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    days = []
    cursor = start.date()
    end_date = end.date()
    while cursor <= end_date:
        days.append(cursor.isoformat())
        cursor = cursor + dt.timedelta(days=1)
    if not days:
        return aggregates
    with db.session() as session:
        rows = (
            session.execute(
                select(DailyHighlightsRecord).where(DailyHighlightsRecord.day.in_(days))
            )
            .scalars()
            .all()
        )
    aggregates["daily_highlights"] = [
        {
            "day": row.day,
            "summary": row.data_json.get("summary", ""),
            "highlights": row.data_json.get("highlights", []),
        }
        for row in rows
    ]
    return aggregates


def _build_thread_aggregates(
    retrieval: ThreadRetrievalService,
    query: str,
    time_range: Optional[tuple[dt.datetime, dt.datetime]],
) -> dict[str, Any]:
    broad = bool(time_range) or len((query or "").split()) <= 3
    if not broad:
        return {}
    try:
        candidates = retrieval.retrieve(query or "", time_range, limit=5)
    except Exception:
        return {}
    if not candidates:
        return {}
    return {
        "threads": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "summary": item.summary,
                "ts_start": item.ts_start.isoformat(),
                "ts_end": item.ts_end.isoformat() if item.ts_end else None,
                "citations": item.citations,
            }
            for item in candidates
        ]
    }


def _merge_aggregates(base: dict | None, incoming: dict | None) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (incoming or {}).items():
        merged[key] = value
    return merged


def _build_answer_payload(
    answer: str,
    citations: list[str],
    *,
    warnings: list[str],
    used_llm: bool,
    context_pack: dict[str, Any],
    output_format: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if output_format not in {"json", "tron"}:
        return None, None
    payload = {
        "answer": answer,
        "citations": citations,
        "warnings": warnings,
        "used_llm": used_llm,
        "context_pack": context_pack,
        "evidence": build_evidence_payload(context_pack),
    }
    if output_format == "tron":
        from ..format.tron import encode_tron

        return payload, encode_tron(payload)
    return payload, None


def _select_context_pack_text(
    config: AppConfig,
    decision: object,
    requested_format: str,
    json_text: str,
    tron_text: str | None,
) -> str:
    if requested_format != "tron":
        if not bool(getattr(decision, "cloud", False)):
            return json_text
        if config.output.allow_tron_compression and tron_text:
            return tron_text
        return json_text
    cloud = bool(getattr(decision, "cloud", False))
    if not cloud:
        return tron_text or json_text
    if config.output.allow_tron_compression:
        return tron_text or json_text
    return json_text


def _build_timeline_answer(evidence: list[EvidenceItem]) -> tuple[str, list[str]]:
    if not evidence:
        return "No events found in the requested time range.", []
    sorted_items = sorted(evidence, key=lambda item: item.timestamp)
    lines: list[str] = []
    citations: list[str] = []
    for item in sorted_items:
        snippet = " ".join((item.text or "").split())
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        title = item.title or ""
        label = f"{item.app}: {title}".strip(": ")
        if snippet:
            line = f"{item.timestamp} — {label} — {snippet} [{item.evidence_id}]"
        else:
            line = f"{item.timestamp} — {label} [{item.evidence_id}]"
        lines.append(line)
        citations.append(item.evidence_id)
    return "\n".join(lines), citations


def _prompt_strategy_info(payload: object | None) -> PromptStrategyInfo | None:
    if payload is None:
        return None
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    if not isinstance(payload, dict):
        return None
    return PromptStrategyInfo(
        strategy=str(payload.get("strategy", "baseline")),
        repeat_factor=int(payload.get("repeat_factor", 1)),
        step_by_step_used=bool(payload.get("step_by_step_used", False)),
        safe_mode_degraded=bool(payload.get("safe_mode_degraded", False)),
        degraded_reason=payload.get("degraded_reason"),
    )


def _fetch_spans(
    db: DatabaseManager,
    event_id: str,
    matched_span_keys: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    with db.session() as session:
        stmt = select(OCRSpanRecord).where(OCRSpanRecord.capture_id == event_id)
        if matched_span_keys:
            stmt = stmt.where(OCRSpanRecord.span_key.in_(matched_span_keys))
        stmt = stmt.order_by(OCRSpanRecord.start.asc())
        rows = session.execute(stmt).scalars().all()
    return [
        {
            "span_key": row.span_key,
            "span_id": row.span_key,
            "start": row.start,
            "end": row.end,
            "conf": row.confidence,
            "bbox": row.bbox,
            "text": row.text,
        }
        for row in rows
    ]


def _frame_size_from_tags(tags: dict | None) -> tuple[int, int] | None:
    if not isinstance(tags, dict):
        return None
    meta = tags.get("capture_meta")
    if not isinstance(meta, dict):
        return None
    width = meta.get("frame_width")
    height = meta.get("frame_height")
    try:
        width_val = int(width)
        height_val = int(height)
    except (TypeError, ValueError):
        return None
    if width_val <= 0 or height_val <= 0:
        return None
    return width_val, height_val


def _span_bbox(raw: object) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        coords = [raw.get(key) for key in ("x0", "y0", "x1", "y1")]
        if any(val is None for val in coords):
            return None
        try:
            return [int(float(val)) for val in coords]
        except (TypeError, ValueError):
            return None
    if isinstance(raw, list):
        values = [val for val in raw if isinstance(val, (int, float))]
        if not values:
            return None
        return [int(float(val)) for val in values]
    return None


def _bbox_norm(bbox: list[int] | None, frame_size: tuple[int, int] | None) -> list[float] | None:
    if not bbox or not frame_size:
        return None
    width, height = frame_size
    if width <= 0 or height <= 0:
        return None
    if len(bbox) >= 8:
        xs = bbox[0::2]
        ys = bbox[1::2]
        if not xs or not ys:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    elif len(bbox) >= 4:
        x0, y0, x1, y1 = bbox[:4]
    else:
        return None
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if width == 0 or height == 0:
        return None
    return [
        round(x0 / width, 6),
        round(y0 / height, 6),
        round(x1 / width, 6),
        round(y1 / height, 6),
    ]


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


def _no_evidence_message(query: str, has_time_range: bool, reason: str | None) -> str:
    _ = query
    _ = reason
    if has_time_range:
        return "No evidence found in the selected time range. Try expanding the time range."
    return "No evidence found. Try rephrasing the query or adding a time range."


def _spans_for_event(
    spans: list[dict],
    snippet: str,
    snippet_offset: int,
    query: str,
    matched_span_keys: list[str],
    *,
    frame_size: tuple[int, int] | None = None,
) -> list[EvidenceSpan]:
    evidence_spans: list[EvidenceSpan] = []
    query_lower = query.lower()
    candidate_spans = spans
    if matched_span_keys:
        filtered = [span for span in spans if str(span.get("span_key")) in set(matched_span_keys)]
        if filtered:
            candidate_spans = filtered
    elif query_lower:
        candidate_spans = [
            span for span in spans if query_lower in str(span.get("text", "")).lower()
        ]
    for span in candidate_spans:
        start = int(span.get("start", 0)) - snippet_offset
        end = int(span.get("end", 0)) - snippet_offset
        if start < 0 or end > len(snippet) or end <= start:
            continue
        bbox = _span_bbox(span.get("bbox"))
        bbox_norm = _bbox_norm(bbox, frame_size)
        evidence_spans.append(
            EvidenceSpan(
                span_id=str(span.get("span_id", "S?")),
                start=start,
                end=end,
                conf=float(span.get("conf", span.get("confidence", 0.9))),
                bbox=bbox,
                bbox_norm=bbox_norm,
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
        overlaps = [rep for rep in replacements if span.start < rep[1] and span.end > rep[0]]
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
                bbox=span.bbox,
                bbox_norm=span.bbox_norm,
            )
        )
    return remapped


def _extract_citations(answer_text: str) -> list[str]:
    import re

    citations = re.findall(r"(?:\\[|【)(E\\d+)(?::L\\d+-L\\d+)?(?:\\]|】)", answer_text or "")
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
                select(QueryHistoryRecord).where(QueryHistoryRecord.normalized_text == normalized)
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
            {
                "span_id": span.span_id,
                "start": span.start,
                "end": span.end,
                "conf": span.conf,
            }
            for span in item.spans
        ],
        "text": item.text,
    }
    if item.retrieval:
        payload["retrieval"] = item.retrieval
    if include_screenshots and event:
        payload["screenshot_path"] = event.screenshot_path
        payload["focus_path"] = event.focus_path
    return payload
