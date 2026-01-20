"""FastAPI app for the LLM gateway."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import AppConfig, is_loopback_host
from ..plugins import PluginManager
from ..logging_utils import get_logger
from ..observability.metrics import (
    gateway_failures_total,
    gateway_latency_ms,
    gateway_requests_total,
)
from .models import ChatCompletionRequest, EmbeddingRequest, GatewayHealth
from .service import GatewayRouter, UpstreamError

_LOG = get_logger("gateway.api")


def create_gateway_app(
    config: AppConfig,
    *,
    router: GatewayRouter | None = None,
    plugin_manager: PluginManager | None = None,
) -> FastAPI:
    router = router or GatewayRouter(config, plugin_manager=plugin_manager)
    app = FastAPI(title="Autocapture LLM Gateway")

    @app.middleware("http")
    async def _limit_body(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
            except ValueError:
                size = None
            if size is not None and size > config.gateway.max_body_bytes:
                return Response(status_code=413, content="Payload too large")
        return await call_next(request)

    def _require_api_key(request: Request) -> None:
        if not config.gateway.require_api_key:
            return
        api_key = config.gateway.api_key or ""
        header = request.headers.get("Authorization") or ""
        if not api_key or header != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="unauthorized")

    def _require_internal(request: Request) -> None:
        host = request.client.host if request.client else ""
        if host and is_loopback_host(host):
            return
        token = (
            request.headers.get("X-Internal-Token") or request.headers.get("Authorization") or ""
        )
        internal_token = config.gateway.internal_token or config.gateway.api_key or ""
        if not internal_token:
            raise HTTPException(status_code=403, detail="internal_token_required")
        if token == internal_token:
            return
        if token == f"Bearer {internal_token}":
            return
        raise HTTPException(status_code=403, detail="internal_unauthorized")

    @app.on_event("startup")
    async def _startup_probe() -> None:
        if not config.gateway.startup_probe:
            return
        await router.probe_upstreams()

    @app.get("/health", response_model=GatewayHealth)
    def health() -> GatewayHealth:
        return GatewayHealth(
            status="ok",
            registry_enabled=router.registry_enabled(),
            providers=[provider.id for provider in config.model_registry.providers],
        )

    @app.get("/healthz", response_model=GatewayHealth)
    def healthz() -> GatewayHealth:
        return health()

    @app.get("/ready")
    def ready() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        _require_api_key(request)
        return JSONResponse(content=await router.handle_models())

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Any:
        _require_api_key(request)
        start = time.monotonic()
        try:
            response = await router.handle_proxy_request(payload.model_dump())
            gateway_requests_total.labels("chat", "200").inc()
            return JSONResponse(content=response)
        except UpstreamError as exc:
            status = exc.status_code or 502
            gateway_requests_total.labels("chat", str(status)).inc()
            gateway_failures_total.labels("chat", str(status)).inc()
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            gateway_latency_ms.labels("chat").observe((time.monotonic() - start) * 1000)

    @app.post("/v1/completions")
    async def completions(payload: dict[str, Any], request: Request) -> Any:
        _require_api_key(request)
        start = time.monotonic()
        try:
            response = await router.handle_completions(payload)
            gateway_requests_total.labels("completions", "200").inc()
            return JSONResponse(content=response)
        except UpstreamError as exc:
            status = exc.status_code or 502
            gateway_requests_total.labels("completions", str(status)).inc()
            gateway_failures_total.labels("completions", str(status)).inc()
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            gateway_latency_ms.labels("completions").observe((time.monotonic() - start) * 1000)

    @app.post("/v1/embeddings")
    async def embeddings(payload: EmbeddingRequest, request: Request) -> Any:
        _require_api_key(request)
        start = time.monotonic()
        try:
            response = await router.handle_embeddings(payload.model_dump())
            gateway_requests_total.labels("embeddings", "200").inc()
            return JSONResponse(content=response)
        except UpstreamError as exc:
            status = exc.status_code or 502
            gateway_requests_total.labels("embeddings", str(status)).inc()
            gateway_failures_total.labels("embeddings", str(status)).inc()
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            gateway_latency_ms.labels("embeddings").observe((time.monotonic() - start) * 1000)

    @app.post("/internal/stage/{stage_id}/chat.completions")
    async def stage_chat(stage_id: str, payload: ChatCompletionRequest, request: Request) -> Any:
        start = time.monotonic()
        tenant_id = request.headers.get("X-Tenant-Id") or payload.tenant_id
        _require_internal(request)
        try:
            response = await router.handle_stage_request(
                stage_id,
                payload.model_dump(),
                tenant_id=tenant_id,
            )
            gateway_requests_total.labels("stage", "200").inc()
            return JSONResponse(content=response)
        except UpstreamError as exc:
            status = exc.status_code or 502
            gateway_requests_total.labels("stage", str(status)).inc()
            gateway_failures_total.labels("stage", str(status)).inc()
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        finally:
            gateway_latency_ms.labels("stage").observe((time.monotonic() - start) * 1000)

    return app


__all__ = ["create_gateway_app"]
