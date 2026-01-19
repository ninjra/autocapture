"""FastAPI app for graph retrieval workers."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import AppConfig
from ..logging_utils import get_logger
from ..observability.metrics import (
    graph_failures_total,
    graph_latency_ms,
    graph_requests_total,
)
from .models import GraphIndexRequest, GraphQueryRequest
from .service import GraphService

_LOG = get_logger("graph.api")

_ADAPTERS = {"graphrag", "hypergraphrag", "hyperrag"}


def create_graph_app(
    config: AppConfig,
    *,
    service: GraphService | None = None,
) -> FastAPI:
    service = service or GraphService(config)
    app = FastAPI(title="Autocapture Graph Workers")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "enabled": config.graph_service.enabled}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "enabled": config.graph_service.enabled}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    def _validate_adapter(adapter: str) -> str:
        name = adapter.strip().lower()
        if name not in _ADAPTERS:
            raise HTTPException(status_code=404, detail="adapter_not_found")
        return name

    @app.post("/{adapter}/index")
    def index(adapter: str, payload: GraphIndexRequest) -> Any:
        adapter = _validate_adapter(adapter)
        start = time.monotonic()
        try:
            response = service.index(payload, adapter=adapter)
            graph_requests_total.labels(adapter, "index", "200").inc()
            return JSONResponse(content=response.model_dump())
        except RuntimeError as exc:
            graph_requests_total.labels(adapter, "index", "503").inc()
            graph_failures_total.labels(adapter, "index", "worker_unavailable").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            graph_requests_total.labels(adapter, "index", "400").inc()
            graph_failures_total.labels(adapter, "index", "invalid_request").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            graph_requests_total.labels(adapter, "index", "500").inc()
            graph_failures_total.labels(adapter, "index", "error").inc()
            _LOG.warning("Graph index failed ({}): {}", adapter, exc)
            raise HTTPException(status_code=500, detail="index_failed") from exc
        finally:
            graph_latency_ms.labels(adapter, "index").observe((time.monotonic() - start) * 1000)

    @app.post("/{adapter}/query")
    def query(adapter: str, payload: GraphQueryRequest) -> Any:
        adapter = _validate_adapter(adapter)
        start = time.monotonic()
        try:
            response = service.query(payload, adapter=adapter)
            graph_requests_total.labels(adapter, "query", "200").inc()
            return JSONResponse(content=response.model_dump())
        except RuntimeError as exc:
            graph_requests_total.labels(adapter, "query", "503").inc()
            graph_failures_total.labels(adapter, "query", "worker_unavailable").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            graph_requests_total.labels(adapter, "query", "400").inc()
            graph_failures_total.labels(adapter, "query", "invalid_request").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            graph_requests_total.labels(adapter, "query", "500").inc()
            graph_failures_total.labels(adapter, "query", "error").inc()
            _LOG.warning("Graph query failed ({}): {}", adapter, exc)
            raise HTTPException(status_code=500, detail="query_failed") from exc
        finally:
            graph_latency_ms.labels(adapter, "query").observe((time.monotonic() - start) * 1000)

    return app


__all__ = ["create_graph_app"]
