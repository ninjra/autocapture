"""FastAPI assembly using the wired container."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .container import AppContainer
from .lifespan import build_lifespan
from .middleware.stack import install_middleware
from .routers.core import build_router
from .security_helpers import build_security_manager
from ..paths import resource_root


def create_app(container: AppContainer) -> FastAPI:
    config = container.config
    app_kwargs: dict[str, object] = {
        "title": "Autocapture Memory Engine",
        "lifespan": build_lifespan(container),
    }
    if config.api.require_api_key and config.mode.mode != "remote":
        app_kwargs["docs_url"] = None
        app_kwargs["redoc_url"] = None
        app_kwargs["openapi_url"] = None
    app = FastAPI(**app_kwargs)
    app.state.container = container
    app.state.db = container.db
    app.state.vector_index = container.vector_index
    app.state.embedder = container.embedder
    app.state.worker_supervisor = container.worker_supervisor
    app.state.memory_store = container.memory_store
    app.state.memory_compiler = container.memory_compiler
    app.state.memory_service_client = container.memory_service_client
    app.state.plugins = container.plugins
    security_manager = build_security_manager(config)
    install_middleware(app, config=config, security_manager=security_manager)
    ui_dir = resource_root() / "autocapture" / "ui" / "web"
    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=ui_dir), name="static")
    app.include_router(build_router(container, security_manager=security_manager))
    return app
