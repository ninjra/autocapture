from __future__ import annotations

from loguru import logger

from autocapture.api.container import build_container
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.embeddings.service import EmbeddingService
from autocapture.plugins import PluginManager
from autocapture.storage.database import DatabaseManager


def test_routing_resolves_sqlite_backends_and_logs() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:"))
    config.embed.text_model = "local-test"
    config.routing.vector_backend = "local"
    config.routing.spans_v2_backend = "local"
    config.routing.table_extractor = "disabled"
    config.qdrant.enabled = True
    db = DatabaseManager(config.database)
    plugins = PluginManager(config)
    embedder = EmbeddingService(config.embed)
    messages: list[str] = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), level="INFO")
    try:
        _ = build_container(
            config,
            db_manager=db,
            plugin_manager=plugins,
            embedder=embedder,
        )
    finally:
        logger.remove(handler_id)
    log_text = "".join(messages)
    assert "Routing: vector_backend=local" in log_text
    assert "Resolved vector.backend 'local' -> autocapture.builtin.vector_sqlite" in log_text
    assert "Resolved spans_v2.backend 'local' -> autocapture.builtin.spans_v2_sqlite" in log_text


def test_plugin_manager_reuses_backend_instances() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:"))
    config.embed.text_model = "local-test"
    plugins = PluginManager(config)
    db = DatabaseManager(config.database)
    backend_a = plugins.resolve_extension(
        "vector.backend",
        "local",
        factory_kwargs={"dim": 16, "db": db},
    )
    backend_b = plugins.resolve_extension(
        "vector.backend",
        "local",
        factory_kwargs={"dim": 16, "db": db},
    )
    assert backend_a is backend_b
