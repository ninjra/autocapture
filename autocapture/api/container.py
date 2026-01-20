"""API composition root for service wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..agents.answer_graph import AnswerGraph
from ..agents.jobs import AgentJobQueue
from ..config import AppConfig
from ..embeddings.service import EmbeddingService
from ..encryption import EncryptionManager
from ..indexing.pruner import IndexPruner
from ..indexing.vector_index import VectorIndex
from ..logging_utils import get_logger
from ..media.store import MediaStore
from ..memory.compiler import ContextCompiler
from ..memory.entities import EntityResolver, SecretStore
from ..memory.prompts import PromptLibraryService, PromptRegistry
from ..memory.retrieval import RetrievalService
from ..memory.store import MemoryStore
from ..memory.threads import ThreadRetrievalService
from ..memory_service.client import MemoryServiceClient
from ..observability.otel import init_otel
from ..policy import PolicyEnvelope
from ..plugins import PluginManager
from ..security.token_vault import TokenVaultStore
from ..storage.database import DatabaseManager
from ..storage.retention import RetentionManager


@dataclass(frozen=True)
class AppContainer:
    config: AppConfig
    db: DatabaseManager
    db_owned: bool
    plugins: PluginManager
    embedder: EmbeddingService
    vector_index: VectorIndex
    reranker: object | None
    retrieval: RetrievalService
    thread_retrieval: ThreadRetrievalService
    encryption_mgr: EncryptionManager
    secret_store: bytes
    token_vault: TokenVaultStore
    entities: EntityResolver
    agent_jobs: AgentJobQueue
    prompt_registry: PromptRegistry
    memory_service_client: MemoryServiceClient | None
    policy_envelope: PolicyEnvelope
    answer_graph: AnswerGraph
    retention: RetentionManager
    index_pruner: IndexPruner
    media_store: MediaStore
    memory_store: MemoryStore | None
    memory_compiler: ContextCompiler | None
    worker_supervisor: object | None


def build_container(
    config: AppConfig,
    db_manager: DatabaseManager | None = None,
    *,
    embedder: EmbeddingService | None = None,
    vector_index: VectorIndex | None = None,
    worker_supervisor: object | None = None,
    plugin_manager: PluginManager | None = None,
) -> AppContainer:
    init_otel(config.features.enable_otel)
    log = get_logger("api")
    db_owned = db_manager is None
    db = db_manager or DatabaseManager(config.database)
    plugins = plugin_manager or PluginManager(config)
    embedder_obj = embedder
    if embedder_obj is None:
        embedder_id = (config.routing.embedding or "local").strip().lower()
        try:
            embedder_obj = plugins.resolve_extension("embedder.text", embedder_id)
        except Exception as exc:
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
            log.warning("Vector backend plugin failed: {}", exc)
            backend = None
        vector_index = VectorIndex(config, dim, backend=backend)
    reranker = None
    reranker_id = (config.routing.reranker or "disabled").strip().lower()
    try:
        reranker = plugins.resolve_extension("reranker", reranker_id)
    except Exception as exc:
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
        embedder=getattr(retrieval, "embedder", None),
        vector_index=getattr(retrieval, "vector_index", None),
    )
    encryption_mgr = EncryptionManager(config.encryption)
    secret_store = SecretStore(Path(config.capture.data_dir)).get_or_create()
    token_vault = TokenVaultStore(config, db)
    entities = EntityResolver(db, secret_store, token_vault=token_vault)
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
    policy_envelope = PolicyEnvelope(config)
    answer_graph = AnswerGraph(
        config,
        retrieval,
        db=db,
        thread_retrieval=thread_retrieval,
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
        vector_index=getattr(retrieval, "vector_index", None),
        spans_index=getattr(retrieval, "spans_index", None),
    )
    media_store = MediaStore(config.capture, config.encryption)
    memory_store: MemoryStore | None = None
    memory_compiler: ContextCompiler | None = None
    if config.memory.enabled:
        try:
            memory_store = MemoryStore(config.memory)
            memory_compiler = ContextCompiler(memory_store, config.memory)
        except Exception as exc:
            log.warning("Memory store init failed: {}", exc)
    return AppContainer(
        config=config,
        db=db,
        db_owned=db_owned,
        plugins=plugins,
        embedder=embedder_obj,
        vector_index=vector_index,
        reranker=reranker,
        retrieval=retrieval,
        thread_retrieval=thread_retrieval,
        encryption_mgr=encryption_mgr,
        secret_store=secret_store,
        token_vault=token_vault,
        entities=entities,
        agent_jobs=agent_jobs,
        prompt_registry=prompt_registry,
        memory_service_client=memory_service_client,
        policy_envelope=policy_envelope,
        answer_graph=answer_graph,
        retention=retention,
        index_pruner=index_pruner,
        media_store=media_store,
        memory_store=memory_store,
        memory_compiler=memory_compiler,
        worker_supervisor=worker_supervisor,
    )
