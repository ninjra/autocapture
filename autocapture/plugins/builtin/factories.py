"""Builtin plugin factories wrapping core implementations."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from ...config import AppConfig, ModelStageConfig, is_loopback_host
from ...llm.providers import LLMProvider, OllamaProvider, OpenAICompatibleProvider, OpenAIProvider
from ...gateway.client import GatewayProvider
from ...llm.prompt_strategy import PromptStrategySettings
from ...llm.governor import LLMGovernor
from ...security.secret_store import SecretStore as EnvSecretStore
from ...vision.extractors import (
    DeepSeekOCRExtractor,
    RapidOCRScreenExtractor,
    VLMExtractor,
)
from ...vision.rapidocr import RapidOCRExtractor
from ...vision.types import ExtractionResult, VISION_SCHEMA_VERSION
from ...embeddings.service import EmbeddingService
from ...memory.reranker import CrossEncoderReranker
from ...memory.compression import CompressedAnswer, extractive_answer
from ...memory.verification import RulesVerifier
from ...gateway.decode import decode_backend_from_settings, extract_backend_settings
from ...memory.graph_adapters import GraphAdapterClient
from ...memory.retrieval import RetrievalService
from ...indexing.vector_index import QdrantBackend, VectorBackend
from ...indexing.spans_v2 import QdrantSpansV2Backend
from ...paths import resource_root
from ..policy import PolicyGate
from ..sdk.context import PluginContext, LLMProviderInfo
from ...training.pipelines import pipeline_from_settings


class DisabledExtractor:
    def extract(self, _image) -> ExtractionResult:
        return ExtractionResult(
            text="",
            spans=[],
            tags={
                "vision_extract": {
                    "schema_version": VISION_SCHEMA_VERSION,
                    "engine": "disabled",
                    "parse_failed": True,
                    "parse_format": "disabled",
                    "reason": "disabled",
                    "regions": [],
                    "visible_text": "",
                    "content_flags": [],
                    "tiles": [],
                }
            },
        )


class DisabledOCREngine:
    def extract(self, _image):
        return []


class ExtractiveCompressor:
    def compress(self, evidence) -> CompressedAnswer:
        return extractive_answer(evidence)


class AbstractiveCompressor:
    def compress(self, evidence) -> CompressedAnswer:
        compressed = extractive_answer(evidence)
        if not compressed.answer:
            return compressed
        # Simple deterministic summarization: truncate and de-duplicate lines.
        seen = set()
        lines = []
        for line in compressed.answer.splitlines():
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
            if len(lines) >= 5:
                break
        summary = " ".join(lines)
        if len(summary) > 800:
            summary = summary[:800].rstrip() + "…"
        return CompressedAnswer(answer=summary, citations=compressed.citations)


class RulesVerifierAdapter(RulesVerifier):
    pass


def create_ollama_provider(
    context: PluginContext,
    *,
    stage: str,
    stage_config: ModelStageConfig,
    prompt_strategy: PromptStrategySettings,
    governor: LLMGovernor | None,
    routing_override: str | None = None,
) -> tuple[LLMProvider, LLMProviderInfo]:
    config: AppConfig = context.config
    base_url = stage_config.base_url or config.llm.ollama_url
    model = stage_config.model or config.llm.ollama_model
    provider = OllamaProvider(
        base_url,
        model,
        keep_alive_s=config.llm.ollama_keep_alive_s,
        timeout_s=config.llm.timeout_s,
        retries=config.llm.retries,
        prompt_strategy=prompt_strategy,
        governor=governor,
    )
    info = LLMProviderInfo(
        provider_id="ollama",
        model=model,
        base_url=base_url,
        cloud=False,
    )
    return provider, info


def create_openai_provider(
    context: PluginContext,
    *,
    stage: str,
    stage_config: ModelStageConfig,
    prompt_strategy: PromptStrategySettings,
    governor: LLMGovernor | None,
    routing_override: str | None = None,
) -> tuple[LLMProvider, LLMProviderInfo]:
    config: AppConfig = context.config
    policy: PolicyGate = context.policy
    secrets = EnvSecretStore()
    api_key = stage_config.api_key or config.llm.openai_api_key
    if not api_key:
        record = secrets.get("OPENAI_API_KEY")
        api_key = record.value if record else None
    if not api_key:
        raise RuntimeError(f"Stage '{stage}' requires OpenAI API key")
    policy.guard_cloud_text(
        stage=stage,
        stage_config=stage_config,
        provider="openai",
        base_url=None,
        cloud=True,
    )
    model = stage_config.model or config.llm.openai_model
    provider = OpenAIProvider(
        api_key,
        model,
        timeout_s=config.llm.timeout_s,
        retries=config.llm.retries,
        prompt_strategy=prompt_strategy,
        governor=governor,
    )
    info = LLMProviderInfo(
        provider_id="openai",
        model=model,
        base_url=None,
        cloud=True,
    )
    return provider, info


def create_openai_compatible_provider(
    context: PluginContext,
    *,
    stage: str,
    stage_config: ModelStageConfig,
    prompt_strategy: PromptStrategySettings,
    governor: LLMGovernor | None,
    routing_override: str | None = None,
) -> tuple[LLMProvider, LLMProviderInfo]:
    config: AppConfig = context.config
    policy: PolicyGate = context.policy
    secrets = EnvSecretStore()
    base_url = stage_config.base_url or config.llm.openai_compatible_base_url
    api_key = stage_config.api_key or config.llm.openai_compatible_api_key
    if not api_key:
        record = secrets.get("OPENAI_COMPATIBLE_API_KEY")
        api_key = record.value if record else None
    if not base_url:
        raise RuntimeError(f"Stage '{stage}' requires openai_compatible base_url")
    cloud = _is_cloud_endpoint(base_url)
    policy.guard_cloud_text(
        stage=stage,
        stage_config=stage_config,
        provider="openai_compatible",
        base_url=base_url,
        cloud=cloud,
    )
    model = stage_config.model or config.llm.openai_compatible_model
    provider = OpenAICompatibleProvider(
        base_url,
        model,
        api_key=api_key,
        timeout_s=config.llm.timeout_s,
        retries=config.llm.retries,
        prompt_strategy=prompt_strategy,
        governor=governor,
    )
    info = LLMProviderInfo(
        provider_id="openai_compatible",
        model=model,
        base_url=base_url,
        cloud=cloud,
    )
    return provider, info


def create_gateway_provider(
    context: PluginContext,
    *,
    stage: str,
    stage_config: ModelStageConfig,
    prompt_strategy: PromptStrategySettings,
    governor: LLMGovernor | None,
    routing_override: str | None = None,
    provider_alias: str | None = None,
) -> tuple[LLMProvider, LLMProviderInfo]:
    config: AppConfig = context.config
    base_url = stage_config.base_url or f"http://{config.gateway.bind_host}:{config.gateway.port}"
    api_key = stage_config.api_key or config.gateway.internal_token or config.gateway.api_key
    provider = GatewayProvider(
        base_url,
        stage,
        api_key=api_key,
        timeout_s=config.llm.timeout_s,
        retries=config.llm.retries,
        prompt_strategy=prompt_strategy,
        governor=governor,
    )
    info = LLMProviderInfo(
        provider_id=provider_alias or "gateway",
        model=stage_config.model or "gateway",
        base_url=base_url,
        cloud=False,
    )
    return provider, info


def create_vlm_extractor(context: PluginContext, **kwargs) -> VLMExtractor:
    config: AppConfig = context.config
    policy: PolicyGate = context.policy
    backend = config.vision_extract.vlm
    provider = backend.provider
    base_url = backend.base_url or config.llm.ollama_url
    policy.guard_cloud_images(
        provider=provider,
        base_url=base_url,
        allow_cloud=backend.allow_cloud,
    )
    return VLMExtractor(config)


def create_deepseek_extractor(context: PluginContext, **kwargs) -> DeepSeekOCRExtractor:
    config: AppConfig = context.config
    policy: PolicyGate = context.policy
    backend = config.vision_extract.deepseek_ocr
    provider = backend.provider
    base_url = backend.base_url or config.llm.ollama_url
    policy.guard_cloud_images(
        provider=provider,
        base_url=base_url,
        allow_cloud=backend.allow_cloud,
    )
    return DeepSeekOCRExtractor(config)


def create_rapidocr_extractor(context: PluginContext, **kwargs) -> RapidOCRScreenExtractor:
    config: AppConfig = context.config
    ocr_engine = kwargs.get("ocr_engine")
    return RapidOCRScreenExtractor(config, ocr_engine=ocr_engine)


def create_disabled_extractor(context: PluginContext, **kwargs) -> DisabledExtractor:
    _ = context
    return DisabledExtractor()


def create_rapidocr_engine(context: PluginContext, **kwargs) -> RapidOCRExtractor:
    config: AppConfig = context.config
    return RapidOCRExtractor(config.ocr)


def create_disabled_ocr_engine(context: PluginContext, **kwargs) -> DisabledOCREngine:
    _ = context
    return DisabledOCREngine()


def create_embedder_local(context: PluginContext, **kwargs) -> EmbeddingService:
    config: AppConfig = context.config
    pause_controller = kwargs.get("pause_controller")
    return EmbeddingService(config.embed, pause_controller=pause_controller)


def create_embedder_disabled(context: PluginContext, **kwargs) -> None:
    _ = context
    return None


def create_reranker_local(context: PluginContext, **kwargs) -> CrossEncoderReranker:
    config: AppConfig = context.config
    return CrossEncoderReranker(config.reranker)


def create_reranker_disabled(context: PluginContext, **kwargs) -> None:
    _ = context
    return None


def create_compressor_extractive(context: PluginContext, **kwargs) -> ExtractiveCompressor:
    _ = context
    return ExtractiveCompressor()


def create_compressor_abstractive(context: PluginContext, **kwargs) -> AbstractiveCompressor:
    _ = context
    return AbstractiveCompressor()


def create_verifier_rules(context: PluginContext, **kwargs) -> RulesVerifierAdapter:
    _ = context
    return RulesVerifierAdapter()


def create_retrieval_local(context: PluginContext, **kwargs) -> RetrievalService:
    config: AppConfig = context.config
    db = kwargs.get("db")
    embedder = kwargs.get("embedder")
    vector_index = kwargs.get("vector_index")
    reranker = kwargs.get("reranker")
    spans_index = kwargs.get("spans_index")
    runtime_governor = kwargs.get("runtime_governor")
    plugin_manager = kwargs.get("plugin_manager")
    return RetrievalService(
        db,
        config,
        embedder=embedder,
        vector_index=vector_index,
        reranker=reranker,
        spans_index=spans_index,
        runtime_governor=runtime_governor,
        plugin_manager=plugin_manager,
    )


def create_graph_adapter_graphrag(context: PluginContext, **kwargs) -> GraphAdapterClient:
    config: AppConfig = context.config
    return GraphAdapterClient("graphrag", config.retrieval.graph_adapters.graphrag)


def create_graph_adapter_hypergraphrag(context: PluginContext, **kwargs) -> GraphAdapterClient:
    config: AppConfig = context.config
    return GraphAdapterClient("hypergraphrag", config.retrieval.graph_adapters.hypergraphrag)


def create_graph_adapter_hyperrag(context: PluginContext, **kwargs) -> GraphAdapterClient:
    config: AppConfig = context.config
    return GraphAdapterClient("hyperrag", config.retrieval.graph_adapters.hyperrag)


def create_vector_backend_qdrant(context: PluginContext, **kwargs) -> VectorBackend | None:
    config: AppConfig = context.config
    dim = kwargs.get("dim")
    if dim is None:
        raise ValueError("Vector backend requires dim")
    return QdrantBackend(config, int(dim))


def create_vector_backend_sqlite(context: PluginContext, **kwargs) -> VectorBackend:
    config: AppConfig = context.config
    dim = kwargs.get("dim")
    if dim is None:
        raise ValueError("Vector backend requires dim")
    from ...indexing.sqlite_backends import SqliteVectorBackend
    from ...storage.database import DatabaseManager

    db = kwargs.get("db") or DatabaseManager(config.database)
    return SqliteVectorBackend(db, dim=int(dim), config=config)


def create_spans_v2_backend_qdrant(context: PluginContext, **kwargs):
    config: AppConfig = context.config
    dim = kwargs.get("dim")
    if dim is None:
        raise ValueError("Spans v2 backend requires dim")
    return QdrantSpansV2Backend(config, int(dim))


def create_spans_v2_backend_sqlite(context: PluginContext, **kwargs):
    config: AppConfig = context.config
    dim = kwargs.get("dim")
    if dim is None:
        raise ValueError("Spans v2 backend requires dim")
    from ...indexing.sqlite_backends import SqliteSpansV2Backend
    from ...storage.database import DatabaseManager

    db = kwargs.get("db") or DatabaseManager(config.database)
    return SqliteSpansV2Backend(db, dim=int(dim), config=config)


def create_table_extractor_stub(context: PluginContext, **kwargs):
    _ = context
    from ...enrichment.table_extractor import StubTableExtractor

    return StubTableExtractor()


def create_prompt_bundle_builtin(context: PluginContext, **kwargs) -> Path:
    _ = context
    return resource_root() / "autocapture" / "prompts" / "derived"


def _decode_backend_from_context(context: PluginContext, backend_id: str):
    settings = extract_backend_settings(context.plugin_settings, backend_id)
    try:
        return decode_backend_from_settings(backend_id, settings)
    except Exception:
        for provider in context.config.model_registry.providers:
            if provider.id == backend_id:
                return provider
        raise RuntimeError(f"Decode backend '{backend_id}' missing configuration")


def create_decode_backend_swift(context: PluginContext, **kwargs):
    return _decode_backend_from_context(context, "swift")


def create_decode_backend_lookahead(context: PluginContext, **kwargs):
    return _decode_backend_from_context(context, "lookahead")


def create_decode_backend_medusa(context: PluginContext, **kwargs):
    return _decode_backend_from_context(context, "medusa")


def _training_settings(context: PluginContext, pipeline_id: str) -> dict:
    settings = context.plugin_settings if isinstance(context.plugin_settings, dict) else {}
    pipelines = settings.get("pipelines") if isinstance(settings, dict) else None
    if isinstance(pipelines, dict):
        scoped = pipelines.get(pipeline_id)
        if isinstance(scoped, dict):
            return scoped
    return settings if isinstance(settings, dict) else {}


def create_training_pipeline_lora(context: PluginContext, **kwargs):
    settings = _training_settings(context, "lora")
    return pipeline_from_settings("lora", settings)


def create_training_pipeline_qlora(context: PluginContext, **kwargs):
    settings = _training_settings(context, "qlora")
    return pipeline_from_settings("qlora", settings)


def create_training_pipeline_dpo(context: PluginContext, **kwargs):
    settings = _training_settings(context, "dpo")
    return pipeline_from_settings("dpo", settings)


def _is_cloud_endpoint(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return bool(host) and not is_loopback_host(host)
