"""Provider routing for pluggable pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass

from ..logging_utils import get_logger
from ..security.secret_store import SecretStore as EnvSecretStore

from ..config import AppConfig, LLMConfig, ProviderRoutingConfig, PrivacyConfig
from ..llm.providers import LLMProvider, OpenAICompatibleProvider, OpenAIProvider, OllamaProvider
from ..llm.governor import get_global_governor
from ..llm.prompt_strategy import PromptStrategySettings


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_text: bool = False
    supports_embeddings: bool = False
    supports_rerank: bool = False
    supports_ocr: bool = False
    cloud: bool = False


@dataclass(frozen=True)
class ProviderSelection:
    kind: str
    provider_id: str
    capabilities: ProviderCapabilities


@dataclass(frozen=True)
class RoutingDecision:
    llm_provider: str


class ProviderRouter:
    def __init__(
        self,
        routing: ProviderRoutingConfig,
        llm_config: LLMConfig,
        *,
        config: AppConfig | None = None,
        offline: bool,
        privacy: PrivacyConfig,
        prompt_strategy: PromptStrategySettings | None = None,
    ) -> None:
        self._config = config
        self._routing = routing
        self._llm_config = llm_config
        self._offline = offline
        self._privacy = privacy
        self._prompt_strategy = prompt_strategy or PromptStrategySettings.from_llm_config(
            llm_config
        )
        self._governor = get_global_governor(config) if config else None
        self._log = get_logger("provider_router")
        self._secrets = EnvSecretStore()

    def select_llm(self) -> tuple[LLMProvider, RoutingDecision]:
        if self._routing.llm == "openai" and not self._privacy.cloud_enabled and not self._offline:
            raise RuntimeError(
                "Cloud provider blocked. Enable a cloud profile (privacy.cloud_enabled=true) "
                "to allow OpenAI usage."
            )
        if self._routing.llm == "openai_compatible":
            base_url = self._llm_config.openai_compatible_base_url
            if not base_url:
                raise RuntimeError("openai_compatible_base_url is required for openai_compatible")
            api_key = self._llm_config.openai_compatible_api_key
            if not api_key:
                record = self._secrets.get("OPENAI_COMPATIBLE_API_KEY")
                api_key = record.value if record else None
            selection = RoutingDecision(llm_provider="openai_compatible")
            self._log.info("Provider selection: kind=llm provider={}", selection.llm_provider)
            return (
                OpenAICompatibleProvider(
                    base_url,
                    self._llm_config.openai_compatible_model,
                    api_key=api_key,
                    timeout_s=self._llm_config.timeout_s,
                    retries=self._llm_config.retries,
                    prompt_strategy=self._prompt_strategy,
                    governor=self._governor,
                ),
                selection,
            )
        if self._routing.llm.startswith("openai") and self._llm_config.openai_api_key:
            selection = RoutingDecision(llm_provider="openai")
            self._log.info("Provider selection: kind=llm provider={}", selection.llm_provider)
            return (
                OpenAIProvider(
                    self._llm_config.openai_api_key,
                    self._llm_config.openai_model,
                    timeout_s=self._llm_config.timeout_s,
                    retries=self._llm_config.retries,
                    prompt_strategy=self._prompt_strategy,
                    governor=self._governor,
                ),
                selection,
            )
        if self._routing.llm.startswith("openai") and not self._llm_config.openai_api_key:
            record = self._secrets.get("OPENAI_API_KEY")
            if record:
                selection = RoutingDecision(llm_provider="openai")
                self._log.info("Provider selection: kind=llm provider={}", selection.llm_provider)
                return (
                    OpenAIProvider(
                        record.value,
                        self._llm_config.openai_model,
                        timeout_s=self._llm_config.timeout_s,
                        retries=self._llm_config.retries,
                        prompt_strategy=self._prompt_strategy,
                        governor=self._governor,
                    ),
                    selection,
                )
        selection = RoutingDecision(llm_provider="ollama")
        self._log.info("Provider selection: kind=llm provider={}", selection.llm_provider)
        return (
            OllamaProvider(
                self._llm_config.ollama_url,
                self._llm_config.ollama_model,
                timeout_s=self._llm_config.timeout_s,
                retries=self._llm_config.retries,
                prompt_strategy=self._prompt_strategy,
                governor=self._governor,
            ),
            selection,
        )

    def select_embedding(self) -> ProviderSelection:
        provider_id = (self._routing.embedding or "local").strip().lower()
        caps = ProviderCapabilities(supports_embeddings=True, cloud=False)
        selection = ProviderSelection(kind="embedding", provider_id=provider_id, capabilities=caps)
        self._log.info(
            "Provider selection: kind=embedding provider={} cloud={}",
            selection.provider_id,
            selection.capabilities.cloud,
        )
        return selection

    def select_reranker(self) -> ProviderSelection:
        provider_id = (self._routing.reranker or "disabled").strip().lower()
        caps = ProviderCapabilities(supports_rerank=True, cloud=False)
        selection = ProviderSelection(kind="reranker", provider_id=provider_id, capabilities=caps)
        self._log.info(
            "Provider selection: kind=reranker provider={} cloud={}",
            selection.provider_id,
            selection.capabilities.cloud,
        )
        return selection

    def select_ocr(self) -> ProviderSelection:
        provider_id = (self._routing.ocr or "local").strip().lower()
        caps = ProviderCapabilities(supports_ocr=True, cloud=False)
        selection = ProviderSelection(kind="ocr", provider_id=provider_id, capabilities=caps)
        self._log.info(
            "Provider selection: kind=ocr provider={} cloud={}",
            selection.provider_id,
            selection.capabilities.cloud,
        )
        return selection
