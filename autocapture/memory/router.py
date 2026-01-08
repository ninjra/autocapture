"""Provider routing for pluggable pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LLMConfig, ProviderRoutingConfig, PrivacyConfig
from ..llm.providers import LLMProvider, OpenAIProvider, OllamaProvider


@dataclass(frozen=True)
class RoutingDecision:
    llm_provider: str


class ProviderRouter:
    def __init__(
        self,
        routing: ProviderRoutingConfig,
        llm_config: LLMConfig,
        *,
        offline: bool,
        privacy: PrivacyConfig,
    ) -> None:
        self._routing = routing
        self._llm_config = llm_config
        self._offline = offline
        self._privacy = privacy

    def select_llm(self) -> tuple[LLMProvider, RoutingDecision]:
        if self._offline and not self._privacy.cloud_enabled:
            if self._routing.llm.startswith("openai"):
                raise RuntimeError(
                    "Offline mode enabled; cloud provider blocked. Enable a cloud profile "
                    "(privacy.cloud_enabled=true and offline=false) to allow egress."
                )
        if self._routing.llm.startswith("openai") and self._llm_config.openai_api_key:
            return (
                OpenAIProvider(
                    self._llm_config.openai_api_key,
                    self._llm_config.openai_model,
                    timeout_s=self._llm_config.timeout_s,
                    retries=self._llm_config.retries,
                ),
                RoutingDecision(llm_provider="openai"),
            )
        return (
            OllamaProvider(
                self._llm_config.ollama_url,
                self._llm_config.ollama_model,
                timeout_s=self._llm_config.timeout_s,
                retries=self._llm_config.retries,
            ),
            RoutingDecision(llm_provider="ollama"),
        )
