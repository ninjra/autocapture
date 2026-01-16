"""Provider routing for pluggable pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LLMConfig, ProviderRoutingConfig, PrivacyConfig
from ..llm.providers import LLMProvider, OpenAICompatibleProvider, OpenAIProvider, OllamaProvider
from ..llm.prompt_strategy import PromptStrategySettings


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
        prompt_strategy: PromptStrategySettings | None = None,
    ) -> None:
        self._routing = routing
        self._llm_config = llm_config
        self._offline = offline
        self._privacy = privacy
        self._prompt_strategy = prompt_strategy or PromptStrategySettings.from_llm_config(
            llm_config
        )

    def select_llm(self) -> tuple[LLMProvider, RoutingDecision]:
        if self._routing.llm == "openai" and not self._privacy.cloud_enabled:
            raise RuntimeError(
                "Cloud provider blocked. Enable a cloud profile (privacy.cloud_enabled=true) "
                "to allow OpenAI usage."
            )
        if self._routing.llm == "openai_compatible":
            base_url = self._llm_config.openai_compatible_base_url
            if not base_url:
                raise RuntimeError("openai_compatible_base_url is required for openai_compatible")
            return (
                OpenAICompatibleProvider(
                    base_url,
                    self._llm_config.openai_compatible_model,
                    api_key=self._llm_config.openai_compatible_api_key,
                    timeout_s=self._llm_config.timeout_s,
                    retries=self._llm_config.retries,
                    prompt_strategy=self._prompt_strategy,
                ),
                RoutingDecision(llm_provider="openai_compatible"),
            )
        if self._routing.llm.startswith("openai") and self._llm_config.openai_api_key:
            return (
                OpenAIProvider(
                    self._llm_config.openai_api_key,
                    self._llm_config.openai_model,
                    timeout_s=self._llm_config.timeout_s,
                    retries=self._llm_config.retries,
                    prompt_strategy=self._prompt_strategy,
                ),
                RoutingDecision(llm_provider="openai"),
            )
        return (
            OllamaProvider(
                self._llm_config.ollama_url,
                self._llm_config.ollama_model,
                timeout_s=self._llm_config.timeout_s,
                retries=self._llm_config.retries,
                prompt_strategy=self._prompt_strategy,
            ),
            RoutingDecision(llm_provider="ollama"),
        )
