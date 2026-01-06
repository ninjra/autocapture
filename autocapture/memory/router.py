"""Provider routing for pluggable pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LLMConfig, ProviderRoutingConfig
from ..llm.providers import LLMProvider, OpenAIProvider, OllamaProvider


@dataclass(frozen=True)
class RoutingDecision:
    llm_provider: str


class ProviderRouter:
    def __init__(
        self,
        routing: ProviderRoutingConfig,
        llm_config: LLMConfig,
    ) -> None:
        self._routing = routing
        self._llm_config = llm_config

    def select_llm(self) -> tuple[LLMProvider, RoutingDecision]:
        if self._routing.llm.startswith("openai") and self._llm_config.openai_api_key:
            return (
                OpenAIProvider(
                    self._llm_config.openai_api_key, self._llm_config.openai_model
                ),
                RoutingDecision(llm_provider="openai"),
            )
        return (
            OllamaProvider(self._llm_config.ollama_url, self._llm_config.ollama_model),
            RoutingDecision(llm_provider="ollama"),
        )
