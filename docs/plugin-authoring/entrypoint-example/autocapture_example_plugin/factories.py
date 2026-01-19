"""Example factory implementations for an entry-point plugin."""

from __future__ import annotations

from typing import Any

from autocapture.plugins.sdk import LLMProvider, LLMProviderInfo, PluginContext


class ExampleProvider(LLMProvider):
    async def generate_answer(
        self,
        system_prompt: str,
        query: str,
        context_pack_text: str,
        *,
        temperature: float | None = None,
        priority: str = "foreground",
    ) -> str:
        _ = (system_prompt, context_pack_text, temperature, priority)
        return f"[example] {query}"


def create_example_provider(
    context: PluginContext,
    *,
    stage: str,
    stage_config: Any,
    prompt_strategy: Any,
    governor: Any,
    routing_override: str | None = None,
) -> tuple[LLMProvider, LLMProviderInfo]:
    _ = (prompt_strategy, governor, routing_override)
    model = getattr(stage_config, "model", None) or context.plugin_settings.get(
        "model", "example-model"
    )
    provider = ExampleProvider()
    info = LLMProviderInfo(
        provider_id="example-llm",
        model=str(model),
        base_url=None,
        cloud=False,
    )
    return provider, info


def _health(_context: PluginContext | None = None) -> dict[str, Any]:
    return {"ok": True, "detail": "example plugin ready"}


# Attach optional health check for the plugin manager.
create_example_provider.health = _health
