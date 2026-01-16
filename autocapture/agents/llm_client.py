"""Lightweight LLM client for agent jobs."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx

from ..config import AppConfig
from ..llm.prompt_repetition import apply_prompt_repetition
from ..resilience import RetryPolicy, is_retryable_exception, retry_sync


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class AgentLLMClient:
    def __init__(self, config: AppConfig, *, http_client: httpx.Client | None = None) -> None:
        self._config = config
        self._retry = RetryPolicy(max_retries=config.llm.retries)
        self._http_client = http_client

    def generate_text(self, system_prompt: str, user_prompt: str, context: str) -> LLMResponse:
        provider = self._config.llm.provider
        if provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, context)
        if provider == "openai_compatible":
            return self._generate_openai_compatible(system_prompt, user_prompt, context)
        return self._generate_ollama(system_prompt, user_prompt, context)

    def generate_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
    ) -> LLMResponse:
        provider = self._config.agents.vision.provider
        if provider == "openai_compatible":
            return self._generate_openai_compatible(
                system_prompt,
                user_prompt,
                "",
                image_bytes=image_bytes,
                model=self._config.agents.vision.model,
                base_url=self._config.agents.vision.base_url,
                api_key=self._config.agents.vision.api_key,
            )
        return self._generate_ollama(
            system_prompt,
            user_prompt,
            "",
            image_bytes=image_bytes,
            model=self._config.agents.vision.model,
            base_url=self._config.agents.vision.base_url or self._config.llm.ollama_url,
        )

    def _generate_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        *,
        image_bytes: bytes | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> LLMResponse:
        model = model or self._config.llm.ollama_model
        base_url = (base_url or self._config.llm.ollama_url).rstrip("/")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": context})
        if image_bytes:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            messages[-1]["images"] = [encoded]
        payload = {
            "model": model,
            "messages": apply_prompt_repetition(
                messages, enabled=self._config.llm.prompt_repetition
            ),
            "stream": False,
        }

        def _request() -> LLMResponse:
            if self._http_client is not None:
                response = self._http_client.post(f"{base_url}/api/chat", json=payload)
            else:
                with httpx.Client(timeout=self._config.llm.timeout_s) as client:
                    response = client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return LLMResponse(
                text=data.get("message", {}).get("content", "").strip(),
                provider="ollama",
                model=model,
            )

        return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)

    def _generate_openai(self, system_prompt: str, user_prompt: str, context: str) -> LLMResponse:
        api_key = self._config.llm.openai_api_key
        if not api_key:
            raise RuntimeError("OpenAI API key not configured")
        input_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": context}]},
        ]
        payload = {
            "model": self._config.llm.openai_model,
            "input": apply_prompt_repetition(
                input_messages, enabled=self._config.llm.prompt_repetition
            ),
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        def _request() -> LLMResponse:
            if self._http_client is not None:
                response = self._http_client.post(
                    "https://api.openai.com/v1/responses", json=payload, headers=headers
                )
            else:
                with httpx.Client(timeout=self._config.llm.timeout_s) as client:
                    response = client.post(
                        "https://api.openai.com/v1/responses", json=payload, headers=headers
                    )
            response.raise_for_status()
            data = response.json()
            text = ""
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text += content.get("text", "")
            return LLMResponse(
                text=text.strip(), provider="openai", model=self._config.llm.openai_model
            )

        return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)

    def _generate_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        *,
        image_bytes: bytes | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> LLMResponse:
        base_url = (base_url or self._config.llm.openai_compatible_base_url or "").rstrip("/")
        if not base_url:
            raise RuntimeError("openai_compatible_base_url is not set")
        model = model or self._config.llm.openai_compatible_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": context})
        if image_bytes:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            )
        payload = {
            "model": model,
            "messages": apply_prompt_repetition(
                messages, enabled=self._config.llm.prompt_repetition
            ),
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def _request() -> LLMResponse:
            if self._http_client is not None:
                response = self._http_client.post(
                    f"{base_url}/v1/chat/completions", json=payload, headers=headers
                )
            else:
                with httpx.Client(timeout=self._config.llm.timeout_s) as client:
                    response = client.post(
                        f"{base_url}/v1/chat/completions", json=payload, headers=headers
                    )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return LLMResponse(text=text.strip(), provider="openai_compatible", model=model)

        return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)
