"""LLM provider integrations for answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx

from ..logging_utils import get_logger
from ..resilience import (
    CircuitBreaker,
    RetryPolicy,
    is_retryable_exception,
    is_retryable_http_status,
    retry_async,
)


@dataclass(frozen=True)
class Citation:
    segment_id: str
    ts_range: str
    snippets: list[str]
    thumbs: list[str]


@dataclass(frozen=True)
class LLMAnswer:
    answer: str
    citations: list[Citation]


@dataclass(frozen=True)
class ContextChunk:
    segment_id: str
    ts_range: str
    snippet: str


class LLMProvider(ABC):
    """Base interface for answer generation."""

    @abstractmethod
    async def generate_answer(
        self,
        system_prompt: str,
        query: str,
        context_pack_text: str,
        *,
        temperature: float | None = None,
    ) -> str:
        """Generate an answer string for the provided prompt and context."""


def _format_evidence_message(context_pack_text: str) -> str:
    return f"EVIDENCE_JSON:\n```json\n{context_pack_text}\n```"


def _is_retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_status(exc.response.status_code)
    return is_retryable_exception(exc)


class OllamaProvider(LLMProvider):
    """Use a local Ollama instance for answers."""

    def __init__(self, base_url: str, model: str, *, timeout_s: float, retries: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._log = get_logger("llm.ollama")
        self._timeout = timeout_s
        self._retry_policy = RetryPolicy(max_retries=retries)
        self._breaker = CircuitBreaker()

    async def generate_answer(
        self,
        system_prompt: str,
        query: str,
        context_pack_text: str,
        *,
        temperature: float | None = None,
    ) -> str:
        temperature = 0.2 if temperature is None else temperature
        if not self._breaker.allow():
            raise RuntimeError("LLM circuit open")
        try:
            return await self._generate_openai(system_prompt, query, context_pack_text, temperature)
        except Exception as exc:
            self._log.warning("Ollama OpenAI endpoint failed: {}", exc)
        return await self._generate_native(system_prompt, query, context_pack_text, temperature)

    async def _generate_openai(
        self, system: str, query: str, context_pack_text: str, temperature: float
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {
                    "role": "user",
                    "content": _format_evidence_message(context_pack_text),
                },
            ],
            "temperature": temperature,
        }

        async def _request() -> str:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/v1/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        try:
            result = await retry_async(
                _request,
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
            return result
        except Exception as exc:
            self._breaker.record_failure(exc)
            raise

    async def _generate_native(
        self, system: str, query: str, context_pack_text: str, temperature: float
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {
                    "role": "user",
                    "content": _format_evidence_message(context_pack_text),
                },
            ],
            "stream": False,
            "temperature": temperature,
        }

        async def _request() -> str:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
            return data["message"]["content"].strip()

        try:
            result = await retry_async(
                _request,
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
            return result
        except Exception as exc:
            self._breaker.record_failure(exc)
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI Responses API provider."""

    def __init__(self, api_key: str, model: str, *, timeout_s: float, retries: int) -> None:
        self._api_key = api_key
        self._model = model
        self._log = get_logger("llm.openai")
        self._timeout = timeout_s
        self._retry_policy = RetryPolicy(max_retries=retries)
        self._breaker = CircuitBreaker()

    async def generate_answer(
        self,
        system_prompt: str,
        query: str,
        context_pack_text: str,
        *,
        temperature: float | None = None,
    ) -> str:
        temperature = 0.2 if temperature is None else temperature
        if not self._breaker.allow():
            raise RuntimeError("LLM circuit open")
        payload = {
            "model": self._model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _format_evidence_message(context_pack_text),
                        }
                    ],
                },
            ],
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async def _request() -> dict:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/responses",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        try:
            data = await retry_async(
                _request,
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
        except Exception as exc:
            self._breaker.record_failure(exc)
            self._log.warning("OpenAI request failed: {}", exc)
            raise
        return extract_response_text(data)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible chat/completions provider (llama.cpp, Open WebUI)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None,
        timeout_s: float,
        retries: int,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._log = get_logger("llm.openai_compatible")
        self._timeout = timeout_s
        self._retry_policy = RetryPolicy(max_retries=retries)
        self._breaker = CircuitBreaker()

    async def generate_answer(
        self,
        system_prompt: str,
        query: str,
        context_pack_text: str,
        *,
        temperature: float | None = None,
    ) -> str:
        temperature = 0.2 if temperature is None else temperature
        if not self._breaker.allow():
            raise RuntimeError("LLM circuit open")
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "user", "content": _format_evidence_message(context_pack_text)},
            ],
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async def _request() -> dict:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        try:
            data = await retry_async(
                _request,
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
        except Exception as exc:
            self._breaker.record_failure(exc)
            self._log.warning("OpenAI-compatible request failed: {}", exc)
            raise
        return data["choices"][0]["message"]["content"].strip()


def build_prompt(query: str, context: Sequence[ContextChunk]) -> tuple[str, str]:
    system_prompt = (
        "You are an assistant answering questions about a personal activity archive. "
        "Always respond in natural language and include citations by segment id and "
        "timestamp range. If you are uncertain, say so and ask the user to click a citation."
    )
    context_lines = [
        f"[{chunk.segment_id} | {chunk.ts_range}] {chunk.snippet}" for chunk in context
    ]
    user_prompt = (
        "Question:\n"
        f"{query}\n\n"
        "Context snippets (with citations):\n" + "\n".join(context_lines) + "\n\n"
        "Answer with inline citations like [segment_id @ 2024-01-01T12:00:00Z]."
    )
    return system_prompt, user_prompt


def extract_response_text(payload: dict) -> str:
    if "output_text" in payload and payload["output_text"]:
        return str(payload["output_text"]).strip()
    output = payload.get("output", [])
    parts: list[str] = []
    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                parts.append(str(text))
    if parts:
        return "\n".join(parts).strip()
    raise KeyError("No output text in OpenAI response")


def build_citations(context: Iterable[ContextChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in context:
        citations.append(
            Citation(
                segment_id=chunk.segment_id,
                ts_range=chunk.ts_range,
                snippets=[chunk.snippet],
                thumbs=[],
            )
        )
    return citations
