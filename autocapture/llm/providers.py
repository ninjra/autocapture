"""LLM provider integrations for answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx

from ..logging_utils import get_logger


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


class LLMProvider:
    """Base interface for answer generation."""

    async def generate_answer(
        self, system_prompt: str, query: str, context_pack_text: str
    ) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Use a local Ollama instance for answers."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._log = get_logger("llm.ollama")

    async def generate_answer(
        self, system_prompt: str, query: str, context_pack_text: str
    ) -> str:
        try:
            return await self._generate_openai(system_prompt, query, context_pack_text)
        except httpx.HTTPError as exc:
            self._log.warning("Ollama OpenAI endpoint failed: %s", exc)
        except KeyError as exc:
            self._log.warning("Unexpected Ollama OpenAI response: %s", exc)
        return await self._generate_native(system_prompt, query, context_pack_text)

    async def _generate_openai(
        self, system: str, query: str, context_pack_text: str
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "user", "content": context_pack_text},
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions", json=payload
            )
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def _generate_native(
        self, system: str, query: str, context_pack_text: str
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "user", "content": context_pack_text},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return data["message"]["content"].strip()


class OpenAIProvider(LLMProvider):
    """OpenAI Responses API provider."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._log = get_logger("llm.openai")

    async def generate_answer(
        self, system_prompt: str, query: str, context_pack_text: str
    ) -> str:
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
                    "content": [{"type": "text", "text": context_pack_text}],
                },
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        return extract_response_text(data)


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
        "Context snippets (with citations):\n"
        + "\n".join(context_lines)
        + "\n\n"
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
