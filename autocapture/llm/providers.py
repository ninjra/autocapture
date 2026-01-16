"""LLM provider integrations for answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

import json
import time

from ..logging_utils import get_logger
from .prompt_strategy import (
    PromptStrategy,
    PromptStrategyMetadata,
    PromptStrategySettings,
    apply_prompt_strategy,
)
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

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float,
        retries: int,
        prompt_strategy: PromptStrategySettings,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._log = get_logger("llm.ollama")
        self._timeout = timeout_s
        self._retry_policy = RetryPolicy(max_retries=retries)
        self._breaker = CircuitBreaker()
        self._prompt_strategy = prompt_strategy
        self.last_prompt_metadata: PromptStrategyMetadata | None = None
        self.last_prompt_metadata_stage1: PromptStrategyMetadata | None = None

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
        user_prompt = build_user_prompt(query, context_pack_text)
        if self._prompt_strategy.step_by_step_two_stage and self._prompt_strategy.enable_step_by_step:
            return await self._generate_two_stage(system_prompt, user_prompt, temperature)
        return await self._generate_single_stage(system_prompt, user_prompt, temperature)

    async def _generate_single_stage(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        strategy_result = apply_prompt_strategy(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            self._prompt_strategy,
            task_type="answer",
        )
        self.last_prompt_metadata = strategy_result.metadata
        messages = strategy_result.messages
        start = time.monotonic()
        try:
            result, usage = await self._generate_openai(messages, temperature)
            _log_llm_response(
                self._log, strategy_result.metadata, result, usage, _elapsed_ms(start)
            )
            return result
        except Exception as exc:
            self._log.warning("Ollama OpenAI endpoint failed: {}", exc)
        result, usage = await self._generate_native(messages, temperature)
        _log_llm_response(self._log, strategy_result.metadata, result, usage, _elapsed_ms(start))
        return result

    async def _generate_two_stage(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        stage1_prompt = _stage1_prompt(user_prompt)
        stage1_result = apply_prompt_strategy(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": stage1_prompt}],
            self._prompt_strategy,
            task_type="answer_stage1",
            step_by_step_requested=True,
        )
        self.last_prompt_metadata_stage1 = stage1_result.metadata
        start = time.monotonic()
        try:
            stage1_text, usage = await self._generate_openai(stage1_result.messages, temperature)
            _log_llm_response(
                self._log, stage1_result.metadata, stage1_text, usage, _elapsed_ms(start)
            )
        except Exception as exc:
            self._log.warning("Ollama OpenAI endpoint failed: {}", exc)
            stage1_text, usage = await self._generate_native(stage1_result.messages, temperature)
            _log_llm_response(
                self._log, stage1_result.metadata, stage1_text, usage, _elapsed_ms(start)
            )
        final_answer = _extract_final_answer(stage1_text)
        stage2_prompt = _stage2_prompt(user_prompt, final_answer)
        stage2_result = apply_prompt_strategy(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": stage2_prompt}],
            self._prompt_strategy,
            task_type="answer_stage2",
            step_by_step_requested=False,
            override_strategy=PromptStrategy.BASELINE,
        )
        self.last_prompt_metadata = stage2_result.metadata
        start = time.monotonic()
        try:
            stage2_text, usage = await self._generate_openai(stage2_result.messages, temperature)
            _log_llm_response(
                self._log, stage2_result.metadata, stage2_text, usage, _elapsed_ms(start)
            )
            return stage2_text
        except Exception as exc:
            self._log.warning("Ollama OpenAI endpoint failed: {}", exc)
        stage2_text, usage = await self._generate_native(stage2_result.messages, temperature)
        _log_llm_response(
            self._log, stage2_result.metadata, stage2_text, usage, _elapsed_ms(start)
        )
        return stage2_text

    async def _generate_openai(
        self, messages: list[dict[str, Any]], temperature: float
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }

        async def _request() -> str:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/v1/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
            return (
                data["choices"][0]["message"]["content"].strip(),
                data.get("usage", {}),
            )

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
        self, messages: list[dict[str, Any]], temperature: float
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }

        async def _request() -> str:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
            usage = {
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            }
            return data["message"]["content"].strip(), usage

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

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float,
        retries: int,
        prompt_strategy: PromptStrategySettings,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._log = get_logger("llm.openai")
        self._timeout = timeout_s
        self._retry_policy = RetryPolicy(max_retries=retries)
        self._breaker = CircuitBreaker()
        self._prompt_strategy = prompt_strategy
        self.last_prompt_metadata: PromptStrategyMetadata | None = None
        self.last_prompt_metadata_stage1: PromptStrategyMetadata | None = None

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
        user_prompt = build_user_prompt(query, context_pack_text)
        if self._prompt_strategy.step_by_step_two_stage and self._prompt_strategy.enable_step_by_step:
            return await self._generate_two_stage(system_prompt, user_prompt, temperature)
        return await self._generate_single_stage(system_prompt, user_prompt, temperature)

    async def _generate_single_stage(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        input_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        strategy_result = apply_prompt_strategy(
            input_messages, self._prompt_strategy, task_type="answer"
        )
        self.last_prompt_metadata = strategy_result.metadata
        payload = {
            "model": self._model,
            "input": strategy_result.messages,
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

        start = time.monotonic()
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
        result = extract_response_text(data)
        usage = data.get("usage", {})
        _log_llm_response(
            self._log, strategy_result.metadata, result, usage, _elapsed_ms(start)
        )
        return result

    async def _generate_two_stage(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        stage1_prompt = _stage1_prompt(user_prompt)
        input_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": stage1_prompt}]},
        ]
        stage1_result = apply_prompt_strategy(
            input_messages,
            self._prompt_strategy,
            task_type="answer_stage1",
            step_by_step_requested=True,
        )
        self.last_prompt_metadata_stage1 = stage1_result.metadata
        stage1_payload = {
            "model": self._model,
            "input": stage1_result.messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async def _request(payload: dict) -> dict:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/responses",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        start = time.monotonic()
        try:
            data = await retry_async(
                lambda: _request(stage1_payload),
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
        except Exception as exc:
            self._breaker.record_failure(exc)
            self._log.warning("OpenAI request failed: {}", exc)
            raise
        stage1_text = extract_response_text(data)
        _log_llm_response(
            self._log, stage1_result.metadata, stage1_text, data.get("usage", {}), _elapsed_ms(start)
        )
        final_answer = _extract_final_answer(stage1_text)
        stage2_prompt = _stage2_prompt(user_prompt, final_answer)
        stage2_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": stage2_prompt}]},
        ]
        stage2_result = apply_prompt_strategy(
            stage2_messages,
            self._prompt_strategy,
            task_type="answer_stage2",
            step_by_step_requested=False,
            override_strategy=PromptStrategy.BASELINE,
        )
        self.last_prompt_metadata = stage2_result.metadata
        stage2_payload = {
            "model": self._model,
            "input": stage2_result.messages,
            "temperature": temperature,
        }
        start = time.monotonic()
        try:
            data = await retry_async(
                lambda: _request(stage2_payload),
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
        except Exception as exc:
            self._breaker.record_failure(exc)
            self._log.warning("OpenAI request failed: {}", exc)
            raise
        stage2_text = extract_response_text(data)
        _log_llm_response(
            self._log, stage2_result.metadata, stage2_text, data.get("usage", {}), _elapsed_ms(start)
        )
        return stage2_text


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
        prompt_strategy: PromptStrategySettings,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._log = get_logger("llm.openai_compatible")
        self._timeout = timeout_s
        self._retry_policy = RetryPolicy(max_retries=retries)
        self._breaker = CircuitBreaker()
        self._prompt_strategy = prompt_strategy
        self.last_prompt_metadata: PromptStrategyMetadata | None = None
        self.last_prompt_metadata_stage1: PromptStrategyMetadata | None = None

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
        user_prompt = build_user_prompt(query, context_pack_text)
        if self._prompt_strategy.step_by_step_two_stage and self._prompt_strategy.enable_step_by_step:
            return await self._generate_two_stage(system_prompt, user_prompt, temperature)
        return await self._generate_single_stage(system_prompt, user_prompt, temperature)

    async def _generate_single_stage(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        strategy_result = apply_prompt_strategy(messages, self._prompt_strategy, task_type="answer")
        self.last_prompt_metadata = strategy_result.metadata
        payload = {
            "model": self._model,
            "messages": strategy_result.messages,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async def _request(payload: dict) -> dict:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        start = time.monotonic()
        try:
            data = await retry_async(
                lambda: _request(payload),
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
        except Exception as exc:
            self._breaker.record_failure(exc)
            self._log.warning("OpenAI-compatible request failed: {}", exc)
            raise
        result = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        _log_llm_response(
            self._log, strategy_result.metadata, result, usage, _elapsed_ms(start)
        )
        return result

    async def _generate_two_stage(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        stage1_prompt = _stage1_prompt(user_prompt)
        stage1_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": stage1_prompt},
        ]
        stage1_result = apply_prompt_strategy(
            stage1_messages,
            self._prompt_strategy,
            task_type="answer_stage1",
            step_by_step_requested=True,
        )
        self.last_prompt_metadata_stage1 = stage1_result.metadata
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async def _request(payload: dict) -> dict:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        stage1_payload = {
            "model": self._model,
            "messages": stage1_result.messages,
            "temperature": temperature,
        }
        start = time.monotonic()
        try:
            data = await retry_async(
                lambda: _request(stage1_payload),
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
        except Exception as exc:
            self._breaker.record_failure(exc)
            self._log.warning("OpenAI-compatible request failed: {}", exc)
            raise
        stage1_text = data["choices"][0]["message"]["content"].strip()
        _log_llm_response(
            self._log, stage1_result.metadata, stage1_text, data.get("usage", {}), _elapsed_ms(start)
        )
        final_answer = _extract_final_answer(stage1_text)
        stage2_prompt = _stage2_prompt(user_prompt, final_answer)
        stage2_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": stage2_prompt},
        ]
        stage2_result = apply_prompt_strategy(
            stage2_messages,
            self._prompt_strategy,
            task_type="answer_stage2",
            step_by_step_requested=False,
            override_strategy=PromptStrategy.BASELINE,
        )
        self.last_prompt_metadata = stage2_result.metadata
        stage2_payload = {
            "model": self._model,
            "messages": stage2_result.messages,
            "temperature": temperature,
        }
        start = time.monotonic()
        try:
            data = await retry_async(
                lambda: _request(stage2_payload),
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
        except Exception as exc:
            self._breaker.record_failure(exc)
            self._log.warning("OpenAI-compatible request failed: {}", exc)
            raise
        stage2_text = data["choices"][0]["message"]["content"].strip()
        _log_llm_response(
            self._log, stage2_result.metadata, stage2_text, data.get("usage", {}), _elapsed_ms(start)
        )
        return stage2_text


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


def build_user_prompt(query: str, context_pack_text: str) -> str:
    if context_pack_text:
        return f"{query}\n\n{_format_evidence_message(context_pack_text)}"
    return query


def _stage1_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "Reason step by step to derive the answer. Then on a new line write "
        "'Final: <your answer>'."
    )


def _stage2_prompt(user_prompt: str, final_answer: str) -> str:
    return (
        f"{user_prompt}\n\n"
        f"Draft answer: {final_answer}\n\n"
        "Respond with the final answer only. Do not include reasoning."
    )


def _extract_final_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("final:"):
            return stripped.split(":", 1)[1].strip()
        if lowered.startswith("answer:"):
            return stripped.split(":", 1)[1].strip()
        return stripped
    return text.strip()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _log_llm_response(
    log,
    metadata: PromptStrategyMetadata,
    response_text: str,
    usage: dict[str, Any],
    latency_ms: float,
) -> None:
    response_tokens_estimate = _estimate_tokens(response_text)
    payload = {
        "event": "prompt_strategy.response",
        "strategy": metadata.strategy.value,
        "repeat_factor": metadata.repeat_factor,
        "step_by_step_used": metadata.step_by_step_used,
        "prompt_tokens_estimate": metadata.prompt_tokens_estimate,
        "response_tokens_estimate": response_tokens_estimate,
        "prompt_hash": metadata.prompt_hash_after,
        "usage": usage,
        "task_type": metadata.task_type,
        "latency_ms": latency_ms,
    }
    log.info(json.dumps(payload))


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000


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
