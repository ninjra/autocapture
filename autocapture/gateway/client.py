"""Gateway client provider for internal routing."""

from __future__ import annotations

from typing import Any

import httpx

from ..llm.providers import LLMProvider, build_user_prompt
from ..llm.prompt_strategy import (
    PromptStrategy,
    PromptStrategyMetadata,
    PromptStrategySettings,
    apply_prompt_strategy,
)
from ..llm.governor import LLMGovernor
from ..logging_utils import get_logger
from ..resilience import CircuitBreaker, RetryPolicy, is_retryable_http_status, retry_async


class GatewayProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        stage: str,
        *,
        api_key: str | None,
        timeout_s: float,
        retries: int,
        prompt_strategy: PromptStrategySettings,
        governor: LLMGovernor | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._stage = stage
        self._api_key = api_key
        self._log = get_logger("llm.gateway")
        self._timeout = timeout_s
        self._retry_policy = RetryPolicy(max_retries=retries)
        self._breaker = CircuitBreaker()
        self._prompt_strategy = prompt_strategy
        self._governor = governor
        self.last_prompt_metadata: PromptStrategyMetadata | None = None
        self.last_prompt_metadata_stage1: PromptStrategyMetadata | None = None

    async def generate_answer(
        self,
        system_prompt: str,
        query: str,
        context_pack_text: str,
        *,
        temperature: float | None = None,
        priority: str = "foreground",
    ) -> str:
        temperature = 0.2 if temperature is None else temperature
        if not self._breaker.allow():
            raise RuntimeError("LLM circuit open")
        user_prompt = build_user_prompt(query, context_pack_text)
        if self._governor:
            async with self._governor.reserve_async(priority):
                return await self._generate_internal(system_prompt, user_prompt, temperature)
        return await self._generate_internal(system_prompt, user_prompt, temperature)

    async def _generate_internal(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        if (
            self._prompt_strategy.step_by_step_two_stage
            and self._prompt_strategy.enable_step_by_step
        ):
            return await self._generate_two_stage(system_prompt, user_prompt, temperature)
        return await self._generate_single_stage(system_prompt, user_prompt, temperature)

    async def _generate_single_stage(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        strategy_result = apply_prompt_strategy(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            self._prompt_strategy,
            task_type="answer",
        )
        self.last_prompt_metadata = strategy_result.metadata
        payload = {
            "messages": strategy_result.messages,
            "temperature": temperature,
        }
        response = await self._request(payload)
        return response

    async def _generate_two_stage(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        from ..llm.providers import _stage1_prompt, _stage2_prompt, _extract_final_answer

        stage1_prompt = _stage1_prompt(user_prompt)
        stage1_result = apply_prompt_strategy(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": stage1_prompt},
            ],
            self._prompt_strategy,
            task_type="answer_stage1",
            step_by_step_requested=True,
        )
        self.last_prompt_metadata_stage1 = stage1_result.metadata
        stage1_payload = {
            "messages": stage1_result.messages,
            "temperature": temperature,
        }
        stage1_text = await self._request(stage1_payload)
        final_answer = _extract_final_answer(stage1_text)
        stage2_prompt = _stage2_prompt(user_prompt, final_answer)
        stage2_result = apply_prompt_strategy(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": stage2_prompt},
            ],
            self._prompt_strategy,
            task_type="answer_stage2",
            step_by_step_requested=False,
            override_strategy=PromptStrategy.BASELINE,
        )
        self.last_prompt_metadata = stage2_result.metadata
        stage2_payload = {
            "messages": stage2_result.messages,
            "temperature": temperature,
        }
        return await self._request(stage2_payload)

    async def _request(self, payload: dict[str, Any]) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async def _do_request() -> str:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/internal/stage/{self._stage}/chat.completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
            return _extract_text(data)

        try:
            result = await retry_async(
                _do_request,
                policy=self._retry_policy,
                is_retryable=_is_retryable_http_error,
            )
            self._breaker.record_success()
            return result
        except Exception as exc:
            self._breaker.record_failure(exc)
            raise


def _is_retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_status(exc.response.status_code)
    return isinstance(exc, httpx.TransportError)


def _extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return str(data)
