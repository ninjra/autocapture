"""Vision model client helpers."""

from __future__ import annotations

import base64
import json
import time

import httpx

from ..logging_utils import get_logger
from ..llm.prompt_strategy import PromptStrategySettings, apply_prompt_strategy
from ..llm.governor import LLMGovernor
from ..resilience import RetryPolicy, is_retryable_exception, retry_sync


class VisionClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str | None,
        api_key: str | None,
        timeout_s: float,
        retries: int,
        prompt_strategy: PromptStrategySettings,
        http_client: httpx.Client | None = None,
        governor: LLMGovernor | None = None,
        priority: str = "background",
    ) -> None:
        self._provider = provider
        self._model = model
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._retry = RetryPolicy(max_retries=retries)
        self._log = get_logger("vision.client")
        self._prompt_strategy = prompt_strategy
        self._http_client = http_client
        self._governor = governor
        self._priority = priority

    def generate(self, system_prompt: str, user_prompt: str, images: list[bytes]) -> str:
        if self._governor:
            with self._governor.reserve(self._priority):
                return self._generate(system_prompt, user_prompt, images)
        return self._generate(system_prompt, user_prompt, images)

    def _generate(self, system_prompt: str, user_prompt: str, images: list[bytes]) -> str:
        if self._provider == "ollama":
            return self._generate_ollama(system_prompt, user_prompt, images)
        if self._provider == "openai_compatible":
            return self._generate_openai_compatible(system_prompt, user_prompt, images)
        if self._provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, images)
        raise RuntimeError(f"Unsupported vision provider: {self._provider}")

    def _generate_ollama(self, system_prompt: str, user_prompt: str, images: list[bytes]) -> str:
        base_url = self._base_url or "http://127.0.0.1:11434"
        messages = [{"role": "system", "content": system_prompt}]
        if images:
            encoded = [base64.b64encode(image).decode("utf-8") for image in images]
            messages.append({"role": "user", "content": user_prompt, "images": encoded})
        else:
            messages.append({"role": "user", "content": user_prompt})
        strategy_result = apply_prompt_strategy(
            messages, self._prompt_strategy, task_type="vision", step_by_step_requested=False
        )
        payload = {"model": self._model, "messages": strategy_result.messages, "stream": False}
        start = time.monotonic()

        def _request() -> str:
            if self._http_client is not None:
                response = self._http_client.post(f"{base_url}/api/chat", json=payload)
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("message", {}).get("content", "").strip()
            _log_vision_response(self._log, strategy_result.metadata, result, start)
            return result

        return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)

    def _generate_openai_compatible(
        self, system_prompt: str, user_prompt: str, images: list[bytes]
    ) -> str:
        if not self._base_url:
            raise RuntimeError("openai_compatible base_url is required for vision extraction")
        content = [{"type": "text", "text": user_prompt}]
        for image in images:
            encoded = base64.b64encode(image).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        strategy_result = apply_prompt_strategy(
            messages, self._prompt_strategy, task_type="vision", step_by_step_requested=False
        )
        payload = {
            "model": self._model,
            "messages": strategy_result.messages,
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        start = time.monotonic()

        def _request() -> str:
            if self._http_client is not None:
                response = self._http_client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        f"{self._base_url}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"].strip()
            _log_vision_response(self._log, strategy_result.metadata, result, start)
            return result

        return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)

    def _generate_openai(self, system_prompt: str, user_prompt: str, images: list[bytes]) -> str:
        base_url = self._base_url or "https://api.openai.com"
        if not self._api_key:
            raise RuntimeError("OpenAI API key not configured for vision extraction")
        content = [{"type": "text", "text": user_prompt}]
        for image in images:
            encoded = base64.b64encode(image).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        strategy_result = apply_prompt_strategy(
            messages, self._prompt_strategy, task_type="vision", step_by_step_requested=False
        )
        payload = {
            "model": self._model,
            "messages": strategy_result.messages,
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        start = time.monotonic()

        def _request() -> str:
            if self._http_client is not None:
                response = self._http_client.post(
                    f"{base_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        f"{base_url.rstrip('/')}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"].strip()
            _log_vision_response(self._log, strategy_result.metadata, result, start)
            return result

        try:
            return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)
        except Exception as exc:
            self._log.warning("OpenAI vision request failed: {}", exc)
            raise


def _log_vision_response(log, metadata, text: str, start: float) -> None:
    response_tokens_estimate = max(1, len(text) // 4) if text else 0
    payload = {
        "event": "prompt_strategy.response",
        "strategy": metadata.strategy.value,
        "repeat_factor": metadata.repeat_factor,
        "step_by_step_used": metadata.step_by_step_used,
        "prompt_tokens_estimate": metadata.prompt_tokens_estimate,
        "response_tokens_estimate": response_tokens_estimate,
        "prompt_hash": metadata.prompt_hash_after,
        "task_type": metadata.task_type,
        "latency_ms": (time.monotonic() - start) * 1000,
    }
    log.info(json.dumps(payload))
