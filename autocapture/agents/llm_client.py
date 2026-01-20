"""Lightweight LLM client for agent jobs."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx

from ..config import AppConfig
import json
import time

from ..llm.prompt_strategy import PromptStrategySettings, apply_prompt_strategy
from ..llm.governor import LLMGovernor, get_global_governor
from ..logging_utils import get_logger
from ..resilience import RetryPolicy, is_retryable_exception, retry_sync
from ..policy import PolicyEnvelope


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class AgentLLMClient:
    def __init__(
        self,
        config: AppConfig,
        *,
        http_client: httpx.Client | None = None,
        governor: LLMGovernor | None = None,
    ) -> None:
        self._config = config
        self._retry = RetryPolicy(max_retries=config.llm.retries)
        self._http_client = http_client
        self._prompt_strategy = PromptStrategySettings.from_llm_config(
            config.llm, data_dir=config.capture.data_dir
        )
        self._log = get_logger("agent.llm_client")
        self._governor = governor or get_global_governor(config)
        self._policy = PolicyEnvelope(config)

    def generate_text(self, system_prompt: str, user_prompt: str, context: str) -> LLMResponse:
        provider = self._config.llm.provider
        base_url = None
        if provider == "openai_compatible":
            base_url = self._config.llm.openai_compatible_base_url
        elif provider == "ollama":
            base_url = self._config.llm.ollama_url
        cloud = self._policy.infer_cloud_from_endpoint(base_url, provider)
        if provider == "openai":
            return self._policy.execute_call_sync(
                stage=None,
                call=lambda: self._generate_openai(system_prompt, user_prompt, context),
                cloud=cloud,
            )
        if provider == "openai_compatible":
            return self._policy.execute_call_sync(
                stage=None,
                call=lambda: self._generate_openai_compatible(system_prompt, user_prompt, context),
                cloud=cloud,
            )
        return self._policy.execute_call_sync(
            stage=None,
            call=lambda: self._generate_ollama(system_prompt, user_prompt, context),
            cloud=cloud,
        )

    def generate_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
    ) -> LLMResponse:
        provider = self._config.agents.vision.provider
        base_url = self._config.agents.vision.base_url
        if provider != "openai_compatible":
            base_url = base_url or self._config.llm.ollama_url
        cloud = self._policy.infer_cloud_from_endpoint(base_url, provider)
        if provider == "openai_compatible":
            return self._policy.execute_vision_sync(
                stage=None,
                call=lambda: self._generate_openai_compatible(
                    system_prompt,
                    user_prompt,
                    "",
                    image_bytes=image_bytes,
                    model=self._config.agents.vision.model,
                    base_url=self._config.agents.vision.base_url,
                    api_key=self._config.agents.vision.api_key,
                ),
                cloud=cloud,
            )
        return self._policy.execute_vision_sync(
            stage=None,
            call=lambda: self._generate_ollama(
                system_prompt,
                user_prompt,
                "",
                image_bytes=image_bytes,
                model=self._config.agents.vision.model,
                base_url=base_url,
            ),
            cloud=cloud,
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
        combined_prompt = user_prompt
        if context:
            combined_prompt = f"{user_prompt}\n\n{context}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_prompt},
        ]
        if image_bytes:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            messages[-1]["images"] = [encoded]
        strategy_result = apply_prompt_strategy(messages, self._prompt_strategy, task_type="agent")
        payload = {"model": model, "messages": strategy_result.messages, "stream": False}
        start = time.monotonic()

        def _request() -> LLMResponse:
            with self._governor.reserve("background"):
                if self._http_client is not None:
                    response = self._http_client.post(f"{base_url}/api/chat", json=payload)
                else:
                    with httpx.Client(timeout=self._config.llm.timeout_s) as client:
                        response = client.post(f"{base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                response_obj = LLMResponse(
                    text=data.get("message", {}).get("content", "").strip(),
                    provider="ollama",
                    model=model,
                )
                _log_agent_response(self._log, strategy_result.metadata, response_obj.text, start)
                return response_obj

        return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)

    def _generate_openai(self, system_prompt: str, user_prompt: str, context: str) -> LLMResponse:
        api_key = self._config.llm.openai_api_key
        if not api_key:
            raise RuntimeError("OpenAI API key not configured")
        combined_prompt = user_prompt
        if context:
            combined_prompt = f"{user_prompt}\n\n{context}"
        input_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": combined_prompt}]},
        ]
        strategy_result = apply_prompt_strategy(
            input_messages, self._prompt_strategy, task_type="agent"
        )
        payload = {"model": self._config.llm.openai_model, "input": strategy_result.messages}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        start = time.monotonic()

        def _request() -> LLMResponse:
            with self._governor.reserve("background"):
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
                response_obj = LLMResponse(
                    text=text.strip(), provider="openai", model=self._config.llm.openai_model
                )
                _log_agent_response(self._log, strategy_result.metadata, response_obj.text, start)
                return response_obj

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
        combined_prompt = user_prompt
        if context:
            combined_prompt = f"{user_prompt}\n\n{context}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_prompt},
        ]
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
        strategy_result = apply_prompt_strategy(messages, self._prompt_strategy, task_type="agent")
        payload = {"model": model, "messages": strategy_result.messages, "temperature": 0.2}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        start = time.monotonic()

        def _request() -> LLMResponse:
            with self._governor.reserve("background"):
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
                response_obj = LLMResponse(
                    text=text.strip(), provider="openai_compatible", model=model
                )
                _log_agent_response(self._log, strategy_result.metadata, response_obj.text, start)
                return response_obj

        return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)


def _log_agent_response(log, metadata, text: str, start: float) -> None:
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
