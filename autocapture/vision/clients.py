"""Vision model client helpers."""

from __future__ import annotations

import base64

import httpx

from ..logging_utils import get_logger
from ..llm.prompt_repetition import apply_prompt_repetition
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
        prompt_repetition: bool = False,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._retry = RetryPolicy(max_retries=retries)
        self._log = get_logger("vision.client")
        self._prompt_repetition = prompt_repetition
        self._http_client = http_client

    def generate(self, system_prompt: str, user_prompt: str, images: list[bytes]) -> str:
        if self._provider == "ollama":
            return self._generate_ollama(system_prompt, user_prompt, images)
        if self._provider == "openai_compatible":
            return self._generate_openai_compatible(system_prompt, user_prompt, images)
        if self._provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, images)
        raise RuntimeError(f"Unsupported vision provider: {self._provider}")

    def _generate_ollama(self, system_prompt: str, user_prompt: str, images: list[bytes]) -> str:
        base_url = self._base_url or "http://127.0.0.1:11434"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if images:
            encoded = [base64.b64encode(image).decode("utf-8") for image in images]
            messages[-1]["images"] = encoded
        payload = {
            "model": self._model,
            "messages": apply_prompt_repetition(messages, enabled=self._prompt_repetition),
            "stream": False,
        }

        def _request() -> str:
            if self._http_client is not None:
                response = self._http_client.post(f"{base_url}/api/chat", json=payload)
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()

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
        payload = {
            "model": self._model,
            "messages": apply_prompt_repetition(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                enabled=self._prompt_repetition,
            ),
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

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
            return data["choices"][0]["message"]["content"].strip()

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
        payload = {
            "model": self._model,
            "messages": apply_prompt_repetition(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                enabled=self._prompt_repetition,
            ),
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}

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
            return data["choices"][0]["message"]["content"].strip()

        try:
            return retry_sync(_request, policy=self._retry, is_retryable=is_retryable_exception)
        except Exception as exc:
            self._log.warning("OpenAI vision request failed: {}", exc)
            raise
