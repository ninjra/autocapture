"""Gateway routing and upstream proxy logic."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from ..config import AppConfig
from ..logging_utils import get_logger
from ..model_ops.registry import ModelRegistry, RegistryError, StageModel
from ..resilience import RetryPolicy, retry_async, is_retryable_exception, is_retryable_http_status
from ..answer.claims import parse_claims_json
from ..answer.claim_validation import ClaimValidator, EvidenceLineInfo
from ..observability.otel import otel_span

_LOG = get_logger("gateway")


@dataclass(frozen=True)
class UpstreamResponse:
    payload: dict[str, Any]
    content: str


class UpstreamError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GatewayRouter:
    def __init__(
        self,
        config: AppConfig,
        *,
        registry: ModelRegistry | None = None,
        http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or ModelRegistry(config.model_registry)
        self._validator = ClaimValidator(config.verification.citation_validator)
        self._client_factory = http_client_factory
        self._provider_breakers: dict[str, int] = {}
        self._provider_open_until: dict[str, float] = {}
        self._decode_limits: dict[str, asyncio.Semaphore] = {}
        self._provider_limits: dict[str, asyncio.Semaphore] = {}
        self._provider_limits_capacity: dict[str, int] = {}
        self._gpu_sem = asyncio.Semaphore(max(1, int(config.gateway.gpu_max_concurrency)))

    def registry_enabled(self) -> bool:
        return self._registry.enabled

    def _client(self, timeout_s: float) -> httpx.AsyncClient:
        timeout_s = min(float(timeout_s), float(self._config.gateway.request_timeout_s))
        if self._client_factory:
            return self._client_factory(timeout_s=timeout_s)
        return httpx.AsyncClient(timeout=timeout_s)

    def _breaker_open(self, provider_id: str) -> bool:
        open_until = self._provider_open_until.get(provider_id)
        if open_until is None:
            return False
        if time.monotonic() >= open_until:
            self._provider_open_until.pop(provider_id, None)
            self._provider_breakers[provider_id] = 0
            return False
        return True

    def _record_failure(self, provider_id: str, threshold: int, reset_timeout_s: float) -> None:
        failures = self._provider_breakers.get(provider_id, 0) + 1
        self._provider_breakers[provider_id] = failures
        if failures >= threshold:
            self._provider_open_until[provider_id] = time.monotonic() + reset_timeout_s

    def _record_success(self, provider_id: str) -> None:
        self._provider_breakers[provider_id] = 0
        self._provider_open_until.pop(provider_id, None)

    async def handle_stage_request(
        self,
        stage_id: str,
        request: dict[str, Any],
        *,
        tenant_id: str | None,
    ) -> dict[str, Any]:
        if not self._registry.enabled:
            raise UpstreamError("model_registry disabled", status_code=400)
        try:
            candidates = self._registry.stage_candidates(stage_id)
        except RegistryError as exc:
            raise UpstreamError(str(exc), status_code=404) from exc
        if not candidates:
            raise UpstreamError(f"stage '{stage_id}' has no candidates", status_code=404)
        evidence_map = _extract_evidence_map(request.get("messages") or [])
        last_error: Exception | None = None
        for candidate in candidates:
            provider = candidate.provider
            policy = candidate.stage
            if self._breaker_open(provider.id):
                continue
            if not _cloud_allowed(self._config, policy.allow_cloud, provider):
                last_error = UpstreamError("cloud_blocked", status_code=403)
                continue
            try:
                response = await self._attempt_candidate(
                    candidate,
                    request,
                    tenant_id=tenant_id,
                    evidence_map=evidence_map,
                )
                self._record_success(provider.id)
                return response.payload
            except Exception as exc:
                last_error = exc
                self._record_failure(
                    provider.id,
                    provider.circuit_breaker.failure_threshold,
                    provider.circuit_breaker.reset_timeout_s,
                )
                continue
        if isinstance(last_error, UpstreamError):
            raise last_error
        raise UpstreamError("upstream_failed", status_code=502)

    async def handle_proxy_request(self, request: dict[str, Any]) -> dict[str, Any]:
        model_id = (request.get("model") or "").strip()
        if not model_id:
            raise UpstreamError("model required", status_code=400)
        if self._registry.enabled and self._registry.stage(model_id):
            return await self.handle_stage_request(
                model_id, request, tenant_id=request.get("tenant_id")
            )
        if self._registry.enabled and model_id in self._registry.model_ids():
            candidate = self._registry.direct_candidate(model_id)
            return await self._proxy_model(candidate, request, tenant_id=request.get("tenant_id"))
        # Fallback to configured openai-compatible
        return await self._proxy_fallback(request)

    async def handle_embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        model_id = (request.get("model") or "").strip()
        if not model_id:
            raise UpstreamError("model required", status_code=400)
        if self._registry.enabled and model_id in self._registry.model_ids():
            candidate = self._registry.direct_candidate(model_id)
            response = await self._call_upstream(
                candidate.provider,
                request,
                tenant_id=request.get("tenant_id"),
                endpoint="/v1/embeddings",
            )
            return response.payload
        response = await self._call_upstream(
            _provider_from_fallback(self._config),
            request,
            tenant_id=request.get("tenant_id"),
            endpoint="/v1/embeddings",
        )
        return response.payload

    async def handle_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        model_id = (request.get("model") or "").strip()
        if not model_id:
            raise UpstreamError("model required", status_code=400)
        if self._registry.enabled and model_id in self._registry.model_ids():
            candidate = self._registry.direct_candidate(model_id)
            response = await self._call_upstream(
                candidate.provider,
                request,
                tenant_id=request.get("tenant_id"),
                endpoint="/v1/completions",
            )
            return response.payload
        response = await self._call_upstream(
            _provider_from_fallback(self._config),
            request,
            tenant_id=request.get("tenant_id"),
            endpoint="/v1/completions",
        )
        return response.payload

    async def handle_models(self) -> dict[str, Any]:
        data: list[dict[str, Any]] = []
        if self._registry.enabled:
            for model in self._registry.models():
                data.append(
                    {
                        "id": model.id,
                        "object": "model",
                        "owned_by": model.provider_id,
                        "context_length": model.context_tokens,
                        "metadata": {
                            "supports_json": model.supports_json,
                            "supports_tools": model.supports_tools,
                            "supports_vision": model.supports_vision,
                            "quantization": (
                                model.quantization.model_dump()
                                if model.quantization is not None
                                else None
                            ),
                            "lora": (model.lora.model_dump() if model.lora is not None else None),
                            "runtime": model.runtime.model_dump(),
                            "lmcache_enabled": model.lmcache_enabled,
                        },
                    }
                )
        return {"object": "list", "data": data}

    async def _proxy_model(
        self, candidate: StageModel, request: dict[str, Any], *, tenant_id: str | None
    ) -> dict[str, Any]:
        payload = dict(request)
        payload["model"] = candidate.model.upstream_model_name
        response = await self._call_upstream(candidate.provider, payload, tenant_id=tenant_id)
        return response.payload

    async def _proxy_fallback(self, request: dict[str, Any]) -> dict[str, Any]:
        base_url = self._config.llm.openai_compatible_base_url
        if not base_url:
            raise UpstreamError("openai_compatible_base_url not configured", status_code=400)
        provider = _provider_from_fallback(self._config)
        response = await self._call_upstream(provider, request, tenant_id=request.get("tenant_id"))
        return response.payload

    async def _attempt_candidate(
        self,
        candidate: StageModel,
        request: dict[str, Any],
        *,
        tenant_id: str | None,
        evidence_map: dict[str, EvidenceLineInfo],
    ) -> UpstreamResponse:
        stage = candidate.stage
        if stage.decode.strategy not in {"standard", "swift", "lookahead", "medusa"}:
            raise UpstreamError("decode_strategy_not_allowed", status_code=400)
        base_payload = dict(request)
        base_payload["model"] = candidate.model.upstream_model_name
        base_payload = _apply_sampling(base_payload, stage)
        base_payload = _apply_lora(base_payload, candidate)
        base_payload = _apply_response_format(base_payload, stage)
        base_payload = _apply_lmcache(base_payload, candidate, tenant_id)
        errors: list[str] = []
        max_attempts = max(1, int(getattr(stage, "max_attempts", 1)))
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            payload = dict(base_payload)
            if attempt > 1 and errors:
                payload = _build_repair_payload(payload, errors)
                payload["temperature"] = 0.0
            if stage.decode.strategy != "standard":
                backend = self._registry.decode_backend(stage)
                if backend is None:
                    raise UpstreamError("decode_backend_missing", status_code=500)
                decode_limit = stage.decode.max_concurrency
                if stage.decode.strategy == "medusa":
                    decode_limit = 1
                if not await self._acquire_decode_slot(
                    stage.decode.backend_provider_id, decode_limit
                ):
                    raise UpstreamError("decode_backend_busy", status_code=429)
                try:
                    payload["decoding_strategy"] = stage.decode.strategy
                    response = await self._call_with_limits(
                        backend,
                        payload,
                        tenant_id=tenant_id,
                        gpu_required=True,
                    )
                finally:
                    self._release_decode_slot(stage.decode.backend_provider_id)
            else:
                response = await self._call_with_limits(
                    candidate.provider,
                    payload,
                    tenant_id=tenant_id,
                    gpu_required=_gpu_required(candidate.model),
                )
            if (
                stage.requirements.require_json
                and stage.requirements.claims_schema == "claims_json_v1"
            ):
                try:
                    parsed = parse_claims_json(response.content)
                except Exception:
                    errors = ["claims_parse_failed"]
                    if stage.repair_on_failure and attempt < max_attempts:
                        continue
                    raise UpstreamError("claims_parse_failed", status_code=422)
                validation = self._validator.validate(parsed.payload, evidence_map=evidence_map)
                if not validation.valid:
                    errors = validation.errors
                    if stage.repair_on_failure and attempt < max_attempts:
                        continue
                    raise UpstreamError("claims_validation_failed", status_code=422)
            if stage.requirements.require_citations and not _contains_citations(response.content):
                raise UpstreamError("citations_missing", status_code=422)
            return response
        raise UpstreamError("claims_validation_failed", status_code=422)

    async def _call_upstream(
        self,
        provider,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        endpoint: str = "/v1/chat/completions",
    ) -> UpstreamResponse:
        api_key = provider.api_key
        if not api_key and provider.api_key_env:
            api_key = os.environ.get(provider.api_key_env)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider.headers:
            headers.update(provider.headers)
        timeout_s = provider.timeout_s
        base_url = provider.base_url or ""
        if provider.type == "openai" and not base_url:
            base_url = "https://api.openai.com"

        async def _request() -> dict[str, Any]:
            async with self._client(timeout_s) as client:
                response = await client.post(
                    f"{base_url.rstrip('/')}{endpoint}",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        with otel_span("gateway.upstream", {"provider_id": provider.id}):
            try:
                data = await retry_async(
                    _request,
                    policy=RetryPolicy(max_retries=provider.retries),
                    is_retryable=_retryable_exc,
                )
            except httpx.HTTPStatusError as exc:
                raise UpstreamError(
                    "upstream_http_error", status_code=exc.response.status_code
                ) from exc
            except Exception as exc:
                raise UpstreamError("upstream_error") from exc
        content = _extract_content(data)
        return UpstreamResponse(payload=data, content=content)

    async def _call_with_limits(
        self,
        provider,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        gpu_required: bool,
    ) -> UpstreamResponse:
        if not await self._acquire_provider_slot(provider.id, provider.max_concurrency):
            raise UpstreamError("provider_busy", status_code=429)
        gpu_acquired = False
        if gpu_required:
            if not await self._acquire_gpu_slot():
                self._release_provider_slot(provider.id)
                raise UpstreamError("gpu_busy", status_code=429)
            gpu_acquired = True
        try:
            return await self._call_upstream(provider, payload, tenant_id=tenant_id)
        finally:
            if gpu_acquired:
                self._release_gpu_slot()
            self._release_provider_slot(provider.id)

    async def _acquire_provider_slot(self, provider_id: str, max_concurrency: int) -> bool:
        if (
            provider_id not in self._provider_limits
            or self._provider_limits_capacity.get(provider_id) != max_concurrency
        ):
            self._provider_limits[provider_id] = asyncio.Semaphore(max_concurrency)
            self._provider_limits_capacity[provider_id] = max_concurrency
        sem = self._provider_limits[provider_id]
        if sem.locked():
            return False
        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.001)
            return True
        except Exception:
            return False

    def _release_provider_slot(self, provider_id: str) -> None:
        sem = self._provider_limits.get(provider_id)
        if sem is None:
            return
        try:
            sem.release()
        except ValueError:
            return

    async def _acquire_gpu_slot(self) -> bool:
        sem = self._gpu_sem
        if sem.locked():
            return False
        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.001)
            return True
        except Exception:
            return False

    def _release_gpu_slot(self) -> None:
        try:
            self._gpu_sem.release()
        except ValueError:
            return

    async def _acquire_decode_slot(self, backend_id: str, max_concurrency: int) -> bool:
        if backend_id not in self._decode_limits:
            self._decode_limits[backend_id] = asyncio.Semaphore(max_concurrency)
        sem = self._decode_limits[backend_id]
        if sem.locked():
            return False
        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.001)
            return True
        except Exception:
            return False

    def _release_decode_slot(self, backend_id: str) -> None:
        sem = self._decode_limits.get(backend_id)
        if sem is None:
            return
        try:
            sem.release()
        except ValueError:
            return

    async def probe_upstreams(self) -> dict[str, str]:
        results: dict[str, str] = {}
        if not self._registry.enabled:
            return results
        for provider in self._registry.providers():
            if not _cloud_allowed(self._config, True, provider):
                results[provider.id] = "skipped"
                continue
            try:
                await self._probe_provider(provider)
                self._record_success(provider.id)
                results[provider.id] = "ok"
            except Exception as exc:
                self._provider_open_until[provider.id] = time.monotonic() + float(
                    provider.circuit_breaker.reset_timeout_s
                )
                self._provider_breakers[provider.id] = provider.circuit_breaker.failure_threshold
                _LOG.warning("Gateway probe failed for {}: {}", provider.id, exc)
                results[provider.id] = "failed"
        return results

    async def _probe_provider(self, provider) -> None:
        api_key = provider.api_key
        if not api_key and provider.api_key_env:
            api_key = os.environ.get(provider.api_key_env)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider.headers:
            headers.update(provider.headers)
        base_url = provider.base_url or ""
        if provider.type == "openai" and not base_url:
            base_url = "https://api.openai.com"
        timeout_s = min(
            float(provider.timeout_s),
            float(self._config.gateway.upstream_probe_timeout_s),
        )
        async with self._client(timeout_s) as client:
            response = await client.get(f"{base_url.rstrip('/')}/v1/models", headers=headers)
            response.raise_for_status()


class ProviderFallback:
    def __init__(
        self,
        *,
        id: str,
        base_url: str,
        api_key: str | None,
        timeout_s: float,
        retries: int,
        max_concurrency: int,
    ):
        self.id = id
        self.type = "openai_compatible"
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_env = None
        self.timeout_s = timeout_s
        self.retries = retries
        self.headers: dict[str, str] = {}
        self.allow_cloud = True
        self.circuit_breaker = type("CB", (), {"failure_threshold": 5, "reset_timeout_s": 30.0})()
        self.max_concurrency = max_concurrency


def _provider_from_fallback(config: AppConfig) -> ProviderFallback:
    return ProviderFallback(
        id="openai_compatible",
        base_url=config.llm.openai_compatible_base_url or "",
        api_key=config.llm.openai_compatible_api_key,
        timeout_s=config.llm.timeout_s,
        retries=config.llm.retries,
        max_concurrency=4,
    )


def _retryable_exc(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_http_status(exc.response.status_code)
    return is_retryable_exception(exc)


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
    return json.dumps(payload)


def _apply_sampling(payload: dict[str, Any], stage) -> dict[str, Any]:
    if stage.sampling.temperature is not None:
        payload["temperature"] = stage.sampling.temperature
    if stage.sampling.top_p is not None:
        payload["top_p"] = stage.sampling.top_p
    if stage.sampling.max_tokens is not None:
        payload["max_tokens"] = stage.sampling.max_tokens
    if stage.sampling.seed is not None:
        payload["seed"] = stage.sampling.seed
    return payload


def _apply_response_format(payload: dict[str, Any], stage) -> dict[str, Any]:
    if stage.requirements.require_json:
        payload.setdefault("response_format", {"type": "json_object"})
    return payload


def _apply_lora(payload: dict[str, Any], candidate: StageModel) -> dict[str, Any]:
    lora_adapter = payload.get("lora_adapter_id") or payload.get("lora_adapter")
    if not lora_adapter:
        return payload
    lora_cfg = candidate.model.lora
    if not lora_cfg or not lora_cfg.enabled:
        raise UpstreamError("lora_not_enabled", status_code=400)
    if lora_cfg.enforce_allowlist and lora_adapter not in lora_cfg.allowed_adapters:
        raise UpstreamError("lora_adapter_not_allowed", status_code=400)
    payload["lora_adapter"] = lora_adapter
    return payload


def _apply_lmcache(
    payload: dict[str, Any], candidate: StageModel, tenant_id: str | None
) -> dict[str, Any]:
    if not candidate.model.lmcache_enabled:
        return payload
    tenant = tenant_id or payload.get("tenant_id") or "default"
    messages = list(payload.get("messages") or [])
    prefix = {"role": "system", "content": f"LMCache tenant: {tenant}"}
    payload["messages"] = [prefix] + messages
    return payload


def _build_repair_payload(payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    messages = list(payload.get("messages") or [])
    repair_msg = {
        "role": "system",
        "content": (
            "Your previous response failed validation. "
            f"Errors: {', '.join(errors)}. "
            "Return corrected JSON only."
        ),
    }
    return {**payload, "messages": messages + [repair_msg], "temperature": 0.0}


def _extract_evidence_map(messages: list[dict[str, Any]]) -> dict[str, EvidenceLineInfo]:
    text = "\n".join(str(msg.get("content") or "") for msg in messages)
    match = re.search(r"EVIDENCE_JSON:\s*```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    evidence = payload.get("evidence") or []
    mapping: dict[str, EvidenceLineInfo] = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("id") or "")
        if not evidence_id:
            continue
        text_block = item.get("text")
        if not isinstance(text_block, str):
            text_block = ""
        lines = text_block.splitlines()
        if not lines and text_block:
            lines = [text_block]
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        citable = bool(meta.get("citable", True))
        mapping[evidence_id] = EvidenceLineInfo(
            evidence_id=evidence_id,
            lines=lines,
            citable=citable,
        )
    return mapping


def _extract_evidence_ids(messages: list[dict[str, Any]]) -> set[str]:
    return set(_extract_evidence_map(messages))


def _contains_citations(text: str) -> bool:
    return bool(re.search(r"(?:\\[|【)E\\d+(?::L\\d+-L\\d+)?(?:\\]|】)", text or ""))


def _cloud_allowed(config: AppConfig, allow_cloud: bool, provider) -> bool:
    if provider.type == "openai":
        is_cloud = True
    else:
        is_cloud = bool(provider.base_url and not _is_loopback(provider.base_url))
    if not is_cloud:
        return True
    if not allow_cloud or not getattr(provider, "allow_cloud", True):
        return False
    if config.offline:
        return False
    if not config.privacy.cloud_enabled:
        return False
    return True


def _is_loopback(base_url: str) -> bool:
    return "127.0.0.1" in base_url or "localhost" in base_url


def _gpu_required(model) -> bool:
    device = str(getattr(getattr(model, "runtime", None), "device", "") or "").lower()
    if device in {"cpu"}:
        return False
    return True
