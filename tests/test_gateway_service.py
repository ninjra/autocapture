from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from autocapture.config import (
    AppConfig,
    ModelRegistryConfig,
    ModelSpec,
    ProviderSpec,
    StagePolicy,
    StageRequirementsConfig,
)
from autocapture.gateway.service import GatewayRouter, UpstreamError


def _registry_config(repair_on_failure: bool = True) -> ModelRegistryConfig:
    provider = ProviderSpec(id="local", type="openai_compatible", base_url="http://localhost")
    model = ModelSpec(id="m1", provider_id="local", upstream_model_name="model-1")
    stage = StagePolicy(
        id="final_answer",
        primary_model_id="m1",
        requirements=StageRequirementsConfig(require_json=True, claims_schema="claims_json_v1"),
        repair_on_failure=repair_on_failure,
    )
    return ModelRegistryConfig(
        enabled=True,
        providers=[provider],
        models=[model],
        stages=[stage],
    )


def _evidence_message() -> dict:
    payload = {"evidence": [{"id": "E1", "text": "Evidence"}]}
    return {
        "role": "user",
        "content": f"EVIDENCE_JSON:\n```json\n{json.dumps(payload)}\n```",
    }


def test_gateway_repairs_invalid_claims() -> None:
    app = FastAPI()
    calls = {"count": 0}

    @app.post("/v1/chat/completions")
    async def completions(_payload: dict) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            content = (
                "```json\n"
                '{"schema_version":1,"claims":[{"text":"Claim","evidence_ids":[]}]}'
                "\n```"
            )
        else:
            content = (
                "```json\n"
                '{"schema_version":1,"claims":[{"text":"Claim","evidence_ids":["E1"]}]}'
                "\n```"
            )
        return {"choices": [{"message": {"content": content}}]}

    def client_factory(timeout_s: float) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(
            transport=transport, base_url="http://localhost", timeout=timeout_s
        )

    config = AppConfig(model_registry=_registry_config())
    router = GatewayRouter(config, http_client_factory=client_factory)
    payload = {"model": "final_answer", "messages": [_evidence_message()]}
    response = asyncio.run(router.handle_stage_request("final_answer", payload, tenant_id=None))
    assert response["choices"][0]["message"]["content"]
    assert calls["count"] == 2


def test_gateway_rejects_invalid_claims_without_repair() -> None:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(_payload: dict) -> dict:
        content = (
            "```json\n" '{"schema_version":1,"claims":[{"text":"Claim","evidence_ids":[]}]}' "\n```"
        )
        return {"choices": [{"message": {"content": content}}]}

    def client_factory(timeout_s: float) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(
            transport=transport, base_url="http://localhost", timeout=timeout_s
        )

    config = AppConfig(model_registry=_registry_config(repair_on_failure=False))
    router = GatewayRouter(config, http_client_factory=client_factory)
    payload = {"model": "final_answer", "messages": [_evidence_message()]}
    with pytest.raises(UpstreamError) as excinfo:
        asyncio.run(router.handle_stage_request("final_answer", payload, tenant_id=None))
    assert excinfo.value.status_code == 422
