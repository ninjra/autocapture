from __future__ import annotations

import pytest

from tools.diffusionvl_server import DryRunRunner, build_app, parse_args


def test_diffusionvl_parse_args() -> None:
    args = parse_args(
        [
            "--dry-run",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--model",
            "stub-model",
            "--dtype",
            "float16",
            "--steps",
            "12",
            "--gen-length",
            "256",
            "--temperature",
            "0.1",
        ]
    )
    assert args.dry_run is True
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.model == "stub-model"
    assert args.dtype == "float16"
    assert args.steps == 12
    assert args.gen_length == 256
    assert args.temperature == 0.1


@pytest.mark.anyio
async def test_diffusionvl_dry_run_routes(async_client_factory) -> None:
    app = build_app(DryRunRunner("stub-model"))
    async with async_client_factory(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "stub-model", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "stub-model"
    assert payload["choices"][0]["message"]["content"]
