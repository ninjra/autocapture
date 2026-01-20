from __future__ import annotations

import asyncio
import json
from pathlib import Path

from autocapture.llm.transport import ReplayTransport


def test_replay_transport_returns_fixture(tmp_path: Path) -> None:
    response = {"output_text": "ok"}
    payload = {"response": response}
    fixture = tmp_path / "case1.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    transport = ReplayTransport(tmp_path, case_id="case1")
    result = asyncio.run(
        transport.post_json(
            "https://example.com",
            {"model": "x", "messages": []},
            None,
            timeout_s=1,
        )
    )
    assert result == response


def test_replay_transport_missing_fixture(tmp_path: Path) -> None:
    transport = ReplayTransport(tmp_path, case_id="missing")
    try:
        asyncio.run(
            transport.post_json(
                "https://example.com",
                {"model": "x", "messages": []},
                None,
                timeout_s=1,
            )
        )
    except FileNotFoundError:
        return
    assert False, "expected FileNotFoundError"
