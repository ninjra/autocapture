from __future__ import annotations

import json
from pathlib import Path

import httpx

from autocapture.promptops.runner import (
    PromptProposal,
    SourceSnapshot,
    _open_github_pr,
)


def test_promptops_github_pr_creation() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/repos/org/repo"):
            return httpx.Response(200, json={"default_branch": "main"})
        if request.url.path.endswith("/git/ref/heads/main"):
            return httpx.Response(200, json={"object": {"sha": "abc123"}})
        if request.url.path.endswith("/git/refs") and request.method == "POST":
            return httpx.Response(201, json={"ref": "refs/heads/promptops/test"})
        if "/contents/" in request.url.path and request.method == "GET":
            return httpx.Response(200, json={"sha": "file-sha"})
        if "/contents/" in request.url.path and request.method == "PUT":
            body = json.loads(request.content.decode("utf-8"))
            assert "content" in body
            return httpx.Response(201, json={"content": {"sha": "new-sha"}})
        if request.url.path.endswith("/pulls"):
            return httpx.Response(201, json={"html_url": "https://github.com/org/repo/pull/1"})
        return httpx.Response(404, json={"error": "unexpected"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    proposals = [
        PromptProposal(
            name="ANSWER_WITH_CONTEXT_PACK",
            raw_path=Path("prompts/raw/answer_with_context_pack.yaml"),
            derived_path=Path("autocapture/prompts/derived/answer_with_context_pack.yaml"),
            raw_content="name: test\nversion: v2\nsystem_prompt: test\nraw_template: test\nderived_template: test\ntags: []\nrationale: test\n",
            derived_content="name: test\nversion: v2\nsystem_prompt: test\nraw_template: test\nderived_template: test\ntags: []\nrationale: test\n",
            rationale="test",
        )
    ]
    sources = [
        SourceSnapshot(
            source="https://example.com",
            fetched_at="2024-01-01T00:00:00Z",
            status="ok",
            sha256="deadbeef",
            path="/tmp/source.txt",
            error=None,
            is_local=False,
            excerpt="hello",
        )
    ]
    eval_results = {
        "baseline": {"verifier_pass_rate": 0.5},
        "proposed": {"verifier_pass_rate": 0.6},
    }

    pr_url = _open_github_pr(
        "org/repo",
        "token",
        proposals,
        sources,
        eval_results,
        http_client=client,
    )

    assert pr_url == "https://github.com/org/repo/pull/1"
    assert any(path.endswith("/pulls") for _, path in calls)
