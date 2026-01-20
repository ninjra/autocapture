from __future__ import annotations

from pathlib import Path


def test_server_has_no_routes_or_middleware() -> None:
    content = Path("autocapture/api/server.py").read_text(encoding="utf-8")
    forbidden = [
        "@app.",
        "@router.",
        "app.get(",
        "app.post(",
        "include_router(",
        "app.mount(",
        "app.add_middleware(",
    ]
    hits = [pattern for pattern in forbidden if pattern in content]
    assert not hits, f"server.py must remain a thin shim; found {hits}"


def test_answer_graph_has_no_private_retrieval_access() -> None:
    content = Path("autocapture/agents/answer_graph.py").read_text(encoding="utf-8")
    forbidden = [
        "retrieval._",
        "self._retrieval._",
        'getattr(retrieval, "_',
        'getattr(self._retrieval, "_',
    ]
    hits = [pattern for pattern in forbidden if pattern in content]
    assert not hits, f"AnswerGraph must not access RetrievalService private fields; found {hits}"
