from __future__ import annotations

import datetime as dt
from pathlib import Path

from fastapi.testclient import TestClient

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.router import RoutingDecision
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


class BadCitationLLM:
    async def generate_answer(self, system_prompt: str, query: str, context_pack_text: str) -> str:
        return "Answer with invalid citation [E999]"


def test_answer_citations_subset(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.capture.data_dir = tmp_path
    config.embeddings.model = "local-test"
    db = DatabaseManager(config.database)

    with db.session() as session:
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="Docs",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="Meeting notes about roadmap",
                ocr_spans=[{"span_id": "S1", "span_key": "S1", "text": "roadmap", "start": 23, "end": 30, "conf": 0.9}],
                embedding_vector=None,
                tags={},
            )
        )

    def _mock_select(self):
        return BadCitationLLM(), RoutingDecision(llm_provider="mock")

    monkeypatch.setattr("autocapture.memory.router.ProviderRouter.select_llm", _mock_select)

    app = create_app(config, db_manager=db)
    client = TestClient(app)

    response = client.post("/api/answer", json={"query": "roadmap", "extractive_only": False})
    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"]
    assert all(cite.startswith("E") for cite in payload["citations"])
