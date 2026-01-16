import datetime as dt
from pathlib import Path

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.router import RoutingDecision
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord, OCRSpanRecord


class MockLLM:
    async def generate_answer(self, system_prompt: str, query: str, context_pack_text: str) -> str:
        return "Answer based on evidence [E1]"


@pytest.mark.anyio
async def test_retrieve_context_pack_answer(
    monkeypatch, tmp_path: Path, async_client_factory
) -> None:
    config = AppConfig()
    config.database = DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}")
    config.capture.data_dir = tmp_path
    config.embed.text_model = "local-test"
    db = DatabaseManager(config.database)

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="event-1",
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=None,
                foreground_process="Notion",
                foreground_window="Notes",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.flush()
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="Notion",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="Sample meeting notes about roadmap",
                embedding_vector=None,
                tags={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id="event-1",
                span_key="S1",
                start=0,
                end=6,
                text="Sample",
                confidence=0.9,
                bbox={},
            )
        )

    def _mock_select(self):
        return MockLLM(), RoutingDecision(llm_provider="mock")

    monkeypatch.setattr("autocapture.memory.router.ProviderRouter.select_llm", _mock_select)

    app = create_app(config, db_manager=db)
    async with async_client_factory(app) as client:
        response = await client.post(
            "/api/answer", json={"query": "roadmap", "extractive_only": False}
        )
    assert response.status_code == 200
    payload = response.json()
    assert "Answer based on evidence" in payload["answer"]
    assert payload["used_context_pack"]["version"] == 1
