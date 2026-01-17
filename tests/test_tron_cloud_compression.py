from __future__ import annotations

from autocapture.agents.answer_graph import _select_context_pack_text
from autocapture.config import AppConfig


class _Decision:
    def __init__(self, cloud: bool) -> None:
        self.cloud = cloud


def test_tron_forced_for_cloud_when_allowed() -> None:
    config = AppConfig()
    config.output.allow_tron_compression = True
    warnings: list[str] = []
    result = _select_context_pack_text(
        config,
        _Decision(cloud=True),
        "json",
        json_text='{"ok": true}',
        tron_text="TRON_OK",
        warnings=warnings,
        stage="final_answer",
    )
    assert result == "TRON_OK"
    assert "tron_forced_for_cloud_final_answer" in warnings


def test_tron_disabled_for_cloud_when_disallowed() -> None:
    config = AppConfig()
    config.output.allow_tron_compression = False
    warnings: list[str] = []
    result = _select_context_pack_text(
        config,
        _Decision(cloud=True),
        "tron",
        json_text='{"ok": true}',
        tron_text="TRON_OK",
        warnings=warnings,
        stage="final_answer",
    )
    assert result == '{"ok": true}'
    assert "tron_disabled_for_cloud_final_answer" in warnings
