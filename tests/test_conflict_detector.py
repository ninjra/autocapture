import datetime as dt

from autocapture.answer.conflict import detect_conflicts
from autocapture.memory.context_pack import EvidenceItem, EvidenceSpan


def _make_item(evidence_id: str, text: str, timestamp: dt.datetime) -> EvidenceItem:
    span = EvidenceSpan(span_id="S1", start=0, end=10, conf=0.9, bbox=[0, 0, 1, 1])
    return EvidenceItem(
        evidence_id=evidence_id,
        event_id=evidence_id,
        timestamp=timestamp.isoformat(),
        ts_end=None,
        app="App",
        title="Title",
        domain=None,
        score=0.9,
        spans=[span],
        text=text,
        screenshot_path=None,
        screenshot_hash=None,
        retrieval=None,
    )


def test_conflict_detector_flags_conflict():
    ts = dt.datetime(2026, 1, 17, 12, 0, tzinfo=dt.timezone.utc)
    items = [
        _make_item("E1", "Project Alpha cost $100", ts),
        _make_item("E2", "Project Alpha cost $150", ts),
    ]
    result = detect_conflicts(items)
    assert result.conflict is True
    assert result.changed_over_time is False


def test_conflict_detector_change_over_time():
    ts1 = dt.datetime(2026, 1, 17, 12, 0, tzinfo=dt.timezone.utc)
    ts2 = dt.datetime(2026, 1, 17, 13, 0, tzinfo=dt.timezone.utc)
    items = [
        _make_item("E1", "Status: open", ts1),
        _make_item("E2", "Status: closed", ts2),
    ]
    result = detect_conflicts(items)
    assert result.conflict is False
    assert result.changed_over_time is True
