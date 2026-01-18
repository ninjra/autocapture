"""Deterministic retrieval eval harness for CI gates."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import datetime as dt

from .config import AppConfig, DatabaseConfig
from .logging_utils import get_logger
from .memory.retrieval import RetrievalService
from .storage.database import DatabaseManager
from .storage.models import CaptureRecord, EventRecord, OCRSpanRecord
from .text.normalize import normalize_text
from .vision.types import build_ocr_payload
from .indexing.lexical_index import LexicalIndex


@dataclass(frozen=True)
class RetrievalEvalMetrics:
    recall_at_k: float
    mrr: float
    no_evidence_accuracy: float
    total_cases: int

    def to_dict(self) -> dict:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "no_evidence_accuracy": self.no_evidence_accuracy,
            "total_cases": self.total_cases,
        }


def run_retrieval_eval(
    dataset_path: Path,
    *,
    k: int = 5,
) -> RetrievalEvalMetrics:
    log = get_logger("eval.gate")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp.name}", sqlite_wal=False),
    )
    config.embed.text_model = "local-test"
    config.qdrant.enabled = False
    config.retrieval.fusion_enabled = False
    config.retrieval.sparse_enabled = False
    config.retrieval.late_enabled = False
    config.features.enable_thresholding = True

    db = DatabaseManager(config.database)
    _seed_fixture_corpus(db)
    LexicalIndex(db).bulk_upsert(_load_events(db))
    retrieval = RetrievalService(db, config)

    cases = _load_cases(dataset_path)
    total = len(cases)
    if total == 0:
        return RetrievalEvalMetrics(0.0, 0.0, 0.0, 0)

    recall_hits = 0
    rr_total = 0.0
    no_evidence_total = 0
    no_evidence_hits = 0

    for case in cases:
        query = str(case.get("query", "")).strip()
        expected = case.get("expected_evidence_ids") or []
        expect_no_evidence = bool(case.get("expected_no_evidence", False))
        batch = retrieval.retrieve(query, None, None, limit=k)
        results = batch.results
        if expect_no_evidence:
            no_evidence_total += 1
            if batch.no_evidence or not results:
                no_evidence_hits += 1
            continue
        if not results:
            continue
        event_ids = [item.event.event_id for item in results]
        hit = None
        for idx, event_id in enumerate(event_ids, start=1):
            if event_id in expected:
                hit = idx
                break
        if hit is not None:
            recall_hits += 1
            rr_total += 1.0 / hit

    recall_at_k = recall_hits / max(total - no_evidence_total, 1)
    mrr = rr_total / max(total - no_evidence_total, 1)
    no_evidence_accuracy = (
        no_evidence_hits / max(no_evidence_total, 1) if no_evidence_total else 1.0
    )
    metrics = RetrievalEvalMetrics(
        recall_at_k=recall_at_k,
        mrr=mrr,
        no_evidence_accuracy=no_evidence_accuracy,
        total_cases=total,
    )
    log.info(
        "Retrieval eval metrics: recall@{}={} mrr={} no_evidence_accuracy={}",
        k,
        metrics.recall_at_k,
        metrics.mrr,
        metrics.no_evidence_accuracy,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="autocapture.eval_gate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--extended", action="store_true", default=False)
    args = parser.parse_args(argv)

    if args.extended:
        log = get_logger("eval.gate")
        log.info("Extended eval mode not configured; running deterministic set.")

    metrics = run_retrieval_eval(args.dataset, k=args.k)
    baseline = load_baseline(args.baseline)
    assert_gate(metrics, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def load_baseline(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in data.items()}


def assert_gate(metrics: RetrievalEvalMetrics, baseline: dict[str, float]) -> None:
    if metrics.recall_at_k < baseline.get("recall_at_k", 0.0):
        raise AssertionError("recall_at_k below baseline")
    if metrics.mrr < baseline.get("mrr", 0.0):
        raise AssertionError("mrr below baseline")
    if metrics.no_evidence_accuracy < baseline.get("no_evidence_accuracy", 0.0):
        raise AssertionError("no_evidence_accuracy below baseline")


def _load_cases(dataset_path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def _seed_fixture_corpus(db: DatabaseManager) -> None:
    now = dt.datetime(2026, 1, 17, 12, 0, tzinfo=dt.timezone.utc)
    fixtures = [
        ("EVT-1", ["Project roadmap", "ticket-123 assigned to Alpha"]),
        ("EVT-2", ["Support case", "ABC-999 closed"]),
        ("EVT-3", ["Meeting notes", "Alpha Beta"]),
    ]
    with db.session() as session:
        for event_id, lines in fixtures:
            text, spans = build_ocr_payload(
                [(line, 0.9, [0, 0, 10, 10]) for line in lines]
            )
            session.add(
                CaptureRecord(
                    id=event_id,
                    captured_at=now,
                    image_path=None,
                    foreground_process="App",
                    foreground_window="Window",
                    monitor_id="m1",
                    is_fullscreen=False,
                    ocr_status="done",
                    monitor_bounds=[0, 0, 100, 100],
                    privacy_flags={},
                )
            )
            session.add(
                EventRecord(
                    event_id=event_id,
                    ts_start=now,
                    ts_end=None,
                    app_name="App",
                    window_title="Window",
                    url=None,
                    domain=None,
                    screenshot_path=None,
                    screenshot_hash=None,
                    ocr_text=text,
                    ocr_text_normalized=normalize_text(text),
                    tags={},
                )
            )
            for span in spans:
                session.add(
                    OCRSpanRecord(
                        capture_id=event_id,
                        span_key=str(span.get("span_key")),
                        start=int(span.get("start", 0)),
                        end=int(span.get("end", 0)),
                        text=str(span.get("text", "")),
                        confidence=float(span.get("conf", 0.9)),
                        bbox=span.get("bbox", []),
                        engine="fixture",
                        frame_hash=None,
                    )
                )


def _load_events(db: DatabaseManager) -> Iterable[EventRecord]:
    with db.session() as session:
        return list(session.query(EventRecord).all())
