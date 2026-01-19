import asyncio

from autocapture.answer.provenance import verify_provenance
from autocapture.next10.harness import build_harness
from autocapture.storage.ledger import LedgerWriter


def test_provenance_chain_validates():
    corpus, retrieval, answer_graph = build_harness(enable_rerank=True)
    result = asyncio.run(
        answer_graph.run(
            "alpha cost",
            time_range=None,
            filters=None,
            k=4,
            sanitized=True,
            extractive_only=True,
            routing={"llm": "disabled"},
            output_format="text",
            context_pack_format="json",
        )
    )
    assert result.answer_id
    ledger = LedgerWriter(corpus.db)
    assert ledger.validate_chain(result.answer_id) is True

    span_ids = []
    for item in result.context_pack.get("evidence", []):
        meta = item.get("meta") or {}
        for span in meta.get("spans", []):
            if span.get("span_id"):
                span_ids.append(span.get("span_id"))
    status = verify_provenance(corpus.db, query_id=result.query_id or "", span_ids=span_ids)
    assert status.valid_span_ids
