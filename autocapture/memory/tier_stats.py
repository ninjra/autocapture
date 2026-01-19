"""Tier stats updates for adaptive skipping."""

from __future__ import annotations

from sqlalchemy import select

from ..storage.database import DatabaseManager
from ..storage.models import RetrievalHitRecord, TierPlanDecisionRecord, TierStatsRecord


def update_tier_stats(
    db: DatabaseManager,
    *,
    query_id: str,
    query_class: str,
    cited_span_ids: set[str],
) -> None:
    if not query_id:
        return
    tiers = ["FAST", "FUSION", "RERANK"]
    with db.session() as session:
        hits = (
            session.execute(
                select(RetrievalHitRecord).where(RetrievalHitRecord.query_id == query_id)
            )
            .scalars()
            .all()
        )
        plan = (
            session.execute(
                select(TierPlanDecisionRecord)
                .where(TierPlanDecisionRecord.query_id == query_id)
                .order_by(TierPlanDecisionRecord.created_at.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        stage_ms = plan.plan_json.get("stage_ms", {}) if plan else {}
        spans_by_tier: dict[str, set[str]] = {tier: set() for tier in tiers}
        for hit in hits:
            if hit.span_id:
                spans_by_tier.setdefault(hit.tier, set()).add(hit.span_id)
        seen: set[str] = set()
        for tier in tiers:
            cited = spans_by_tier.get(tier, set()) & cited_span_ids
            helped = bool(cited - seen)
            seen |= cited
            record = session.get(TierStatsRecord, (query_class, tier))
            if record is None:
                record = TierStatsRecord(query_class=query_class, tier=tier)
                session.add(record)
            window_n = int(record.window_n or 0)
            new_n = window_n + 1
            help_value = 1.0 if helped else 0.0
            record.help_rate = ((record.help_rate or 0.0) * window_n + help_value) / new_n
            stage_value = float(stage_ms.get(tier, 0.0))
            record.p50_ms = ((record.p50_ms or 0.0) * window_n + stage_value) / new_n
            record.p95_ms = ((record.p95_ms or 0.0) * window_n + stage_value) / new_n
            record.window_n = new_n
    return
