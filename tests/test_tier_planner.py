from autocapture.config import AppConfig
from autocapture.memory.retrieval import _plan_tiers
from autocapture.runtime_budgets import BudgetManager
from autocapture.storage.models import TierStatsRecord


def test_tier_planner_always_includes_fast():
    config = AppConfig()
    config.retrieval.fusion_enabled = True
    config.routing.reranker = "enabled"
    config.reranker.enabled = True
    budgets = BudgetManager(config)
    state = budgets.start()
    plan, skipped, _ = _plan_tiers(config, "auto", budgets, state, "FACT", {})
    assert "FAST" in plan["tiers"]
    assert "FAST" not in skipped.get("tiers", [])


def test_tier_planner_numeric_guard_keeps_rerank():
    config = AppConfig()
    config.retrieval.fusion_enabled = True
    config.routing.reranker = "enabled"
    config.reranker.enabled = True
    budgets = BudgetManager(config)
    state = budgets.start()
    stats = {
        "RERANK": TierStatsRecord(
            query_class="FACT_NUMERIC_TIMEBOUND",
            tier="RERANK",
            help_rate=0.0,
            p95_ms=9999.0,
            window_n=config.next10.tier_stats_window,
        )
    }
    plan, skipped, _ = _plan_tiers(
        config, "auto", budgets, state, "FACT_NUMERIC_TIMEBOUND", stats
    )
    assert "RERANK" in plan["tiers"]
    assert "RERANK" not in skipped.get("tiers", [])


def test_tier_planner_skips_low_help_fusion():
    config = AppConfig()
    config.retrieval.fusion_enabled = True
    budgets = BudgetManager(config)
    state = budgets.start()
    stats = {
        "FUSION": TierStatsRecord(
            query_class="FACT",
            tier="FUSION",
            help_rate=0.0,
            p95_ms=9999.0,
            window_n=config.next10.tier_stats_window,
        )
    }
    plan, skipped, reasons = _plan_tiers(config, "auto", budgets, state, "FACT", stats)
    assert "FUSION" not in plan["tiers"]
    assert "FUSION" in skipped.get("tiers", [])
    assert reasons["tiers"].get("FUSION") in {"low_help_rate_high_latency", "budget_low"}
