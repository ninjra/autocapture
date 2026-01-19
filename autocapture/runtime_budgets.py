"""Deterministic stage budgets and degrade markers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AppConfig
from .logging_utils import get_logger


@dataclass
class BudgetSnapshot:
    total_ms: int
    stages: dict[str, int]
    degrade: dict[str, Any]


@dataclass
class BudgetState:
    started_at: float = field(default_factory=time.monotonic)
    stage_ms_used: dict[str, float] = field(default_factory=dict)
    degraded_stages: list[str] = field(default_factory=list)


class BudgetManager:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._log = get_logger("budgets")
        self._snapshot = self._load_defaults(config.next10.budgets_defaults_path)

    def snapshot(self) -> BudgetSnapshot:
        return self._snapshot

    def start(self) -> BudgetState:
        return BudgetState()

    def remaining_ms(self, state: BudgetState) -> float:
        elapsed = (time.monotonic() - state.started_at) * 1000
        return max(0.0, float(self._snapshot.total_ms) - elapsed)

    def budget_ms(self, stage: str) -> int:
        return int(self._snapshot.stages.get(stage, 0))

    def record_stage(self, state: BudgetState, stage: str, elapsed_ms: float) -> None:
        state.stage_ms_used[stage] = float(elapsed_ms)
        budget_ms = self.budget_ms(stage)
        if budget_ms and elapsed_ms > budget_ms:
            if stage not in state.degraded_stages:
                state.degraded_stages.append(stage)

    def mark_degraded(self, state: BudgetState, stage: str) -> None:
        if stage not in state.degraded_stages:
            state.degraded_stages.append(stage)

    def should_skip_dense(self, state: BudgetState) -> bool:
        min_remaining = int(self._snapshot.degrade.get("min_remaining_ms_for_dense", 0))
        if min_remaining <= 0:
            return False
        return self.remaining_ms(state) < min_remaining

    def should_skip_rerank(self, state: BudgetState) -> bool:
        min_remaining = int(self._snapshot.degrade.get("min_remaining_ms_for_rerank", 0))
        if min_remaining <= 0:
            return False
        return self.remaining_ms(state) < min_remaining

    def reduce_k(self, k: int) -> int:
        factor = float(self._snapshot.degrade.get("reduce_k_factor", 1.0))
        if factor <= 0 or factor >= 1:
            return k
        return max(1, int(round(k * factor)))

    @staticmethod
    def _load_defaults(path: Path) -> BudgetSnapshot:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        total_ms = int(payload.get("total_request_ms", 1500))
        stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
        degrade = payload.get("degrade", {}) if isinstance(payload.get("degrade"), dict) else {}
        return BudgetSnapshot(total_ms=total_ms, stages=stages, degrade=degrade)
