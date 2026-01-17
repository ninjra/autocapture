"""Offline A/B evaluation harness for prompt strategies."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from ..config import AppConfig
from ..logging_utils import get_logger
from ..memory.router import ProviderRouter
from ..llm.prompt_strategy import (
    PromptStrategy,
    PromptStrategySettings,
    apply_prompt_strategy,
    render_prompt_text,
)

_LOG = get_logger("promptops.ab_eval")


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    task_type: str
    prompt: str
    expected: str
    requires_repeat: bool = False
    requires_step_by_step: bool = False


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    task_type: str
    strategy: str
    passed: bool
    expected: str
    prediction: str
    prompt_tokens_estimate: int
    response_tokens_estimate: int
    latency_ms: float
    prompt_hash_before: str
    prompt_hash_after: str
    repeat_factor: int
    step_by_step_used: bool
    safe_mode_degraded: bool

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "strategy": self.strategy,
            "passed": self.passed,
            "expected": self.expected,
            "prediction": self.prediction,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "response_tokens_estimate": self.response_tokens_estimate,
            "latency_ms": self.latency_ms,
            "prompt_hash_before": self.prompt_hash_before,
            "prompt_hash_after": self.prompt_hash_after,
            "repeat_factor": self.repeat_factor,
            "step_by_step_used": self.step_by_step_used,
            "safe_mode_degraded": self.safe_mode_degraded,
        }


class StubModel:
    def answer(self, prompt_text: str, case: EvalCase) -> str:
        text = prompt_text.lower()
        if case.requires_step_by_step and "let's think step by step" not in text:
            return "unsure"
        if case.requires_repeat and "---" not in text:
            return "unknown"
        return case.expected


def load_eval_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw:
        cases.append(
            EvalCase(
                case_id=item["case_id"],
                task_type=item["task_type"],
                prompt=item["prompt"],
                expected=item["expected"],
                requires_repeat=bool(item.get("requires_repeat", False)),
                requires_step_by_step=bool(item.get("requires_step_by_step", False)),
            )
        )
    return cases


def run_ab_eval(
    strategies: Iterable[PromptStrategy],
    *,
    cases_path: Path,
    output_path: Path,
    config: AppConfig | None = None,
    use_live: bool = False,
) -> list[EvalCaseResult]:
    config = config or AppConfig()
    settings = PromptStrategySettings.from_llm_config(config.llm, data_dir=config.capture.data_dir)
    if not settings.enable_step_by_step:
        settings = replace(settings, enable_step_by_step=True)
    stub = StubModel()
    results: list[EvalCaseResult] = []
    cases = load_eval_cases(cases_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for strategy in strategies:
            provider = None
            if use_live:
                if os.environ.get("RUN_LIVE_EVALS") != "1":
                    raise RuntimeError("RUN_LIVE_EVALS=1 is required for live evals.")
                provider = ProviderRouter(
                    config.routing,
                    config.llm,
                    config=config,
                    offline=config.offline,
                    privacy=config.privacy,
                    prompt_strategy=_settings_for_strategy(settings, strategy),
                ).select_llm()[0]
            for case in cases:
                start = time.monotonic()
                step_requested = strategy in {
                    PromptStrategy.STEP_BY_STEP,
                    PromptStrategy.STEP_BY_STEP_PLUS_REPEAT_2X,
                }
                prompt_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": case.prompt},
                ]
                result = apply_prompt_strategy(
                    prompt_messages,
                    settings,
                    task_type=case.task_type,
                    step_by_step_requested=step_requested,
                    override_strategy=strategy,
                )
                prompt_text = render_prompt_text(result.messages)
                if provider is None:
                    prediction = stub.answer(prompt_text, case)
                    metadata = result.metadata
                else:
                    prediction = asyncio.run(
                        provider.generate_answer(
                            "You are a helpful assistant.",
                            case.prompt,
                            "",
                            priority="background",
                        )
                    )
                    metadata = getattr(provider, "last_prompt_metadata", result.metadata)
                latency_ms = (time.monotonic() - start) * 1000
                passed = prediction.strip().lower() == case.expected.strip().lower()
                response_tokens_estimate = max(1, len(prediction) // 4) if prediction else 0
                record = EvalCaseResult(
                    case_id=case.case_id,
                    task_type=case.task_type,
                    strategy=metadata.strategy.value,
                    passed=passed,
                    expected=case.expected,
                    prediction=prediction,
                    prompt_tokens_estimate=metadata.prompt_tokens_estimate,
                    response_tokens_estimate=response_tokens_estimate,
                    latency_ms=latency_ms,
                    prompt_hash_before=metadata.prompt_hash_before,
                    prompt_hash_after=metadata.prompt_hash_after,
                    repeat_factor=metadata.repeat_factor,
                    step_by_step_used=metadata.step_by_step_used,
                    safe_mode_degraded=metadata.safe_mode_degraded,
                )
                handle.write(json.dumps(record.to_dict()) + "\n")
                results.append(record)
                _LOG.info(json.dumps({"event": "promptops.ab_eval_case", **record.to_dict()}))
    return results


def _settings_for_strategy(
    base: PromptStrategySettings, strategy: PromptStrategy
) -> PromptStrategySettings:
    if strategy == PromptStrategy.BASELINE:
        return replace(base, strategy_default=PromptStrategy.BASELINE, strategy_auto_mode=False)
    if strategy == PromptStrategy.REPEAT_2X:
        return replace(base, strategy_default=PromptStrategy.REPEAT_2X, strategy_auto_mode=False)
    if strategy == PromptStrategy.REPEAT_3X:
        return replace(base, strategy_default=PromptStrategy.REPEAT_3X, strategy_auto_mode=False)
    if strategy == PromptStrategy.STEP_BY_STEP:
        return replace(
            base,
            strategy_default=PromptStrategy.STEP_BY_STEP,
            enable_step_by_step=True,
            strategy_auto_mode=False,
        )
    if strategy == PromptStrategy.STEP_BY_STEP_PLUS_REPEAT_2X:
        return replace(
            base,
            strategy_default=PromptStrategy.STEP_BY_STEP_PLUS_REPEAT_2X,
            enable_step_by_step=True,
            strategy_auto_mode=False,
        )
    return base


def parse_strategies(raw: str) -> list[PromptStrategy]:
    mapping = {
        "baseline": PromptStrategy.BASELINE,
        "repeat2": PromptStrategy.REPEAT_2X,
        "repeat3": PromptStrategy.REPEAT_3X,
        "step": PromptStrategy.STEP_BY_STEP,
        "step+repeat2": PromptStrategy.STEP_BY_STEP_PLUS_REPEAT_2X,
        "repeat_2x": PromptStrategy.REPEAT_2X,
        "repeat_3x": PromptStrategy.REPEAT_3X,
        "step_by_step": PromptStrategy.STEP_BY_STEP,
        "step_by_step_plus_repeat_2x": PromptStrategy.STEP_BY_STEP_PLUS_REPEAT_2X,
    }
    strategies: list[PromptStrategy] = []
    for item in (part.strip().lower() for part in raw.split(",")):
        if not item:
            continue
        if item not in mapping:
            raise ValueError(f"Unknown strategy: {item}")
        strategies.append(mapping[item])
    return strategies
