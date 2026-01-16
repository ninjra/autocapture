from __future__ import annotations

from pathlib import Path

import yaml

from autocapture.config import AppConfig, CaptureConfig, DatabaseConfig, PromptOpsConfig
from autocapture.evals import EvalMetrics
from autocapture.promptops.evals import EvalCaseResult, EvalRunResult
from autocapture.promptops.runner import PromptOpsRunner, PromptProposal, SourceSnapshot
from autocapture.storage.database import DatabaseManager


def _build_config(tmp_path: Path, *, max_attempts: int) -> AppConfig:
    return AppConfig(
        capture=CaptureConfig(
            data_dir=tmp_path / "data",
            staging_dir=tmp_path / "staging",
        ),
        database=DatabaseConfig(url="sqlite:///:memory:"),
        promptops=PromptOpsConfig(
            enabled=True,
            max_attempts=max_attempts,
            eval_repeats=1,
        ),
    )


def _make_proposal(marker: str) -> PromptProposal:
    raw_path = Path("prompts/raw/answer_with_context_pack.yaml")
    derived_path = Path("autocapture/prompts/derived/answer_with_context_pack.yaml")
    data = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    for key in ("system_prompt", "raw_template", "derived_template"):
        data[key] = f"{data[key]}\n{marker}"
    data["version"] = "v99"
    content = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return PromptProposal(
        name=data["name"],
        raw_path=raw_path,
        derived_path=derived_path,
        raw_content=content,
        derived_content=content,
        rationale="test",
    )


def _build_case(query: str) -> EvalCaseResult:
    return EvalCaseResult(
        query=query,
        refused=False,
        citations_found=["E1"],
        citation_hit=True,
        verifier_pass=True,
        verifier_errors=[],
        latency_ms=1000.0,
    )


def test_promptops_agentic_selects_best_candidate(tmp_path: Path, monkeypatch) -> None:
    config = _build_config(tmp_path, max_attempts=3)
    db = DatabaseManager(config.database)
    runner = PromptOpsRunner(config, db)

    source = SourceSnapshot(
        source="local",
        fetched_at="now",
        status="ok",
        sha256="abc",
        path="path",
        error=None,
        is_local=True,
        excerpt="notes",
    )
    monkeypatch.setattr(runner, "_fetch_sources", lambda run_id, sources: [source])

    attempt_state = {"current": 0}

    def fake_propose(*_args, **_kwargs):
        attempt_state["current"] += 1
        return [_make_proposal(f"attempt-{attempt_state['current']}")]

    monkeypatch.setattr(runner, "_propose_prompts", fake_propose)

    def fake_run_eval_detailed(*_args, **kwargs):
        overrides = kwargs.get("overrides")
        if overrides is None:
            metrics = EvalMetrics(0.7, 0.7, 0.1, 1000.0)
        else:
            attempt = attempt_state["current"]
            if attempt == 1:
                metrics = EvalMetrics(0.72, 0.71, 0.1, 1000.0)
            elif attempt == 2:
                metrics = EvalMetrics(0.75, 0.73, 0.09, 900.0)
            else:
                metrics = EvalMetrics(0.72, 0.71, 0.12, 1100.0)
        return EvalRunResult(metrics=metrics, cases=[_build_case("query")])

    import autocapture.promptops.runner as runner_module

    monkeypatch.setattr(runner_module, "run_eval_detailed", fake_run_eval_detailed)

    run = runner.run_once()
    assert run is not None
    assert run.status == "completed_no_pr"
    assert run.eval_results["selected_attempt"] == 2
    assert len(run.eval_results["attempts"]) == config.promptops.max_attempts
    patch_path = config.capture.data_dir / "promptops" / "patches" / f"promptops_{run.run_id}.diff"
    assert patch_path.exists()
    assert "attempt-2" in patch_path.read_text(encoding="utf-8")


def test_promptops_agentic_skips_when_no_candidate_passes(tmp_path: Path, monkeypatch) -> None:
    config = _build_config(tmp_path, max_attempts=2)
    db = DatabaseManager(config.database)
    runner = PromptOpsRunner(config, db)

    source = SourceSnapshot(
        source="local",
        fetched_at="now",
        status="ok",
        sha256="abc",
        path="path",
        error=None,
        is_local=True,
        excerpt="notes",
    )
    monkeypatch.setattr(runner, "_fetch_sources", lambda run_id, sources: [source])

    attempt_state = {"current": 0}

    def fake_propose(*_args, **_kwargs):
        attempt_state["current"] += 1
        return [_make_proposal(f"attempt-{attempt_state['current']}")]

    monkeypatch.setattr(runner, "_propose_prompts", fake_propose)

    def fake_run_eval_detailed(*_args, **kwargs):
        overrides = kwargs.get("overrides")
        if overrides is None:
            metrics = EvalMetrics(0.7, 0.7, 0.1, 1000.0)
        else:
            metrics = EvalMetrics(0.7, 0.7, 0.1, 1000.0)
        return EvalRunResult(metrics=metrics, cases=[_build_case("query")])

    import autocapture.promptops.runner as runner_module

    monkeypatch.setattr(runner_module, "run_eval_detailed", fake_run_eval_detailed)

    run = runner.run_once()
    assert run is not None
    assert run.status == "skipped_no_acceptable_proposal"
    patch_dir = config.capture.data_dir / "promptops" / "patches"
    assert not patch_dir.exists() or not list(patch_dir.glob("*.diff"))
