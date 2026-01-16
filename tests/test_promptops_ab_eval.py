from pathlib import Path

from autocapture.config import AppConfig
from autocapture.promptops.ab_eval import load_eval_cases, parse_strategies, run_ab_eval


def test_ab_eval_runs_and_writes_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "report.jsonl"
    cases_path = Path("evals/promptops_ab_cases.json")
    strategies = parse_strategies("baseline,repeat2")
    results = run_ab_eval(
        strategies,
        cases_path=cases_path,
        output_path=output_path,
        config=AppConfig(),
    )
    case_count = len(load_eval_cases(cases_path))
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(results)
    assert len(results) == 2 * case_count
