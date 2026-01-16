"""CLI for PromptOps A/B prompt strategy evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from autocapture import load_config
from autocapture.config import AppConfig
from autocapture.promptops.ab_eval import parse_strategies, run_ab_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptOps A/B prompt strategy evals.")
    parser.add_argument(
        "--strategies",
        default="baseline,repeat2,repeat3,step,step+repeat2",
        help="Comma-separated list of strategies.",
    )
    parser.add_argument(
        "--cases",
        default="evals/promptops_ab_cases.json",
        help="Path to eval cases JSON.",
    )
    parser.add_argument(
        "--output",
        default="evals/promptops_ab_report.jsonl",
        help="Output JSONL path for per-case results.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path (defaults to autocapture.yml or built-in defaults).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live LLM providers (requires RUN_LIVE_EVALS=1).",
    )
    args = parser.parse_args()

    config: AppConfig
    if args.config:
        config = load_config(Path(args.config))
    else:
        config = AppConfig()

    strategies = parse_strategies(args.strategies)
    run_ab_eval(
        strategies,
        cases_path=Path(args.cases),
        output_path=Path(args.output),
        config=config,
        use_live=args.live,
    )


if __name__ == "__main__":
    main()
