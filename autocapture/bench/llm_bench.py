"""Deterministic offline LLM benchmark command."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ..llm.prompt_strategy import PromptStrategy, PromptStrategySettings
from ..llm.providers import OpenAICompatibleProvider, OpenAIProvider
from ..llm.transport import HttpxTransport, LLMTransport, RecordingTransport, ReplayTransport
from ..policy import PolicyEnvelope
from .timing import TimingTracer

_DEFAULT_SYSTEM_PROMPT = "You are a concise assistant."


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    system_prompt: str
    user_prompt: str
    context: str
    provider: str
    model: str
    temperature: float


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autocapture LLM benchmark")
    parser.add_argument("--case-id", default="case1", help="Fixture case id.")
    parser.add_argument(
        "--fixture",
        default=None,
        help="Optional path to bench fixture (defaults to bench/fixtures/<case-id>.json).",
    )
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Replay fixture directory (default: bench/fixtures/responses).",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable offline replay mode (default: true).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record live responses to fixtures (disabled in CI).",
    )
    parser.add_argument("--output", default=None, help="Output file path override.")
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider override (openai|openai_compatible).",
    )
    parser.add_argument("--model", default=None, help="Model override.")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature override.")
    parser.add_argument("--timeout-s", type=float, default=20.0, help="Request timeout seconds.")
    parser.add_argument("--retries", type=int, default=0, help="Retry attempts for live mode.")
    parser.add_argument("--api-key", default=None, help="API key override for live mode.")
    parser.add_argument("--base-url", default=None, help="Base URL for openai_compatible.")
    parser.add_argument("--trace-timing", action="store_true", help="Enable JSONL timing trace.")
    parser.add_argument(
        "--trace-timing-file",
        default=None,
        help="Write timing JSONL to file (defaults to stderr when tracing enabled).",
    )
    parser.add_argument(
        "--trace-timing-redact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redact trace fields (default: true).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    repo_root = _repo_root()
    fixture_path = Path(args.fixture) if args.fixture else _default_fixture(repo_root, args.case_id)
    replay_dir = (
        Path(args.replay_dir) if args.replay_dir else repo_root / "bench" / "fixtures" / "responses"
    )
    output_path = (
        Path(args.output)
        if args.output
        else repo_root / "bench" / "results" / f"output_{args.case_id}.txt"
    )

    if args.record and args.offline:
        raise SystemExit("--record requires --no-offline")
    if args.record and _in_ci():
        raise SystemExit("--record disabled in CI")

    case = _load_case(fixture_path, args)
    trace_enabled = args.trace_timing or bool(args.trace_timing_file)
    tracer = TimingTracer(
        enabled=trace_enabled,
        redact=args.trace_timing_redact,
        file_path=Path(args.trace_timing_file) if args.trace_timing_file else None,
    )
    policy = PolicyEnvelope(None)

    with tracer:
        with tracer.span("total", case_id=case.case_id, mode="offline" if args.offline else "live"):
            with tracer.span("load_config", case_id=case.case_id):
                prompt_settings = _prompt_settings()

            with tracer.span(
                "init_runtime",
                provider=case.provider,
                mode="offline" if args.offline else "live",
            ):
                transport = _build_transport(args.offline, args.record, replay_dir, case.case_id)
                provider = _build_provider(case, args, transport, prompt_settings)

            with tracer.span("scan_files", context_chars=len(case.context)):
                context = case.context

            with tracer.span(
                "build_request",
                system_prompt_chars=len(case.system_prompt),
                user_prompt_chars=len(case.user_prompt),
            ):
                system_prompt = case.system_prompt
                user_prompt = case.user_prompt

            with tracer.span("api_call", provider=case.provider, offline=args.offline):
                answer = asyncio.run(
                    policy.execute_stage(
                        stage=None,
                        provider=provider,
                        decision=None,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        context_pack_text=context,
                        temperature=case.temperature,
                    )
                )

            with tracer.span("apply_patch", output_chars=len(answer)):
                output_text = answer.strip()

            with tracer.span("render_output", output_path=str(output_path)):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(output_text + "\n", encoding="utf-8")

    return 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_fixture(repo_root: Path, case_id: str) -> Path:
    return repo_root / "bench" / "fixtures" / f"{case_id}.json"


def _load_case(path: Path, args: argparse.Namespace) -> BenchCase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Fixture {path} must contain a JSON object")
    case_id = str(payload.get("case_id") or args.case_id)
    system_prompt = str(payload.get("system_prompt") or _DEFAULT_SYSTEM_PROMPT)
    user_prompt = str(payload.get("user_prompt") or "")
    context = str(payload.get("context") or "")
    provider = str(args.provider or payload.get("provider") or "openai")
    model = str(args.model or payload.get("model") or "gpt-4.1-mini")
    temperature = float(
        args.temperature if args.temperature is not None else payload.get("temperature", 0.2)
    )
    if not user_prompt:
        raise ValueError(f"Fixture {path} missing user_prompt")
    return BenchCase(
        case_id=case_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context=context,
        provider=provider,
        model=model,
        temperature=temperature,
    )


def _prompt_settings() -> PromptStrategySettings:
    return PromptStrategySettings(
        strategy_default=PromptStrategy.BASELINE,
        prompt_repeat_factor=1,
        enable_step_by_step=False,
        step_by_step_phrase="Let's think step by step.",
        step_by_step_two_stage=False,
        max_prompt_chars_for_repetition=12000,
        max_tokens_headroom=512,
        max_context_tokens=8192,
        force_no_reasoning=False,
        strategy_auto_mode=False,
        repetition_delimiter="\n\n---\n\n",
        store_prompt_transforms=False,
        prompt_store_redaction=True,
        data_dir=None,
    )


def _build_transport(offline: bool, record: bool, replay_dir: Path, case_id: str) -> LLMTransport:
    if offline:
        return ReplayTransport(replay_dir, case_id=case_id)
    transport: LLMTransport = HttpxTransport()
    if record:
        transport = RecordingTransport(transport, replay_dir, case_id=case_id)
    return transport


def _build_provider(
    case: BenchCase,
    args: argparse.Namespace,
    transport: LLMTransport,
    prompt_settings: PromptStrategySettings,
):
    if case.provider == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key and not args.offline:
            raise RuntimeError("OPENAI_API_KEY required for live runs")
        if not api_key:
            api_key = "offline"
        return OpenAIProvider(
            api_key,
            case.model,
            timeout_s=args.timeout_s,
            retries=args.retries,
            prompt_strategy=prompt_settings,
            transport=transport,
        )
    if case.provider == "openai_compatible":
        base_url = args.base_url or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        if not base_url:
            base_url = "http://127.0.0.1:11434"
        api_key = args.api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        return OpenAICompatibleProvider(
            base_url,
            case.model,
            api_key=api_key,
            timeout_s=args.timeout_s,
            retries=args.retries,
            prompt_strategy=prompt_settings,
            transport=transport,
        )
    raise ValueError(f"Unsupported provider: {case.provider}")


def _in_ci() -> bool:
    for key in ("CI", "GITHUB_ACTIONS", "BUILD_BUILDID", "TF_BUILD"):
        if os.environ.get(key):
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
