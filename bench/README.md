# SPEC-1 LLM Bench

This folder contains the deterministic, offline LLM benchmark harness used for CI regression gating.

## Canonical command (single run)
```
poetry run python -m autocapture.bench.llm_bench \
  --case-id case1 \
  --offline \
  --replay-dir bench/fixtures/responses \
  --output bench/results/output_case1.txt
```

## Harness scripts
- Bash (Linux/WSL/macOS): `bench/run.sh`
- PowerShell (Windows): `bench/run.ps1`

Both scripts write:
- `bench/results/timing.tsv`
- `bench/results/summary.json`
- optional traces to `bench/results/traces/*.jsonl` when `BENCH_TRACE=1`

Environment controls:
- `BENCH_CASE_ID` (default: `case1`)
- `BENCH_ITERATIONS` (default: `20`)
- `BENCH_WARMUP` (default: `3`)
- `BENCH_TRACE` (default: `0`)

## Offline replay
Offline mode uses fixture responses in `bench/fixtures/responses`. It never performs network I/O and
fails fast if the fixture is missing. Response fixtures are sanitized and do not include prompts.

## Timing traces
Enable traces with:
```
BENCH_TRACE=1 bench/run.sh
```
Or add `--trace-timing --trace-timing-file <path>` to the canonical command. Trace fields are
redacted by default and should not include prompt text or file contents.

## Baseline refresh
1. Run the harness locally (offline): `bench/run.sh`
2. Copy `bench/results/summary.json` values into `bench/baseline.json` for the matching case.
3. Commit the updated baseline.

## Security notes
- Do not commit `bench/results/**`.
- `--record` is disabled in CI and intended only for controlled staging runs.
- Traces are redacted by default; avoid storing secrets in fixtures.
