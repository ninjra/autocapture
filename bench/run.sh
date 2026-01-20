#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/bench/results"
TRACE_DIR="$RESULTS_DIR/traces"
CASE_ID="${BENCH_CASE_ID:-case1}"
ITERATIONS="${BENCH_ITERATIONS:-20}"
WARMUP="${BENCH_WARMUP:-3}"
TRACE="${BENCH_TRACE:-0}"
REPLAY_DIR="$ROOT_DIR/bench/fixtures/responses"
TIMING_FILE="$RESULTS_DIR/timing.tsv"
SUMMARY_FILE="$RESULTS_DIR/summary.json"
OUTPUT_FILE="$RESULTS_DIR/output_${CASE_ID}.txt"

mkdir -p "$RESULTS_DIR" "$TRACE_DIR"

printf "iteration\tms\n" > "$TIMING_FILE"

run_once() {
  local trace_file="$1"
  if [[ -n "$trace_file" ]]; then
    poetry run python -m autocapture.bench.llm_bench \
      --case-id "$CASE_ID" \
      --offline \
      --replay-dir "$REPLAY_DIR" \
      --output "$OUTPUT_FILE" \
      --trace-timing \
      --trace-timing-file "$trace_file"
  else
    poetry run python -m autocapture.bench.llm_bench \
      --case-id "$CASE_ID" \
      --offline \
      --replay-dir "$REPLAY_DIR" \
      --output "$OUTPUT_FILE"
  fi
}

for ((i=1; i<=WARMUP; i++)); do
  run_once ""
done

for ((i=1; i<=ITERATIONS; i++)); do
  trace_file=""
  if [[ "$TRACE" == "1" ]]; then
    trace_file="$TRACE_DIR/run_${i}.jsonl"
  fi
  start_ns=$(date +%s%N)
  run_once "$trace_file"
  end_ns=$(date +%s%N)
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
  printf "%s\t%s\n" "$i" "$elapsed_ms" >> "$TIMING_FILE"
done

poetry run python -m autocapture.bench.summary \
  --input "$TIMING_FILE" \
  --output "$SUMMARY_FILE" \
  --case-id "$CASE_ID" \
  --mode "offline"
