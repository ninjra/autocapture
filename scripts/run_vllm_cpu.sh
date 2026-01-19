#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL_CPU:-${1:-}}"
PORT="${PORT_CPU:-8003}"

if [[ -z "${MODEL}" ]]; then
  echo "Usage: scripts/run_vllm_cpu.sh <model-name>" >&2
  exit 1
fi

python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("vllm") is None:
    print("vLLM is not installed. Install the vLLM package in this Poetry env.", file=sys.stderr)
    sys.exit(1)
PY

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model "${MODEL}" \
  --device cpu
