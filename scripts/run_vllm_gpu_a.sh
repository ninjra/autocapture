#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL_A:-${1:-}}"
PORT="${PORT_A:-8001}"
GPU="${GPU_A:-0}"

if [[ -z "${MODEL}" ]]; then
  echo "Usage: scripts/run_vllm_gpu_a.sh <model-name>" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

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
  --model "${MODEL}"
