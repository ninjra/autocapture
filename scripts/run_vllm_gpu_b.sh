#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL_B:-${1:-}}"
PORT="${PORT_B:-8002}"
GPU="${GPU_B:-1}"
KV_TRANSFER_CONFIG="${LMCACHE_KV_CONFIG:-{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}}"

if [[ -z "${MODEL}" ]]; then
  echo "Usage: scripts/run_vllm_gpu_b.sh <model-name>" >&2
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
  --model "${MODEL}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"
