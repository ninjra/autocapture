#!/usr/bin/env bash
set -euo pipefail

CORPUS_ID="${1:-default}"
API_URL="${GRAPH_SERVICE_URL:-http://127.0.0.1:8020}"

curl -sS -X POST "${API_URL}/graphrag/index" \
  -H "Content-Type: application/json" \
  -d "{\"corpus_id\":\"${CORPUS_ID}\"}"
