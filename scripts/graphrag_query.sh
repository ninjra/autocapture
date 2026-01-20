#!/usr/bin/env bash
set -euo pipefail

QUERY="${1:-}"
CORPUS_ID="${2:-default}"
API_URL="${GRAPH_SERVICE_URL:-http://127.0.0.1:8020}"

if [[ -z "${QUERY}" ]]; then
  echo "Usage: scripts/graphrag_query.sh <query> [corpus_id]" >&2
  exit 1
fi

curl -sS -X POST "${API_URL}/graphrag/query" \
  -H "Content-Type: application/json" \
  -d "{\"corpus_id\":\"${CORPUS_ID}\",\"query\":\"${QUERY}\",\"limit\":20}"
