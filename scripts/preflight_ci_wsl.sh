#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found in WSL. Install Poetry and re-run." >&2
  exit 1
fi

export AUTOCAPTURE_TEST_MODE=1
export AUTOCAPTURE_GPU_MODE=off

echo "== Autocapture CI preflight (WSL/Linux) =="
poetry install --with dev --extras "sqlcipher"
poetry run black --check .
poetry run ruff check .
poetry run pytest -q -m "not gpu"
poetry run python .tools/memory_guard.py --check
poetry run python tools/verify_checksums.py
poetry run python tools/license_guardrails.py
poetry run python tools/release_gate.py
poetry run python -m autocapture.promptops.validate
poetry run python tools/pillar_gate.py
poetry run python tools/privacy_scanner.py
poetry run python tools/provenance_gate.py
poetry run python tools/coverage_gate.py
poetry run python tools/latency_gate.py
poetry run python tools/retrieval_sensitivity.py
poetry run python tools/no_evidence_gate.py
poetry run python tools/conflict_gate.py
poetry run python tools/integrity_gate.py
