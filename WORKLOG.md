# Worklog

## YYYY-MM-DD
- Summary:
- Files touched:
- Tests:
- Notes:

## 2026-01-15
- Summary: Added dev-safe embedder fallback + smoke command, tightened dev.ps1 check formatting gate, and disabled optional backends in dev mode.
- Files touched: README.md, autocapture/config.py, autocapture/embeddings/service.py, autocapture/indexing/vector_index.py, autocapture/main.py, autocapture/smoke.py, autocapture/worker/supervisor.py, dev.ps1, tests/test_embedding_fallback.py, tests/test_smoke.py.
- Tests: `poetry run black --check .`, `poetry run ruff check .`, `poetry run pytest -q` (timed out), `poetry run pytest -q tests/test_embedding_fallback.py tests/test_smoke.py`, `APP_ENV=dev LOCALAPPDATA=... poetry run autocapture smoke`.
- Notes: PowerShell not available to run `dev.ps1` directly (pwsh/powershell missing); `poetry run autocapture` failed here due to headless `mss` display error; `poetry run pytest -q` hangs in this environment (FastAPI TestClient does not return).
