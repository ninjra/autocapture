# SPEC-260118 Discovery Log
Date: 2026-01-20
Branch: main

## Repo state
- git status: clean
- branch: main

## Service patterns (FastAPI)
- Local API app: `autocapture/api/server.py:400-1700` (create_app + /api/context-pack, /api/retrieve, /api/answer).
- Gateway app: `autocapture/gateway/app.py:26-175` (create_gateway_app + /v1/* + /metrics).
- Graph worker app: `autocapture/graph/app.py:27-109` (create_graph_app + /{adapter}/index|query + /metrics).
- Service entrypoints: `autocapture/main.py:603-658` (uvicorn for api/gateway/graph-worker).

## Runtime context assembly
- Context pack builder: `autocapture/memory/context_pack.py:159-183`.
- AnswerGraph constructs context packs: `autocapture/agents/answer_graph.py:150-299`.
- /api/context-pack endpoint: `autocapture/api/server.py:1512-1605`.
- Memory snapshots optional: `_maybe_build_memory_snapshot` in `autocapture/api/server.py:577-627`.

## Normalization + chunking
- Text normalization helper: `autocapture/text/normalize.py:27-34`.
- OCR ingest normalizes text: `autocapture/worker/event_worker.py:276-389`.
- OCR span chunking + citable spans with offsets: `autocapture/worker/event_worker.py:755-862`.
- Deterministic memory store chunking: `autocapture/memory/store.py:81-220` and `_chunk_text` `autocapture/memory/store.py:750-778`.

## Policy gates / redaction
- Capture privacy gate: `autocapture/capture/privacy.py:21-55`.
- Memory policy engine (blocked/exclude/redact): `autocapture/memory/policy.py:11-62`.
- Prompt-injection redaction for context packs: `autocapture/memory/prompt_injection.py:9-58` and `autocapture/memory/context_pack.py:60-100`.
- Secret redaction helpers: `autocapture/security/redaction.py:8-60`.

## Artifact + provenance storage (SQLAlchemy)
- Artifact records: `autocapture/storage/models.py:645-664`.
- Citable spans (start/end offsets): `autocapture/storage/models.py:667-699`.
- OCR span table (raw spans): `autocapture/storage/models.py:383-418`.
- Existing memory store schema (SQLite): `autocapture/memory/schema.py:21-152`.

## Migrations / DB
- Alembic config: `alembic.ini:1-35`.
- Alembic run in DatabaseManager: `autocapture/storage/database.py:99-127`.
- Example migration + FTS table creation: `alembic/versions/0014_spec1_runs_and_citations.py:19-155`.
- Docker compose includes Postgres: `docker-compose.yml:1-17`.

## Observability
- Prometheus metrics registry: `autocapture/observability/metrics.py:1-120`.
- Structured logging with redaction: `autocapture/logging_utils.py:1-104`.

## CI / commands
- Local checks: `Makefile:5-8`.
- CI workflow: `.github/workflows/ci.yml:1-164`.
- Dev check workflow: `.github/workflows/dev-check.yml:1-25`.

