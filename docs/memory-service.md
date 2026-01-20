# Memory Service (Organizational Memory)

The Memory Service is an additive, deterministic store for non-personal organizational memory.
It ingests structured memory proposals with provenance anchors and serves bounded, citable
"memory cards" for runtime context building.

## Scope & Guardrails
- **Non-personal only**: rejects people, preferences, PII, and secrets.
- **Fail-closed**: missing policy labels or provenance pointers are rejected.
- **Deterministic**: stable hashing, stable ranking, stable packing.
- **Offline-safe**: embedding/rerank providers default to deterministic stubs.

## Endpoints
- `POST /v1/memory/ingest`
- `POST /v1/memory/query`
- `POST /v1/memory/feedback`
- `GET /v1/memory/health`
- `GET /metrics`

## Run Locally
1. Ensure Postgres is running and `pgvector` is enabled.
2. Run migrations (same repo Alembic workflow):
   - `poetry run alembic upgrade head`
3. Start the service:
   - `poetry run autocapture memory-service`

You can override the service database URL via `memory_service.database_url` in config.

## Configuration Highlights
```yaml
memory_service:
  enabled: false
  bind_host: "127.0.0.1"
  port: 8030
  database_url: null
  default_namespace: "default"
  enable_ingest: true
  enable_query: true
  enable_feedback: true
  embedder:
    provider: "stub"
    dim: 256
  reranker:
    provider: "disabled"
  retrieval:
    max_cards: 8
    max_tokens: 1200
    max_per_type: 2
  policy:
    allowed_audiences: ["internal"]
    sensitivity_order: ["low", "medium", "high"]
    reject_person_text: false

features:
  enable_memory_service_write_hook: false
  enable_memory_service_read_hook: false
```

## Provenance & Citations
Each memory proposal must include **at least one** provenance pointer:
```json
{
  "artifact_version_id": "artifact_version_...",
  "chunk_id": "span_...",
  "start_offset": 120,
  "end_offset": 240,
  "excerpt_hash": "sha256..."
}
```
The worker write hook populates `artifact_versions` and `artifact_chunks` so the Memory Service
can verify provenance (`orphan_provenance` and `excerpt_hash_mismatch` are hard rejects).

## Deterministic Retrieval
The query path uses a hybrid candidate set (vector + keyword + graph), then applies stable
ranking weights and tie-breaks by `memory_id`. Packing enforces:
- `max_cards`, `max_tokens`, `max_per_type`
- fixed type priority order
- stable trimming (no randomization)

## Policy Rejections
Common reject reasons include:
- `policy_pii_detected`
- `policy_secret_detected`
- `policy_person_entity_detected`
- `policy_person_text_detected` (when enabled)
- `policy_preference_detected`
- `policy_audience_missing` / `policy_sensitivity_invalid`
- `missing_provenance`

## Integration Hooks
Two hooks are guarded by feature flags:
- **Write hook**: event worker calls the Memory Service after OCR persistence to ingest
  extraction proposals (no-op stub by default).
- **Read hook**: context builder fetches memory cards and inserts them into the context pack.

Enable flags in config to activate.
