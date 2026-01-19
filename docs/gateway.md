# LLM Gateway

The gateway is an OpenAI-compatible proxy and the single enforcement point for
stage policy, citations, and fallback routing.

## Run
```bash
poetry run autocapture gateway
```

Defaults are controlled by `gateway.*` in `autocapture.yml` / `config/example.yml`.

## Endpoints
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /internal/stage/{stage_id}/chat.completions`
- `GET /health`, `GET /ready`, `GET /metrics`

## Configuration
- `gateway.*`: bind host/port, request size limit, upstream probe timeout.
- `model_registry.providers`: upstream base URLs, retries, circuit breaker config.
- `model_registry.models`: provider linkage, LoRA allowlist, quantization metadata.
- `model_registry.stages`: stage policy (fallbacks, decode strategy, citations/claims).

## Claims + citations
To require claim-level JSON outputs:
- `model_registry.stages[*].requirements.require_json = true`
- `model_registry.stages[*].requirements.claims_schema = "claims_json_v1"`

The gateway validates claims deterministically and optionally repairs invalid outputs.
