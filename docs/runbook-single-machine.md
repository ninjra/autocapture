# Single-Machine Runbook (SPEC-SINGLE-MACHINE)

This runbook brings up the full single-machine stack (WSL2-friendly) with mandatory services.

## Prerequisites
- Poetry installed and configured.
- Docker Desktop with WSL2 integration.

## Install dependencies
```bash
poetry install --with dev
```

## Start infra services (Qdrant + Prometheus + Loki + Grafana)
```bash
docker compose -f infra/compose.yaml up -d
```

## Start graph workers
```bash
scripts/run_graph_worker.sh
```

## Index + query GraphRAG (Graph service API)
```bash
scripts/graphrag_index.sh default
scripts/graphrag_query.sh "your query" default
```

## Start Gateway + API
```bash
scripts/run_gateway.sh
scripts/run_api.sh
```

## Configure model registry
- Update `autocapture.yml` and `config/example.yml` `model_registry.*` entries to match the
  model names and ports you start with `run_vllm_*` scripts.

## Start vLLM instances
```bash
scripts/run_vllm_gpu_a.sh <model-name>
scripts/run_vllm_gpu_b.sh <model-name>
scripts/run_vllm_cpu.sh <model-name>
```

Notes:
- GPU B uses LMCache by default. Override the JSON via `LMCACHE_KV_CONFIG` if needed.

## Smoke checks
```bash
curl -fsS http://127.0.0.1:8010/healthz
curl -fsS http://127.0.0.1:8010/readyz
curl -fsS http://127.0.0.1:8010/v1/models
curl -fsS http://127.0.0.1:8020/healthz
curl -fsS http://127.0.0.1:8020/readyz
curl -fsS http://127.0.0.1:9090/-/ready
```

## Notes
- Gateway internal stage calls require `X-Internal-Token` (or `Authorization: Bearer <token>`) when not loopback.
- Graph worker CLI wrappers live in `scripts/*_worker.sh` and proxy to `autocapture.graph.cli_worker`.
- Metrics are exposed at `/metrics` on the Gateway and Graph ports; Prometheus scrapes via `infra/prometheus.yml`.
