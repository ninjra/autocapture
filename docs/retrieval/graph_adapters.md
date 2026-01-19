# Graph adapters

Graph adapters let retrieval call graph-style workers (GraphRAG / HyperGraphRAG / Hyper-RAG).
Adapters are disabled by default and must be enabled explicitly in configuration.

## Run graph workers
```bash
poetry run autocapture graph-worker
```

## Configure adapters
```yaml
retrieval:
  graph_adapters:
    graphrag:
      enabled: true
      base_url: "http://127.0.0.1:8020"
```

Each adapter uses `/{adapter}/query` on the configured `base_url`.

## Indexing
To build/refresh graph workspaces:
```
POST /graphrag/index
{
  "corpus_id": "default",
  "time_range": {"start": "...", "end": "..."}
}
```

## Querying
```
POST /graphrag/query
{
  "corpus_id": "default",
  "query": "what changed last week",
  "limit": 20
}
```

Responses return `hits[]` with `event_id`, `score`, and `snippet`.
