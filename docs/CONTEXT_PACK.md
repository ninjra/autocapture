# Context Pack v1

## JSON Shape
`/api/context-pack` returns:

```json
{
  "version": 1,
  "query": "...",
  "generated_at": "ISO",
  "evidence": [
    {
      "id": "E1",
      "ts_start": "ISO",
      "ts_end": null,
      "source": "...",
      "title": "...",
      "text": "verbatim extracted snippet",
      "meta": {
        "event_id": "uuid",
        "domain": "...",
        "score": 0.87,
        "screenshot_path": "path-or-null",
        "screenshot_hash": "hash-or-null",
        "spans": [{"span_id": "S12", "start": 120, "end": 260, "conf": 0.93}]
      }
    }
  ],
  "warnings": []
}
```

## TRON Payload (Optional)

Context packs can be serialized as TRON via `output.context_pack_format=tron`.

## Memory snapshot (optional)

When enabled, `/api/context-pack` includes a `memory_snapshot` payload containing the
deterministic memory snapshot plus its manifest. If memory hotness is enabled and
requested, the manifest includes a `memory_hotness` block with ranking metadata.

Request fields:
- `memory_hotness_mode`: `off` | `as_of` | `dynamic`
- `memory_hotness_as_of_utc`: required when `mode=as_of`; optional when `mode=dynamic`
