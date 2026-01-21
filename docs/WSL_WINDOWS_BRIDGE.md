# WSL <-> Windows Bridge

This guide covers posting Windows-captured screenshots into the WSL backend for OCR/indexing.
The WSL API remains local-first and only accepts bridge uploads when explicitly configured.

## Configure the WSL backend
1) Set a bridge token in your config (`autocapture.yml`):
```
api:
  bridge_token: "replace-with-strong-secret"
```
2) Start the API from WSL:
```
poetry run autocapture api
```

## Post from Windows
Send a multipart request with a JSON metadata part and an image file:

```
curl -X POST http://<wsl-host>:8008/api/events/ingest ^
  -H "X-Bridge-Token: replace-with-strong-secret" ^
  -F "metadata={\"app_name\":\"Chrome\",\"window_title\":\"Example\",\"monitor_id\":\"1\"}" ^
  -F "image=@screenshot.png"
```

Metadata fields:
- `observation_id` (optional UUID for idempotency)
- `captured_at` (ISO8601 timestamp, optional; defaults to now)
- `app_name`, `window_title`, `monitor_id`, `is_fullscreen`
- `url`, `domain` (optional; persisted to capture metadata and propagated to events)

If the app/window is excluded by privacy filters, the server will skip storage before disk write.

## Storage & retention
The dashboard uses `/api/storage` to report:
- `screenshot_ttl_days`
- `media_usage_bytes`
- `media_path`
