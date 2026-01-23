# Web Console Only Refactor Checklist

Date: 2026-01-23
Branch: refactor/web-console-only

## Scope
- Remove WPF build surface; keep FastAPI-served web UI as the only UI.
- Add events browsing stack + media variants + focus endpoint.
- Introduce Vite/React/TS console source (build output to autocapture/ui/web).
- Add CLI parity command: autocapture ui open.

## Baseline anchors (current repo reality)
- FastAPI mounts /static from autocapture/ui/web: autocapture/api/app.py
- Root / serves index.html from autocapture/ui/web: autocapture/api/routers/core.py
- Unlock guard uses Authorization: Bearer <token> or ?unlock=...: autocapture/api/security_helpers.py
- Unlock endpoint: POST /api/unlock: autocapture/api/routers/core.py
- Protected prefixes include /api/state, /api/settings, /api/storage, /api/plugins, /api/audit: autocapture/api/middleware/stack.py
- Existing event detail: GET /api/event/{event_id}: autocapture/api/routers/core.py
- Existing screenshot endpoint: GET /api/screenshot/{event_id}?variant=full|focus: autocapture/api/routers/core.py

## Decisions (proposed)
- Cursor encoding: base64url JSON with {ts_start, event_id, direction}.
- Stable sort key: ts_start DESC, event_id DESC (tie-breaker).
- Pagination defaults: limit=100 when omitted; max limit=500 (clamp).
- OCR search: reuse LexicalIndex (event_fts) when available; fallback to LIKE in LexicalIndex.
- Thumbnails: generate on-demand from existing media; width = config.capture.thumbnail_width.
- Thumbnail cache: data_dir/state/thumbs (or config/cache if added); encrypt cached thumbs if encryption_mgr is available.
- Vite base path: /static/ so assets resolve under FastAPI mount; index served at /.
- Built UI output: keep committed output in autocapture/ui/web with stable file names to minimize churn.

## Module checklist

### WPF decommission
- Archive (preferred) or remove:
  - archive/wpf-shell/autocapture.sln
  - archive/wpf-shell/src/Autocapture.Shell/**
- Update any doc references (only spec mention currently).
- Add guard in CI/scripts to prevent reintroduction (rg in workflows).

### Events browsing stack
- Storage queries:
  - autocapture/storage/queries/events.py (new)
  - Use EventRecord fields: ts_start, app_name, window_title, domain, url, ocr_text, tags
  - Cursor-based pagination + filters + q search via LexicalIndex
- UX service:
  - autocapture/ux/events_service.py (new)
  - list_events, get_facets, get_event_detail
  - Build thumb URLs (screenshot/focus) for list responses
- Router:
  - autocapture/api/routers/events.py (new)
  - GET /api/events
  - GET /api/events/facets
  - GET /api/events/{event_id}
  - Register in autocapture/api/app.py (or router build).
- Legacy alias:
  - /api/event/{event_id} should call new service or be aliased.

### Media variants + focus endpoint
- Service:
  - autocapture/ux/media_service.py (new)
  - Path validation under config.capture.data_dir
  - Decrypt .acenc via encryption_mgr
  - Thumb generation (Pillow) + cache
- Router:
  - Update /api/screenshot/{event_id} to accept variant=thumb|full
  - Add /api/focus/{event_id}?variant=thumb|full

### Storage stats
- Add GET /api/storage/stats (router + service)
- Use cached folder size computation (similar to existing /api/storage).

### Settings schema enrichment
- Update settings schema builder to include:
  - sensitive, requires_restart, danger_level
- Effective settings should redact sensitive values (already uses redact_payload).

### Plugins schema/details/preview/apply
- Add endpoints:
  - GET /api/plugins/schema
  - GET /api/plugins/{id}
  - POST /api/plugins/{id}/preview
  - POST /api/plugins/{id}/apply
- Use PreviewTokenManager (same pattern as settings/delete).
- Persist configs under settings.json plugins.configs.

### Web console (Vite/React/TS)
- Source:
  - autocapture/ui/console/** (new)
  - Vite config builds into ../web
  - Base path /static/
- Pages: Dashboard, Ask, Search, Explorer, Highlights, Settings, Plugins, Maintenance, Audit, Logs/Perf.
- Explorer:
  - virtualized grid, thumb-only tiles, full image on demand.
- Unlock handling:
  - store session token only; attach Authorization header each request.

### Packaging + CI
- Keep pyproject.toml include of autocapture/ui/web/**
- Keep pyinstaller.spec bundling autocapture/ui/web
- Add Node build step in CI (no runtime Node).

### CLI parity
- Add autocapture ui open in autocapture/main.py
- Use webbrowser to open local URL and print it.

## Tests (targeted)
- Backend:
  - tests for /api/events list/facets/detail
  - tests for /api/screenshot and /api/focus variants
  - tests for cursor pagination
- UI:
  - npm/pnpm build output -> autocapture/ui/web
- Smoke:
  - GET / returns HTML
  - /static asset served

## Stop conditions
- If FastAPI mount/index paths differ from baseline, pause and re-map.
- If unlock guard contract differs (header/cookie), pause and align.
- If EventRecord schema lacks required fields, pause to confirm migrations needed.
