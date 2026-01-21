-- Autocapture tracking store schema.
-- Originating_MOD_IDs: [MISSING_VALUE]
-- Excluded_MISSING_VALUE_Items: [MISSING_VALUE]
-- Source: autocapture/tracking/store.py

CREATE TABLE IF NOT EXISTS host_events (
  id TEXT PRIMARY KEY,
  ts_start_ms INTEGER NOT NULL,
  ts_end_ms INTEGER NOT NULL,
  kind TEXT NOT NULL,
  session_id TEXT,
  app_name TEXT,
  window_title TEXT,
  payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_host_events_ts_start ON host_events(ts_start_ms);
CREATE INDEX IF NOT EXISTS idx_host_events_kind ON host_events(kind);
CREATE INDEX IF NOT EXISTS idx_host_events_app ON host_events(app_name);
