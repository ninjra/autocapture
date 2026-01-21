-- Autocapture SQLite RAG core index schema (lexical/FTS).
-- Originating_MOD_IDs: [MISSING_VALUE]
-- Excluded_MISSING_VALUE_Items: [MISSING_VALUE]
-- Source: autocapture/indexing/lexical_index.py, autocapture/indexing/thread_index.py

CREATE VIRTUAL TABLE IF NOT EXISTS event_fts
USING fts5(event_id UNINDEXED, ocr_text, window_title, app_name, domain, url, agent_text);

CREATE VIRTUAL TABLE IF NOT EXISTS span_fts
USING fts5(span_id UNINDEXED, event_id UNINDEXED, frame_id UNINDEXED, text);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(chunk_id UNINDEXED, event_id UNINDEXED, text);

CREATE VIRTUAL TABLE IF NOT EXISTS thread_fts
USING fts5(thread_id UNINDEXED, title, summary, entities, tasks);
