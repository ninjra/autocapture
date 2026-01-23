"""SQLite feature probes."""

from __future__ import annotations

from sqlalchemy import text


def sqlite_fts5_available(engine) -> bool:
    if engine.dialect.name != "sqlite":
        return False
    with engine.begin() as conn:
        try:
            enabled = conn.execute(
                text("SELECT sqlite_compileoption_used('ENABLE_FTS5')")
            ).scalar()
            if int(enabled or 0) == 1:
                return True
        except Exception:
            pass
        try:
            conn.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(content)"))
            conn.execute(text("DROP TABLE IF EXISTS _fts5_probe"))
            return True
        except Exception:
            return False
