"""SQLite vector/spans_v2 backend tables.

Revision ID: 0016_sqlite_vector_backends
Revises: 0015_spec260118_memory_service
Create Date: 2026-01-21
"""

from __future__ import annotations

from alembic import op

revision = "0016_sqlite_vector_backends"
down_revision = "0015_spec260118_memory_service"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    op.execute(
        "CREATE TABLE IF NOT EXISTS vec_spans ("
        "point_id TEXT PRIMARY KEY,"
        "capture_id TEXT NOT NULL,"
        "span_key TEXT NOT NULL,"
        "embedding_model TEXT NOT NULL,"
        "vector BLOB NOT NULL,"
        "norm REAL NOT NULL,"
        "signature INTEGER NOT NULL,"
        "bucket INTEGER NOT NULL,"
        "app_name TEXT,"
        "domain TEXT,"
        "payload_json TEXT"
        ")"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_vec_spans_capture ON vec_spans(capture_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_vec_spans_model ON vec_spans(embedding_model)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_vec_spans_bucket ON vec_spans(bucket)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_vec_spans_app ON vec_spans(app_name)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_vec_spans_domain ON vec_spans(domain)")

    op.execute(
        "CREATE TABLE IF NOT EXISTS vec_spans_v2_meta ("
        "span_id TEXT PRIMARY KEY,"
        "capture_id TEXT NOT NULL,"
        "span_key TEXT NOT NULL,"
        "embedding_model TEXT NOT NULL,"
        "app TEXT,"
        "domain TEXT,"
        "window_title TEXT,"
        "frame_id TEXT,"
        "frame_hash TEXT,"
        "ts TEXT,"
        "bbox_norm_json TEXT,"
        "text TEXT,"
        "tags_json TEXT"
        ")"
    )
    op.execute(
        "CREATE TABLE IF NOT EXISTS vec_spans_v2_dense ("
        "span_id TEXT PRIMARY KEY,"
        "vector BLOB NOT NULL,"
        "norm REAL NOT NULL,"
        "signature INTEGER NOT NULL,"
        "bucket INTEGER NOT NULL,"
        "FOREIGN KEY(span_id) REFERENCES vec_spans_v2_meta(span_id) ON DELETE CASCADE"
        ")"
    )
    op.execute(
        "CREATE TABLE IF NOT EXISTS vec_spans_v2_sparse ("
        "span_id TEXT NOT NULL,"
        "token_id INTEGER NOT NULL,"
        "weight REAL NOT NULL,"
        "PRIMARY KEY (span_id, token_id),"
        "FOREIGN KEY(span_id) REFERENCES vec_spans_v2_meta(span_id) ON DELETE CASCADE"
        ")"
    )
    op.execute(
        "CREATE TABLE IF NOT EXISTS vec_spans_v2_late ("
        "span_id TEXT NOT NULL,"
        "token_index INTEGER NOT NULL,"
        "vector BLOB NOT NULL,"
        "norm REAL NOT NULL,"
        "PRIMARY KEY (span_id, token_index),"
        "FOREIGN KEY(span_id) REFERENCES vec_spans_v2_meta(span_id) ON DELETE CASCADE"
        ")"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_capture ON vec_spans_v2_meta(capture_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_model ON vec_spans_v2_meta(embedding_model)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_app ON vec_spans_v2_meta(app)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_domain ON vec_spans_v2_meta(domain)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_bucket ON vec_spans_v2_dense(bucket)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_sparse_token ON vec_spans_v2_sparse(token_id)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_v2_sparse_token")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_v2_bucket")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_v2_domain")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_v2_app")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_v2_model")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_v2_capture")
    op.execute("DROP TABLE IF EXISTS vec_spans_v2_late")
    op.execute("DROP TABLE IF EXISTS vec_spans_v2_sparse")
    op.execute("DROP TABLE IF EXISTS vec_spans_v2_dense")
    op.execute("DROP TABLE IF EXISTS vec_spans_v2_meta")

    op.execute("DROP INDEX IF EXISTS ix_vec_spans_domain")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_app")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_bucket")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_model")
    op.execute("DROP INDEX IF EXISTS ix_vec_spans_capture")
    op.execute("DROP TABLE IF EXISTS vec_spans")
