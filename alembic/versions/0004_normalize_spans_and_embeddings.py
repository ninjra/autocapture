"""Normalize spans and embeddings."""

from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa

revision = "0004_normalize_spans_and_embeddings"
down_revision = "0003_indexes_and_uniques"
branch_labels = None
depends_on = None


def _has_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    return any(col["name"] == column for col in inspector.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    dialect = bind.dialect.name

    if "ocr_spans" in inspector.get_table_names():
        with op.batch_alter_table("ocr_spans") as batch:
            batch.create_index("ix_ocr_spans_capture_id", ["capture_id"])

    if _has_column(inspector, "events", "ocr_spans"):
        spans_table = sa.table(
            "ocr_spans",
            sa.column("capture_id", sa.String),
            sa.column("span_key", sa.String),
            sa.column("start", sa.Integer),
            sa.column("end", sa.Integer),
            sa.column("text", sa.Text),
            sa.column("confidence", sa.Float),
            sa.column("bbox", sa.JSON),
        )

        def _parse_spans(raw):
            if raw is None:
                return []
            if isinstance(raw, list):
                return raw
            if isinstance(raw, str):
                raw = raw.strip()
                if not raw:
                    return []
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return []
            return []

        offset = 0
        chunk_size = 200
        while True:
            rows = bind.execute(
                sa.text(
                    "SELECT event_id, ocr_spans FROM events "
                    "WHERE ocr_spans IS NOT NULL AND ocr_spans != '[]' "
                    "LIMIT :limit OFFSET :offset"
                ),
                {"limit": chunk_size, "offset": offset},
            ).fetchall()
            if not rows:
                break
            insert_rows = []
            for event_id, raw_spans in rows:
                for span in _parse_spans(raw_spans):
                    span_key = str(span.get("span_key") or span.get("span_id") or "")
                    if not span_key:
                        continue
                    insert_rows.append(
                        {
                            "capture_id": str(event_id),
                            "span_key": span_key,
                            "start": int(span.get("start", 0)),
                            "end": int(span.get("end", 0)),
                            "text": str(span.get("text", "")),
                            "confidence": float(span.get("conf", span.get("confidence", 0.0))),
                            "bbox": span.get("bbox", []),
                        }
                    )
            if insert_rows:
                if dialect == "sqlite":
                    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                    stmt = sqlite_insert(spans_table).values(insert_rows)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["capture_id", "span_key"]
                    )
                    bind.execute(stmt)
                elif dialect == "postgresql":
                    from sqlalchemy.dialects.postgresql import insert as pg_insert

                    stmt = pg_insert(spans_table).values(insert_rows)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["capture_id", "span_key"]
                    )
                    bind.execute(stmt)
                else:
                    bind.execute(spans_table.insert(), insert_rows)
            offset += chunk_size

    if _has_column(inspector, "embeddings", "span_id"):
        bind.execute(
            sa.text(
                "UPDATE embeddings "
                "SET span_key = ("
                "  SELECT span_key FROM ocr_spans WHERE ocr_spans.id = embeddings.span_id"
                ") "
                "WHERE span_key IS NULL AND span_id IS NOT NULL"
            )
        )

    bind.execute(
        sa.text(
            "DELETE FROM embeddings "
            "WHERE span_key IS NULL "
            "   OR NOT EXISTS ("
            "       SELECT 1 FROM ocr_spans "
            "       WHERE ocr_spans.capture_id = embeddings.capture_id "
            "         AND ocr_spans.span_key = embeddings.span_key"
            "   )"
        )
    )

    if _has_column(inspector, "events", "ocr_spans"):
        with op.batch_alter_table("events") as batch:
            batch.drop_column("ocr_spans")

    if _has_column(inspector, "embeddings", "span_id"):
        with op.batch_alter_table("embeddings") as batch:
            batch.drop_column("span_id")

    if _has_column(inspector, "embeddings", "span_key"):
        with op.batch_alter_table("embeddings") as batch:
            batch.alter_column(
                "span_key",
                existing_type=sa.String(length=64),
                nullable=False,
            )
            batch.create_foreign_key(
                "fk_embeddings_capture_span",
                "ocr_spans",
                ["capture_id", "span_key"],
                ["capture_id", "span_key"],
                ondelete="CASCADE",
            )

    if _has_column(inspector, "hnsw_mapping", "span_id"):
        with op.batch_alter_table("hnsw_mapping") as batch:
            batch.drop_column("span_id")


def downgrade() -> None:
    with op.batch_alter_table("hnsw_mapping") as batch:
        batch.add_column(sa.Column("span_id", sa.Integer(), nullable=True))

    with op.batch_alter_table("embeddings") as batch:
        batch.drop_constraint("fk_embeddings_capture_span", type_="foreignkey")
        batch.alter_column("span_key", existing_type=sa.String(length=64), nullable=True)
        batch.add_column(sa.Column("span_id", sa.Integer(), nullable=True))

    with op.batch_alter_table("events") as batch:
        batch.add_column(sa.Column("ocr_spans", sa.JSON(), nullable=True))

    op.drop_index("ix_ocr_spans_capture_id", table_name="ocr_spans")
