"""Indexes and idempotency constraints."""

from __future__ import annotations

from alembic import op

revision = "0003_indexes_and_uniques"
down_revision = "0002_worker_leases_and_hnsw_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_captures_ocr_status", "captures", ["ocr_status"])
    op.create_index(
        "ix_captures_ocr_status_heartbeat",
        "captures",
        ["ocr_status", "ocr_heartbeat_at"],
    )
    op.create_index("ix_embeddings_status", "embeddings", ["status"])
    op.create_index(
        "ix_embeddings_status_heartbeat",
        "embeddings",
        ["status", "heartbeat_at"],
    )
    with op.batch_alter_table("ocr_spans") as batch:
        batch.create_unique_constraint(
            "uq_ocr_spans_capture_span_key", ["capture_id", "span_key"]
        )
    with op.batch_alter_table("embeddings") as batch:
        batch.create_unique_constraint(
            "uq_embeddings_capture_span_model",
            ["capture_id", "span_key", "model"],
        )


def downgrade() -> None:
    with op.batch_alter_table("embeddings") as batch:
        batch.drop_constraint("uq_embeddings_capture_span_model", type_="unique")
    with op.batch_alter_table("ocr_spans") as batch:
        batch.drop_constraint("uq_ocr_spans_capture_span_key", type_="unique")
    op.drop_index("ix_embeddings_status_heartbeat", table_name="embeddings")
    op.drop_index("ix_embeddings_status", table_name="embeddings")
    op.drop_index("ix_captures_ocr_status_heartbeat", table_name="captures")
    op.drop_index("ix_captures_ocr_status", table_name="captures")
