"""Worker lease columns and HNSW unique mapping."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_worker_leases_and_hnsw_unique"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("captures", sa.Column("ocr_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("captures", sa.Column("ocr_heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("captures", sa.Column("ocr_attempts", sa.Integer(), server_default="0", nullable=False))
    op.add_column("captures", sa.Column("ocr_last_error", sa.Text(), nullable=True))
    op.add_column("embeddings", sa.Column("attempts", sa.Integer(), server_default="0", nullable=False))
    op.add_column("embeddings", sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("embeddings", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    with op.batch_alter_table("hnsw_mapping") as batch:
        batch.create_unique_constraint("uq_hnsw_event_span", ["event_id", "span_key"])


def downgrade() -> None:
    with op.batch_alter_table("hnsw_mapping") as batch:
        batch.drop_constraint("uq_hnsw_event_span", type_="unique")
    op.drop_column("embeddings", "heartbeat_at")
    op.drop_column("embeddings", "processing_started_at")
    op.drop_column("embeddings", "attempts")
    op.drop_column("captures", "ocr_last_error")
    op.drop_column("captures", "ocr_attempts")
    op.drop_column("captures", "ocr_heartbeat_at")
    op.drop_column("captures", "ocr_started_at")
