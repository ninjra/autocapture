"""Phase 0 contracts and backfill checkpoints."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010_phase0_contracts"
down_revision = "0009_runtime_state_and_retrieval_traces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column("frame_hash", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "events",
        sa.Column("ocr_text_normalized", sa.Text(), nullable=True),
    )

    op.add_column(
        "captures",
        sa.Column("event_id", sa.String(length=36), nullable=True),
    )
    op.create_index("ix_captures_event_id", "captures", ["event_id"])
    op.add_column(
        "captures",
        sa.Column("created_at_utc", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "captures",
        sa.Column("monotonic_ts", sa.Float(), nullable=True),
    )
    op.add_column(
        "captures",
        sa.Column("monitor_bounds", sa.JSON(), nullable=True),
    )
    op.add_column(
        "captures",
        sa.Column("privacy_flags", sa.JSON(), nullable=True, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "captures",
        sa.Column("frame_hash", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "captures",
        sa.Column("schema_version", sa.String(length=16), nullable=True, server_default="v1"),
    )

    op.add_column(
        "ocr_spans",
        sa.Column("engine", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "ocr_spans",
        sa.Column("frame_hash", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "ocr_spans",
        sa.Column("schema_version", sa.String(length=16), nullable=True, server_default="v1"),
    )

    op.add_column(
        "embeddings",
        sa.Column("frame_hash", sa.String(length=128), nullable=True),
    )

    op.create_table(
        "backfill_checkpoints",
        sa.Column("name", sa.String(length=64), primary_key=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )


def downgrade() -> None:
    op.drop_table("backfill_checkpoints")

    op.drop_column("embeddings", "frame_hash")

    op.drop_column("ocr_spans", "schema_version")
    op.drop_column("ocr_spans", "frame_hash")
    op.drop_column("ocr_spans", "engine")

    op.drop_column("captures", "schema_version")
    op.drop_column("captures", "frame_hash")
    op.drop_column("captures", "privacy_flags")
    op.drop_column("captures", "monitor_bounds")
    op.drop_column("captures", "monotonic_ts")
    op.drop_column("captures", "created_at_utc")
    op.drop_index("ix_captures_event_id", table_name="captures")
    op.drop_column("captures", "event_id")

    op.drop_column("events", "ocr_text_normalized")
    op.drop_column("events", "frame_hash")
