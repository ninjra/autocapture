"""Add focus paths and thread summary tables."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0008_threads_and_focus_paths"
down_revision = "0007_agents_and_security"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("events") as batch:
        batch.add_column(sa.Column("focus_path", sa.Text(), nullable=True))
    with op.batch_alter_table("captures") as batch:
        batch.add_column(sa.Column("focus_path", sa.Text(), nullable=True))

    op.create_table(
        "threads",
        sa.Column("thread_id", sa.String(length=64), primary_key=True),
        sa.Column("ts_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ts_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("app_name", sa.String(length=256), nullable=False),
        sa.Column("window_title", sa.String(length=512), nullable=False),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_threads_ts_start", "threads", ["ts_start"])

    op.create_table(
        "thread_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "thread_id",
            sa.String(length=64),
            sa.ForeignKey("threads.thread_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "event_id",
            sa.String(length=36),
            sa.ForeignKey("events.event_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.UniqueConstraint("thread_id", "event_id", name="uq_thread_events_thread_event"),
    )
    op.create_index("ix_thread_events_event_id", "thread_events", ["event_id"])
    op.create_index("ix_thread_events_thread_id", "thread_events", ["thread_id"])

    op.create_table(
        "thread_summaries",
        sa.Column(
            "thread_id",
            sa.String(length=64),
            sa.ForeignKey("threads.thread_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("schema_version", sa.String(length=16), nullable=False),
        sa.Column("data_json", sa.JSON(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("thread_summaries")
    op.drop_index("ix_thread_events_thread_id", table_name="thread_events")
    op.drop_index("ix_thread_events_event_id", table_name="thread_events")
    op.drop_table("thread_events")
    op.drop_index("ix_threads_ts_start", table_name="threads")
    op.drop_table("threads")
    with op.batch_alter_table("captures") as batch:
        batch.drop_column("focus_path")
    with op.batch_alter_table("events") as batch:
        batch.drop_column("focus_path")
