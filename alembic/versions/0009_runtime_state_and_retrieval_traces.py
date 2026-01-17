"""Add runtime state and retrieval trace tables."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0009_runtime_state_and_retrieval_traces"
down_revision = "0008_threads_and_focus_paths"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrieval_traces",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("rewrites_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("fused_results_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "runtime_state",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("current_mode", sa.String(length=32), nullable=False),
        sa.Column("pause_reason", sa.String(length=64), nullable=True),
        sa.Column("since_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_fullscreen_hwnd", sa.String(length=64), nullable=True),
        sa.Column("last_fullscreen_process", sa.String(length=256), nullable=True),
        sa.Column("last_fullscreen_title", sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("runtime_state")
    op.drop_table("retrieval_traces")
