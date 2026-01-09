"""Event embedding lease columns.

Adds lease/heartbeat fields to `events` so crashed embedding workers can recover
rows stuck in `embedding_status='processing'`.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_event_embedding_leases"
down_revision = "0005_drop_vector_mapping"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column("embedding_attempts", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "events",
        sa.Column("embedding_last_error", sa.Text(), nullable=True),
    )
    op.add_column(
        "events",
        sa.Column("embedding_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "events",
        sa.Column("embedding_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_events_embedding_status_heartbeat",
        "events",
        ["embedding_status", "embedding_heartbeat_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_events_embedding_status_heartbeat", table_name="events")
    op.drop_column("events", "embedding_heartbeat_at")
    op.drop_column("events", "embedding_started_at")
    op.drop_column("events", "embedding_last_error")
    op.drop_column("events", "embedding_attempts")
