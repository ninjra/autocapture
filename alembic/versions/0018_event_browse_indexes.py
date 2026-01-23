"""Add browse-friendly indexes for events.

Revision ID: 0018_event_browse_indexes
Revises: 0017_capture_url_domain
Create Date: 2026-01-23
"""

from __future__ import annotations

from alembic import op

revision = "0018_event_browse_indexes"
down_revision = "0017_capture_url_domain"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_events_app_name", "events", ["app_name"])
    op.create_index("ix_events_domain", "events", ["domain"])
    op.create_index("ix_events_ts_start_app_name", "events", ["ts_start", "app_name"])
    op.create_index("ix_events_ts_start_domain", "events", ["ts_start", "domain"])


def downgrade() -> None:
    op.drop_index("ix_events_ts_start_domain", table_name="events")
    op.drop_index("ix_events_ts_start_app_name", table_name="events")
    op.drop_index("ix_events_domain", table_name="events")
    op.drop_index("ix_events_app_name", table_name="events")
