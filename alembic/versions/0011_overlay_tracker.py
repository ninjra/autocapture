"""Overlay tracker tables."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0011_overlay_tracker"
down_revision = "0010_phase0_contracts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "overlay_projects",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", name="uq_overlay_projects_name"),
    )

    op.create_table(
        "overlay_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("display_name", sa.String(length=512), nullable=True),
        sa.Column("last_process_name", sa.String(length=256), nullable=False),
        sa.Column("last_window_title_raw", sa.String(length=512), nullable=True),
        sa.Column("last_browser_url_raw", sa.String(length=1024), nullable=True),
        sa.Column("identity_type", sa.String(length=16), nullable=True),
        sa.Column("identity_key", sa.String(length=512), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=False, server_default="idle"),
        sa.Column("last_activity_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("snooze_until_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["overlay_projects.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_overlay_items_last_activity", "overlay_items", ["last_activity_at_utc"])
    op.create_index(
        "ix_overlay_items_project_last_activity",
        "overlay_items",
        ["project_id", "last_activity_at_utc"],
    )
    op.create_index("ix_overlay_items_snooze_until", "overlay_items", ["snooze_until_utc"])
    op.create_index("ix_overlay_items_project_id", "overlay_items", ["project_id"])

    op.create_table(
        "overlay_item_identities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("identity_type", sa.String(length=16), nullable=False),
        sa.Column("identity_key", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["overlay_items.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "identity_type",
            "identity_key",
            name="uq_overlay_item_identities_key",
        ),
    )
    op.create_index(
        "ix_overlay_item_identities_item",
        "overlay_item_identities",
        ["item_id"],
    )

    op.create_table(
        "overlay_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("ts_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("process_name", sa.String(length=256), nullable=False),
        sa.Column("raw_window_title", sa.String(length=512), nullable=True),
        sa.Column("raw_browser_url", sa.String(length=1024), nullable=True),
        sa.Column("identity_type", sa.String(length=16), nullable=True),
        sa.Column("identity_key", sa.String(length=512), nullable=True),
        sa.Column("collector", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.String(length=16), nullable=False, server_default="v1"),
        sa.Column("app_version", sa.String(length=64), nullable=True),
        sa.Column(
            "payload_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.ForeignKeyConstraint(["item_id"], ["overlay_items.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["overlay_projects.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_overlay_events_ts", "overlay_events", ["ts_utc"])
    op.create_index(
        "ix_overlay_events_item_ts",
        "overlay_events",
        ["item_id", "ts_utc"],
    )
    op.create_index("ix_overlay_events_item_id", "overlay_events", ["item_id"])
    op.create_index("ix_overlay_events_project_id", "overlay_events", ["project_id"])

    op.create_table(
        "overlay_kv",
        sa.Column("key", sa.String(length=64), primary_key=True),
        sa.Column(
            "value_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("overlay_kv")

    op.drop_index("ix_overlay_events_project_id", table_name="overlay_events")
    op.drop_index("ix_overlay_events_item_id", table_name="overlay_events")
    op.drop_index("ix_overlay_events_item_ts", table_name="overlay_events")
    op.drop_index("ix_overlay_events_ts", table_name="overlay_events")
    op.drop_table("overlay_events")

    op.drop_index("ix_overlay_item_identities_item", table_name="overlay_item_identities")
    op.drop_table("overlay_item_identities")

    op.drop_index("ix_overlay_items_project_id", table_name="overlay_items")
    op.drop_index("ix_overlay_items_snooze_until", table_name="overlay_items")
    op.drop_index(
        "ix_overlay_items_project_last_activity",
        table_name="overlay_items",
    )
    op.drop_index("ix_overlay_items_last_activity", table_name="overlay_items")
    op.drop_table("overlay_items")

    op.drop_table("overlay_projects")
