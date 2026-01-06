"""Initial schema."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("event_id", sa.String(length=36), primary_key=True),
        sa.Column("ts_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ts_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("app_name", sa.String(length=256), nullable=False),
        sa.Column("window_title", sa.String(length=512), nullable=False),
        sa.Column("url", sa.String(length=1024), nullable=True),
        sa.Column("domain", sa.String(length=256), nullable=True),
        sa.Column("screenshot_path", sa.Text(), nullable=True),
        sa.Column("screenshot_hash", sa.String(length=128), nullable=False),
        sa.Column("ocr_text", sa.Text(), nullable=False),
        sa.Column("ocr_spans", sa.JSON(), nullable=False),
        sa.Column("embedding_vector", sa.JSON(), nullable=True),
        sa.Column("embedding_status", sa.String(length=16), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_events_ts_start", "events", ["ts_start"])

    op.create_table(
        "entities",
        sa.Column("entity_id", sa.String(length=36), primary_key=True),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("canonical_name", sa.String(length=512), nullable=False),
        sa.Column("canonical_token", sa.String(length=64), nullable=False, unique=True),
        sa.Column("parent_entity_id", sa.String(length=36), nullable=True),
        sa.Column("attributes", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "entity_aliases",
        sa.Column("alias_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "entity_id",
            sa.String(length=36),
            sa.ForeignKey("entities.entity_id", ondelete="CASCADE"),
        ),
        sa.Column("alias_text", sa.String(length=512), nullable=False),
        sa.Column("alias_norm", sa.String(length=512), nullable=False),
        sa.Column("alias_type", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "daily_aggregates",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("day", sa.String(length=10), nullable=False),
        sa.Column("app_name", sa.String(length=256), nullable=False),
        sa.Column("domain", sa.String(length=256), nullable=True),
        sa.Column("metric_name", sa.String(length=64), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("derived_from", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_daily_aggregates_day", "daily_aggregates", ["day"])

    op.create_table(
        "prompt_library",
        sa.Column("prompt_id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("raw_template", sa.Text(), nullable=False),
        sa.Column("derived_template", sa.Text(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "prompt_ops_runs",
        sa.Column("run_id", sa.String(length=64), primary_key=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sources_fetched", sa.JSON(), nullable=False),
        sa.Column("proposals", sa.JSON(), nullable=False),
        sa.Column("eval_results", sa.JSON(), nullable=False),
        sa.Column("pr_url", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
    )

    op.create_table(
        "captures",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("foreground_process", sa.String(length=256), nullable=False),
        sa.Column("foreground_window", sa.String(length=512), nullable=False),
        sa.Column("monitor_id", sa.String(length=64), nullable=False),
        sa.Column("is_fullscreen", sa.Boolean(), nullable=False),
        sa.Column("ocr_status", sa.String(length=16), nullable=False),
    )
    op.create_index("ix_captures_captured_at", "captures", ["captured_at"])

    op.create_table(
        "ocr_spans",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "capture_id",
            sa.String(length=36),
            sa.ForeignKey("captures.id", ondelete="CASCADE"),
        ),
        sa.Column("span_key", sa.String(length=64), nullable=False),
        sa.Column("start", sa.Integer(), nullable=False),
        sa.Column("end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=False),
    )

    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "capture_id",
            sa.String(length=36),
            sa.ForeignKey("captures.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "span_id", sa.Integer(), sa.ForeignKey("ocr_spans.id", ondelete="SET NULL")
        ),
        sa.Column("vector", sa.JSON(), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("span_key", sa.String(length=64), nullable=True),
    )

    op.create_table(
        "segments",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("video_path", sa.Text(), nullable=True),
        sa.Column("encoder", sa.String(length=64), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=True),
    )

    op.create_table(
        "observations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column(
            "segment_id",
            sa.String(length=36),
            sa.ForeignKey("segments.id", ondelete="SET NULL"),
        ),
        sa.Column("cursor_x", sa.Integer(), nullable=False),
        sa.Column("cursor_y", sa.Integer(), nullable=False),
        sa.Column("monitor_id", sa.String(length=64), nullable=False),
    )

    op.create_table(
        "query_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_query_history_normalized_text", "query_history", ["normalized_text"]
    )


def downgrade() -> None:
    op.drop_index("ix_query_history_normalized_text", table_name="query_history")
    op.drop_table("query_history")
    op.drop_table("observations")
    op.drop_table("segments")
    op.drop_table("embeddings")
    op.drop_table("ocr_spans")
    op.drop_index("ix_captures_captured_at", table_name="captures")
    op.drop_table("captures")
    op.drop_table("prompt_ops_runs")
    op.drop_table("prompt_library")
    op.drop_index("ix_daily_aggregates_day", table_name="daily_aggregates")
    op.drop_table("daily_aggregates")
    op.drop_table("entity_aliases")
    op.drop_table("entities")
    op.drop_index("ix_events_ts_start", table_name="events")
    op.drop_table("events")
