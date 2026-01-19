"""Next-10 contracts, ledger, and tiering tables."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0012_next10_contracts"
down_revision = "0011_overlay_tracker"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "schema_migrations",
        sa.Column("version", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (1, CURRENT_TIMESTAMP)")

    op.create_table(
        "frame_records",
        sa.Column("frame_id", sa.String(length=36), primary_key=True),
        sa.Column("event_id", sa.String(length=36), nullable=True),
        sa.Column("captured_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("monotonic_ts", sa.Float(), nullable=False),
        sa.Column("monitor_id", sa.String(length=64), nullable=True),
        sa.Column("monitor_bounds", sa.JSON(), nullable=True),
        sa.Column("app_name", sa.String(length=256), nullable=True),
        sa.Column("window_title", sa.String(length=512), nullable=True),
        sa.Column("media_path", sa.Text(), nullable=True),
        sa.Column("privacy_flags", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("frame_hash", sa.String(length=128), nullable=True),
        sa.Column("excluded", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("masked", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.event_id"], ondelete="SET NULL"),
    )
    op.create_index("ix_frame_records_event_id", "frame_records", ["event_id"])
    op.create_index(
        "ix_frame_records_captured_at",
        "frame_records",
        ["captured_at_utc"],
    )

    op.create_table(
        "artifact_records",
        sa.Column("artifact_id", sa.String(length=128), primary_key=True),
        sa.Column("frame_id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("engine", sa.String(length=64), nullable=True),
        sa.Column("engine_version", sa.String(length=64), nullable=True),
        sa.Column("derived_from", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "upstream_artifact_ids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["frame_id"], ["frame_records.frame_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["events.event_id"], ondelete="SET NULL"),
    )
    op.create_index("ix_artifact_records_frame", "artifact_records", ["frame_id"])

    op.create_table(
        "citable_spans",
        sa.Column("span_id", sa.String(length=128), primary_key=True),
        sa.Column("artifact_id", sa.String(length=128), nullable=False),
        sa.Column("frame_id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=True),
        sa.Column("span_hash", sa.String(length=128), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("bbox_norm", sa.JSON(), nullable=True),
        sa.Column("tombstoned", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("expires_at_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("legacy_span_key", sa.String(length=64), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["artifact_records.artifact_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["frame_id"],
            ["frame_records.frame_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["event_id"], ["events.event_id"], ondelete="SET NULL"),
        sa.UniqueConstraint("span_hash", name="uq_citable_spans_span_hash"),
    )
    op.create_index("ix_citable_spans_event", "citable_spans", ["event_id"])
    op.create_index("ix_citable_spans_frame", "citable_spans", ["frame_id"])

    op.create_table(
        "query_records",
        sa.Column("query_id", sa.String(length=128), primary_key=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("filters_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("query_class", sa.String(length=64), nullable=True),
        sa.Column("budgets_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_query_records_created_at", "query_records", ["created_at"])

    op.create_table(
        "tier_plan_decisions",
        sa.Column("decision_id", sa.String(length=128), primary_key=True),
        sa.Column("query_id", sa.String(length=128), nullable=False),
        sa.Column("plan_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("skipped_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("reasons_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("budgets_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["query_id"], ["query_records.query_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_tier_plan_decisions_query", "tier_plan_decisions", ["query_id"])

    op.create_table(
        "retrieval_hits",
        sa.Column("hit_id", sa.String(length=128), primary_key=True),
        sa.Column("query_id", sa.String(length=128), nullable=False),
        sa.Column("tier", sa.String(length=32), nullable=False),
        sa.Column("span_id", sa.String(length=128), nullable=True),
        sa.Column("event_id", sa.String(length=36), nullable=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("scores_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("citable", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["query_id"], ["query_records.query_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["span_id"], ["citable_spans.span_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["event_id"], ["events.event_id"], ondelete="SET NULL"),
    )
    op.create_index("ix_retrieval_hits_query", "retrieval_hits", ["query_id"])
    op.create_index("ix_retrieval_hits_tier_rank", "retrieval_hits", ["tier", "rank"])
    op.create_index("ix_retrieval_hits_span", "retrieval_hits", ["span_id"])

    op.create_table(
        "answer_records",
        sa.Column("answer_id", sa.String(length=128), primary_key=True),
        sa.Column("query_id", sa.String(length=128), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("coverage_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("confidence_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("budgets_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("stale", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("answer_format_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["query_id"], ["query_records.query_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_answer_records_query", "answer_records", ["query_id"])

    op.create_table(
        "answer_citations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("answer_id", sa.String(length=128), nullable=False),
        sa.Column("sentence_id", sa.String(length=128), nullable=False),
        sa.Column("sentence_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("span_id", sa.String(length=128), nullable=True),
        sa.Column("citable", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["answer_id"], ["answer_records.answer_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["span_id"], ["citable_spans.span_id"], ondelete="SET NULL"),
        sa.UniqueConstraint("answer_id", "sentence_id", "span_id", name="uq_answer_citations"),
    )
    op.create_index("ix_answer_citations_answer", "answer_citations", ["answer_id"])

    op.create_table(
        "provenance_ledger_entries",
        sa.Column("entry_id", sa.String(length=128), primary_key=True),
        sa.Column("answer_id", sa.String(length=128), nullable=True),
        sa.Column("entry_type", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("prev_hash", sa.String(length=128), nullable=True),
        sa.Column("entry_hash", sa.String(length=128), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["answer_id"], ["answer_records.answer_id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_provenance_ledger_answer",
        "provenance_ledger_entries",
        ["answer_id"],
    )
    op.create_index(
        "ix_provenance_ledger_hash",
        "provenance_ledger_entries",
        ["entry_hash"],
    )

    op.create_table(
        "tier_stats",
        sa.Column("query_class", sa.String(length=64), primary_key=True),
        sa.Column("tier", sa.String(length=32), primary_key=True),
        sa.Column("window_n", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("help_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("p50_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("p95_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_provenance_append_only_update
        BEFORE UPDATE ON provenance_ledger_entries
        BEGIN
            SELECT RAISE(ABORT, 'append-only');
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_provenance_append_only_delete
        BEFORE DELETE ON provenance_ledger_entries
        BEGIN
            SELECT RAISE(ABORT, 'append-only');
        END;
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_provenance_append_only_delete")
    op.execute("DROP TRIGGER IF EXISTS trg_provenance_append_only_update")

    op.drop_table("tier_stats")
    op.drop_index("ix_provenance_ledger_hash", table_name="provenance_ledger_entries")
    op.drop_index("ix_provenance_ledger_answer", table_name="provenance_ledger_entries")
    op.drop_table("provenance_ledger_entries")

    op.drop_index("ix_answer_citations_answer", table_name="answer_citations")
    op.drop_table("answer_citations")

    op.drop_index("ix_answer_records_query", table_name="answer_records")
    op.drop_table("answer_records")

    op.drop_index("ix_retrieval_hits_span", table_name="retrieval_hits")
    op.drop_index("ix_retrieval_hits_tier_rank", table_name="retrieval_hits")
    op.drop_index("ix_retrieval_hits_query", table_name="retrieval_hits")
    op.drop_table("retrieval_hits")

    op.drop_index("ix_tier_plan_decisions_query", table_name="tier_plan_decisions")
    op.drop_table("tier_plan_decisions")

    op.drop_index("ix_query_records_created_at", table_name="query_records")
    op.drop_table("query_records")

    op.drop_index("ix_citable_spans_frame", table_name="citable_spans")
    op.drop_index("ix_citable_spans_event", table_name="citable_spans")
    op.drop_table("citable_spans")

    op.drop_index("ix_artifact_records_frame", table_name="artifact_records")
    op.drop_table("artifact_records")

    op.drop_index("ix_frame_records_captured_at", table_name="frame_records")
    op.drop_index("ix_frame_records_event_id", table_name="frame_records")
    op.drop_table("frame_records")

    op.drop_table("schema_migrations")
