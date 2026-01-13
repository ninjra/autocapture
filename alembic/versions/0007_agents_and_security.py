"""Add agent jobs/results, highlights, token vault, and encryption tables."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0007_agents_and_security"
down_revision = "0006_event_embedding_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("job_key", sa.String(length=128), nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=True),
        sa.Column("day", sa.String(length=10), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("leased_by", sa.String(length=64), nullable=True),
        sa.Column("leased_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("result_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("job_key", name="uq_agent_jobs_job_key"),
    )
    op.create_index("ix_agent_jobs_status", "agent_jobs", ["status"])
    op.create_index("ix_agent_jobs_scheduled_for", "agent_jobs", ["scheduled_for"])
    op.create_index("ix_agent_jobs_type", "agent_jobs", ["job_type"])
    op.create_index("ix_agent_jobs_event_id", "agent_jobs", ["event_id"])

    op.create_table(
        "agent_results",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=True),
        sa.Column("day", sa.String(length=10), nullable=True),
        sa.Column("schema_version", sa.String(length=16), nullable=False),
        sa.Column("output_json", sa.JSON(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_results_type", "agent_results", ["job_type"])
    op.create_index("ix_agent_results_event", "agent_results", ["event_id"])
    op.create_index("ix_agent_results_day", "agent_results", ["day"])

    op.create_table(
        "event_enrichments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("result_id", sa.String(length=36), nullable=False),
        sa.Column("schema_version", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("event_id", name="uq_event_enrichments_event_id"),
    )
    op.create_index("ix_event_enrichments_event_id", "event_enrichments", ["event_id"])

    op.create_table(
        "daily_highlights",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("day", sa.String(length=10), nullable=False),
        sa.Column("schema_version", sa.String(length=16), nullable=False),
        sa.Column("data_json", sa.JSON(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("day", name="uq_daily_highlights_day"),
    )

    op.create_table(
        "token_vault",
        sa.Column("token", sa.String(length=64), primary_key=True),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("value_ciphertext", sa.Text(), nullable=False),
        sa.Column("value_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("token_vault")
    op.drop_table("daily_highlights")
    op.drop_index("ix_event_enrichments_event_id", table_name="event_enrichments")
    op.drop_table("event_enrichments")
    op.drop_index("ix_agent_results_day", table_name="agent_results")
    op.drop_index("ix_agent_results_event", table_name="agent_results")
    op.drop_index("ix_agent_results_type", table_name="agent_results")
    op.drop_table("agent_results")
    op.drop_index("ix_agent_jobs_event_id", table_name="agent_jobs")
    op.drop_index("ix_agent_jobs_type", table_name="agent_jobs")
    op.drop_index("ix_agent_jobs_scheduled_for", table_name="agent_jobs")
    op.drop_index("ix_agent_jobs_status", table_name="agent_jobs")
    op.drop_table("agent_jobs")
