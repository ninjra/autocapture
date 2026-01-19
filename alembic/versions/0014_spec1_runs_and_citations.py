"""SPEC-1 run logging, claim citations, and chunks FTS.

Revision ID: 0014_spec1_runs_and_citations
Revises: 0013_spec1_gateway_claims
Create Date: 2026-01-19
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0014_spec1_runs_and_citations"
down_revision = "0013_spec1_gateway_claims"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("answer_claim_citations", sa.Column("line_start", sa.Integer(), nullable=True))
    op.add_column("answer_claim_citations", sa.Column("line_end", sa.Integer(), nullable=True))
    op.add_column("answer_claim_citations", sa.Column("confidence", sa.Float(), nullable=True))

    op.create_table(
        "request_run",
        sa.Column("request_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "query_id",
            sa.String(length=128),
            sa.ForeignKey("query_records.query_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="started"),
        sa.Column("warnings_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "stage_run",
        sa.Column("run_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "request_id",
            sa.String(length=128),
            sa.ForeignKey("request_run.request_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("provider_id", sa.String(length=128), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=True),
        sa.Column("attempt_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_stage_run_request", "stage_run", ["request_id"])
    op.create_index("ix_stage_run_stage", "stage_run", ["stage"])

    op.create_table(
        "retrieval_run",
        sa.Column("run_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "request_id",
            sa.String(length=128),
            sa.ForeignKey("request_run.request_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "query_id",
            sa.String(length=128),
            sa.ForeignKey("query_records.query_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("k", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("result_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("engine_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_retrieval_run_request", "retrieval_run", ["request_id"])

    op.create_table(
        "provider_health",
        sa.Column("provider_id", sa.String(length=128), primary_key=True),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("circuit_open_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "evidence_items",
        sa.Column("item_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "request_id",
            sa.String(length=128),
            sa.ForeignKey("request_run.request_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "query_id",
            sa.String(length=128),
            sa.ForeignKey("query_records.query_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("evidence_id", sa.String(length=32), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("line_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("injection_risk", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("citable", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("kind", sa.String(length=32), nullable=False, server_default="source"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evidence_items_request", "evidence_items", ["request_id"])

    op.create_table(
        "claims",
        sa.Column("claim_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "request_id",
            sa.String(length=128),
            sa.ForeignKey("request_run.request_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("entailment_verdict", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "claim_citations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "claim_id",
            sa.String(length=128),
            sa.ForeignKey("claims.claim_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("evidence_id", sa.String(length=32), nullable=False),
        sa.Column("line_start", sa.Integer(), nullable=False),
        sa.Column("line_end", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_claim_citations_claim", "claim_citations", ["claim_id"])

    op.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
        "USING fts5(chunk_id UNINDEXED, event_id UNINDEXED, text)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks_fts")
    op.drop_index("ix_claim_citations_claim", table_name="claim_citations")
    op.drop_table("claim_citations")
    op.drop_table("claims")
    op.drop_index("ix_evidence_items_request", table_name="evidence_items")
    op.drop_table("evidence_items")
    op.drop_table("provider_health")
    op.drop_index("ix_retrieval_run_request", table_name="retrieval_run")
    op.drop_table("retrieval_run")
    op.drop_index("ix_stage_run_stage", table_name="stage_run")
    op.drop_index("ix_stage_run_request", table_name="stage_run")
    op.drop_table("stage_run")
    op.drop_table("request_run")
    op.drop_column("answer_claim_citations", "confidence")
    op.drop_column("answer_claim_citations", "line_end")
    op.drop_column("answer_claim_citations", "line_start")
