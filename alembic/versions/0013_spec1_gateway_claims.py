"""SPEC-1 claim/line-map/provider call tables.

Revision ID: 0013_spec1_gateway_claims
Revises: 0012_next10_contracts
Create Date: 2026-01-19
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0013_spec1_gateway_claims"
down_revision = "0012_next10_contracts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "answer_claims",
        sa.Column("claim_id", sa.String(length=128), primary_key=True),
        sa.Column("answer_id", sa.String(length=128), sa.ForeignKey("answer_records.answer_id", ondelete="CASCADE")),
        sa.Column("claim_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("entailment_verdict", sa.String(length=32), nullable=True),
        sa.Column("entailment_rationale", sa.Text(), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("answer_id", "claim_id", name="uq_answer_claims_answer_claim"),
    )
    op.create_index("ix_answer_claims_answer", "answer_claims", ["answer_id"]) 

    op.create_table(
        "answer_claim_citations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("claim_id", sa.String(length=128), sa.ForeignKey("answer_claims.claim_id", ondelete="CASCADE")),
        sa.Column("span_id", sa.String(length=128), sa.ForeignKey("citable_spans.span_id", ondelete="SET NULL"), nullable=True),
        sa.Column("evidence_id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("claim_id", "span_id", "evidence_id", name="uq_answer_claim_citations"),
    )
    op.create_index("ix_answer_claim_citations_claim", "answer_claim_citations", ["claim_id"]) 

    op.create_table(
        "evidence_line_map",
        sa.Column("map_id", sa.String(length=128), primary_key=True),
        sa.Column("query_id", sa.String(length=128), sa.ForeignKey("query_records.query_id", ondelete="CASCADE")),
        sa.Column("evidence_id", sa.String(length=32), nullable=False),
        sa.Column("span_id", sa.String(length=128), sa.ForeignKey("citable_spans.span_id", ondelete="SET NULL"), nullable=True),
        sa.Column("line_count", sa.Integer(), nullable=False),
        sa.Column("line_offsets_json", sa.JSON(), nullable=False),
        sa.Column("text_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("query_id", "evidence_id", name="uq_evidence_line_map_query_evidence"),
    )
    op.create_index("ix_evidence_line_map_query", "evidence_line_map", ["query_id"]) 

    op.create_table(
        "provider_calls",
        sa.Column("call_id", sa.String(length=128), primary_key=True),
        sa.Column("query_id", sa.String(length=128), sa.ForeignKey("query_records.query_id", ondelete="SET NULL"), nullable=True),
        sa.Column("answer_id", sa.String(length=128), sa.ForeignKey("answer_records.answer_id", ondelete="SET NULL"), nullable=True),
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
    op.create_index("ix_provider_calls_stage", "provider_calls", ["stage"]) 
    op.create_index("ix_provider_calls_query", "provider_calls", ["query_id"]) 


def downgrade() -> None:
    op.drop_index("ix_provider_calls_query", table_name="provider_calls")
    op.drop_index("ix_provider_calls_stage", table_name="provider_calls")
    op.drop_table("provider_calls")
    op.drop_index("ix_evidence_line_map_query", table_name="evidence_line_map")
    op.drop_table("evidence_line_map")
    op.drop_index("ix_answer_claim_citations_claim", table_name="answer_claim_citations")
    op.drop_table("answer_claim_citations")
    op.drop_index("ix_answer_claims_answer", table_name="answer_claims")
    op.drop_table("answer_claims")
