"""SPEC-260118 Memory Service tables (Postgres-only).

Revision ID: 0015_spec260118_memory_service
Revises: 0014_spec1_runs_and_citations
Create Date: 2026-01-20
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0015_spec260118_memory_service"
down_revision = "0014_spec1_runs_and_citations"
branch_labels = None
depends_on = None


class Vector(sa.types.UserDefinedType):
    def __init__(self, dim: int) -> None:
        self._dim = int(dim)

    def get_col_spec(self, **_kw) -> str:  # pragma: no cover - used by Alembic DDL
        return f"vector({self._dim})"


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect != "postgresql":
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "artifact_versions",
        sa.Column("artifact_version_id", sa.String(length=128), primary_key=True),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("artifact_id", sa.String(length=128), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("labels_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("metadata_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_artifact_versions_namespace_hash",
        "artifact_versions",
        ["namespace", "content_hash"],
    )

    op.create_table(
        "artifact_chunks",
        sa.Column("chunk_id", sa.String(length=128), primary_key=True),
        sa.Column(
            "artifact_version_id",
            sa.String(length=128),
            sa.ForeignKey("artifact_versions.artifact_version_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("excerpt_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_artifact_chunks_version",
        "artifact_chunks",
        ["artifact_version_id"],
    )

    op.create_table(
        "memory_items",
        sa.Column("memory_id", sa.String(length=128), primary_key=True),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column("content_text", sa.Text(), nullable=False),
        sa.Column("content_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="validated"),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("trust_tier", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("audiences", sa.ARRAY(sa.String(length=64)), nullable=False),
        sa.Column("sensitivity", sa.String(length=32), nullable=False),
        sa.Column("sensitivity_rank", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("policy_labels", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memory_items_namespace", "memory_items", ["namespace"])
    op.create_index("ix_memory_items_type", "memory_items", ["memory_type"])
    op.create_index("ix_memory_items_status", "memory_items", ["status"])

    op.execute(
        "ALTER TABLE memory_items "
        "ADD COLUMN content_tsv tsvector GENERATED ALWAYS AS "
        "(to_tsvector('english', coalesce(content_text, ''))) STORED"
    )
    op.execute("CREATE INDEX ix_memory_items_tsv ON memory_items USING GIN (content_tsv)")

    op.create_table(
        "memory_embeddings",
        sa.Column(
            "memory_id",
            sa.String(length=128),
            sa.ForeignKey("memory_items.memory_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("embedding", Vector(256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("memory_id", "model"),
    )
    op.execute(
        "CREATE INDEX ix_memory_embeddings_hnsw ON memory_embeddings "
        "USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=128)"
    )

    op.create_table(
        "memory_provenance",
        sa.Column(
            "memory_id",
            sa.String(length=128),
            sa.ForeignKey("memory_items.memory_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "artifact_version_id",
            sa.String(length=128),
            sa.ForeignKey("artifact_versions.artifact_version_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            sa.String(length=128),
            sa.ForeignKey("artifact_chunks.chunk_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("excerpt_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("memory_id", "chunk_id"),
    )
    op.create_index("ix_memory_provenance_memory", "memory_provenance", ["memory_id"])

    op.create_table(
        "entities",
        sa.Column("entity_id", sa.String(length=128), primary_key=True),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("namespace", "kind", "name", name="uq_entities_namespace_kind_name"),
    )

    op.create_table(
        "memory_entities",
        sa.Column(
            "memory_id",
            sa.String(length=128),
            sa.ForeignKey("memory_items.memory_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "entity_id",
            sa.String(length=128),
            sa.ForeignKey("entities.entity_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("memory_id", "entity_id"),
    )

    op.create_table(
        "edges",
        sa.Column("edge_id", sa.String(length=128), primary_key=True),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column(
            "from_entity_id",
            sa.String(length=128),
            sa.ForeignKey("entities.entity_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "to_entity_id",
            sa.String(length=128),
            sa.ForeignKey("entities.entity_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relation", sa.String(length=64), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_edges_from", "edges", ["from_entity_id"])
    op.create_index("ix_edges_to", "edges", ["to_entity_id"])

    op.create_table(
        "memory_feedback",
        sa.Column("feedback_id", sa.String(length=128), primary_key=True),
        sa.Column("memory_id", sa.String(length=128), nullable=False),
        sa.Column("namespace", sa.String(length=128), nullable=False),
        sa.Column("useful", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memory_feedback_memory", "memory_feedback", ["memory_id"])


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect != "postgresql":
        return

    op.drop_index("ix_memory_feedback_memory", table_name="memory_feedback")
    op.drop_table("memory_feedback")
    op.drop_index("ix_edges_to", table_name="edges")
    op.drop_index("ix_edges_from", table_name="edges")
    op.drop_table("edges")
    op.drop_table("memory_entities")
    op.drop_table("entities")
    op.drop_index("ix_memory_provenance_memory", table_name="memory_provenance")
    op.drop_table("memory_provenance")
    op.execute("DROP INDEX IF EXISTS ix_memory_embeddings_hnsw")
    op.drop_table("memory_embeddings")
    op.execute("DROP INDEX IF EXISTS ix_memory_items_tsv")
    op.execute("ALTER TABLE memory_items DROP COLUMN IF EXISTS content_tsv")
    op.drop_index("ix_memory_items_status", table_name="memory_items")
    op.drop_index("ix_memory_items_type", table_name="memory_items")
    op.drop_index("ix_memory_items_namespace", table_name="memory_items")
    op.drop_table("memory_items")
    op.drop_index("ix_artifact_chunks_version", table_name="artifact_chunks")
    op.drop_table("artifact_chunks")
    op.drop_index("ix_artifact_versions_namespace_hash", table_name="artifact_versions")
    op.drop_table("artifact_versions")
