"""Drop legacy vector mapping table if present."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "0005_drop_vector_mapping"
down_revision = "0004_normalize_spans_and_embeddings"
branch_labels = None
depends_on = None

TABLE_NAME = "h" "nsw_mapping"
INDEX_NAME = "ix_" + TABLE_NAME + "_event_id"
UNIQUE_NAME = "uq_" + TABLE_NAME + "_event_span"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE_NAME in inspector.get_table_names():
        op.drop_table(TABLE_NAME)


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if TABLE_NAME in inspector.get_table_names():
        return
    op.create_table(
        TABLE_NAME,
        sa.Column("label", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("span_key", sa.String(length=64), nullable=False),
    )
    op.create_index(INDEX_NAME, TABLE_NAME, ["event_id"])
    op.create_unique_constraint(UNIQUE_NAME, TABLE_NAME, ["event_id", "span_key"])
