"""Add URL/domain metadata to captures.

Revision ID: 0017_capture_url_domain
Revises: 0016_sqlite_vector_backends
Create Date: 2026-01-21
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0017_capture_url_domain"
down_revision = "0016_sqlite_vector_backends"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("captures", sa.Column("url", sa.String(length=1024), nullable=True))
    op.add_column("captures", sa.Column("domain", sa.String(length=256), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("captures") as batch:
        batch.drop_column("domain")
        batch.drop_column("url")
